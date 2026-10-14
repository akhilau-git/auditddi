import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import esm_embeddings as esm  # noqa: E402
from src.features import pubchem_names as pcn  # noqa: E402
from src.features import uniprot_sequences as ups  # noqa: E402
from src.features.drug_features import CYP_COLS, biology_known, build_cyp_block, build_store, coverage_report, feature_matrix  # noqa: E402


class R:
    def __init__(self, status=200, text="", js=None):
        self.status_code, self.text, self._js = status, text, js

    def json(self):
        return self._js


class FakeWeb:
    SEQ = {"P1": "MKT" * 20, "P2": "AAGG" * 10, "P3": "WWYY" * 8}

    def __init__(self, batch_ok=True):
        self.calls, self.batch_ok = [], batch_ok

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if "uniprotkb/accessions" in url:
            if not self.batch_ok:
                return R(500)
            return R(200, "".join(f">sp|{a}|{a}_HUMAN X OS=Homo sapiens OX=9606\n{self.SEQ[a]}\n" for a in params["accessions"].split(",") if a in self.SEQ))
        if url.endswith(".fasta"):
            a = url.split("/")[-1][:-6]
            return R(200, f">sp|{a}|{a}_HUMAN X OS=Homo sapiens OX=9606\n{self.SEQ[a]}\n") if a in self.SEQ else R(404)
        if "/property/Title/JSON" in url:
            key = url.split("/inchikey/")[1].split("/")[0]
            return R(404) if key == "KEYd1" else R(200, js={"PropertyTable": {"Properties": [{"CID": 7, "Title": f"Title-{key}"}]}})
        if "/synonyms/JSON" in url:
            return R(200, js={"InformationList": {"Information": [{"Synonym": ["syn1", "z" * 80]}]}})
        return R(404)


class TestUniprot(unittest.TestCase):
    def test_local_then_rest_and_missing_reported(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "fa").mkdir()
            (d / "fa" / "a.fasta").write_text(">sp|P1|P1_HUMAN X OS=Homo sapiens OX=9606\n" + "MKT" * 20 + "\n")
            pd.DataFrame({"drug_idx": [0, 1, 2, 3], "uniprot": ["P1", "P2", "P3", "P404"]}).to_csv(d / "t.csv", index=False)
            ups.main(["--targets", str(d / "t.csv"), "--local-fasta", str(d / "fa"), "--out", str(d / "o")], session=FakeWeb())
            rep = json.loads((d / "o" / "uniprot_report.json").read_text())
            self.assertEqual((rep["with_sequence"], rep["from_local"], rep["from_rest"], rep["missing"]), (3, 1, 2, 1))
            self.assertEqual(rep["missing_examples"], ["P404"])
            back = ups.parse_fasta((d / "o" / "target_sequences.fasta").read_text())
            self.assertEqual(back["P3"]["sequence"], "WWYY" * 8)                      # wrapped FASTA round-trips

    def test_batch_failure_falls_back_to_single_requests(self):
        got = ups.fetch(["P1", "P2"], FakeWeb(batch_ok=False), pause=0)
        self.assertEqual(set(got), {"P1", "P2"})

    def test_offline_never_calls_the_network(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame({"drug_idx": [0], "uniprot": ["P1"]}).to_csv(d / "t.csv", index=False)
            w = FakeWeb()
            ups.main(["--targets", str(d / "t.csv"), "--out", str(d / "o"), "--offline"], session=w)
            self.assertEqual(w.calls, [])


def fake_embedder(seqs):
    return np.stack([[len(s), s.count("A") + 1.0, float(sum(map(ord, s)) % 97)] for s in seqs]).astype(np.float32)


class TestEsm(unittest.TestCase):
    def test_cache_truncation_and_flags(self):
        seqs = {"P1": "MKT" * 20, "P2": "A" * 50, "P3": "W" * 30}
        calls = []

        def emb(x):
            calls.append(len(x))
            return fake_embedder(x)

        with tempfile.TemporaryDirectory() as d:
            c = Path(d) / "cache.npz"
            e1, r1 = esm.embed_proteins(seqs, emb, "m", c, max_len=40)
            self.assertEqual((r1["embedded_now"], r1["truncated"]), (3, 2))        # P1 (60) and P2 (50) exceed 40
            self.assertEqual(e1["P1"][0], 40)                                      # embedded the first 40 residues only
            e2, r2 = esm.embed_proteins(seqs, emb, "m", c, max_len=40)
            self.assertEqual((r2["embedded_now"], r2["from_cache"]), (0, 3))       # nothing recomputed
            self.assertEqual(calls, [3])
            e3, r3 = esm.embed_proteins(seqs, emb, "other-model", c, max_len=40)  # a different model must not reuse the cache
            self.assertEqual(r3["embedded_now"], 3)

    def test_drug_aggregation_weights_and_missing(self):
        emb = {"A": np.array([1.0, 0.0], np.float32), "B": np.array([0.0, 1.0], np.float32)}
        t = pd.DataFrame({"drug_idx": [0, 0, 1, 2], "uniprot": ["A", "B", "A", "NOSEQ"], "moa": [1, 0, 0, 1]})
        M, has = esm.drug_embeddings(4, t, emb, moa_weight=2.0)
        self.assertEqual(has.tolist(), [True, True, False, False])
        self.assertTrue(np.allclose(M[0], [2 / 3, 1 / 3]))                         # MOA target counts double
        self.assertTrue(np.allclose(M[1], [1.0, 0.0]))
        self.assertTrue(np.allclose(M[2], [0, 0]))                                  # a target without a sequence: zero vector, flagged unknown
        self.assertTrue(np.allclose(M[3], [0, 0]))

    def test_cli_with_injected_embedder(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "s.fasta").write_text(">P1\nMKTMKT\n>P2\nAAGG\n")
            pd.DataFrame({"drug_idx": [0, 1], "uniprot": ["P1", "P2"]}).to_csv(d / "t.csv", index=False)
            esm.main(["--sequences", str(d / "s.fasta"), "--targets", str(d / "t.csv"), "--n-drugs", "3", "--out", str(d / "o")], embedder=fake_embedder)
            with np.load(d / "o" / "drug_esm.npz") as z:
                self.assertEqual(z["esm"].shape, (3, 3))
                self.assertEqual(z["has_esm"].tolist(), [True, True, False])


class TestPubchem(unittest.TestCase):
    def test_names_cache_and_unknown_not_guessed(self):
        drugs = pd.DataFrame({"drug_idx": [0, 1], "smiles": ["a", "d1"]})
        with tempfile.TemporaryDirectory() as d:
            w = FakeWeb()
            key = lambda s: "KEY" + s  # noqa: E731
            df = pcn.fetch_names(drugs, w, Path(d) / "c.json", key, pause=0)
            self.assertEqual(df.set_index("drug_idx").loc[0, "title"], "Title-KEYa")
            self.assertTrue(pd.isna(df.set_index("drug_idx").loc[1, "title"]))      # 404 -> no title, no guess
            self.assertEqual(df.set_index("drug_idx").loc[0, "synonyms"], "syn1")   # over-long junk synonym dropped
            n = len(w.calls)
            pcn.fetch_names(drugs, w, Path(d) / "c.json", key, pause=0)
            self.assertEqual(len(w.calls), n)                                       # fully cached


class TestCypAndEsmBlocks(unittest.TestCase):
    def test_cyp_block_known_only_with_label_text(self):
        lf = pd.DataFrame({"drug_idx": [0, 2], "n_labels": [3, 1], "pk_text_chars": [500, 0], "cyp3A4_substrate": [1, 0], "cyp2D6_inhibitor": [0, 1]})
        M, has = build_cyp_block(4, lf)
        self.assertEqual(has.tolist(), [True, False, False, False])                  # drug 2 has a label but no pharmacology text: unknown
        self.assertEqual(int(M[0, CYP_COLS.index("cyp3A4_substrate")]), 1)

    def test_store_with_cyp_and_esm(self):
        drugs = pd.DataFrame({"drug_idx": range(3), "smiles": list("ABC")})
        tg = pd.DataFrame({"drug_idx": [0], "uniprot": ["P1"], "organism": "Homo sapiens"})
        lf = pd.DataFrame({"drug_idx": [1], "n_labels": [1], "pk_text_chars": [10], "cyp3A4_inhibitor": [1]})
        e = {"esm": np.ones((3, 5), np.float32), "has_esm": np.array([True, False, False])}
        st = build_store(drugs, np.ones((3, 4), np.uint8), np.ones(3, bool), tg, pd.DataFrame({"canonical_smiles": [], "genes_list": []}),
                         canon_fn=lambda x: x, cyp=lf, esm=e)
        rep = coverage_report(st)
        self.assertEqual((rep["has_cyp_label"], rep["has_esm"], rep["esm_dim"]), (1, 1, 5))
        self.assertEqual(biology_known(st).tolist(), [True, True, False])
        F = feature_matrix(st, ["ecfp", "cyp", "esm"])
        self.assertEqual(F.shape, (3, 4 + len(CYP_COLS) + 1 + 5 + 1))
        with self.assertRaises(AssertionError):
            build_store(drugs, np.ones((3, 4), np.uint8), np.ones(3, bool), tg, pd.DataFrame({"canonical_smiles": [], "genes_list": []}),
                        canon_fn=lambda x: x, esm={"esm": np.ones((2, 5)), "has_esm": np.array([True, False])})


if __name__ == "__main__":
    unittest.main(verbosity=2)
