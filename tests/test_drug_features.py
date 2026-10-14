import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.fingerprints import smiles_list_hash  # noqa: E402
from src.baselines.run_baselines import main as run_main  # noqa: E402
from src.data_prep.build_event_table import build_event_table, save_event_table  # noqa: E402
from src.data_prep.make_splits import main as splits_main  # noqa: E402
from src.features.drug_features import (  # noqa: E402
    biology_known,
    build_gene_block,
    build_store,
    build_target_block,
    coverage_report,
    feature_matrix,
    load_store,
    parse_gene_list,
    save_store,
    stratum_of_pairs,
)

ident = lambda s: s  # noqa: E731  (RDKit-free canonicaliser for tests)


class TestParsing(unittest.TestCase):
    def test_gene_list_formats(self):
        self.assertEqual(parse_gene_list('["CYP3A4","CYP2D6"]'), ["CYP2D6", "CYP3A4"])
        self.assertEqual(parse_gene_list("['CYP3A4', 'ABCB1']"), ["ABCB1", "CYP3A4"])
        self.assertEqual(parse_gene_list("CYP3A4;CYP2C9, UGT1A1"), ["CYP2C9", "CYP3A4", "UGT1A1"])
        self.assertEqual(parse_gene_list("[]"), [])
        self.assertEqual(parse_gene_list(None), [])
        self.assertEqual(parse_gene_list(float("nan")), [])


class TestBlocks(unittest.TestCase):
    def setUp(self):
        self.drugs = pd.DataFrame({"drug_idx": range(5), "smiles": list("ABCDE")})
        self.targets = pd.DataFrame({"drug_idx": [0, 0, 1, 3], "uniprot": ["P1", "P2", "P2", "P3"],
                                     "organism": ["Homo sapiens", "Homo sapiens", "Homo sapiens", "E. coli"]})
        self.prof = pd.DataFrame({"canonical_smiles": ["B", "C", "Z"], "genes_list": ['["G1","G2"]', "G2", '["G9"]']})

    def test_target_block_no_fallback_and_flags(self):
        M, vocab, has = build_target_block(5, self.targets)
        self.assertEqual(vocab, ["P1", "P2", "P3"])
        self.assertEqual(has.tolist(), [True, True, False, True, False])
        self.assertEqual(int(M[2].sum()), 0)            # drug 2 has NO target: stays all-zero, never filled
        self.assertEqual(int(M[4].sum()), 0)

    def test_human_only(self):
        M, vocab, has = build_target_block(5, self.targets, human_only=True)
        self.assertNotIn("P3", vocab)
        self.assertFalse(has[3])

    def test_gene_block_exact_match_and_unmatched_reported(self):
        M, vocab, has, meta = build_gene_block(self.drugs, self.prof, canon_fn=ident)
        self.assertEqual(vocab, ["G1", "G2"])
        self.assertEqual(has.tolist(), [False, True, True, False, False])
        self.assertEqual(meta["unmatched_profiles"], 1)   # 'Z' matches no drug; G9 is dropped, not guessed

    def test_store_roundtrip_hash_guard_and_report(self):
        ecfp = np.random.default_rng(0).integers(0, 2, (5, 16)).astype(np.uint8)
        st = build_store(self.drugs, ecfp, np.ones(5, bool), self.targets, self.prof, canon_fn=ident)
        rep = coverage_report(st)
        self.assertEqual((rep["has_target"], rep["has_gene"], rep["has_both"], rep["has_neither"]), (3, 2, 1, 1))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.npz"
            save_store(st, p)
            st2 = load_store(p, self.drugs)
            self.assertTrue(np.array_equal(st["target"], st2["target"]))
            with self.assertRaises(ValueError):
                load_store(p, pd.DataFrame({"drug_idx": range(5), "smiles": list("ABCDX")}))

    def test_feature_matrix_and_strata(self):
        ecfp = np.ones((5, 4), dtype=np.uint8)
        st = build_store(self.drugs, ecfp, np.ones(5, bool), self.targets, self.prof, canon_fn=ident)
        F = feature_matrix(st, ["ecfp", "target", "gene"])
        self.assertEqual(F.shape, (5, 4 + 3 + 1 + 2 + 1))
        known = biology_known(st)
        self.assertEqual(known.tolist(), [True, True, True, True, False])
        s = stratum_of_pairs(known, np.array([0, 0, 4]), np.array([1, 4, 4]))
        self.assertEqual(s.tolist(), [2, 1, 0])
        with self.assertRaises(ValueError):
            feature_matrix(st, ["pdb"])


def planted_bio(signal: str, seed=0, n_drugs=170, n_events=24, bits=48, n_targets=12):
    """signal = 'targets' : each drug's targets shift event risk (fingerprints are noise)
       signal = 'none'    : events depend on fingerprints only (targets are noise)"""
    rng = np.random.default_rng(seed)
    F = (rng.random((n_drugs, bits)) < 0.25).astype(np.uint8)
    has = rng.random(n_drugs) < 0.55
    T = np.zeros((n_drugs, n_targets), dtype=np.uint8)
    for i in np.where(has)[0]:
        T[i, rng.choice(n_targets, rng.integers(1, 4), replace=False)] = 1
    Wt = rng.normal(size=(n_targets, n_events)) * 1.6
    Wf = rng.normal(size=(bits, n_events)) * 0.5
    rows = []
    for i in range(n_drugs):
        for j in range(i + 1, n_drugs):
            if rng.random() < 0.45:
                z = -1.4 + ((((T[i] + T[j]) @ Wt) * 0.8) if signal == "targets" else ((F[i] + F[j]) @ Wf / 3))
                p = 1 / (1 + np.exp(-z))
                ev = np.where(rng.random(n_events) < p)[0]
                if len(ev) == 0:
                    ev = [int(rng.integers(n_events))]
                rows += [(f"D{i:03d}", f"D{j:03d}", f"ev{e}") for e in ev]
    t, _ = build_event_table(pd.DataFrame(rows, columns=["source", "target", "interaction_type"]))
    order = t.drugs.drug_idx.to_numpy()                      # drugs are sorted by SMILES string = D000.. in order
    return t, F[order], T[order], has[order]


class TestExperimentControls(unittest.TestCase):
    def _run(self, signal):
        t, F, T, has = planted_bio(signal)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_event_table(t, d / "table", {})
            splits_main(["--table", str(d / "table"), "--out", str(d / "splits"), "--seeds", "0", "--kinds", "cold_drug",
                         "--frac-test", "0.3", "--min-pairs-full", "120"])
            tg = pd.DataFrame([(i, f"P{k}", "Homo sapiens") for i in range(len(T)) for k in np.where(T[i])[0]],
                              columns=["drug_idx", "uniprot", "organism"])
            st = build_store(t.drugs, F, np.ones(len(F), bool), tg, pd.DataFrame({"canonical_smiles": [], "genes_list": []}), canon_fn=ident)
            save_store(st, d / "store.npz")
            out = d / "out"
            for feats in (["ecfp"], ["ecfp", "target"]):
                run_main(["--table", str(d / "table"), "--splits", str(d / "splits"), "--out", str(out), "--feature-store", str(d / "store.npz"),
                          "--features", *feats, "--names", "cold_drug_seed0", "--models", "logreg", "--epochs", "25", "--min-pos", "3"])
            res = pd.read_csv(out / "results.csv")
            return res

    @staticmethod
    def _auc(res, feats, part):
        r = res[(res.features == feats) & (res.partition == part)]
        return float(r.macro_auroc.iloc[0]) if len(r) else float("nan")

    def test_positive_control_detects_real_biology_signal(self):
        res = self._run("targets")
        self.assertGreater(self._auc(res, "ecfp+target", "test_s2"), self._auc(res, "ecfp", "test_s2") + 0.05)
        # and the gain lives where biology exists, not where it is missing
        both = self._auc(res, "ecfp+target", "test_s2|bio2")
        none = self._auc(res, "ecfp+target", "test_s2|bio0")
        if not np.isnan(none):
            self.assertGreater(both, none)

    def test_negative_control_does_not_invent_a_gain(self):
        res = self._run("none")
        gain = self._auc(res, "ecfp+target", "test_s2") - self._auc(res, "ecfp", "test_s2")
        self.assertLess(gain, 0.04)

    def test_resume_keys_separate_feature_sets(self):
        res = self._run("targets")
        self.assertEqual(set(res.features), {"ecfp", "ecfp+target"})
        self.assertTrue(res.stratified.all())



class TestEmptyVocabularyFailsLoudly(unittest.TestCase):
    def test_make_splits_refuses_empty_vocab(self):
        t, F, T, has = planted_bio("none", n_drugs=40)
        with tempfile.TemporaryDirectory() as d:
            save_event_table(t, Path(d) / "table", {})
            with self.assertRaises(ValueError):
                splits_main(["--table", str(Path(d) / "table"), "--out", str(Path(d) / "s"), "--seeds", "0",
                             "--kinds", "cold_drug", "--min-pairs-full", "999999"])


class TestBuildStoreCLI(unittest.TestCase):
    def test_cli_builds_store_and_reports_coverage(self):
        from src.features.build_feature_store import main as build_main

        t, F, T, has = planted_bio("targets", n_drugs=30)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_event_table(t, d / "table", {})
            np.savez_compressed(d / "ecfp.npz", bits=F, valid=np.ones(len(F), bool), smiles_sha256=np.array(smiles_list_hash(t.drugs["smiles"].tolist())))
            pd.DataFrame([(i, f"P{k}", "Homo sapiens") for i in range(len(T)) for k in np.where(T[i])[0]],
                         columns=["drug_idx", "uniprot", "organism"]).to_csv(d / "tg.csv", index=False)
            pd.DataFrame({"canonical_smiles": [t.drugs.smiles[0]], "genes_list": ['["CYP3A4"]']}).to_csv(d / "pg.csv", index=False)
            build_main(["--table", str(d / "table"), "--ecfp", str(d / "ecfp.npz"), "--targets", str(d / "tg.csv"),
                        "--pharmgkb", str(d / "pg.csv"), "--out", str(d / "f" / "store.npz"), "--identity-canon"])
            rep = __import__("json").loads((d / "f" / "store.json").read_text())
            self.assertEqual(rep["has_target"], int(has.sum()))
            self.assertEqual(rep["has_gene"], 1)
            self.assertIn("sha256", rep["inputs"]["targets"])
            # a store built for another drug list must be refused
            np.savez_compressed(d / "bad.npz", bits=F, valid=np.ones(len(F), bool), smiles_sha256=np.array("x"))
            with self.assertRaises(ValueError):
                build_main(["--table", str(d / "table"), "--ecfp", str(d / "bad.npz"), "--targets", str(d / "tg.csv"),
                            "--pharmgkb", str(d / "pg.csv"), "--out", str(d / "f2.npz"), "--identity-canon"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
