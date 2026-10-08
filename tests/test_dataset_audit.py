import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_audit as da  # noqa: E402

ident = lambda s: s  # noqa: E731


class Resp:
    def __init__(self, status=200, text="", js=None):
        self.status_code, self.text, self._js = status, text, js

    def json(self):
        if self._js is None:
            raise ValueError("no json")
        return self._js


class FakeSession:
    """Canned responses; counts calls so tests can check caching."""
    FASTA = {"P1": "MKT" * 10, "P2": "AAGG" * 8, "P3": "WWYY" * 5}

    def __init__(self, uniprot_batch_ok=True):
        self.calls = []
        self.batch_ok = uniprot_batch_ok

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if "uniprotkb/accessions" in url:
            if not self.batch_ok:
                return Resp(500)
            txt = ""
            for a in params["accessions"].split(","):
                if a in self.FASTA:
                    txt += f">sp|{a}|{a}_HUMAN Protein {a} OS=Homo sapiens OX=9606 GN=G{a} PE=1 SV=1\n{self.FASTA[a]}\n"
            return Resp(200, txt)
        if "uniprotkb/" in url and url.endswith(".fasta"):
            a = url.split("/")[-1][:-6]
            return Resp(200, f">sp|{a}|{a}_HUMAN P OS=Homo sapiens OX=9606\n{self.FASTA[a]}\n") if a in self.FASTA else Resp(404)
        if "/property/Title/JSON" in url:
            key = url.split("/inchikey/")[1].split("/")[0]
            smi = key.replace("KEY", "")
            if smi == "D004":
                return Resp(404)
            return Resp(200, js={"PropertyTable": {"Properties": [{"CID": 1000 + int(smi[1:]), "Title": f"Drug{smi[1:]}"}]}})
        if "/synonyms/JSON" in url:
            cid = int(url.split("/cid/")[1].split("/")[0])
            return Resp(200, js={"InformationList": {"Information": [{"CID": cid, "Synonym": [f"Alias{cid}", "x" * 90]}]}})
        if "rxnav" in url:
            return Resp(200, '{"idGroup":{"rxnormId":["1191"]}}')
        if "api.fda.gov" in url:
            return Resp(429)
        if "kegg" in url:
            raise TimeoutError("boom")
        return Resp(200, ">sp|P05067|A4_HUMAN\nMK" if "uniprot" in url else "ok")


def make_root(d: Path):
    from src.data_prep.build_event_table import build_event_table, save_event_table
    rows = [(f"D{i:03d}", f"D{j:03d}", f"ev{(i + j) % 4}") for i in range(6) for j in range(i + 1, 6)]
    t, rep = build_event_table(pd.DataFrame(rows, columns=["source", "target", "interaction_type"]))
    save_event_table(t, d / "event_table_v1", rep)
    (d / "pharmgkb").mkdir()
    for f in ("relationships.tsv", "chemicals.tsv", "genes.tsv"):
        (d / "pharmgkb" / f).write_text("a\tb\n")
    pd.DataFrame({"canonical_smiles": ["D000", "D001", "ZZZ"], "genes_list": ["[]", "[]", "[]"]}).to_csv(d / "pharmgkb_twosides_gene_profiles.csv", index=False)
    (d / "BindingDB").mkdir()
    pd.DataFrame({"source": ["D002", "D009"], "target": ["T1", "T2"], "affinity_type": ["Ki", "IC50"], "affinity_value": [1, 2]}).to_csv(d / "BindingDB" / "drug_target_edges.csv", index=False)
    (d / "PDB").mkdir()
    for i in range(3):
        (d / "PDB" / f"{i}.pdb").write_text("ATOM")
    (d / "faers" / "ASCII").mkdir(parents=True)
    (d / "faers" / "ASCII" / "DRUG23Q4.txt").write_text("x")
    (d / "GEO").mkdir()
    (d / "GEO" / "Brain_GSE1.txt.gz").write_text("x")
    (d / "p1_targets").mkdir()
    pd.DataFrame({"drug_idx": [0, 0, 1, 2], "uniprot": ["P1", "P2", "P3", "P9"], "organism": ["Homo sapiens"] * 4}).to_csv(d / "p1_targets" / "drug_targets_chembl_mechanism.csv", index=False)
    (d / "uniprot").mkdir()
    (d / "uniprot" / "a.fasta").write_text(">sp|P1|P1_HUMAN Protein 1 OS=Homo sapiens OX=9606 GN=A PE=1 SV=1\nMKTMKTMKT\nMKT\n")
    (d / "ddinter" / "raw").mkdir(parents=True)
    pd.DataFrame({"DDInterID_A": 1, "Drug_A": ["Drug000", "Drug000", "Drug001", "Other"], "DDInterID_B": 2,
                  "Drug_B": ["Drug001", "Drug002", "Drug007", "Drug000"], "Level": ["Major", "Minor", "Moderate", "Major"]}).to_csv(d / "ddinter" / "raw" / "ddinter_downloads_code_A.csv", index=False)
    pd.DataFrame({"DDInterID_A": 1, "Drug_A": ["drug003"], "DDInterID_B": 2, "Drug_B": ["DRUG001 "], "Level": ["Minor"]}).to_csv(d / "ddinter" / "raw" / "ddinter_downloads_code_B.csv", index=False)
    return t


class TestFasta(unittest.TestCase):
    def test_parse_and_scan(self):
        txt = ">sp|P12345|ABC_HUMAN Some protein OS=Homo sapiens OX=9606 GN=ABC PE=1 SV=2\nMKT\nLLA\n>tr|Q9XYZ1|Q9XYZ1_ECOLI Other OS=Escherichia coli OX=562 GN=z\nAAA\n"
        r = da.parse_fasta(txt)
        self.assertEqual(set(r), {"P12345", "Q9XYZ1"})
        self.assertEqual(r["P12345"]["length"], 6)
        self.assertEqual(r["P12345"]["organism"], "Homo sapiens")
        self.assertEqual(r["Q9XYZ1"]["organism"], "Escherichia coli")

    def test_fetch_batch_and_fallback(self):
        for ok in (True, False):
            s = FakeSession(uniprot_batch_ok=ok)
            got = da.fetch_uniprot_fasta(["P2", "P3", "NOPE"], s, batch=10, pause=0)
            self.assertEqual(set(got), {"P2", "P3"})
            self.assertEqual(got["P2"]["sequence"], "AAGG" * 8)


class TestSections(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.t = make_root(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_uniprot_section_counts_and_outputs(self):
        r = da.uniprot_section(self.root, FakeSession(), fetch=True)
        self.assertEqual(r["target_accessions_needed"], 4)
        self.assertEqual(r["already_in_local_fasta"], 1)          # P1
        self.assertEqual(r["fetched_now"], 2)                      # P2, P3
        self.assertEqual(r["still_missing"], ["P9"])               # unknown to UniProt: reported, never invented
        self.assertAlmostEqual(r["sequence_coverage_of_targets"], 0.75)
        self.assertTrue((self.root / "dataset_audit" / "chembl_targets_fetched.fasta").exists())
        back = da.parse_fasta((self.root / "dataset_audit" / "chembl_targets_fetched.fasta").read_text())
        self.assertEqual(back["P3"]["sequence"], "WWYY" * 5)       # FASTA wrapping round-trips

    def test_pubchem_names_cache_and_missing(self):
        s = FakeSession()
        key = lambda smi: "KEY" + smi  # noqa: E731
        df = da.pubchem_names(self.root, s, inchikey_fn=key, pause=0)
        self.assertEqual(len(df), 6)
        self.assertTrue(df[df.drug_idx == 4].title.isna().all())   # D004 -> 404 -> no title, no guess
        self.assertTrue(df[df.drug_idx == 0].synonyms.iloc[0].startswith("Alias1000"))
        self.assertNotIn("x" * 90, df.synonyms.iloc[0])            # over-long junk synonyms dropped
        n_calls = len(s.calls)
        da.pubchem_names(self.root, s, inchikey_fn=key, pause=0)   # fully cached: no new calls
        self.assertEqual(len(s.calls), n_calls)

    def test_ddinter_inventory_and_overlap(self):
        inv = da.ddinter_inventory(self.root)
        self.assertEqual(inv["letters_present"], ["A", "B"])
        self.assertIn("C", inv["letters_missing"])
        self.assertEqual(inv["rows"], 5)
        names = da.pubchem_names(self.root, FakeSession(), inchikey_fn=lambda s: "KEY" + s, pause=0)
        dd = da.load_ddinter_raw(self.root)
        o = da.ddinter_overlap(dd, names, self.t.pairs)
        # matches: Drug000->0, Drug001->1, Drug002->2, Drug003->3 ; Drug007/Other unknown
        # pairs with both ours: (0,1) twice (A row1 and B row), (0,2) ; B row 'drug003'-'DRUG001' -> (1,3)
        self.assertEqual(o["ddinter_pairs_with_both_drugs_ours"], 3)
        self.assertEqual(o["of_which_in_twosides_pairs"], 3)       # all pairs of the 6 synthetic drugs exist
        self.assertEqual(o["our_drugs_found_in_ddinter"], 4)

    def test_ambiguous_synonym_is_not_used(self):
        names = pd.DataFrame({"drug_idx": [0, 1], "title": ["Alpha", "Beta"], "synonyms": ["shared", "shared"]})
        dd = pd.DataFrame({"a": ["shared"], "b": ["alpha"], "Level": ["Major"]})
        pairs = pd.DataFrame({"drug_a": [0], "drug_b": [1]})
        o = da.ddinter_overlap(dd, names, pairs)
        self.assertEqual(o["ddinter_pairs_with_both_drugs_ours"], 0)   # 'shared' maps to two drugs -> ignored

    def test_api_checks_classify_failures(self):
        df = da.api_checks(FakeSession())
        r = df.set_index("source")
        self.assertTrue(r.loc["RxNorm", "reachable"])
        self.assertFalse(r.loc["openFDA", "reachable"])
        self.assertEqual(r.loc["openFDA", "note"], "HTTP 429")
        self.assertFalse(r.loc["KEGG", "reachable"])
        self.assertEqual(r.loc["KEGG", "note"], "TimeoutError")

    def test_local_checks_and_summary(self):
        loc = da.local_checks(self.root, canon_fn=ident)
        self.assertEqual(loc["TWOSIDES"]["n_drugs"], 6)
        self.assertEqual(loc["PharmGKB"]["drugs_with_real_gene_profile"], 2)   # ZZZ is not one of our drugs
        self.assertEqual(loc["BindingDB"]["drugs_matched"], 1)                 # D009 is not ours
        self.assertEqual(loc["PDB"]["structure_files"], 3)
        self.assertEqual(loc["FAERS"]["quarters"], ["23Q4"])
        self.assertEqual(loc["ChEMBL"]["drugs_with_real_targets"], 3)
        s = da.summary(loc, da.uniprot_section(self.root, FakeSession()), None, da.api_checks(FakeSession()), n_drugs=6)
        st = s.set_index("source").status
        self.assertEqual(st["TWOSIDES"], "CHECK")        # only 4 events here, below the 1300 threshold: flagged, not OK
        self.assertEqual(st["PDB"], "MISSING")
        self.assertEqual(st["DDInter"], "PARTIAL")
        self.assertEqual(st["KEGG"], "MISSING")
        self.assertEqual(st["GEO"], "NOT USED")
        self.assertEqual(st["UniProt"], "PARTIAL")       # 75% < 95%

    def test_inventory_hashes_small_files(self):
        inv = da.inventory(self.root, ["PDB", "nonexistent"])
        self.assertEqual(int((inv.path == "(folder missing)").sum()), 1)
        self.assertTrue(inv[inv.source_dir == "PDB"].sha256.notna().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
