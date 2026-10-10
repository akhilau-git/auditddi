import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.drugcentral_targets import build_targets, main, match_structures, name_bridge, split_accessions, union_targets  # noqa: E402

ident = lambda s: str(s)  # noqa: E731


def toy():
    drugs = pd.DataFrame({"drug_idx": [0, 1, 2, 3], "smiles": ["A", "B", "C", "D"]})
    st = pd.DataFrame({"SMILES": ["A", "B", "B", "Z"], "InChI": "x", "InChIKey": "k", "ID": [10, 20, 21, 30],
                       "INN": ["alpha", "beta", "beta-salt", "zeta"], "CAS_RN": ["1-1", "2-2", "2-3", "9-9"]})
    tg = pd.DataFrame({"DRUG_NAME": ["alpha"] * 3 + ["beta"] * 2 + ["zeta"], "STRUCT_ID": [10, 10, 10, 20, 21, 30],
                       "TARGET_NAME": "t", "TARGET_CLASS": ["GPCR"] * 6, "ACCESSION": ["P1", "P2|P3", "P1", "P4", "P4", "P9"],
                       "GENE": ["G1", "G2", "G1", "G4", "G4", "G9"], "MOA": [1, 0, 1, 1, 1, 1],
                       "ACTION_TYPE": "ANTAGONIST", "ORGANISM": "Homo sapiens"})
    return drugs, st, tg


class TestMatching(unittest.TestCase):
    def test_accession_split(self):
        self.assertEqual(split_accessions("P1|P2"), ["P1", "P2"])
        self.assertEqual(split_accessions("P1, P2;P3"), ["P1", "P2", "P3"])
        self.assertEqual(split_accessions(None), [])
        self.assertEqual(split_accessions(float("nan")), [])

    def test_match_keeps_unmatched_unmatched(self):
        drugs, st, _ = toy()
        m = match_structures(drugs, st, ident)
        self.assertEqual(sorted(m.drug_idx.unique()), [0, 1])           # C and D have no DrugCentral structure: no guess
        self.assertEqual(sorted(m[m.drug_idx == 1].drugcentral_id), [20, 21])   # two structures share a skeleton: both kept

    def test_targets_split_complexes_and_dedupe(self):
        drugs, st, tg = toy()
        t = build_targets(match_structures(drugs, st, ident), tg)
        self.assertEqual(sorted(t[t.drug_idx == 0].uniprot), ["P1", "P2", "P3"])   # P1 appears twice -> once; 'P2|P3' split
        self.assertEqual(sorted(t[t.drug_idx == 1].uniprot), ["P4"])               # two structures, same target -> once
        self.assertNotIn(3, set(t.drug_idx))                                       # unmatched drug has no targets
        self.assertNotIn("P9", set(t.uniprot))                                     # target of an unrelated structure never leaks in

    def test_moa_only_filters_off_target_rows(self):
        drugs, st, tg = toy()
        t = build_targets(match_structures(drugs, st, ident), tg, moa_only=True)
        self.assertEqual(sorted(t[t.drug_idx == 0].uniprot), ["P1"])               # 'P2|P3' row has MOA=0

    def test_union_records_both_sources(self):
        drugs, st, tg = toy()
        dc = build_targets(match_structures(drugs, st, ident), tg)
        ch = pd.DataFrame({"drug_idx": [0, 2], "uniprot": ["P1", "P7"], "organism": "Homo sapiens"})
        u = union_targets(ch, dc)
        self.assertEqual(u[(u.drug_idx == 0) & (u.uniprot == "P1")].source.iloc[0], "chembl_mechanism+drugcentral")
        self.assertEqual(u[(u.drug_idx == 2) & (u.uniprot == "P7")].source.iloc[0], "chembl_mechanism")
        self.assertEqual(u.drop_duplicates(["drug_idx", "uniprot"]).shape[0], len(u))

    def test_name_bridge_one_row_per_drug(self):
        drugs, st, _ = toy()
        b = name_bridge(match_structures(drugs, st, ident))
        self.assertEqual(len(b), 2)
        self.assertEqual(b[b.drug_idx == 0].inn.iloc[0], "alpha")

    def test_cli_and_report(self):
        drugs, st, tg = toy()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "table").mkdir(); drugs.to_csv(d / "table" / "drugs.csv", index=False)
            (d / "dc").mkdir(); st.to_csv(d / "dc" / "structures.smiles.tsv", sep="\t", index=False)
            tg.to_csv(d / "dc" / "drug_target_interaction.tsv.gz", sep="\t", index=False, compression="gzip")
            pd.DataFrame({"drug_idx": [2], "uniprot": ["P7"], "organism": ["Homo sapiens"]}).to_csv(d / "ch.csv", index=False)
            main(["--table", str(d / "table"), "--drugcentral", str(d / "dc"), "--chembl-targets", str(d / "ch.csv"), "--out", str(d / "o"), "--identity-key"])
            rep = json.loads((d / "o" / "drugcentral_match_report.json").read_text())
            self.assertEqual((rep["drugs_matched_to_drugcentral"], rep["drugs_with_drugcentral_targets"], rep["drugs_with_union_targets"]), (2, 2, 3))
            self.assertTrue((d / "o" / "drug_names_drugcentral.csv").exists())


@unittest.skipUnless(__import__("importlib").util.find_spec("rdkit"), "RDKit not installed (runs on Colab)")
class TestRdkitKey(unittest.TestCase):
    def test_salt_and_stereo_still_match_but_different_skeleton_does_not(self):
        from src.features.drugcentral_targets import rdkit_ik14

        self.assertEqual(rdkit_ik14("CC(=O)OC1=CC=CC=C1C(=O)O"), rdkit_ik14("OC(=O)c1ccccc1OC(C)=O"))     # same molecule, different SMILES
        self.assertEqual(rdkit_ik14("C[C@H](N)C(=O)O"), rdkit_ik14("C[C@@H](N)C(=O)O"))        # stereoisomers share the skeleton
        self.assertEqual(rdkit_ik14("CC(=O)Nc1ccc(O)cc1.Cl"), rdkit_ik14("CC(=O)Nc1ccc(O)cc1"))   # counter-ion dropped (largest fragment)
        self.assertNotEqual(rdkit_ik14("CCO"), rdkit_ik14("CCCO"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
