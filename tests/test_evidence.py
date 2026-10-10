import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evidence.ddinter_bridge import bridge, load_ddinter, main as dd_main  # noqa: E402
from src.evidence.openfda_labels import attribute_label, build_name_map, cyp_flags, find_mentions, iter_label_records, main as lab_main, mention_regex, process  # noqa: E402


def rec(generic, substance=None, di=None, pk=None):
    r = {"openfda": {"generic_name": [generic], "substance_name": [substance or generic.upper()]}, "set_id": "x"}
    if di is not None:
        r["drug_interactions"] = [di]
    if pk is not None:
        r["clinical_pharmacology"] = [pk]
    return r


def bulk_json(records):
    return json.dumps({"meta": {"last_updated": "2026-01-01", "results": {"total": len(records)}}, "results": records}, indent=2, ensure_ascii=False)


class TestStreaming(unittest.TestCase):
    def test_same_records_for_every_chunk_size_including_multibyte(self):
        recs = [rec(f"Drug{i}", di="caf\u00e9 \u2013 \u03b1-blocker %d \"quoted\" {x}" % i) for i in range(30)]
        txt = bulk_json(recs).encode("utf-8")
        for chunk in (7, 64, 1000, 1 << 20):
            got = list(iter_label_records(io.BytesIO(txt), chunk=chunk))
            self.assertEqual(got, recs, f"chunk={chunk}")

    def test_empty_and_malformed_inputs_do_not_crash(self):
        self.assertEqual(list(iter_label_records(io.BytesIO(b'{"meta":{},"results":[]}'))), [])
        self.assertEqual(list(iter_label_records(io.BytesIO(b"not json at all"))), [])
        self.assertEqual(list(iter_label_records(io.BytesIO(b""))), [])

    def test_a_record_larger_than_a_chunk(self):
        big = rec("Bigdrug", di="x" * 50000)
        self.assertEqual(list(iter_label_records(io.BytesIO(bulk_json([big, rec("Small")]).encode()), chunk=1000))[0], big)


class TestAttribution(unittest.TestCase):
    def setUp(self):
        names = pd.DataFrame({"drug_idx": [0, 1, 2, 3], "name": ["warfarin", "ketoconazole", "sharedname", "sharedname"]})
        self.nm, self.n_amb = build_name_map(names)

    def test_ambiguous_names_dropped(self):
        self.assertNotIn("sharedname", self.nm)
        self.assertEqual(self.n_amb, 1)

    def test_single_ingredient_match_case_insensitive(self):
        self.assertEqual(attribute_label(rec("Warfarin Sodium", substance="WARFARIN"), self.nm), 0)
        self.assertEqual(attribute_label(rec("warfarin"), self.nm), 0)

    def test_combination_products_and_unknown_drugs_not_attributed(self):
        combo = {"openfda": {"generic_name": ["warfarin, ketoconazole"], "substance_name": ["WARFARIN", "KETOCONAZOLE"]}}
        self.assertIsNone(attribute_label(combo, self.nm))
        self.assertIsNone(attribute_label(rec("aspirin"), self.nm))
        self.assertIsNone(attribute_label({}, self.nm))
        self.assertIsNone(attribute_label(rec("sharedname"), self.nm))


class TestHeuristics(unittest.TestCase):
    def test_cyp_roles(self):
        f = cyp_flags("Drug X is extensively metabolized by CYP3A4 in the liver. It is a strong inhibitor of CYP2D6 and does not induce CYP2C9 expression.")
        self.assertEqual((f["cyp3A4_substrate"], f["cyp2D6_inhibitor"]), (1, 1))
        self.assertEqual(f["cyp2C9_inducer"], 1)                 # keyword heuristic: 'induce' near a CYP mention counts, even when negated
        self.assertEqual(f["cyp1A2_substrate"], 0)
        self.assertEqual(sum(cyp_flags("no enzymes mentioned").values()), 0)
        self.assertEqual(cyp_flags("substrate of CYP 2C19")["cyp2C19_substrate"], 1)    # spaced form

    def test_mentions_exclude_own_name_short_names_and_count(self):
        nm = {"warfarin": 0, "ketoconazole": 1, "iron": 2, "digoxin": 3}
        rx = mention_regex(nm, min_len=5)
        self.assertNotIn("iron", rx.pattern)                         # 4 letters: too ordinary to search for
        m = find_mentions("Ketoconazole increases warfarin levels. Avoid warfarin with Ketoconazole. Warfarin itself.", rx, nm, own=0)
        self.assertEqual(set(m), {1})                                # own name excluded
        self.assertEqual(m[1][0], 2)
        self.assertIn("increases warfarin", m[1][1].lower())
        self.assertEqual(find_mentions("nothing relevant", rx, nm, own=0), {})

    def test_process_end_to_end(self):
        nm, _ = build_name_map(pd.DataFrame({"drug_idx": [0, 1, 2], "name": ["warfarin", "ketoconazole", "digoxin"]}))
        recs = [rec("Warfarin", di="Ketoconazole may increase warfarin effect; digoxin levels unaffected.", pk="Warfarin is a substrate of CYP2C9 and CYP3A4."),
                rec("Ketoconazole", pk="Ketoconazole is a potent inhibitor of CYP3A4."),
                rec("Warfarin", di="Avoid ketoconazole."), rec("aspirin", di="mentions warfarin")]
        feats, pairs, stats = process(recs, nm)
        self.assertEqual(stats, {"records_seen": 4, "records_attributed": 3})
        w = feats[feats.drug_idx == 0].iloc[0]
        self.assertEqual((w.n_labels, w.cyp2C9_substrate, w.cyp3A4_substrate), (2, 1, 1))
        self.assertEqual(feats[feats.drug_idx == 1].iloc[0].cyp3A4_inhibitor, 1)
        got = {(r.drug_label, r.drug_mentioned): r.n_mentions for r in pairs.itertuples()}
        self.assertEqual(got, {(0, 1): 2, (0, 2): 1})                # aspirin's label is not ours: its 'warfarin' mention is ignored


class TestCLIAndDDInter(unittest.TestCase):
    def test_label_cli_over_zip_parts(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "labels").mkdir()
            for i, recs in enumerate([[rec("Warfarin", di="Ketoconazole raises effect", pk="substrate of CYP2C9")], [rec("Ketoconazole", pk="inhibits CYP3A4")]]):
                with zipfile.ZipFile(d / "labels" / f"drug-label-000{i}-of-0002.json.zip", "w") as z:
                    z.writestr(f"drug-label-000{i}-of-0002.json", bulk_json(recs))
            pd.DataFrame({"drug_idx": [0, 1], "inn": ["warfarin", "ketoconazole"]}).to_csv(d / "n.csv", index=False)
            lab_main(["--names", str(d / "n.csv"), "--labels", str(d / "labels"), "--out", str(d / "o")])
            rep = json.loads((d / "o" / "label_report.json").read_text())
            self.assertEqual((rep["parts"], rep["drugs_with_a_label"], rep["unordered_mention_pairs"], rep["drugs_with_any_cyp_flag"]), (2, 2, 1, 2))
            self.assertTrue((d / "o" / "label_pairs.csv").exists())

    def test_ddinter_bridge_unambiguous_worst_severity_and_twosides_flag(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "raw").mkdir(); (d / "table").mkdir()
            pd.DataFrame({"Drug_A": ["Warfarin", "Ketoconazole", "Warfarin", "Unknowndrug"], "Drug_B": ["Ketoconazole", "WARFARIN ", "Digoxin", "Warfarin"],
                          "Level": ["Moderate", "Major", "Minor", "Major"]}).to_csv(d / "raw" / "ddinter_downloads_code_A.csv", index=False)
            pd.DataFrame({"drug_idx": [0, 1, 2], "inn": ["warfarin", "ketoconazole", "digoxin"]}).to_csv(d / "n.csv", index=False)
            pd.DataFrame({"drug_a": [0], "drug_b": [1]}).to_csv(d / "table" / "pairs.csv", index=False)
            dd_main(["--table", str(d / "table"), "--ddinter", str(d / "raw"), "--names", str(d / "n.csv"), "--out", str(d / "o")])
            g = pd.read_csv(d / "o" / "ddinter_pairs.csv").set_index(["drug_a", "drug_b"])
            self.assertEqual(len(g), 2)                                          # 'Unknowndrug' dropped, never guessed
            self.assertEqual(g.loc[(0, 1), "severity"], "major")                 # two records, worst severity kept
            self.assertEqual(g.loc[(0, 1), "n_records"], 2)
            self.assertTrue(bool(g.loc[(0, 1), "in_twosides"]))
            self.assertFalse(bool(g.loc[(0, 2), "in_twosides"]))                 # a DDInter pair TWOSIDES never reported
            rep = json.loads((d / "o" / "ddinter_report.json").read_text())
            self.assertEqual(rep["of_which_NOT_in_twosides"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
