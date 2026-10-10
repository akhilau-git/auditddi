import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.drug_features import atc_levels, biology_known, build_atc_block, build_class_block, build_store, coverage_report, feature_matrix  # noqa: E402
from src.features.rxnorm_by_name import find_atc_codes, main, names_table, resolve, to_long_atc  # noqa: E402


class R:
    def __init__(self, js, status=200):
        self._js, self.status_code = js, status

    def json(self):
        return self._js


class FakeRx:
    CLASS = {"1191": {"rxclassDrugInfoList": {"rxclassDrugInfo": [
        {"minConcept": {"rxcui": "1191"}, "rxclassMinConceptItem": {"classId": "B01AC06", "classType": "ATC1-4"}},
        {"minConcept": {"rxcui": "1191"}, "rxclassMinConceptItem": {"classId": "N02BA01", "classType": "ATC1-4"}},
        {"minConcept": {"rxcui": "1191"}, "rxclassMinConceptItem": {"classId": "N0000175503", "classType": "EPC"}}]}}}
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url.split("/REST/")[1], (params or {}).get("name") or (params or {}).get("term") or (params or {}).get("rxcui")))
        if url.endswith("rxcui.json"):
            return R({"idGroup": {"name": params["name"], "rxnormId": ["1191"]}}) if params["name"] == "aspirin" else R({"idGroup": {}})
        if url.endswith("approximateTerm.json"):
            return R({"approximateGroup": {"candidate": [{"rxcui": "999", "rank": "1"}]}})
        if "rxclass" in url:
            return R(self.CLASS.get(params["rxcui"], {}))
        return R({}, 404)


class TestRxNorm(unittest.TestCase):
    def test_atc_parser_ignores_non_atc_class_ids(self):
        self.assertEqual(find_atc_codes(FakeRx.CLASS["1191"]), ["B01AC06", "N02BA01"])
        self.assertEqual(find_atc_codes({}), [])

    def test_exact_only_by_default_no_guessing(self):
        s = FakeRx()
        names = pd.DataFrame({"drug_idx": [0, 1], "name": ["aspirin", "unknownium"]})
        res = resolve(s, names, approximate=False, pause=0)
        self.assertEqual(res.set_index("drug_idx").loc[0, "rxcui"], "1191")
        self.assertTrue(pd.isna(res.set_index("drug_idx").loc[1, "rxcui"]))            # not matched, not guessed
        self.assertFalse(any(c[0] == "approximateTerm.json" for c in s.calls))
        self.assertEqual(res.set_index("drug_idx").loc[0, "atc_codes"], "B01AC06|N02BA01")

    def test_approximate_is_labelled(self):
        res = resolve(FakeRx(), pd.DataFrame({"drug_idx": [1], "name": ["unknownium"]}), approximate=True, pause=0)
        self.assertEqual(res.match_type.iloc[0], "approximate")

    def test_second_name_used_when_first_fails_and_cache_avoids_repeat_calls(self):
        s = FakeRx()
        names = pd.DataFrame({"drug_idx": [0, 0, 5], "name": ["acetylsalicylic acid brand", "aspirin", "aspirin"]})
        cache = {}
        res = resolve(s, names, pause=0, cache=cache)
        self.assertEqual(res[res.drug_idx == 0].name_used.iloc[0], "aspirin")
        n = len(s.calls)
        resolve(s, names, pause=0, cache=cache)
        self.assertEqual(len(s.calls), n)                                               # fully cached

    def test_names_table_priority_and_long_atc(self):
        t = names_table(pd.DataFrame({"drug_idx": [0], "inn": ["Aspirin"]}), pd.DataFrame({"drug_idx": [0, 1], "title": ["Acetylsalicylic acid", "Foo"]}))
        self.assertEqual(t[t.drug_idx == 0].name.tolist(), ["Aspirin", "Acetylsalicylic acid"])
        long = to_long_atc(pd.DataFrame({"drug_idx": [0, 1], "atc_codes": ["B01AC06|N02BA01", ""]}))
        self.assertEqual(len(long), 2)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame({"drug_idx": [0, 1], "inn": ["aspirin", "unknownium"]}).to_csv(d / "n.csv", index=False)
            main(["--names", str(d / "n.csv"), "--out", str(d / "o")], session=FakeRx())
            rep = json.loads((d / "o" / "rxnorm_report.json").read_text())
            self.assertEqual((rep["with_rxcui"], rep["with_atc"]), (1, 1))


class TestClassAndAtcBlocks(unittest.TestCase):
    def test_atc_levels(self):
        self.assertEqual(atc_levels("N02BA01"), ["N", "N02", "N02B", "N02BA"])
        self.assertEqual(atc_levels("N0000175503"), [])        # not an ATC code
        self.assertEqual(atc_levels("N"), [])                  # a bare letter is not a full code

    def test_class_block_shared_classes_and_flags(self):
        t = pd.DataFrame({"drug_idx": [0, 0, 1, 3], "target_class": ["GPCR", "Kinase", "GPCR", np.nan], "uniprot": list("abcd")})
        M, vocab, has = build_class_block(5, t)
        self.assertEqual(vocab, ["GPCR", "Kinase"])
        self.assertEqual(has.tolist(), [True, True, False, False, False])
        self.assertEqual(int(M[1].sum()), 1)

    def test_store_with_class_and_atc(self):
        drugs = pd.DataFrame({"drug_idx": range(4), "smiles": list("ABCD")})
        tg = pd.DataFrame({"drug_idx": [0, 1], "uniprot": ["P1", "P2"], "organism": "Homo sapiens", "target_class": ["GPCR", "GPCR"]})
        atc = pd.DataFrame({"drug_idx": [0, 2], "atc_code": ["N02BA01", "B01AC06"]})
        st = build_store(drugs, np.ones((4, 8), np.uint8), np.ones(4, bool), tg, pd.DataFrame({"canonical_smiles": [], "genes_list": []}), canon_fn=lambda x: x, atc=atc)
        rep = coverage_report(st)
        self.assertEqual((rep["has_target_class"], rep["has_atc"]), (2, 2))
        self.assertEqual(biology_known(st).tolist(), [True, True, True, False])
        F = feature_matrix(st, ["ecfp", "tclass", "atc"])
        self.assertEqual(F.shape[0], 4)
        with self.assertRaises(ValueError):
            feature_matrix(build_store(drugs, np.ones((4, 8), np.uint8), np.ones(4, bool), tg[["drug_idx", "uniprot", "organism"]],
                                        pd.DataFrame({"canonical_smiles": [], "genes_list": []}), canon_fn=lambda x: x), ["atc"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
