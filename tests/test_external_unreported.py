import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evidence.external_unreported import candidate_pairs, evaluate_candidates, popularity  # noqa: E402


class TestCandidates(unittest.TestCase):
    def test_candidates_exclude_twosides_pairs_and_stay_in_scope(self):
        dd = pd.DataFrame({"drug_a": [1, 2], "drug_b": [2, 5], "severity": ["major", "minor"]})        # scope {1,2,5}
        tw = pd.DataFrame({"drug_a": [5, 9], "drug_b": [1, 8]})                                       # (1,5) reported, others out of scope
        a, b = candidate_pairs(tw, dd)
        got = set(zip(a.tolist(), b.tolist()))
        self.assertEqual(got, {(1, 2), (2, 5)})                                                       # (1,5) removed (order-insensitive)
        self.assertTrue((a < b).all())
        self.assertEqual(len(got), len(a))                                                            # no duplicates

    def test_popularity_uses_training_pairs_only(self):
        A = np.array([0, 0, 1]); B = np.array([1, 2, 2])
        p = popularity(4, A, B, np.array([0, 1]), np.array([2, 3]), np.array([3, 3]))              # train pairs (0,1),(0,2): deg 0:2, 1:1, 2:1
        self.assertAlmostEqual(p[0], np.log1p(1) + np.log1p(0))
        self.assertEqual(p[1], 0.0)


class TestEvaluate(unittest.TestCase):
    def _world(self, informative: bool, n_drugs=120, seed=0):
        rng = np.random.default_rng(seed)
        ii, jj = np.triu_indices(n_drugs, 1)
        keep = rng.random(len(ii)) < 0.5
        a, b = ii[keep].astype(np.int64), jj[keep].astype(np.int64)
        truth = rng.random(len(a)) < 0.2
        dd = pd.DataFrame({"drug_a": a[truth], "drug_b": b[truth], "severity": "major"})
        base = rng.normal(size=(len(a),))
        s = base + (1.2 * truth if informative else 0.0)
        regime = rng.integers(0, 3, len(a)).astype(np.int8)
        return a, b, dd, s, regime

    def test_informative_score_detected_noise_not(self):
        for informative in (True, False):
            a, b, dd, s, regime = self._world(informative)
            df = evaluate_candidates({"model": s, "popularity_baseline": np.random.default_rng(1).normal(size=len(a))}, a, b, regime, dd, None, n_boot=30)
            r = df[(df.flag == "ddinter_any") & (df.subset == "all")].set_index("score")
            if informative:
                self.assertGreater(r.loc["model", "ci_low"], 0.65)
            else:
                self.assertLess(abs(r.loc["model", "auroc"] - 0.5), 0.04)
            self.assertLess(abs(r.loc["popularity_baseline", "auroc"] - 0.5), 0.04)

    def test_regime_subsets_are_reported_separately_and_sizes_add_up(self):
        a, b, dd, s, regime = self._world(True)
        df = evaluate_candidates({"model": s}, a, b, regime, dd, None, n_boot=10)
        sizes = df[(df.flag == "ddinter_any")].set_index("subset").n_pairs
        self.assertEqual(int(sizes["regime0_both_seen"] + sizes["regime1_one_unseen"] + sizes["regime2_both_unseen"]), int(sizes["all"]))

    def test_too_few_positives_gives_nan(self):
        a, b, dd, s, regime = self._world(True)
        df = evaluate_candidates({"model": s}, a, b, regime, dd.iloc[:5], None, n_boot=10, min_pos=30)
        self.assertTrue(df[df.flag == "ddinter_any"].auroc.isna().all())

    def test_popularity_confound_visible(self):
        """If the curated list favours popular drugs, a popularity-only score scores well: the baseline row must show it."""
        rng = np.random.default_rng(2)
        n_drugs = 150
        pop = rng.pareto(1.5, n_drugs) + 1
        ii, jj = np.triu_indices(n_drugs, 1)
        a, b = ii.astype(np.int64), jj.astype(np.int64)
        score = np.log1p(pop[a]) + np.log1p(pop[b])
        truth = rng.random(len(a)) < 1 / (1 + np.exp(-3 * (score - np.median(score))))
        dd = pd.DataFrame({"drug_a": a[truth], "drug_b": b[truth], "severity": "major"})
        df = evaluate_candidates({"popularity_baseline": score, "model": score + rng.normal(0, 0.1, len(a))}, a, b, np.zeros(len(a), np.int8), dd, None, n_boot=10)
        r = df[(df.flag == "ddinter_any") & (df.subset == "all")].set_index("score")
        self.assertGreater(r.loc["popularity_baseline", "auroc"], 0.7)
        self.assertAlmostEqual(r.loc["model", "auroc"], r.loc["popularity_baseline", "auroc"], delta=0.03)   # the model adds nothing beyond popularity


if __name__ == "__main__":
    unittest.main(verbosity=2)
