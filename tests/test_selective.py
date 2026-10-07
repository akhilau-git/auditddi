import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.trust.selective import (  # noqa: E402
    abstention_gain,
    apply_abstention,
    deployed_summary,
    drug_domain_score,
    macro_auroc_subset,
    max_tanimoto_to_set,
    pair_reliability,
    random_abstention_curve,
    risk_coverage,
    threshold_for_coverage,
    top_k_mean_score,
)


def planted(n=4000, m=15, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    rel = rng.random(n)
    L = rng.normal(size=(n, m))
    Y = (rng.random((n, m)) < 1 / (1 + np.exp(-(1.5 * L - 1.2)))).astype(np.int8)
    strength = rel if informative else rng.random(n)
    S = 1 / (1 + np.exp(-(strength[:, None] * 3.0 * L + rng.normal(size=(n, m)))))
    return Y, S, rel


class TestTanimoto(unittest.TestCase):
    def test_hand_computed(self):
        bits = np.array([[1, 1, 0, 0],      # query
                         [1, 0, 0, 0],      # sim 1/2
                         [1, 1, 1, 0]])     # sim 2/3
        s = max_tanimoto_to_set(bits, [0], [1, 2])
        self.assertAlmostEqual(float(s[0]), 2 / 3, places=6)

    def test_exclude_self(self):
        bits = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]])
        self.assertAlmostEqual(float(max_tanimoto_to_set(bits, [0], [0, 1, 2], exclude_self=True)[0]), 2 / 3, places=6)
        self.assertAlmostEqual(float(max_tanimoto_to_set(bits, [0], [0, 1, 2], exclude_self=False)[0]), 1.0, places=6)

    def test_domain_score_and_pair_reliability(self):
        bits = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]])
        score = drug_domain_score(bits, train_drugs=np.array([1, 2]))
        self.assertEqual(score[1], 1.0)
        self.assertEqual(score[2], 1.0)
        self.assertAlmostEqual(float(score[0]), 2 / 3, places=6)      # unseen drug
        self.assertAlmostEqual(float(score[3]), 0.0, places=6)        # nothing in common
        rel = pair_reliability(score, np.array([0]), np.array([1]))   # S2 pair: one unseen drug
        self.assertAlmostEqual(float(rel["ad_min"][0]), 2 / 3, places=6)
        self.assertAlmostEqual(float(rel["ad_mean"][0]), (2 / 3 + 1) / 2, places=6)


class TestRiskCoverage(unittest.TestCase):
    def test_informative_reliability_beats_random_null(self):
        Y, S, rel = planted()
        rc = {r["coverage"]: r for r in risk_coverage(Y, S, rel, (1.0, 0.5, 0.25))}
        null = {r["coverage"]: r for r in random_abstention_curve(Y, S, (1.0, 0.5, 0.25), n_rep=6)}
        self.assertGreater(rc[0.5]["macro_auroc"], rc[1.0]["macro_auroc"] + 0.03)
        self.assertGreater(rc[0.25]["macro_auroc"], rc[0.5]["macro_auroc"])
        self.assertGreater(rc[0.5]["macro_auroc"], null[0.5]["null_mean"] + 5 * max(null[0.5]["null_sd"], 0.002))
        self.assertAlmostEqual(rc[1.0]["macro_auroc"], macro_auroc_subset(Y, S, np.arange(len(Y)))[0], places=10)

    def test_uninformative_reliability_gives_no_gain(self):
        Y, S, rel = planted(informative=False, seed=1)
        rng = np.random.default_rng(5)
        g = abstention_gain(Y, S, rng.random(len(Y)), 0.5, n_boot=40)
        self.assertLess(g["ci_low"], 0.0 + 1e-9 + 0.02)
        self.assertGreater(g["ci_high"], -0.02)
        self.assertLess(abs(g["gain"]), 0.03)

    def test_gain_ci_excludes_zero_when_informative(self):
        Y, S, rel = planted(seed=2)
        g = abstention_gain(Y, S, rel, 0.5, n_boot=40)
        self.assertGreater(g["ci_low"], 0.0)

    def test_constant_reliability_acts_like_random_not_row_order(self):
        Y, S, _ = planted(seed=3)
        const = np.ones(len(Y))
        rc = risk_coverage(Y, S, const, (1.0, 0.5))
        null = random_abstention_curve(Y, S, (0.5,), n_rep=6)[0]
        self.assertAlmostEqual(rc[1]["macro_auroc"], null["null_mean"], delta=4 * max(null["null_sd"], 0.004))

    def test_top20_mean_shape(self):
        S = np.random.default_rng(0).random((10, 50))
        t = top_k_mean_score(S, 20)
        self.assertEqual(t.shape, (10,))
        self.assertTrue(np.allclose(t, np.sort(S, axis=1)[:, -20:].mean(axis=1)))


class TestThresholdFromValidation(unittest.TestCase):
    def test_validation_threshold_transfers(self):
        rng = np.random.default_rng(0)
        rel_val, rel_test = rng.random(5000), rng.random(5000)
        tau = threshold_for_coverage(rel_val, 0.6)
        self.assertAlmostEqual(float((rel_val >= tau).mean()), 0.6, delta=0.01)
        self.assertAlmostEqual(float(apply_abstention(rel_test, tau).mean()), 0.6, delta=0.03)

    def test_deployed_summary_uses_frozen_tau(self):
        Yv, Sv, rv = planted(seed=4)
        Yt, St, rt = planted(seed=5)
        tau = threshold_for_coverage(rv, 0.5)
        d = deployed_summary(Yt, St, rt, tau)
        self.assertAlmostEqual(d["realised_coverage"], 0.5, delta=0.05)
        self.assertGreater(d["macro_auroc_answered"], d["macro_auroc_all"] + 0.03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
