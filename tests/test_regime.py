import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.multilabel_metrics import calibration_macro  # noqa: E402
from src.trust.calibration import EventCalibrator, conformal_report, conformal_sets, conformal_thresholds, logit, sigmoid, split_calibration  # noqa: E402
from src.trust.regime import RegimeAwareTrust, regime_of_pairs, seen_mask  # noqa: E402


def regime_data(n1, n2, E=15, seed=0, base_seed=123):
    """Same score distortion everywhere, but regime 2 pairs carry almost no real signal: the model's
    confidence is justified in regime 1 and badly over-stated in regime 2."""
    base = np.random.default_rng(base_seed).uniform(-3.0, -1.0, E)       # event base rates shared by val and test
    rng = np.random.default_rng(seed)
    n = n1 + n2
    regime = np.r_[np.full(n1, 1), np.full(n2, 2)].astype(np.int8)
    latent = rng.normal(size=(n, E))
    true_signal = np.where(regime == 1, 1.6, 0.15)[:, None]
    Y = (rng.random((n, E)) < sigmoid(base + true_signal * latent)).astype(np.int8)
    S = sigmoid(base + 1.6 * latent).astype(np.float32)           # the model always claims strong signal
    return Y, S, regime


class TestRegimeLabels(unittest.TestCase):
    def test_regime_definition_depends_only_on_training_drugs(self):
        a = np.array([0, 0, 1, 2, 3]); b = np.array([1, 2, 2, 3, 4])
        train_rows = np.array([0, 2])                      # pairs (0,1) and (1,2) -> drugs 0,1,2 are seen
        seen = seen_mask(5, a, b, train_rows)
        self.assertEqual(seen.tolist(), [True, True, True, False, False])
        self.assertEqual(regime_of_pairs(seen, a, b).tolist(), [0, 0, 0, 1, 2])


class TestRegimeAwareTrust(unittest.TestCase):
    def setUp(self):
        Yv, Sv, rv = regime_data(8000, 1500, seed=1)
        Yt, St, rt = regime_data(4000, 1500, seed=2)
        self.Yv, self.Sv, self.rv, self.Yt, self.St, self.rt = Yv, Sv, rv, Yt, St, rt

    def test_pooled_calibration_fails_in_the_hard_regime_regime_aware_does_not(self):
        fit, cal = split_calibration(len(self.Sv), 0)
        pooled = EventCalibrator().fit(self.Sv[fit], self.Yv[fit])
        trust = RegimeAwareTrust().fit(self.Sv, self.Yv, self.rv)
        m2 = self.rt == 2
        cp = calibration_macro(self.Yt[m2], pooled.transform(self.St[m2]))
        cr = calibration_macro(self.Yt[m2], trust.transform(self.St, self.rt)[m2])
        # proper scoring quantities (robust): regime-aware is clearly better where the model is over-confident
        self.assertLess(cr["macro_ece"], cp["macro_ece"] * 0.5)
        self.assertLess(cr["macro_brier"], cp["macro_brier"])
        self.assertLess(cr["macro_nll"], cp["macro_nll"])
        # robust slope: pooled stays far from 1, regime-aware is much closer (the MEAN slope is unstable here, see docs)
        self.assertGreater(abs(cp["median_cal_slope"] - 1), abs(cr["median_cal_slope"] - 1) + 0.3)
        m1 = self.rt == 1
        c1 = calibration_macro(self.Yt[m1], trust.transform(self.St, self.rt)[m1])
        self.assertAlmostEqual(c1["median_cal_slope"], 1.0, delta=0.25)           # the easy regime stays well calibrated

    def test_conformal_coverage_restored_per_regime(self):
        trust = RegimeAwareTrust(alphas=(0.1,)).fit(self.Sv, self.Yv, self.rv)
        P = trust.transform(self.St, self.rt)
        in1, in0 = trust.sets(P, self.rt, 0.1)
        fit, cal = split_calibration(len(self.Sv), 0)
        pooled = EventCalibrator().fit(self.Sv[fit], self.Yv[fit])
        Pp = pooled.transform(self.St)
        q1, q0, _, _ = conformal_thresholds(pooled.transform(self.Sv[cal]), self.Yv[cal], 0.1)
        p1, p0 = conformal_sets(Pp, q1, q0)
        m2 = self.rt == 2
        cov_regime = conformal_report(self.Yt[m2], in1[m2], in0[m2], min_pos=10)["coverage_positive_macro"]
        cov_pooled = conformal_report(self.Yt[m2], p1[m2], p0[m2], min_pos=10)["coverage_positive_macro"]
        self.assertGreaterEqual(cov_regime, 0.84)
        self.assertGreater(cov_regime, cov_pooled)

    def test_hard_regime_gets_less_informative_sets(self):
        trust = RegimeAwareTrust().fit(self.Sv, self.Yv, self.rv)
        P = trust.transform(self.St, self.rt)
        in1, in0 = trust.sets(P, self.rt, 0.1)
        both = (in1 & in0).mean(axis=1)
        self.assertGreater(both[self.rt == 2].mean(), both[self.rt == 1].mean())   # honest: less information => more {0,1}

    def test_small_or_missing_regime_falls_back_and_is_flagged(self):
        Y, S, r = regime_data(3000, 50, seed=3)
        trust = RegimeAwareTrust(min_pairs=200).fit(S, Y, r)
        self.assertTrue(trust.info_[2]["fallback_to_pooled"])
        self.assertTrue(trust.info_[0]["fallback_to_pooled"])                       # no regime-0 validation pairs at all
        P = trust.transform(S, np.zeros(len(S), dtype=np.int8))                     # regime 0 at test time: pooled map used, no crash
        self.assertEqual(P.shape, S.shape)
        in1, in0 = trust.sets(P, np.zeros(len(S), dtype=np.int8), 0.1)
        self.assertEqual(in1.shape, S.shape)

    def test_ranking_never_changes(self):
        from src.evaluation.multilabel_metrics import auroc_per_event

        trust = RegimeAwareTrust().fit(self.Sv, self.Yv, self.rv)
        P = trust.transform(self.St, self.rt)
        m = self.rt == 1
        a0 = auroc_per_event(self.Yt[m], self.St[m]); a1 = auroc_per_event(self.Yt[m], P[m])
        self.assertTrue(np.nanmax(np.abs(a0 - a1)) < 5e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
