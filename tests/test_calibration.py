import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.multilabel_metrics import auroc_per_event, calibration_macro  # noqa: E402
from src.trust.calibration import (  # noqa: E402
    EventCalibrator,
    conformal_report,
    conformal_sets,
    conformal_thresholds,
    fit_platt,
    logit,
    sigmoid,
    split_calibration,
)


def make_scores(n=4000, E=20, seed=0, overconfidence=2.5, signal=1.6):
    """True probabilities p; the model's score is a distorted (overconfident) version."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(-3.0, -1.0, E)
    latent = rng.normal(size=(n, E))
    p = sigmoid(base + signal * latent)
    Y = (rng.random((n, E)) < p).astype(np.int8)
    S = sigmoid(overconfidence * logit(p) + 0.3)
    return Y, S.astype(np.float32), p


class TestPlatt(unittest.TestCase):
    def test_recovers_known_parameters(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 2, 30000)
        y = (rng.random(30000) < sigmoid(-1.0 + 0.5 * x)).astype(float)
        a, b = fit_platt(x, y)
        self.assertAlmostEqual(a, -1.0, delta=0.1)
        self.assertAlmostEqual(b, 0.5, delta=0.05)

    def test_slope_stays_positive_even_for_adversarial_data(self):
        x = np.linspace(-3, 3, 400)
        y = (x < 0).astype(float)                       # perfectly anti-correlated
        a, b = fit_platt(x, y)
        self.assertGreater(b, 0)

    def test_shrinkage_pulls_rare_events_toward_pooled_map(self):
        Y, S, _ = make_scores(n=3000, E=6, seed=1)
        Y[:, 5] = 0
        Y[:3, 5] = 1                                    # event 5 has only 3 positives
        cal0 = EventCalibrator(lam=0.0).fit(S, Y)
        cal3 = EventCalibrator(lam=5.0).fit(S, Y)
        d0 = abs(cal0.b_[5] - cal0.b0_) + abs(cal0.a_[5] - cal0.a0_)
        d3 = abs(cal3.b_[5] - cal3.b0_) + abs(cal3.a_[5] - cal3.a0_)
        self.assertLess(d3, d0)

    def test_event_without_positives_uses_pooled_map(self):
        Y, S, _ = make_scores(n=1500, E=4, seed=2)
        Y[:, 2] = 0
        cal = EventCalibrator().fit(S, Y)
        self.assertEqual((cal.a_[2], cal.b_[2]), (cal.a0_, cal.b0_))


class TestCalibrationEffect(unittest.TestCase):
    def test_calibration_fixes_overconfidence_and_keeps_ranking(self):
        Y, S, _ = make_scores(n=6000, E=16, seed=3)
        fit_idx, test_idx = np.arange(0, 3000), np.arange(3000, 6000)
        cal = EventCalibrator().fit(S[fit_idx], Y[fit_idx])
        Pt = cal.transform(S[test_idx])
        before = calibration_macro(Y[test_idx], S[test_idx])
        after = calibration_macro(Y[test_idx], Pt)
        self.assertLess(after["macro_ece"], before["macro_ece"] * 0.5)
        self.assertAlmostEqual(after["macro_cal_slope"], 1.0, delta=0.2)
        self.assertGreater(abs(before["macro_cal_slope"] - 1.0), abs(after["macro_cal_slope"] - 1.0))
        a0 = auroc_per_event(Y[test_idx], S[test_idx])
        a1 = auroc_per_event(Y[test_idx], Pt)
        self.assertTrue(np.nanmax(np.abs(a0 - a1)) < 5e-3)       # b_e > 0 => same ranking (float32 ties only)
        self.assertTrue((cal.b_ > 0).all())

    def test_split_is_disjoint_and_complete(self):
        f, c = split_calibration(101, seed=4)
        self.assertEqual(len(set(f) & set(c)), 0)
        self.assertEqual(sorted(np.concatenate([f, c]).tolist()), list(range(101)))
        self.assertEqual(split_calibration(101, seed=4)[0].tolist(), f.tolist())


class TestConformal(unittest.TestCase):
    def _setup(self, seed=5):
        Y, S, _ = make_scores(n=12000, E=12, seed=seed)
        fit, cal, test = np.arange(0, 4000), np.arange(4000, 8000), np.arange(8000, 12000)
        calib = EventCalibrator().fit(S[fit], Y[fit])
        return Y, calib.transform(S), cal, test

    def test_valid_coverage_when_exchangeable(self):
        Y, P, cal, test = self._setup()
        q1, q0, n1, n0 = conformal_thresholds(P[cal], Y[cal], alpha=0.1)
        in1, in0 = conformal_sets(P[test], q1, q0)
        r = conformal_report(Y[test], in1, in0, min_pos=20, alpha=0.1)
        self.assertGreaterEqual(r["coverage_positive_macro"], 0.86)
        self.assertGreaterEqual(r["coverage_negative_macro"], 0.86)
        self.assertAlmostEqual(r["frac_singleton_1"] + r["frac_singleton_0"] + r["frac_both"] + r["frac_empty"], 1.0, places=9)
        self.assertGreater(r["frac_singleton_0"], 0.3)                 # the sets are informative, not trivially {0,1}

    def test_too_few_calibration_positives_gives_trivial_certificate(self):
        Y, P, cal, test = self._setup(seed=6)
        Yc = Y[cal].copy()
        Yc[:, 0] = 0
        Yc[:5, 0] = 1                                                    # only 5 positives for event 0
        q1, q0, n1, n0 = conformal_thresholds(P[cal], Yc, alpha=0.1)
        self.assertTrue(np.isinf(q1[0]))
        in1, in0 = conformal_sets(P[test], q1, q0)
        self.assertTrue(in1[:, 0].all())                                 # label 1 always included => positives always covered

    def test_shift_breaks_the_guarantee_and_the_report_shows_it(self):
        """Cold-start failure mode: on new drugs the model is far less confident than it was on the
        calibration pairs, so true positives fall below the inclusion cutoff learned at calibration."""
        Y, P, cal, test = self._setup(seed=7)
        q1, q0, _, _ = conformal_thresholds(P[cal], Y[cal], alpha=0.1)
        exch = conformal_report(Y[test], *conformal_sets(P[test], q1, q0), min_pos=20, alpha=0.1)
        shifted = conformal_report(Y[test], *conformal_sets(P[test] * 0.05, q1, q0), min_pos=20, alpha=0.1)
        self.assertGreaterEqual(exch["coverage_positive_macro"], 0.86)
        self.assertLess(shifted["coverage_positive_macro"], 0.6)          # the guarantee fails, and the report shows it
        self.assertLess(shifted["frac_events_pos_cov_within_5pts_of_nominal"], exch["frac_events_pos_cov_within_5pts_of_nominal"])

    def test_coverage_alone_does_not_prove_usefulness(self):
        """Uninformative scores can still reach nominal positive coverage (by including label 1 widely),
        so conformal coverage must always be reported together with ranking quality."""
        Y, P, cal, test = self._setup(seed=9)
        q1, q0, _, _ = conformal_thresholds(P[cal], Y[cal], alpha=0.1)
        rng = np.random.default_rng(1)
        noise = rng.random(P[test].shape).astype(np.float32) * P[test].max(axis=0)[None, :]
        cov_noise = conformal_report(Y[test], *conformal_sets(noise, q1, q0), min_pos=20, alpha=0.1)["coverage_positive_macro"]
        auc_noise = float(np.nanmean(auroc_per_event(Y[test], noise)))
        auc_good = float(np.nanmean(auroc_per_event(Y[test], P[test])))
        self.assertGreaterEqual(cov_noise, 0.8)             # coverage looks fine ...
        self.assertAlmostEqual(auc_noise, 0.5, delta=0.03)  # ... for a score that carries no signal
        self.assertGreater(auc_good, 0.65)

    def test_lower_alpha_gives_larger_sets(self):
        Y, P, cal, test = self._setup(seed=8)
        sizes = []
        for alpha in (0.2, 0.05):
            q1, q0, _, _ = conformal_thresholds(P[cal], Y[cal], alpha)
            sizes.append(conformal_report(Y[test], *conformal_sets(P[test], q1, q0), min_pos=20, alpha=alpha)["mean_set_size"])
        self.assertGreater(sizes[1], sizes[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
