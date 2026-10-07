import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.multilabel_metrics import (  # noqa: E402
    auprc_per_event,
    auroc_per_event,
    benjamini_hochberg,
    best_global_threshold,
    bootstrap_ci,
    brier,
    calibration_slope_intercept,
    ece_mce,
    evaluable_mask,
    evaluate,
    macro_auroc_metric,
    micro_auprc,
    micro_auroc,
    paired_bootstrap,
    precision_recall_at_k,
    set_metrics,
)


def synth(n=600, m=12, seed=0, signal=1.0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.02, 0.3, m)
    latent = rng.normal(size=(n, m))
    Y = (rng.random((n, m)) < 1 / (1 + np.exp(-(np.log(base / (1 - base)) + 1.5 * latent)))).astype(np.int8)
    S = 1 / (1 + np.exp(-(signal * latent + rng.normal(scale=1.0, size=(n, m)) + np.log(base / (1 - base)))))
    return Y, S


class TestRanking(unittest.TestCase):
    def test_auroc_matches_sklearn_including_ties(self):
        Y, S = synth(seed=1)
        S = np.round(S, 1)                                   # force many ties
        ours = auroc_per_event(Y, S)
        ref = np.array([roc_auc_score(Y[:, j], S[:, j]) for j in range(Y.shape[1])])
        self.assertTrue(np.allclose(ours, ref, atol=1e-10))

    def test_auprc_matches_sklearn(self):
        Y, S = synth(seed=2)
        ref = np.array([average_precision_score(Y[:, j], S[:, j]) for j in range(Y.shape[1])])
        self.assertTrue(np.allclose(auprc_per_event(Y, S), ref))

    def test_micro_matches_sklearn(self):
        Y, S = synth(seed=3)
        self.assertAlmostEqual(micro_auroc(Y, S), roc_auc_score(Y.ravel(), S.ravel()), places=10)
        self.assertAlmostEqual(micro_auprc(Y, S), average_precision_score(Y.ravel(), S.ravel()), places=10)

    def test_single_class_column_is_nan(self):
        Y = np.array([[1, 0], [0, 0], [1, 0]])
        S = np.array([[0.9, 0.1], [0.2, 0.3], [0.8, 0.2]])
        a = auroc_per_event(Y, S)
        self.assertTrue(np.isnan(a[1]))
        self.assertEqual(a[0], 1.0)

    def test_evaluable_mask(self):
        Y = np.zeros((20, 3), dtype=int)
        Y[:6, 0] = 1
        Y[:2, 1] = 1
        self.assertEqual(evaluable_mask(Y, 5).tolist(), [True, False, False])

    def test_precision_recall_at_k_hand_computed(self):
        Y = np.array([[1, 0, 1, 0], [0, 1, 0, 0]])
        S = np.array([[0.9, 0.8, 0.1, 0.0], [0.1, 0.2, 0.9, 0.3]])
        r = precision_recall_at_k(Y, S, ks=(1, 2))
        # pair 1: top1={0} hit 1 ; top2={0,1} hit 1.  pair 2: top1={2} hit 0 ; top2={2,3} hit 0
        self.assertAlmostEqual(r["precision@1"], (1 + 0) / 2)
        self.assertAlmostEqual(r["precision@2"], (0.5 + 0) / 2)
        self.assertAlmostEqual(r["recall@2"], (0.5 + 0) / 2)


class TestSetMetrics(unittest.TestCase):
    def test_hand_computed(self):
        Y = np.array([[1, 1, 0], [0, 1, 0]])
        S = np.array([[0.9, 0.2, 0.8], [0.1, 0.9, 0.1]])
        r = set_metrics(Y, S, 0.5)                    # P = [[1,0,1],[0,1,0]]
        # pair1: tp=1 fp=1 fn=1 -> F1=2/4, J=1/3 ; pair2: tp=1 -> F1=1, J=1
        self.assertAlmostEqual(r["example_f1"], (0.5 + 1.0) / 2)
        self.assertAlmostEqual(r["jaccard"], (1 / 3 + 1.0) / 2)
        self.assertAlmostEqual(r["hamming_loss"], 2 / 6)
        # micro: TP=2 FP=1 FN=1 -> 4/6
        self.assertAlmostEqual(r["micro_f1"], 4 / 6)

    def test_empty_prediction_and_truth_is_perfect(self):
        Y = np.zeros((3, 4), dtype=int)
        S = np.zeros((3, 4))
        r = set_metrics(Y, S, 0.5)
        self.assertEqual(r["example_f1"], 1.0)
        self.assertEqual(r["hamming_loss"], 0.0)

    def test_threshold_selection_uses_given_data_only(self):
        Y, S = synth(seed=4)
        t = best_global_threshold(Y, S)
        self.assertTrue(0.05 <= t <= 0.95)


class TestCalibration(unittest.TestCase):
    def test_perfectly_calibrated(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.01, 0.99, 60000)
        y = (rng.random(60000) < p).astype(float)
        ece, _ = ece_mce(y, p)
        slope, inter = calibration_slope_intercept(y, p)
        self.assertLess(ece, 0.02)
        self.assertAlmostEqual(slope, 1.0, delta=0.05)
        self.assertAlmostEqual(inter, 0.0, delta=0.05)

    def test_overconfident_has_slope_below_one(self):
        rng = np.random.default_rng(1)
        p_true = rng.uniform(0.05, 0.95, 60000)
        y = (rng.random(60000) < p_true).astype(float)
        z = np.log(p_true / (1 - p_true)) * 2.5                 # inflate logits
        p = 1 / (1 + np.exp(-z))
        slope, _ = calibration_slope_intercept(y, p)
        self.assertLess(slope, 0.6)
        self.assertGreater(ece_mce(y, p)[0], 0.05)

    def test_brier_known(self):
        self.assertAlmostEqual(brier(np.array([1.0, 0.0]), np.array([0.8, 0.3])), (0.04 + 0.09) / 2)


class TestStatistics(unittest.TestCase):
    def test_bootstrap_ci_contains_estimate(self):
        Y, S = synth(n=400, seed=5)
        r = bootstrap_ci(macro_auroc_metric(5), Y, S, n_boot=60, seed=1)
        self.assertLessEqual(r["ci_low"], r["estimate"] + 1e-9)
        self.assertGreaterEqual(r["ci_high"], r["estimate"] - 1e-9)
        self.assertEqual(r["unit"], "pair")

    def test_paired_bootstrap_detects_real_difference(self):
        Y, S_good = synth(n=500, seed=6, signal=2.0)
        _, S_bad = synth(n=500, seed=6, signal=0.0)
        r = paired_bootstrap(macro_auroc_metric(5), Y, S_good, S_bad, n_boot=60, seed=2)
        self.assertGreater(r["diff"], 0.1)
        self.assertLess(r["p_value"], 0.05)

    def test_paired_bootstrap_same_model_not_significant(self):
        Y, S = synth(n=300, seed=7)
        r = paired_bootstrap(macro_auroc_metric(5), Y, S, S, n_boot=40, seed=3)
        self.assertEqual(r["diff"], 0.0)
        self.assertGreater(r["p_value"], 0.05)

    def test_benjamini_hochberg_known_example(self):
        q, rej = benjamini_hochberg([0.01, 0.04, 0.03, 0.005], alpha=0.05)
        self.assertTrue(np.allclose(q, [0.02, 0.04, 0.04, 0.02]))
        self.assertTrue(rej.all())


class TestEvaluate(unittest.TestCase):
    def test_evaluate_keys_and_sanity(self):
        Y, S = synth(n=800, seed=8, signal=1.5)
        r = evaluate(Y, S, threshold=0.3)
        for k in ("macro_auroc", "macro_auprc", "micro_auroc", "micro_f1", "precision@5", "macro_ece", "micro_nll", "n_events_evaluable"):
            self.assertIn(k, r)
        self.assertGreater(r["macro_auroc"], 0.6)
        r0 = evaluate(Y, np.random.default_rng(0).random(Y.shape), threshold=0.3)
        self.assertAlmostEqual(r0["macro_auroc"], 0.5, delta=0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
