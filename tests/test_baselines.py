import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.fingerprints import pair_features  # noqa: E402
from src.baselines.models import FrequencyPrior, NumpyMultiLabelNet, RandomForestPCA, make_model  # noqa: E402
from src.evaluation.multilabel_metrics import auroc_per_event  # noqa: E402

try:
    import rdkit  # noqa: F401
    HAVE_RDKIT = True
except Exception:
    HAVE_RDKIT = False


def make_problem(n_drugs=60, bits=64, n_events=10, seed=0, mode="symmetric"):
    rng = np.random.default_rng(seed)
    F = (rng.random((n_drugs, bits)) < 0.25).astype(np.uint8)
    a, b = np.triu_indices(n_drugs, 1)
    X = pair_features(F, a, b, mode)
    Wt = rng.normal(size=(X.shape[1], n_events))
    z = (X - X.mean(0)) @ Wt / np.sqrt(X.shape[1]) * 2.0 - 1.5
    Y = (rng.random(z.shape) < 1 / (1 + np.exp(-z))).astype(np.float32)
    idx = rng.permutation(len(a))
    tr, va, te = idx[:1000], idx[1000:1300], idx[1300:]
    return F, a, b, X, Y, tr, va, te


class TestGradients(unittest.TestCase):
    def _check(self, hidden):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(7, 5))
        Y = (rng.random((7, 4)) < 0.4).astype(np.float64)
        pw = rng.uniform(0.5, 5.0, 4)
        net = NumpyMultiLabelNet(hidden=hidden, dropout=0.0, seed=1)
        net._init(5, 4, Y)
        net.P = {k: rng.normal(size=v.shape) * 0.5 for k, v in net.P.items()}   # float64
        loss, G = net.loss_and_grads(X, Y, pw, train=False)
        for k in net.P:
            num = np.zeros_like(net.P[k])
            it = np.nditer(net.P[k], flags=["multi_index"])
            for _ in it:
                ix = it.multi_index
                old = net.P[k][ix]
                net.P[k][ix] = old + 1e-6
                lp, _ = net.loss_and_grads(X, Y, pw)
                net.P[k][ix] = old - 1e-6
                lm, _ = net.loss_and_grads(X, Y, pw)
                net.P[k][ix] = old
                num[ix] = (lp - lm) / 2e-6
            self.assertTrue(np.allclose(G[k], num, atol=1e-7, rtol=1e-4), f"gradient mismatch in {k} (hidden={hidden})")

    def test_logreg_gradient(self):
        self._check(0)

    def test_mlp_gradient(self):
        self._check(6)


class TestLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.F, cls.a, cls.b, cls.X, cls.Y, cls.tr, cls.va, cls.te = make_problem()

    def _auc(self, model):
        model.fit(self.X[self.tr], self.Y[self.tr], self.X[self.va], self.Y[self.va])
        S = model.predict_proba(self.X[self.te])
        return float(np.nanmean(auroc_per_event(self.Y[self.te].astype(int), S))), S

    def test_prior_is_chance(self):
        auc, _ = self._auc(FrequencyPrior())
        self.assertAlmostEqual(auc, 0.5, places=6)

    def test_logreg_and_mlp_learn_signal(self):
        for name in ("logreg", "mlp32"):
            m = make_model(name, seed=0, epochs=40, patience=6, lr=3e-3, monitor="auroc", batch_size=64)
            auc, S = self._auc(m)
            self.assertGreater(auc, 0.7, name)
            self.assertTrue(((S >= 0) & (S <= 1)).all())

    def test_deterministic(self):
        outs = []
        for _ in range(2):
            m = make_model("mlp16", seed=3, epochs=5, batch_size=64, monitor="auroc")
            m.fit(self.X[self.tr], self.Y[self.tr], self.X[self.va], self.Y[self.va])
            outs.append(m.predict_proba(self.X[self.te]))
        self.assertTrue(np.array_equal(outs[0], outs[1]))

    def test_pos_weight_from_training_labels_only(self):
        m = make_model("logreg", seed=0, epochs=1, monitor="auroc")
        m.fit(self.X[self.tr], self.Y[self.tr], self.X[self.va], self.Y[self.va])
        pos = self.Y[self.tr].sum(0)
        self.assertTrue(np.allclose(m.pos_weight_, (len(self.tr) - pos) / np.maximum(pos, 1), rtol=1e-5))

    def test_early_stopping_restores_best(self):
        m = make_model("mlp16", seed=0, epochs=30, patience=2, lr=5e-2, monitor="auroc", batch_size=64)
        m.fit(self.X[self.tr], self.Y[self.tr], self.X[self.va], self.Y[self.va])
        vals = [h["val_auroc"] for h in m.history_ if "val_auroc" in h]
        self.assertAlmostEqual(m.best_val_, max(vals), places=6)

    def test_rf_pca(self):
        m = RandomForestPCA(n_components=6, n_estimators=30, max_leaf_nodes=200, seed=0, n_jobs=1)
        m.fit(self.X[self.tr], self.Y[self.tr])
        S = m.predict_proba(self.X[self.te])
        self.assertTrue(((S >= 0) & (S <= 1)).all())
        auc = float(np.nanmean(auroc_per_event(self.Y[self.te].astype(int), S)))
        self.assertGreater(auc, 0.6)
        self.assertGreater(m.explained_, 0.3)


class TestPairSymmetry(unittest.TestCase):
    def test_symmetric_features_identical_when_swapped(self):
        F, a, b, *_ = make_problem()
        self.assertTrue(np.array_equal(pair_features(F, a, b, "symmetric"), pair_features(F, b, a, "symmetric")))

    def test_concat_features_differ_when_swapped(self):
        F, a, b, *_ = make_problem()
        self.assertFalse(np.array_equal(pair_features(F, a, b, "concat"), pair_features(F, b, a, "concat")))

    def test_concat_model_is_order_sensitive_symmetric_is_not(self):
        out = {}
        for mode in ("symmetric", "concat"):
            F, a, b, X, Y, tr, va, te = make_problem(mode=mode, seed=1)
            m = make_model("mlp16", seed=0, epochs=15, lr=3e-3, batch_size=64, monitor="auroc").fit(X[tr], Y[tr], X[va], Y[va])
            S1 = m.predict_proba(pair_features(F, a[te], b[te], mode))
            S2 = m.predict_proba(pair_features(F, b[te], a[te], mode))
            out[mode] = float(np.abs(S1 - S2).mean())
        self.assertEqual(out["symmetric"], 0.0)
        self.assertGreater(out["concat"], 1e-4)


@unittest.skipUnless(HAVE_RDKIT, "RDKit not installed (runs on Colab)")
class TestECFP6(unittest.TestCase):
    def test_radius3_and_invalid(self):
        from src.baselines.fingerprints import ecfp6_matrix

        smi = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO", "not_a_smiles"]
        X6, ok = ecfp6_matrix(smi, radius=3)
        X4, _ = ecfp6_matrix(smi, radius=2)
        self.assertEqual(X6.shape, (3, 1024))
        self.assertEqual(ok.tolist(), [True, True, False])
        self.assertEqual(int(X6[2].sum()), 0)
        self.assertGreater(int(X6[0].sum()), int(X4[0].sum()))   # radius 3 sets more bits than radius 2


if __name__ == "__main__":
    unittest.main(verbosity=2)
