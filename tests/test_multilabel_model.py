"""Torch tests for the multi-label model and trainer.  They are SKIPPED when torch is not installed; run them on
your machine or in Colab:   python -m unittest tests.test_multilabel_model -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

from src.baselines.fingerprints import smiles_list_hash  # noqa: E402
from src.data_prep.build_event_table import save_event_table  # noqa: E402
from src.data_prep.make_splits import main as splits_main  # noqa: E402
from src.features.drug_features import build_store, feature_layout, feature_matrix, save_store  # noqa: E402

if HAVE_TORCH:
    from src.multilabel.model import AuditDDIML, drop_biology, load_checkpoint, predict_proba, save_checkpoint  # noqa: E402
    from src.multilabel.train import main as train_main  # noqa: E402
    from tests.test_run_baselines import planted_table  # noqa: E402


def small_store(n=12, bits=20, esm_dim=5, seed=0):
    rng = np.random.default_rng(seed)
    drugs = pd.DataFrame({"drug_idx": range(n), "smiles": [f"S{i}" for i in range(n)]})
    tg = pd.DataFrame({"drug_idx": [0, 1, 2, 3], "uniprot": ["P1", "P2", "P1", "P3"], "organism": "Homo sapiens", "target_class": ["GPCR", "Kinase", "GPCR", "Kinase"]})
    esm = {"esm": rng.normal(size=(n, esm_dim)).astype(np.float32), "has_esm": rng.random(n) < 0.6}
    esm["esm"][~esm["has_esm"]] = 0
    return build_store(drugs, (rng.random((n, bits)) < 0.3).astype(np.uint8), np.ones(n, bool), tg, pd.DataFrame({"canonical_smiles": [], "genes_list": []}),
                       canon_fn=lambda s: s, esm=esm)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.store = small_store()
        self.feats = ["ecfp", "target", "tclass", "esm"]
        self.layout, self.dim = feature_layout(self.store, self.feats)
        self.F = torch.tensor(feature_matrix(self.store, self.feats), dtype=torch.float32)
        self.model = AuditDDIML(self.layout, n_events=7, emb=8, tower_hidden=16, hidden=32, dropout=0.2).eval()

    def test_output_shape_and_raw_logits(self):
        out = self.model(self.F[:5], self.F[5:10])
        self.assertEqual(tuple(out.shape), (5, 7))
        self.assertTrue(torch.isfinite(out).all())

    def test_exact_permutation_symmetry(self):
        a, b = self.F[:8], self.F[4:12]
        with torch.no_grad():
            self.assertTrue(torch.allclose(self.model(a, b), self.model(b, a), atol=1e-6))

    def test_symmetry_also_in_training_mode_without_dropout(self):
        m = AuditDDIML(self.layout, 7, emb=8, tower_hidden=16, hidden=32, dropout=0.0).train()
        a, b = self.F[:8], self.F[4:12]
        with torch.no_grad():
            self.assertTrue(torch.allclose(m(a, b), m(b, a), atol=1e-6))       # batch norm sees the same pair vectors either way

    def test_unknown_biology_is_ignored_whatever_the_numbers_are(self):
        x = self.F[:6].clone()
        for name in ("target", "tclass", "esm"):
            L = self.layout[name]
            x[:, L["flag"]] = 0.0                                              # mark unknown
        garbage = x.clone()
        for name in ("target", "tclass", "esm"):
            L = self.layout[name]
            garbage[:, L["start"]:L["end"]] = torch.randn_like(garbage[:, L["start"]:L["end"]]) * 100
        with torch.no_grad():
            self.assertTrue(torch.allclose(self.model(x, self.F[6:12]), self.model(garbage, self.F[6:12]), atol=1e-6))

    def test_known_biology_does_change_the_output(self):
        x = self.F[:6].clone()
        y = x.clone()
        L = self.layout["esm"]
        y[:, L["start"]:L["end"]] += 5.0
        y[:, L["flag"]] = 1.0
        x[:, L["flag"]] = 1.0
        with torch.no_grad():
            self.assertFalse(torch.allclose(self.model(x, self.F[6:12]), self.model(y, self.F[6:12]), atol=1e-4))

    def test_drop_biology_zeroes_blocks_and_flags_but_not_fingerprints(self):
        x = self.F[:10].clone()
        for name in ("target", "tclass", "esm"):
            x[:, self.layout[name]["flag"]] = 1.0
        d = drop_biology(x, self.layout, p=1.0)
        for name in ("target", "tclass", "esm"):
            L = self.layout[name]
            self.assertTrue(torch.all(d[:, L["start"]:L["flag"] + 1] == 0))
        L = self.layout["ecfp"]
        self.assertTrue(torch.equal(d[:, L["start"]:L["end"]], x[:, L["start"]:L["end"]]))
        self.assertTrue(torch.equal(drop_biology(x, self.layout, p=0.0), x))
        self.assertTrue(torch.equal(x[:, 0], self.F[:10][:, 0]))              # the input tensor itself is not modified

    def test_every_parameter_receives_a_gradient(self):
        m = AuditDDIML(self.layout, 7, emb=8, tower_hidden=16, hidden=32, dropout=0.0).train()
        x = self.F.clone()
        for name in ("target", "tclass", "esm"):
            x[:, self.layout[name]["flag"]] = 1.0                                # make every biology tower active
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x[:8], x[4:12]), torch.rand(8, 7).round())
        loss.backward()
        missing = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        self.assertEqual(missing, [])

    def test_can_overfit_a_tiny_problem(self):
        torch.manual_seed(1)
        m = AuditDDIML(self.layout, 4, emb=16, tower_hidden=32, hidden=64, dropout=0.0).train()
        a, b = torch.randint(0, 12, (48,)), torch.randint(0, 12, (48,))
        y = (torch.rand(48, 4) < 0.5).float()
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        first = None
        for step in range(250):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(self.F[a], self.F[b]), y)
            opt.zero_grad(); loss.backward(); opt.step()
            first = float(loss) if first is None else first
        self.assertLess(float(loss), 0.5 * first)

    def test_output_bias_starts_at_prevalence(self):
        m = AuditDDIML(self.layout, 3, emb=8, tower_hidden=16, hidden=32)
        m.init_output_bias([0.1, 0.5, 0.9])
        self.assertTrue(torch.allclose(torch.sigmoid(m.head[-1].bias), torch.tensor([0.1, 0.5, 0.9]), atol=1e-5))

    def test_predict_proba_batches_and_restores_mode(self):
        m = AuditDDIML(self.layout, 7, emb=8, tower_hidden=16, hidden=32).train()
        a, b = torch.arange(0, 12), torch.arange(11, -1, -1)
        p1 = predict_proba(m, self.F, a, b, batch=5)
        p2 = predict_proba(m, self.F, a, b, batch=12)
        self.assertTrue(m.training)                                            # caller's mode is restored
        self.assertTrue(np.allclose(p1, p2, atol=1e-6))
        self.assertTrue(((p1 >= 0) & (p1 <= 1)).all())
        self.assertEqual(predict_proba(m, self.F, a[:0], b[:0]).shape, (0, 7))

    def test_checkpoint_roundtrip_gives_identical_predictions(self):
        a, b = torch.arange(0, 12), torch.arange(11, -1, -1)
        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(Path(d) / "m.pt", self.model, {"note": "test", "events": ["e%d" % i for i in range(7)]})
            m2, meta = load_checkpoint(Path(d) / "m.pt")
            self.assertEqual(meta["note"], "test")
            self.assertTrue((Path(d) / "m.json").exists())
            self.assertTrue(np.allclose(predict_proba(self.model, self.F, a, b), predict_proba(m2, self.F, a, b), atol=1e-6))

    def test_fingerprint_only_layout_works(self):
        lay, _ = feature_layout(self.store, ["ecfp"])
        m = AuditDDIML(lay, 3, emb=8, tower_hidden=16, hidden=32).eval()
        F = torch.tensor(feature_matrix(self.store, ["ecfp"]), dtype=torch.float32)
        self.assertEqual(tuple(m(F[:4], F[4:8]).shape), (4, 3))

    def test_empty_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            AuditDDIML({}, 3)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestTrainerEndToEnd(unittest.TestCase):
    def test_train_save_scores_and_feed_the_trust_pipeline(self):
        from src.trust.run_calibration import main as cal_main

        t, F = planted_table(n_drugs=110, n_events=20, seed=4)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_event_table(t, d / "table", {})
            splits_main(["--table", str(d / "table"), "--out", str(d / "splits"), "--seeds", "0", "--kinds", "cold_drug", "--min-pairs-full", "300", "--frac-val", "0.2"])
            rng = np.random.default_rng(0)
            n = len(t.drugs)
            esm = {"esm": rng.normal(size=(n, 4)).astype(np.float32), "has_esm": rng.random(n) < 0.7}
            tg = pd.DataFrame({"drug_idx": [0, 1], "uniprot": ["P1", "P2"], "organism": "Homo sapiens"})
            store = build_store(t.drugs, F[t.drugs.drug_idx.to_numpy()], np.ones(n, bool), tg, pd.DataFrame({"canonical_smiles": [], "genes_list": []}), canon_fn=lambda s: s, esm=esm)
            save_store(store, d / "store.npz")
            args = ["--table", str(d / "table"), "--splits", str(d / "splits"), "--name", "cold_drug_seed0", "--feature-store", str(d / "store.npz"),
                    "--features", "ecfp", "esm", "--out", str(d / "out"), "--epochs", "12", "--batch-size", "128", "--emb", "16", "--tower-hidden", "32",
                    "--hidden", "64", "--min-pos", "3", "--cpu", "--save-scores"]
            train_main(args)
            r = pd.read_csv(d / "out" / "results.csv")
            self.assertEqual(set(r.model), {"auditddi_ml"})
            self.assertEqual(set(r.features), {"ecfp+esm"})
            self.assertTrue({"val", "test_s1", "test_s2"} <= set(r.partition))
            s2 = r[r.partition == "test_s2"].macro_auroc.iloc[0]
            self.assertGreater(s2, 0.55)                                            # it learned something on a planted problem
            self.assertTrue((d / "out" / "checkpoints" / "auditddi_ml_cold_drug_seed0_ecfp+esm.pt").exists())
            n_rows = len(r)
            train_main(args)                                                         # second call: skipped, nothing appended
            self.assertEqual(len(pd.read_csv(d / "out" / "results.csv")), n_rows)
            train_main(args + ["--force"])                                           # --force replaces rather than duplicates
            self.assertEqual(len(pd.read_csv(d / "out" / "results.csv")), n_rows)
            # the saved scores plug straight into the calibration / conformal pipeline
            cal_main(["--table", str(d / "table"), "--splits", str(d / "splits"), "--baselines", str(d / "out"), "--names", "cold_drug_seed0",
                      "--models", "auditddi_ml", "--feature-tag", "ecfp+esm", "--min-pos", "3", "--alphas", "0.1"])
            cal = pd.read_csv(d / "out" / "calibration_results.csv")
            self.assertTrue({"raw", "platt", "conformal_alpha0.1"} <= set(cal.stage))



@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestTuneAndUnreported(unittest.TestCase):
    def _world(self, d):
        t, F = planted_table(n_drugs=110, n_events=20, seed=6)
        save_event_table(t, d / "table", {})
        splits_main(["--table", str(d / "table"), "--out", str(d / "splits"), "--seeds", "0", "--kinds", "cold_drug", "--min-pairs-full", "300", "--frac-val", "0.2"])
        n = len(t.drugs)
        rng = np.random.default_rng(0)
        esm = {"esm": rng.normal(size=(n, 4)).astype(np.float32), "has_esm": rng.random(n) < 0.7}
        tg = pd.DataFrame({"drug_idx": [0, 1], "uniprot": ["P1", "P2"], "organism": "Homo sapiens"})
        save_store(build_store(t.drugs, F[t.drugs.drug_idx.to_numpy()], np.ones(n, bool), tg, pd.DataFrame({"canonical_smiles": [], "genes_list": []}),
                               canon_fn=lambda s: s, esm=esm), d / "store.npz")
        return t

    def test_tuning_never_scores_test_and_picks_by_validation(self):
        from src.multilabel.tune import main as tune_main, sample_configs

        self.assertEqual(len(sample_configs(5, 0)), 5)
        self.assertEqual(sample_configs(5, 0), sample_configs(5, 0))                         # deterministic
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._world(d)
            tune_main(["--table", str(d / "table"), "--splits", str(d / "splits"), "--name", "cold_drug_seed0", "--feature-store", str(d / "store.npz"),
                       "--features", "ecfp", "esm", "--out", str(d / "tune"), "--n-configs", "2", "--epochs", "3", "--min-pos", "3", "--cpu"])
            log = pd.read_csv(d / "tune" / "tuning.csv")
            self.assertEqual(len(log), 2)
            for i in range(2):                                                               # no config run produced test results
                self.assertFalse((d / "tune" / f"cfg{i:02d}" / "results.csv").exists())
            self.assertTrue((d / "tune" / "best" / "results.csv").exists())                  # only the final fit scores test
            best = pd.read_csv(d / "tune" / "best" / "results.csv")
            self.assertIn("test_s2", set(best.partition))
            tune_main(["--table", str(d / "table"), "--splits", str(d / "splits"), "--name", "cold_drug_seed0", "--feature-store", str(d / "store.npz"),
                       "--features", "ecfp", "esm", "--out", str(d / "tune"), "--n-configs", "2", "--epochs", "3", "--min-pos", "3", "--cpu", "--no-final"])
            self.assertEqual(len(pd.read_csv(d / "tune" / "tuning.csv")), 2)                  # resumed, nothing re-run

    def test_external_unreported_cli(self):
        from src.evidence.external_unreported import main as ext_main

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            t = self._world(d)
            train_main(["--table", str(d / "table"), "--splits", str(d / "splits"), "--name", "cold_drug_seed0", "--feature-store", str(d / "store.npz"),
                        "--features", "ecfp", "esm", "--out", str(d / "out"), "--epochs", "3", "--emb", "8", "--tower-hidden", "16", "--hidden", "32", "--min-pos", "3", "--cpu"])
            ck = d / "out" / "checkpoints" / "auditddi_ml_cold_drug_seed0_ecfp+esm.pt"
            self.assertTrue(ck.exists())
            rng = np.random.default_rng(1)
            n = len(t.drugs)
            ii, jj = np.triu_indices(n, 1)
            pick = rng.random(len(ii)) < 0.3
            pd.DataFrame({"drug_a": ii[pick], "drug_b": jj[pick], "severity": "major"}).to_csv(d / "dd.csv", index=False)
            df = ext_main(["--table", str(d / "table"), "--splits", str(d / "splits"), "--name", "cold_drug_seed0", "--feature-store", str(d / "store.npz"),
                           "--features", "ecfp", "esm", "--checkpoint", str(ck), "--ddinter", str(d / "dd.csv"), "--out", str(d / "ev"), "--n-boot", "5", "--cpu"])
            self.assertTrue((d / "ev" / "external_unreported.csv").exists())
            self.assertTrue({"popularity_baseline", "mean_prob", "top20_mean", "mean_logit"} <= set(df.score))
            with self.assertRaises(ValueError):                                              # wrong feature set for this checkpoint is refused
                ext_main(["--table", str(d / "table"), "--splits", str(d / "splits"), "--name", "cold_drug_seed0", "--feature-store", str(d / "store.npz"),
                          "--features", "ecfp", "--checkpoint", str(ck), "--ddinter", str(d / "dd.csv"), "--out", str(d / "ev2"), "--cpu"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
