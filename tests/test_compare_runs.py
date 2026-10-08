import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.compare_runs import compare, load_runs, main, paired_seed_comparison  # noqa: E402


def fake_results(effect=0.0, noise=0.004, n_seeds=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_seeds):
        base = 0.70 + rng.normal(0, 0.02)                 # split-to-split variation shared by both feature sets
        for feats, delta in (("ecfp", 0.0), ("ecfp+target", effect)):
            rows.append({"split": f"cold_drug_seed{s}", "kind": "cold_drug", "seed": s, "model": "logreg", "features": feats,
                         "stratified": True, "partition": "test_s2", "macro_auroc": base + delta + rng.normal(0, noise)})
    return pd.DataFrame(rows)


class TestPairedSeeds(unittest.TestCase):
    def test_real_effect_detected_despite_large_split_variance(self):
        df = fake_results(effect=0.03)
        r = compare(df, "ecfp", "macro_auroc")
        row = r[r.features == "ecfp+target"].iloc[0]
        self.assertGreater(row.mean_diff, 0.02)
        self.assertLess(row.p_ttest, 0.01)
        self.assertGreater(row.ci_low, 0)
        self.assertEqual(row.n_improved, 5)

    def test_no_effect_not_significant(self):
        df = fake_results(effect=0.0, seed=3)
        row = compare(df, "ecfp", "macro_auroc").query("features == 'ecfp+target'").iloc[0]
        self.assertGreater(row.p_ttest, 0.05)
        self.assertLess(row.ci_low, 0.0 + 1e-9 + 0.01)
        self.assertGreater(row.ci_high, -0.01)

    def test_wilcoxon_cannot_reach_005_with_five_seeds(self):
        # even a perfectly consistent improvement gives the exact two-sided minimum p = 2/32
        r = paired_seed_comparison(np.zeros(5), np.arange(1, 6) * 0.01)
        self.assertAlmostEqual(r["p_wilcoxon"], 0.0625, places=6)
        self.assertLess(r["p_ttest"], 0.05)

    def test_handles_missing_and_single_seed(self):
        r = paired_seed_comparison(np.array([0.5, np.nan]), np.array([0.6, 0.7]))
        self.assertEqual(r["n_seeds"], 1)
        self.assertTrue(np.isnan(r["p_ttest"]))

    def test_bh_adds_q_values(self):
        df = pd.concat([fake_results(0.03, seed=1), fake_results(0.0, seed=2).assign(partition="test_s1")])
        r = compare(df, "ecfp", "macro_auroc")
        self.assertIn("q_ttest_BH", r.columns)
        q = r.dropna(subset=["q_ttest_BH"])
        self.assertTrue((q.q_ttest_BH >= q.p_ttest - 1e-12).all())

    def test_cli_and_stratified_filter(self):
        df = pd.concat([fake_results(0.03), fake_results(0.0).assign(stratified=False, features="ecfp+junk")])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "results.csv"
            df.to_csv(p, index=False)
            runs = load_runs(p, "logreg", stratified=True)
            self.assertNotIn("ecfp+junk", set(runs.features))
            main(["--results", str(p), "--out", str(Path(d) / "cmp.csv")])
            self.assertTrue((Path(d) / "cmp.csv").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
