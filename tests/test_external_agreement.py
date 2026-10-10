import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evidence.external_agreement import auroc, degree_baseline, evaluate_partition, flags_for_pairs, main, pair_scores  # noqa: E402


class TestPieces(unittest.TestCase):
    def test_auroc_matches_hand_value(self):
        self.assertEqual(auroc(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])), 1.0)
        self.assertEqual(auroc(np.array([1, 0, 1, 0]), np.array([0.1, 0.9, 0.2, 0.8])), 0.0)
        self.assertAlmostEqual(auroc(np.array([1, 0]), np.array([0.5, 0.5])), 0.5)
        self.assertTrue(np.isnan(auroc(np.array([1, 1]), np.array([0.1, 0.2]))))

    def test_degree_baseline_uses_training_pairs_only(self):
        a = np.array([0, 0, 1, 2, 3]); b = np.array([1, 2, 2, 3, 4])
        train = np.array([0, 1])                       # pairs (0,1), (0,2): drug0 degree 2, drug1 1, drug2 1
        s = degree_baseline(5, a, b, train, np.array([2, 4]))       # test pairs (1,2) and (3,4); 3,4 unseen in training
        self.assertAlmostEqual(s[0], 2 * np.log1p(1))
        self.assertEqual(s[1], 0.0)                                  # two unseen drugs: no popularity information at all

    def test_pair_scores_shapes_and_rank_invariance(self):
        S = np.random.default_rng(0).random((50, 30)).astype(np.float32)
        a = pair_scores(S)
        b = pair_scores(S * 0.01)                                    # a pure rescaling of every event
        self.assertTrue(np.allclose(a["mean_rank"], b["mean_rank"]))
        self.assertEqual(set(a), {"mean_prob", "top20_mean", "mean_rank"})

    def test_flags_symmetric_key(self):
        dd = pd.DataFrame({"drug_a": [1], "drug_b": [4], "severity": ["major"]})
        f = flags_for_pairs(np.array([4, 1, 2]), np.array([1, 4, 3]), dd, None)
        self.assertEqual(f["ddinter_any"].tolist(), [True, True, False])
        self.assertEqual(f["ddinter_moderate_major"].tolist(), [True, True, False])
        self.assertNotIn("label_mention", f)


class TestEvaluate(unittest.TestCase):
    def _data(self, informative: bool, n=3000, E=12, seed=0):
        rng = np.random.default_rng(seed)
        n_drugs = 200
        a = rng.integers(0, n_drugs, n); b = (a + 1 + rng.integers(0, n_drugs - 1, n)) % n_drugs
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        truth = rng.random(n) < 0.25
        dd = pd.DataFrame({"drug_a": lo[truth], "drug_b": hi[truth], "severity": "moderate"}).drop_duplicates(["drug_a", "drug_b"])
        S = rng.random((n, E)).astype(np.float32) * 0.2
        if informative:
            S[truth] += 0.25 * rng.random((truth.sum(), E)).astype(np.float32)
        return a, b, dd, S

    def test_signal_detected_and_noise_is_not(self):
        for informative in (True, False):
            a, b, dd, S = self._data(informative)
            rows = np.arange(len(a))
            df = evaluate_partition(S, rows, a, b, np.array([], dtype=int) if False else np.arange(0, 1), 200, dd, None, n_boot=40)
            r = df[(df.flag == "ddinter_any") & (df.score == "mean_rank")].iloc[0]
            if informative:
                self.assertGreater(r.ci_low, 0.6)
            else:
                self.assertLess(abs(r.auroc - 0.5), 0.06)

    def test_constant_baseline_flagged_not_hidden(self):
        a, b, dd, S = self._data(True)
        df = evaluate_partition(S, np.arange(len(a)), a, b, np.array([0]), 200, dd, None, n_boot=20)
        r = df[(df.flag == "ddinter_any") & (df.score == "popularity_baseline")]
        # degree from a single training pair is nearly constant: either flagged constant (AUROC 0.5) or close to chance
        self.assertTrue(bool(r.score_is_constant.iloc[0]) or abs(r.auroc.iloc[0] - 0.5) < 0.08)

    def test_too_few_positives_gives_nan_not_a_number(self):
        a, b, dd, S = self._data(True)
        dd = dd.iloc[:3]
        df = evaluate_partition(S, np.arange(len(a)), a, b, np.array([0]), 200, dd, None, n_boot=10, min_pos=20)
        self.assertTrue(df[df.flag == "ddinter_any"].auroc.isna().all())

    def test_popularity_confound_is_exposed(self):
        """If the curated list simply favours popular drugs, a score that only encodes popularity must look
        'good' too; this is exactly why the baseline column exists."""
        rng = np.random.default_rng(3)
        n_drugs, n = 150, 6000
        pop = rng.pareto(1.5, n_drugs) + 1
        p = pop / pop.sum()
        a = rng.choice(n_drugs, n, p=p); b = rng.choice(n_drugs, n, p=p)
        keep = a != b
        a, b = a[keep], b[keep]
        deg_score = np.log1p(pop[a]) + np.log1p(pop[b])
        truth = rng.random(len(a)) < 1 / (1 + np.exp(-(deg_score - np.median(deg_score))))
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        dd = pd.DataFrame({"drug_a": lo[truth], "drug_b": hi[truth], "severity": "major"}).drop_duplicates(["drug_a", "drug_b"])
        # a 'model' that only knows popularity (+ small noise), and training pairs that reveal popularity
        S = (deg_score[:, None] + rng.normal(0, 0.3, (len(a), 5))).astype(np.float32)
        train = np.arange(len(a))
        df = evaluate_partition(S, np.arange(len(a)), a, b, train, n_drugs, dd, None, n_boot=20)
        r = df[df.flag == "ddinter_any"].set_index("score")
        self.assertGreater(r.loc["popularity_baseline", "auroc"], 0.6)
        self.assertAlmostEqual(r.loc["mean_rank", "auroc"], r.loc["popularity_baseline", "auroc"], delta=0.05)   # model adds nothing beyond popularity


class TestCLI(unittest.TestCase):
    def test_cli(self):
        from src.baselines.fingerprints import smiles_list_hash
        from src.baselines.run_baselines import main as run_main
        from src.data_prep.build_event_table import save_event_table
        from src.data_prep.make_splits import main as splits_main
        from tests.test_run_baselines import planted_table

        t, F = planted_table(n_drugs=90, n_events=20, seed=2)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_event_table(t, d / "table", {})
            splits_main(["--table", str(d / "table"), "--out", str(d / "splits"), "--seeds", "0", "--kinds", "cold_drug", "--min-pairs-full", "300"])
            (d / "out").mkdir()
            np.savez_compressed(d / "out" / "ecfp6_r3_1024.npz", bits=F[t.drugs.drug_idx.to_numpy()], valid=np.ones(len(t.drugs), bool),
                                smiles_sha256=np.array(smiles_list_hash(t.drugs["smiles"].tolist())))
            run_main(["--table", str(d / "table"), "--splits", str(d / "splits"), "--out", str(d / "out"), "--names", "cold_drug_seed0",
                      "--models", "logreg", "--epochs", "6", "--min-pos", "3", "--save-scores"])
            pairs = t.pairs.sample(frac=0.3, random_state=0)
            pd.DataFrame({"drug_a": pairs.drug_a, "drug_b": pairs.drug_b, "severity": "moderate"}).to_csv(d / "dd.csv", index=False)
            main(["--table", str(d / "table"), "--splits", str(d / "splits"), "--baselines", str(d / "out"), "--ddinter", str(d / "dd.csv"),
                  "--names", "cold_drug_seed0", "--models", "logreg", "--n-boot", "10"])
            r = pd.read_csv(d / "out" / "external_agreement.csv")
            self.assertTrue({"popularity_baseline", "mean_rank", "mean_prob", "top20_mean"} <= set(r.score))
            self.assertTrue({"test_s1", "test_s2"} <= set(r.partition))


if __name__ == "__main__":
    unittest.main(verbosity=2)
