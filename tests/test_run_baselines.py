import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.fingerprints import smiles_list_hash  # noqa: E402
from src.baselines.run_baselines import main as run_main  # noqa: E402
from src.data_prep.build_event_table import build_event_table, save_event_table  # noqa: E402
from src.data_prep.make_splits import main as splits_main  # noqa: E402


def planted_table(n_drugs=70, n_events=30, seed=0, bits=64):
    """Events depend on drug bits, so a fingerprint model CAN learn them."""
    rng = np.random.default_rng(seed)
    F = (rng.random((n_drugs, bits)) < 0.25).astype(np.uint8)
    Wt = rng.normal(size=(bits, n_events))
    rows = []
    for i in range(n_drugs):
        for j in range(i + 1, n_drugs):
            if rng.random() < 0.5:
                z = (F[i] + F[j]) @ Wt / 4.0 - 1.0
                p = 1 / (1 + np.exp(-z))
                ev = np.where(rng.random(n_events) < p)[0]
                if len(ev) == 0:
                    ev = [int(rng.integers(n_events))]
                rows += [(f"D{i:03d}", f"D{j:03d}", f"ev{e}") for e in ev]
    t, _ = build_event_table(pd.DataFrame(rows, columns=["source", "target", "interaction_type"]))
    return t, F


class TestRunBaselines(unittest.TestCase):
    def test_end_to_end_and_resume(self):
        t, F = planted_table()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_event_table(t, d / "table", {})
            splits_main(["--table", str(d / "table"), "--out", str(d / "splits"), "--seeds", "0",
                         "--kinds", "transductive", "cold_drug", "--min-pairs-full", "400"])
            # fake fingerprint cache keyed to the real drug list
            np.savez_compressed(d / "out" / "fp.npz" if (d / "out").mkdir() is None else None,
                                bits=F[t.drugs.drug_idx.to_numpy()], valid=np.ones(len(t.drugs), bool),
                                smiles_sha256=np.array(smiles_list_hash(t.drugs["smiles"].tolist())))
            args = ["--table", str(d / "table"), "--splits", str(d / "splits"), "--out", str(d / "out"),
                    "--fingerprints", str(d / "out" / "fp.npz"), "--models", "prior", "logreg",
                    "--pair-modes", "symmetric", "concat", "--epochs", "12", "--min-pos", "3"]
            run_main(args)
            res = pd.read_csv(d / "out" / "results.csv")
            self.assertEqual(set(res.model), {"prior", "logreg"})
            self.assertTrue({"test_s1", "test_s2", "test_all", "val"} <= set(res.partition))
            # prior is at chance, a learned model is clearly above it
            prior = res[(res.model == "prior") & (res.partition == "test_all")].macro_auroc.iloc[0]
            lr = res[(res.model == "logreg") & (res.partition == "test_all") & (res.pair_mode == "symmetric")].macro_auroc.iloc[0]
            self.assertAlmostEqual(prior, 0.5, places=6)
            self.assertGreater(lr, 0.6)
            # order sensitivity recorded only for the concatenation ablation
            conc = res[(res.pair_mode == "concat") & (res.model == "logreg") & res.partition.str.startswith("test")]
            self.assertTrue((conc.order_sensitivity_mean_abs > 0).all())
            self.assertTrue(res[res.pair_mode == "symmetric"].order_sensitivity_mean_abs.isna().all())
            n1 = len(res)
            run_main(args)                                  # resume: nothing new may be appended
            self.assertEqual(len(pd.read_csv(d / "out" / "results.csv")), n1)

    def test_fingerprint_cache_mismatch_is_refused(self):
        from src.baselines.run_baselines import load_fingerprints

        t, F = planted_table(n_drugs=20)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fp.npz"
            np.savez_compressed(p, bits=F, valid=np.ones(20, bool), smiles_sha256=np.array("wrong"))
            with self.assertRaises(ValueError):
                load_fingerprints(t.drugs, p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
