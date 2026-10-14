import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_prep.build_event_table import build_event_table, save_event_table  # noqa: E402
from src.data_prep.make_splits import main as make_splits_main  # noqa: E402
from src.data_prep.multilabel_splits import (  # noqa: E402
    assert_clean,
    audit_split,
    cold_drug_split,
    event_support,
    random_pair_split,
    scaffold_split,
    scaled_min_pairs,
    transductive_split,
)


def synth_table(n_drugs=80, density=0.35, n_events=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_drugs):
        for j in range(i + 1, n_drugs):
            if rng.random() < density:
                k = rng.integers(1, 6)
                for e in rng.choice(n_events, size=k, replace=False):
                    rows.append((f"D{i:03d}", f"D{j:03d}", f"ev{e}"))
    t, _ = build_event_table(pd.DataFrame(rows, columns=["source", "target", "interaction_type"]))
    return t


def fake_scaffold(smiles: str) -> str:
    # groups of 4 consecutive drugs share a scaffold: D000-D003 -> S0 ...
    return f"S{int(smiles[1:]) // 4}"


class TestSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = synth_table()

    # ---- every pair accounted for --------------------------------------
    def test_partition_accounting(self):
        n = self.t.n_pairs
        for res in (
            random_pair_split(self.t.pairs, seed=1),
            transductive_split(self.t.pairs, seed=1),
            cold_drug_split(self.t.pairs, len(self.t.drugs), seed=1),
            scaffold_split(self.t.pairs, self.t.drugs, seed=1, scaffold_fn=fake_scaffold),
        ):
            total = len(res.train) + sum(len(v) for k, v in res.val.items() if k != "all" or "s2" not in res.val)
            total += sum(len(v) for v in res.test.values()) + len(res.dropped)
            self.assertEqual(total, n, res.kind)
            self.assertTrue(all(assert_clean(res, self.t.pairs).values()), res.kind)

    # ---- transductive: no unseen drugs -----------------------------------
    def test_transductive_all_drugs_seen(self):
        res = transductive_split(self.t.pairs, seed=3)
        a, b = self.t.pairs.drug_a.to_numpy(), self.t.pairs.drug_b.to_numpy()
        seen = set(a[res.train]) | set(b[res.train])
        for part in (res.val["all"], res.test["all"]):
            self.assertTrue(set(a[part]) <= seen and set(b[part]) <= seen)
        self.assertEqual(len(set(res.train) & set(res.test["all"])), 0)   # no exact pair overlap

    # ---- cold drug: S1 / S2 definitions -----------------------------------
    def test_cold_drug_s1_s2(self):
        res = cold_drug_split(self.t.pairs, len(self.t.drugs), 0.1, 0.2, seed=2)
        a, b = self.t.pairs.drug_a.to_numpy(), self.t.pairs.drug_b.to_numpy()
        role = res.drug_role
        self.assertTrue((role[a[res.train]] == 0).all() and (role[b[res.train]] == 0).all())
        s1, s2 = res.test["s1"], res.test["s2"]
        self.assertGreater(len(s1), 0)
        self.assertGreater(len(s2), 0)
        self.assertTrue(((role[a[s1]] == 2) & (role[b[s1]] == 2)).all())                  # both unseen
        self.assertTrue((((role[a[s2]] == 2).astype(int) + (role[b[s2]] == 2)) == 1).all())  # exactly one unseen
        n_test_drugs = (role == 2).sum()
        self.assertEqual(n_test_drugs, round(len(self.t.drugs) * 0.2))
        # a test drug never appears in any training pair
        test_drugs = set(np.where(role == 2)[0])
        self.assertFalse(test_drugs & (set(a[res.train]) | set(b[res.train])))

    # ---- scaffold: zero overlap -------------------------------------------
    def test_scaffold_disjoint(self):
        res = scaffold_split(self.t.pairs, self.t.drugs, 0.1, 0.2, seed=4, scaffold_fn=fake_scaffold)
        scaf = res.meta["scaffold_of_drug"]
        role = res.drug_role
        tr = {scaf[i] for i in np.where(role == 0)[0]}
        held = {scaf[i] for i in np.where(role != 0)[0]}
        self.assertFalse(tr & held)
        self.assertTrue(audit_split(res, self.t.pairs)["no_scaffold_overlap_train_vs_held_out"])
        self.assertGreater(len(res.test["s1"]) + len(res.test["s2"]), 0)

    def test_audit_detects_leakage(self):
        res = cold_drug_split(self.t.pairs, len(self.t.drugs), seed=5)
        res.train = np.sort(np.concatenate([res.train, res.test["s1"][:5]]))     # inject leakage
        self.assertFalse(all(audit_split(res, self.t.pairs).values()))
        with self.assertRaises(AssertionError):
            assert_clean(res, self.t.pairs)

    def test_audit_detects_scaffold_leak(self):
        res = scaffold_split(self.t.pairs, self.t.drugs, seed=6, scaffold_fn=fake_scaffold)
        self.assertTrue(audit_split(res, self.t.pairs)["no_scaffold_overlap_train_vs_held_out"])
        scaf = list(res.meta["scaffold_of_drug"])
        held = int(np.where(res.drug_role != 0)[0][0])
        train_drug = int(np.where(res.drug_role == 0)[0][0])
        scaf[train_drug] = scaf[held]                       # a training drug now shares a held-out scaffold
        res.meta["scaffold_of_drug"] = scaf
        self.assertFalse(audit_split(res, self.t.pairs)["no_scaffold_overlap_train_vs_held_out"])

    # ---- reproducibility ---------------------------------------------------
    def test_deterministic_and_seed_sensitive(self):
        a = cold_drug_split(self.t.pairs, len(self.t.drugs), seed=7)
        b = cold_drug_split(self.t.pairs, len(self.t.drugs), seed=7)
        c = cold_drug_split(self.t.pairs, len(self.t.drugs), seed=8)
        self.assertEqual(a.digest(), b.digest())
        self.assertNotEqual(a.digest(), c.digest())

    def test_event_support_flags_rare_events(self):
        from src.data_prep.build_event_table import freeze_vocabulary

        res = random_pair_split(self.t.pairs, seed=1)
        v = freeze_vocabulary(self.t.Y, res.train, self.t.events, min_train_pairs=3)
        sup = event_support(self.t.Y, v, {"train": res.train, "test": res.test["all"]}, min_pos=5)
        self.assertEqual(len(sup), len(v))
        self.assertIn("evaluable_test", sup.columns)

    def test_scaled_min_pairs(self):
        self.assertEqual(scaled_min_pairs(750, 63473, 63473), 750)
        self.assertEqual(scaled_min_pairs(750, 31736, 63473), 375)   # half the pairs -> half the threshold
        self.assertEqual(scaled_min_pairs(750, 31737, 63473), 376)   # always rounds up

    # ---- end to end through the CLI ------------------------------------------
    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            tdir, sdir = Path(d) / "table", Path(d) / "splits"
            save_event_table(self.t, tdir, {})
            make_splits_main(["--table", str(tdir), "--out", str(sdir), "--seeds", "0", "1",
                              "--kinds", "transductive", "cold_drug", "random_pair",
                              "--min-pairs-full", "60"])
            self.assertTrue((sdir / "splits_manifest.json").exists())
            self.assertTrue((sdir / "cold_drug_seed1_vocab.csv").exists())
            v = pd.read_csv(sdir / "cold_drug_seed0_vocab.csv")
            self.assertIn("pos_weight", v.columns)
            self.assertTrue((v.pos_weight > 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
