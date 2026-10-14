import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_prep.build_event_table import (  # noqa: E402
    build_event_table,
    freeze_vocabulary,
    labels_for,
    load_event_table,
    pos_weights,
    save_event_table,
    vocabulary_hash,
)


def toy_edges():
    # drugs A,B,C,D ; events x,y,z
    rows = [
        ("A", "B", "Nausea"),
        ("B", "A", "nausea "),      # reversed + case/space duplicate of the row above
        ("A", "B", "x"),
        ("A", "C", "x"),
        ("C", "A", "y"),
        ("B", "C", "x"),
        ("B", "D", "z"),
        ("C", "D", "x"),
        ("C", "D", "x"),            # exact duplicate
        ("D", "D", "x"),            # self pair -> dropped
        ("A", None, "x"),           # missing -> dropped
    ]
    return pd.DataFrame(rows, columns=["source", "target", "interaction_type"])


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.t, self.rep = build_event_table(toy_edges())

    def test_counts_and_cleaning(self):
        self.assertEqual(self.rep["rows_dropped_missing"], 1)
        self.assertEqual(self.rep["rows_dropped_self_pairs"], 1)
        self.assertEqual(self.rep["n_drugs"], 4)
        self.assertEqual(self.rep["n_pairs"], 5)           # AB AC BC BD CD
        self.assertEqual(self.t.Y.max(), 1)                  # duplicates collapsed to 1
        self.assertGreaterEqual(self.rep["duplicate_rows_collapsed"], 2)

    def test_pair_order_independent(self):
        pairs = self.t.pairs
        self.assertTrue((pairs.drug_a < pairs.drug_b).all())
        self.assertEqual(len(pairs), len(pairs.drop_duplicates(["drug_a", "drug_b"])))

    def test_event_normalisation(self):
        self.assertIn("nausea", set(self.t.events.event))
        self.assertNotIn("Nausea", set(self.t.events.event))

    def test_reversed_rows_same_pair(self):
        d = {s: i for i, s in zip(self.t.drugs.drug_idx, self.t.drugs.smiles)}
        pid = self.t.pairs[(self.t.pairs.drug_a == d['A']) & (self.t.pairs.drug_b == d['B'])].pair_id.iloc[0]
        ev = {e: i for i, e in zip(self.t.events.event_idx, self.t.events.event)}
        self.assertEqual(self.t.Y[pid, ev["nausea"]], 1)
        self.assertEqual(self.t.Y[pid, ev["x"]], 1)

    def test_deterministic(self):
        t2, _ = build_event_table(toy_edges().sample(frac=1.0, random_state=3))
        self.assertTrue(self.t.events.equals(t2.events))
        self.assertTrue(self.t.pairs.equals(t2.pairs))
        self.assertEqual((self.t.Y != t2.Y).nnz, 0)

    def test_roundtrip_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            m = save_event_table(self.t, Path(d), self.rep)
            self.assertIn("labels.npz", m["outputs_sha256"])
            t2 = load_event_table(Path(d))
            self.assertEqual((self.t.Y != t2.Y).nnz, 0)
            self.assertEqual(list(t2.events.event), list(self.t.events.event))


class TestVocabularyLeakage(unittest.TestCase):
    def setUp(self):
        # 100 pairs, 3 events.  event0 common, event1 only appears in rows >= 80,
        # event2 appears in 10 rows.
        rng = np.random.default_rng(0)
        rows = []
        n = 100
        for i in range(n):
            a, b = f"d{i}a", f"d{i}b"
            if i % 2 == 0:
                rows.append((a, b, "e0"))
            if i >= 80:
                rows.append((a, b, "e1"))
            if i % 10 == 0:
                rows.append((a, b, "e2"))
            if i % 2 == 1 and i < 80 and i % 10 != 0:
                rows.append((a, b, "e2x"))   # keep every pair non-empty
        self.t, _ = build_event_table(pd.DataFrame(rows, columns=["source", "target", "interaction_type"]))
        self.n = self.t.n_pairs

    def test_vocab_uses_train_only(self):
        train = np.arange(0, 80)       # e1 is invisible in training
        test = np.arange(80, self.n)
        v = freeze_vocabulary(self.t.Y, train, self.t.events, min_train_pairs=5)
        self.assertNotIn("e1", set(v.event))                 # test-only event cannot enter the vocabulary
        v_all = freeze_vocabulary(self.t.Y, np.arange(self.n), self.t.events, min_train_pairs=5)
        self.assertIn("e1", set(v_all.event))                # sanity: it would, had we peeked at test labels
        self.assertEqual(len(set(train) & set(test)), 0)

    def test_pos_weight_train_only(self):
        train = np.arange(0, 80)
        v = freeze_vocabulary(self.t.Y, train, self.t.events, min_train_pairs=5)
        w = pos_weights(v)
        e0 = v.index[v.event == "e0"][0]
        self.assertAlmostEqual(float(w[e0]), 40 / 40, places=5)   # 40 pos, 40 neg in train
        # changing test labels must not change the weights
        Y2 = self.t.Y.copy().tolil()
        Y2[80:, :] = 0
        v2 = freeze_vocabulary(Y2.tocsr(), train, self.t.events, min_train_pairs=5)
        self.assertTrue(np.allclose(pos_weights(v2), w))
        self.assertEqual(vocabulary_hash(v2), vocabulary_hash(v))

    def test_label_block_shape(self):
        v = freeze_vocabulary(self.t.Y, np.arange(80), self.t.events, min_train_pairs=5)
        Yb = labels_for(self.t.Y, np.arange(80, self.n), v)
        self.assertEqual(Yb.shape, (self.n - 80, len(v)))
        self.assertTrue(set(np.unique(Yb)) <= {0.0, 1.0})

    def test_event_needs_negatives(self):
        v = freeze_vocabulary(self.t.Y, np.arange(0, 10), self.t.events, min_train_pairs=1, min_train_negatives=1)
        self.assertTrue((v.train_neg >= 1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
