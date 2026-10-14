import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.fingerprints import smiles_list_hash
from src.data_prep.build_event_table import build_event_table, save_event_table
from src.data_prep.make_splits import main as splits_main
from src.features.drug_features import save_store
from scripts.run_biological_ablation_experiment import (
    audit_feature_metadata,
    audit_split_non_leakage,
    build_ablation_features,
    run_controlled_experiment,
)


def mock_planted_environment(tmp_dir: Path, n_drugs: int = 50, n_events: int = 15):
    """Create a fully valid synthetic environment with event_table_v1, splits_v1, features_v1."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_drugs):
        for j in range(i + 1, min(n_drugs, i + 8)):
            ev_count = rng.integers(1, 4)
            evs = rng.choice(n_events, size=ev_count, replace=False)
            for e in evs:
                rows.append((f"D{i:03d}", f"D{j:03d}", f"event_{e:02d}"))

    raw_df = pd.DataFrame(rows, columns=["source", "target", "interaction_type"])
    t, _ = build_event_table(raw_df)

    # Save event table
    table_dir = tmp_dir / "event_table_v1"
    save_event_table(t, table_dir, {})

    # Generate splits
    splits_dir = tmp_dir / "splits_v1"
    splits_main([
        "--table", str(table_dir),
        "--out", str(splits_dir),
        "--seeds", "0",
        "--kinds", "cold_drug",
        "--min-pairs-full", "50",
    ])

    # Build drug feature store
    features_dir = tmp_dir / "features_v1"
    features_dir.mkdir(parents=True, exist_ok=True)
    ecfp = rng.integers(0, 2, size=(n_drugs, 1024), dtype=np.uint8)
    target = rng.integers(0, 2, size=(n_drugs, 569), dtype=np.uint8)
    gene = rng.integers(0, 2, size=(n_drugs, 1233), dtype=np.uint8)
    has_t = rng.random(n_drugs) < 0.5
    has_g = rng.random(n_drugs) < 0.4
    target[~has_t] = 0
    gene[~has_g] = 0

    store_dict = {
        "ecfp": ecfp,
        "ecfp_valid": np.ones(n_drugs, dtype=bool),
        "target": target,
        "has_target": has_t,
        "target_vocab": np.array([f"T_{i}" for i in range(569)]),
        "gene": gene,
        "has_gene": has_g,
        "gene_vocab": np.array([f"G_{i}" for i in range(1233)]),
        "smiles_sha256": np.array(smiles_list_hash(t.drugs["smiles"].tolist())),
        "gene_match_meta": np.array(json.dumps({})),
    }
    save_store(store_dict, features_dir / "drug_features.npz")
    return table_dir, splits_dir, features_dir


class TestBiologicalAblationExperiment(unittest.TestCase):
    def test_feature_metadata_and_ablation_matrices(self):
        rng = np.random.default_rng(0)
        n = 20
        store = {
            "ecfp": rng.integers(0, 2, size=(n, 1024), dtype=np.uint8),
            "ecfp_valid": np.ones(n, dtype=bool),
            "target": rng.integers(0, 2, size=(n, 569), dtype=np.uint8),
            "has_target": np.ones(n, dtype=bool),
            "gene": rng.integers(0, 2, size=(n, 1233), dtype=np.uint8),
            "has_gene": np.ones(n, dtype=bool),
        }

        meta = audit_feature_metadata(store)
        self.assertEqual(meta["ecfp"]["dimensionality"], 1024)
        self.assertEqual(meta["target"]["dimensionality"], 569)
        self.assertEqual(meta["gene"]["dimensionality"], 1233)

        # Configs
        f_ecfp, _ = build_ablation_features(store, "ecfp")
        self.assertEqual(f_ecfp.shape, (n, 1024))

        f_target, _ = build_ablation_features(store, "ecfp+target")
        self.assertEqual(f_target.shape, (n, 1024 + 569 + 1))

        f_gene, _ = build_ablation_features(store, "ecfp+gene")
        self.assertEqual(f_gene.shape, (n, 1024 + 1233 + 1))

        f_real, _ = build_ablation_features(store, "ecfp+target+gene")
        self.assertEqual(f_real.shape, (n, 1024 + 569 + 1 + 1233 + 1))

        f_shuff, _ = build_ablation_features(store, "ecfp+shuffled_bio", perm_seed=99)
        self.assertEqual(f_shuff.shape, (n, 1024 + 569 + 1 + 1233 + 1))
        # Structure portion must remain exactly identical to ecfp
        np.testing.assert_array_equal(f_shuff[:, :1024], f_real[:, :1024])

        f_zero, _ = build_ablation_features(store, "ecfp+zeroed_bio")
        self.assertEqual(f_zero.shape, (n, 1024 + 569 + 1 + 1233 + 1))
        np.testing.assert_array_equal(f_zero[:, 1024:], np.zeros((n, 569 + 1 + 1233 + 1)))

    def test_run_controlled_experiment_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            table_dir, splits_dir, features_dir = mock_planted_environment(tmp_path, n_drugs=40, n_events=10)

            out_json = tmp_path / "test_ablation.json"
            out_csv = tmp_path / "test_ablation.csv"

            report = run_controlled_experiment(
                data_root=tmp_path,
                split_name="cold_drug_seed0",
                model_name="logreg",
                epochs=3,
                batch_size=64,
                perm_seed=42,
                min_pos=2,
                out_json=out_json,
                out_csv=out_csv,
            )

            self.assertTrue(out_json.is_file())
            self.assertTrue(out_csv.is_file())
            self.assertIn("controlled_experiments", report)
            self.assertEqual(len(report["controlled_experiments"]), 6)
            self.assertTrue(report["split_anti_leakage_audit"]["s1_both_drugs_unseen"])
            self.assertIn("H1_real_biology_active", report["hypotheses_evaluation"])
            self.assertIn("H2_permutation_collapse", report["hypotheses_evaluation"])
            self.assertIn("H3_zeroed_control_invariance", report["hypotheses_evaluation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
