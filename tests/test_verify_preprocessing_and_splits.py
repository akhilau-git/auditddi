"""Tests for scripts/verify_preprocessing_and_splits.py."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scripts.verify_preprocessing_and_splits import audit_event_table, audit_splits


def test_audit_event_table_on_synthetic_data(tmp_path):
    table_dir = tmp_path / "event_table_v1"
    table_dir.mkdir()

    drugs = pd.DataFrame({"drug_idx": range(645), "smiles": ["CC"] * 645})
    drugs.to_csv(table_dir / "drugs.csv", index=False)

    pairs = pd.DataFrame({"pair_id": range(63473), "drug_a": [0] * 63473, "drug_b": [1] * 63473})
    pairs.to_csv(table_dir / "pairs.csv", index=False)

    events = pd.DataFrame({"event_idx": range(1317), "event": [f"ev_{i}" for i in range(1317)], "n_pairs": [1] * 1317})
    events.to_csv(table_dir / "events_all.csv", index=False)

    # Sparse matrix
    Y = sp.csr_matrix((63473, 1317), dtype=np.uint8)
    np.savez(table_dir / "labels.npz", data=Y.data, indices=Y.indices, indptr=Y.indptr, shape=Y.shape)

    report = audit_event_table(table_dir)
    assert report["status"] == "PASS"
    assert report["checks"]["expected_n_drugs_match"] is True
    assert report["checks"]["expected_n_pairs_match"] is True
    assert report["checks"]["symmetry_drug_a_less_than_drug_b"] is True


def test_audit_splits_on_synthetic_data(tmp_path):
    from src.data_prep.multilabel_splits import (
        cold_drug_split,
        scaffold_split,
        transductive_split,
        save_split,
        audit_split,
    )

    # 20 drugs, all pairs
    n_drugs = 20
    drugs = pd.DataFrame({
        "drug_idx": range(n_drugs),
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"] * 10 + ["C1=CC=CC=C1"] * 10,
    })
    pairs_list = []
    pid = 0
    for i in range(n_drugs):
        for j in range(i + 1, n_drugs):
            pairs_list.append({"pair_id": pid, "drug_a": i, "drug_b": j})
            pid += 1
    pairs = pd.DataFrame(pairs_list)

    splits_dir = tmp_path / "splits_v1"
    splits_dir.mkdir()
    pairs_file = tmp_path / "pairs.csv"
    drugs_file = tmp_path / "drugs.csv"
    pairs.to_csv(pairs_file, index=False)
    drugs.to_csv(drugs_file, index=False)

    # Create dummy vocab for checks
    dummy_vocab = pd.DataFrame({"head_idx": [0], "event": ["ev0"], "event_idx": [0]})

    for kind in ["cold_drug", "scaffold", "transductive"]:
        for seed in [0, 1]:
            if kind == "cold_drug":
                res = cold_drug_split(pairs, n_drugs, seed=seed)
            elif kind == "scaffold":
                res = scaffold_split(pairs, drugs, seed=seed)
            else:
                res = transductive_split(pairs, seed=seed)

            audit_res = audit_split(res, pairs)
            name = f"{kind}_seed{seed}"
            save_split(res, splits_dir, name, audit_res)
            dummy_vocab.to_csv(splits_dir / f"{name}_vocab.csv", index=False)

    report = audit_splits(splits_dir, pairs_file, drugs_file, seeds=[0, 1])
    assert report["status"] == "PASS", f"Audit failed: {report}"
    assert report["total_splits_verified"] == 6
    for split_name, s_info in report["splits_audited"].items():
        assert s_info["status"] == "PASS"
        assert s_info["pair_leakage_zero"] is True
        assert s_info["s1_clean"] is True
        assert s_info["s2_semi_cold_valid"] is True
        assert s_info["scaffold_clean"] is True

