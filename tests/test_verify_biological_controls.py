"""Unit tests for scripts/verify_biological_controls.py."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.verify_biological_controls import audit_feature_store, audit_benchmark_evidence


def test_audit_feature_store(tmp_path):
    store_file = tmp_path / "drug_features.npz"
    n_drugs = 645
    ecfp = np.zeros((n_drugs, 1024), dtype=np.uint8)
    target = np.zeros((n_drugs, 569), dtype=np.uint8)
    gene = np.zeros((n_drugs, 1233), dtype=np.uint8)
    has_t = np.zeros(n_drugs, dtype=bool)
    has_g = np.zeros(n_drugs, dtype=bool)
    has_t[:320] = True
    has_g[:241] = True

    np.savez_compressed(
        store_file,
        ecfp=ecfp,
        target=target,
        gene=gene,
        has_target=has_t,
        has_gene=has_g,
        smiles_sha256=np.array("test"),
    )

    report = audit_feature_store(store_file)
    assert report["status"] == "PASS"
    assert report["checks"]["n_drugs"] == 645
    assert report["checks"]["zero_imputation_target"] is True
    assert report["checks"]["zero_imputation_gene"] is True


def test_audit_benchmark_evidence():
    master_csv = Path("results/final_5seed_master_results.csv")
    report = audit_benchmark_evidence(master_csv)
    if not master_csv.is_file():
        assert report["status"] == "FAIL"
        assert "not found" in report.get("error", "").lower()
        return
    assert report["status"] == "PASS"
    assert report["checks"]["scaffold_s1_significant"] is True
    assert report["checks"]["stratum_monotonicity_valid"] is True
    assert report["checks"]["stratum_bio0_null_control"] is True
