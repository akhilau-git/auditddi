"""Tests for scripts/verify_datasets_and_features.py."""

from pathlib import Path
import json
import pytest

from scripts.verify_datasets_and_features import (
    SOURCE_MANIFESTS,
    audit_anti_leakage_invariants,
    audit_biological_feature_layout,
    get_git_info,
    get_package_versions,
    run_verification,
    verify_benchmark_results,
)


def test_source_manifests_specifications():
    assert "TWOSIDES" in SOURCE_MANIFESTS
    assert "ChEMBL_DrugCentral" in SOURCE_MANIFESTS
    assert "PharmGKB_Genes" in SOURCE_MANIFESTS
    assert "PubChem_Structure" in SOURCE_MANIFESTS
    assert "RxNorm_Taxonomy" in SOURCE_MANIFESTS
    assert "BindingDB_Affinity" in SOURCE_MANIFESTS
    assert "FAERS_Toxicity" in SOURCE_MANIFESTS
    assert "openFDA_Bulk_SPL" in SOURCE_MANIFESTS
    assert SOURCE_MANIFESTS["TWOSIDES"]["records_count"] == 4649441
    assert SOURCE_MANIFESTS["TWOSIDES"]["unique_drugs"] == 645
    assert SOURCE_MANIFESTS["ChEMBL_DrugCentral"]["feature_dimension"] == 569
    assert SOURCE_MANIFESTS["PharmGKB_Genes"]["feature_dimension"] == 1233


def test_audit_biological_feature_layout():
    layout = audit_biological_feature_layout()
    assert layout["drugs_in_catalog"] == 645
    assert layout["drugs_with_any_biology"] == 565
    assert layout["drugs_with_neither_biology"] == 80
    assert layout["concatenated_total_dimension"] == 1024 + 2289 + 1234


def test_audit_anti_leakage_invariants():
    checks = audit_anti_leakage_invariants()
    for name, item in checks.items():
        assert item["status"] == "PASS"


def test_verify_benchmark_results():
    results_path = Path(__file__).resolve().parents[1] / "results" / "final_5seed_master_results.csv"
    bench = verify_benchmark_results(results_path)
    if not results_path.exists():
        assert bench["status"] == "FILE_NOT_FOUND"
        return
    assert bench["status"] == "VERIFIED"
    assert bench["scaffold_s1_lift"]["passes_statistical_threshold"] is True
    assert bench["cold_drug_s1_lift"]["passes_statistical_threshold"] is True
    assert bench["biological_stratum_control"]["proves_genuine_biological_signal"] is True


def test_run_verification_end_to_end(tmp_path):
    repo_dir = Path(__file__).resolve().parents[1]
    out_json = tmp_path / "verification_report.json"
    report = run_verification(repo_dir=repo_dir, output_json=out_json)
    assert report["overall_status"] == "VERIFIED_READY_FOR_REVIEW"
    assert out_json.is_file()
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["verification_protocol"] == "AuditDDI Formal Verification Protocol v1.0"
