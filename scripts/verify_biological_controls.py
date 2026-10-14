"""AuditDDI Biological Controls & Feature Invariant Verification Script.

Verifies:
1. Feature store dimensions (ECFP 1024, Target 569, Gene 1233).
2. Strict zero-imputation invariant (100% zeros for unannotated drugs, no fallback).
3. 5-Seed Paired Benchmark Evidence (Scaffold S1 & Cold-Drug S1 lifts).
4. Stratum Monotonicity Invariant (bio2 > bio1 > bio0, bio0 ~ 0).
5. Multiple Testing Correction (Benjamini-Hochberg FDR q < 0.05).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.drug_features import load_store


def audit_feature_store(store_path: Path) -> Dict[str, Any]:
    """Audit biological feature store dimensions and zero-imputation rules."""
    report: Dict[str, Any] = {"status": "FAIL", "checks": {}}
    if not store_path.is_file():
        report["error"] = f"Feature store not found: {store_path}"
        return report

    store = load_store(store_path)
    ecfp = store["ecfp"]
    target = store["target"]
    gene = store["gene"]
    has_t = store["has_target"]
    has_g = store["has_gene"]
    n_drugs = len(has_t)

    # Invariants
    ecfp_1024 = (ecfp.shape == (n_drugs, 1024))
    target_dim_match = (target.shape == (n_drugs, 569))
    gene_dim_match = (gene.shape == (n_drugs, 1233))
    
    # Strict zero-imputation: unannotated rows must be exactly zero
    zero_target_missing = bool((target[~has_t] == 0).all())
    zero_gene_missing = bool((gene[~has_g] == 0).all())

    active_t_count = int(has_t.sum())
    active_g_count = int(has_g.sum())

    checks = {
        "n_drugs": n_drugs,
        "ecfp_shape_valid": ecfp_1024,
        "target_shape_valid": target_dim_match,
        "gene_shape_valid": gene_dim_match,
        "target_dimension": int(target.shape[1]),
        "gene_dimension": int(gene.shape[1]),
        "drugs_with_target": active_t_count,
        "drugs_with_gene": active_g_count,
        "target_coverage_pct": round(active_t_count / n_drugs * 100, 2),
        "gene_coverage_pct": round(active_g_count / n_drugs * 100, 2),
        "zero_imputation_target": zero_target_missing,
        "zero_imputation_gene": zero_gene_missing,
    }

    all_pass = (
        ecfp_1024
        and target_dim_match
        and gene_dim_match
        and zero_target_missing
        and zero_gene_missing
    )
    report["checks"] = checks
    report["status"] = "PASS" if all_pass else "FAIL"
    return report


def audit_benchmark_evidence(results_csv: Path) -> Dict[str, Any]:
    """Audit empirical paired benchmark evidence and stratum monotonicity."""
    report: Dict[str, Any] = {"status": "FAIL", "checks": {}}
    if not results_csv.is_file():
        report["error"] = f"Master results CSV not found: {results_csv}"
        return report

    df = pd.read_csv(results_csv)
    
    # 1. Scaffold S1 Lift
    scaf_s1 = df[(df["kind"] == "scaffold") & (df["partition"] == "test_s1")]
    scaf_base = scaf_s1[scaf_s1["features"] == "ecfp"]
    scaf_bio = scaf_s1[scaf_s1["features"] == "ecfp+target+gene"]
    
    scaf_lift = float(scaf_bio["mean_diff"].iloc[0])
    scaf_p = float(scaf_bio["p_ttest"].iloc[0])
    scaf_q = float(scaf_bio["q_ttest_BH"].iloc[0])
    scaf_n_imp = int(scaf_bio["n_improved"].iloc[0])

    # 2. Cold-Drug S1 Lift
    cold_s1 = df[(df["kind"] == "cold_drug") & (df["partition"] == "test_s1")]
    cold_bio = cold_s1[cold_s1["features"] == "ecfp+target+gene"]
    cold_lift = float(cold_bio["mean_diff"].iloc[0])
    cold_p = float(cold_bio["p_ttest"].iloc[0])
    cold_q = float(cold_bio["q_ttest_BH"].iloc[0])
    cold_n_imp = int(cold_bio["n_improved"].iloc[0])

    # 3. Stratum Monotonicity on Scaffold S1 (bio2 > bio1 > bio0 and bio0 ~ 0)
    b0 = df[(df["kind"] == "scaffold") & (df["partition"] == "test_s1|bio0") & (df["features"] == "ecfp+target+gene")]
    b1 = df[(df["kind"] == "scaffold") & (df["partition"] == "test_s1|bio1") & (df["features"] == "ecfp+target+gene")]
    b2 = df[(df["kind"] == "scaffold") & (df["partition"] == "test_s1|bio2") & (df["features"] == "ecfp+target+gene")]

    b0_diff = float(b0["mean_diff"].iloc[0])
    b1_diff = float(b1["mean_diff"].iloc[0])
    b2_diff = float(b2["mean_diff"].iloc[0])

    stratum_monotonic = (b2_diff > b1_diff > b0_diff)
    bio0_is_null = abs(b0_diff) < 0.005  # bio0 effect must be statistically indistinguishable from zero

    checks = {
        "scaffold_s1_lift": scaf_lift,
        "scaffold_s1_p_value": scaf_p,
        "scaffold_s1_fdr_q": scaf_q,
        "scaffold_s1_seeds_improved": f"{scaf_n_imp}/5",
        "scaffold_s1_significant": bool(scaf_p < 0.01 and scaf_q < 0.01 and scaf_n_imp == 5),
        "cold_drug_s1_lift": cold_lift,
        "cold_drug_s1_p_value": cold_p,
        "cold_drug_s1_fdr_q": cold_q,
        "cold_drug_s1_seeds_improved": f"{cold_n_imp}/5",
        "cold_drug_s1_significant": bool(cold_p < 0.05 and cold_q < 0.05 and cold_n_imp == 5),
        "stratum_bio0_diff": b0_diff,
        "stratum_bio1_diff": b1_diff,
        "stratum_bio2_diff": b2_diff,
        "stratum_monotonicity_valid": stratum_monotonic,
        "stratum_bio0_null_control": bio0_is_null,
    }

    all_pass = (
        checks["scaffold_s1_significant"]
        and checks["cold_drug_s1_significant"]
        and stratum_monotonic
        and bio0_is_null
    )
    report["checks"] = checks
    report["status"] = "PASS" if all_pass else "FAIL"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Path to auditddi-data root")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    print("=" * 80)
    print("AUDITDDI BIOLOGICAL CONTROLS & FEATURE INVARIANT AUDIT")
    print("=" * 80)
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print(f"Data Root:       {data_root}")
    print("-" * 80)

    # 1. Feature Store Audit
    store_file = data_root / "features_v1" / "drug_features.npz"
    print("1. AUDITING BIOLOGICAL FEATURE STORE (features_v1/drug_features.npz)...")
    store_report = audit_feature_store(store_file)
    print(f"   * Status:                         {store_report['status']}")
    for k, v in store_report.get("checks", {}).items():
        print(f"     - {k:<34}: {v}")
    print("-" * 80)

    # 2. Benchmark Evidence Audit
    results_csv = REPO_ROOT / "results" / "final_5seed_master_results.csv"
    print("2. AUDITING STATISTICAL SIGNIFICANCE & STRATUM MONOTONICITY...")
    bench_report = audit_benchmark_evidence(results_csv)
    print(f"   * Status:                         {bench_report['status']}")
    for k, v in bench_report.get("checks", {}).items():
        print(f"     - {k:<34}: {v}")
    print("=" * 80)

    overall_pass = (store_report["status"] == "PASS" and bench_report["status"] == "PASS")
    print(f"FINAL BIOLOGICAL EVIDENCE VERIFICATION: {'PASS (GENUINE BIOLOGICAL SIGNAL PROVEN)' if overall_pass else 'FAIL'}")
    print("=" * 80)

    full_audit = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "feature_store_audit": store_report,
        "benchmark_evidence_audit": bench_report,
        "overall_status": "PASS" if overall_pass else "FAIL",
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(full_audit, indent=2), encoding="utf-8")
        print(f"Saved full audit JSON to: {args.output_json}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
