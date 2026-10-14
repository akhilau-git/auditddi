"""Live FastAPI Inference & Full-Stack Route Verification Script.

Executes real end-to-end inference and diagnostic tests across:
- GET /health
- GET /ready
- POST /predict (Aspirin + Acetaminophen, Warfarin + Aspirin)
- POST /explain
- POST /api/audit/dossier
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
ACETAMINOPHEN = "CC(=O)NC1=CC=C(C=C1)O"
WARFARIN = "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O"


def run_live_api_smoke_test() -> Dict[str, Any]:
    """Execute live requests against the backend application."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    report: Dict[str, Any] = {"status": "FAIL", "endpoints": {}}

    # 1. GET /health
    r_health = client.get("/health")
    h_ok = (r_health.status_code == 200)
    h_data = r_health.json() if h_ok else {}
    report["endpoints"]["health"] = {
        "status_code": r_health.status_code,
        "ready": h_data.get("ready"),
        "model_loaded": h_data.get("model_loaded"),
        "checkpoint_sha256": h_data.get("model_checkpoint_sha256"),
        "stored_val_auroc": h_data.get("stored_validation_evidence", {}).get("auroc"),
    }

    # 2. GET /ready
    r_ready = client.get("/ready")
    ready_ok = (r_ready.status_code == 200)
    report["endpoints"]["ready"] = {
        "status_code": r_ready.status_code,
        "status": r_ready.json().get("status") if ready_ok else None,
    }

    # 3. POST /predict (Aspirin + Acetaminophen)
    pred_payload = {"smiles_a": ASPIRIN, "smiles_b": ACETAMINOPHEN}
    r_pred = client.post("/predict", json=pred_payload)
    pred_ok = (r_pred.status_code == 200)
    p_data = r_pred.json() if pred_ok else {}
    report["endpoints"]["predict_aspirin_acetaminophen"] = {
        "status_code": r_pred.status_code,
        "model_score": p_data.get("model_score"),
        "interaction_risk_note": p_data.get("interaction_risk_note"),
        "drug_a_toxicity": p_data.get("drug_a_toxicity", {}).get("score"),
        "drug_b_toxicity": p_data.get("drug_b_toxicity", {}).get("score"),
        "calibration_status": p_data.get("score_calibration", {}).get("status"),
    }

    # 4. POST /predict (Warfarin + Aspirin - high clinical significance pair)
    r_warf = client.post("/predict", json={"smiles_a": WARFARIN, "smiles_b": ASPIRIN})
    warf_ok = (r_warf.status_code == 200)
    w_data = r_warf.json() if warf_ok else {}
    report["endpoints"]["predict_warfarin_aspirin"] = {
        "status_code": r_warf.status_code,
        "model_score": w_data.get("model_score"),
        "drug_a_toxicity": w_data.get("drug_a_toxicity", {}).get("score"),
        "drug_b_toxicity": w_data.get("drug_b_toxicity", {}).get("score"),
    }

    # 5. POST /explain
    r_exp = client.post("/explain", json=pred_payload)
    exp_ok = (r_exp.status_code == 200)
    e_data = r_exp.json() if exp_ok else {}
    report["endpoints"]["explain"] = {
        "status_code": r_exp.status_code,
        "explanation_status": e_data.get("status") or ("available" if exp_ok else "failed"),
    }

    # 6. POST /api/audit/dossier
    dossier_url = "/api/audit/dossier" if any(route.path == "/api/audit/dossier" for route in app.routes if hasattr(route, "path")) else "/audit/dossier"
    r_dos = client.post(dossier_url, json=pred_payload)
    dos_ok = (r_dos.status_code == 200)
    d_data = r_dos.json() if dos_ok else {}
    report["endpoints"]["audit_dossier"] = {
        "endpoint": dossier_url,
        "status_code": r_dos.status_code,
        "dominant_cyp": d_data.get("metabolic_collision", {}).get("dominant_cyp"),
        "risk_probability": d_data.get("risk", {}).get("ddi_probability"),
        "decision": d_data.get("decision"),
        "markdown_generated": bool(d_data.get("markdown")),
    }

    all_ok = h_ok and ready_ok and pred_ok and warf_ok and exp_ok and dos_ok
    report["status"] = "PASS" if all_ok else "FAIL"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    print("=" * 80)
    print("AUDITDDI LIVE FASTAPI INFERENCE & ENDPOINT CONTRACT AUDIT")
    print("=" * 80)
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print("-" * 80)

    report = run_live_api_smoke_test()
    print(f"Overall Service Status: {report['status']}")
    print("-" * 80)

    for ep, details in report["endpoints"].items():
        sc = details.get("status_code", "N/A")
        print(f"* Endpoint: {ep:<32} [Status: {sc}]")
        for k, v in details.items():
            if k != "status_code":
                print(f"    - {k:<28}: {v}")
    print("=" * 80)

    overall_pass = (report["status"] == "PASS")
    print(f"FINAL FULL-STACK INFERENCE VERIFICATION: {'PASS (ALL ENDPOINTS OPERATIONAL)' if overall_pass else 'FAIL'}")
    print("=" * 80)

    full_audit = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": report,
        "overall_status": "PASS" if overall_pass else "FAIL",
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(full_audit, indent=2), encoding="utf-8")
        print(f"Saved full audit JSON to: {args.output_json}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
