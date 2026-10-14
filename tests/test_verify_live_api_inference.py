"""Tests for scripts/verify_live_api_inference.py."""

from scripts.verify_live_api_inference import run_live_api_smoke_test


def test_live_api_smoke_test():
    report = run_live_api_smoke_test()
    assert report["status"] == "PASS"
    assert report["endpoints"]["health"]["status_code"] == 200
    assert report["endpoints"]["ready"]["status_code"] == 200
    assert report["endpoints"]["predict_aspirin_acetaminophen"]["status_code"] == 200
    assert report["endpoints"]["audit_dossier"]["status_code"] == 200
