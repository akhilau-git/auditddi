"""Tests for scripts/verify_checkpoint_and_trust.py."""

from pathlib import Path
import pytest
import torch

from scripts.verify_checkpoint_and_trust import (
    audit_checkpoint,
    audit_trust_engine,
    audit_fastapi_routes,
)


def test_audit_checkpoint(tmp_path):
    ckpt_file = tmp_path / "dummy_model.pt"
    torch.save({
        "model_state_dict": {"conv.weight": torch.randn(10, 10)},
        "validation_auroc": 0.897,
        "hidden_channels": 64,
        "in_channels": 74,
        "epoch": 25,
        "seed": 11,
    }, ckpt_file)

    report = audit_checkpoint(ckpt_file)
    assert report["status"] == "PASS"
    assert report["checks"]["validation_auroc"] == 0.897
    assert report["checks"]["validation_auroc_valid"] is True


def test_audit_trust_engine():
    report = audit_trust_engine()
    assert report["status"] == "PASS"
    assert report["checks"]["conformal_calibration_functional"] is True
    assert report["checks"]["applicability_domain_functional"] is True


def test_audit_fastapi_routes():
    report = audit_fastapi_routes()
    assert report["status"] == "PASS"
    assert report["checks"]["expected_routes_registered"] is True
