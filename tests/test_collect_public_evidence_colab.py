from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.collect_public_evidence_colab import collect_source, load_drug_names


def test_load_drug_names_accepts_twosides_catalog(tmp_path: Path) -> None:
    path = tmp_path / "twosides_drugs.csv"
    path.write_text("drug_name,drug_id\nAspirin,1\naspirin,2\nWarfarin,3\n", encoding="utf-8")

    assert load_drug_names(path) == ["Aspirin", "aspirin", "Warfarin"]


def test_collect_source_writes_cached_json_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(url: str, timeout: int) -> tuple[int, bytes]:
        assert "rxnav.nlm.nih.gov" in url
        return 200, b'{"idGroup":{"rxnormId":["1"]}}'

    monkeypatch.setattr("scripts.collect_public_evidence_colab.fetch_json", fake_fetch)
    manifest = collect_source("rxnorm", ["Aspirin"], tmp_path, delay=0, timeout=2)

    assert manifest["drug_count"] == 1
    assert (tmp_path / "rxnorm" / "Aspirin.json").is_file()
    saved = json.loads((tmp_path / "rxnorm" / "source_manifest.json").read_text())
    assert saved["records"][0]["sha256"] == manifest["records"][0]["sha256"]
