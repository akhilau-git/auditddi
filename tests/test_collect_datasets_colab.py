from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.collect_datasets_colab import collect, main


def test_collect_creates_source_folders_and_manifest(tmp_path: Path) -> None:
    manifest = collect(tmp_path, {})

    data_root = tmp_path / "auditddi-data"
    assert (data_root / "twosides").is_dir()
    assert (data_root / "uniprot").is_dir()
    assert (data_root / "collection_manifest.json").is_file()
    assert manifest["sources"]["twosides"]["status"] == "not_requested"
    assert json.loads((data_root / "collection_manifest.json").read_text())["research_only"]


def test_restricted_source_requires_explicit_permission(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="allow-licensed"):
        collect(tmp_path, {"chembl": ["https://example.org/chembl.tsv.gz"]})


def test_cli_rejects_invalid_url(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--project-root", str(tmp_path), "--url", "twosides=file:///not-http"]) == 2
    assert "absolute HTTP" in capsys.readouterr().err
