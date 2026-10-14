from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_existing_datasets import verify


def test_verify_records_files_and_known_hash(tmp_path: Path) -> None:
    twosides = tmp_path / "twosides"
    twosides.mkdir()
    (twosides / "drug_drug_edges.csv").write_text("source,target\n", encoding="utf-8")
    for name in ("event_profile.csv", "side_effects.csv", "twosides_drugs.csv"):
        (twosides / name).write_text("value\n", encoding="utf-8")

    manifest = verify(tmp_path, ["twosides"])

    assert manifest["sources"]["twosides"]["status"] == "verified"
    assert manifest["sources"]["twosides"]["file_count"] == 4
    assert (tmp_path / "verification_manifest.json").is_file()
    assert json.loads((tmp_path / "verification_manifest.json").read_text())["read_only_source_verification"]


def test_verify_marks_missing_source_for_review(tmp_path: Path) -> None:
    manifest = verify(tmp_path, ["uniprot"])

    details = manifest["sources"]["uniprot"]
    assert details["status"] == "needs_review"
    assert details["file_count"] == 0
    assert "uniprot_download_manifest.json" in details["missing_required_files"]


def test_verify_handles_nested_source_files(tmp_path: Path) -> None:
    ddinter = tmp_path / "ddinter"
    (ddinter / "raw").mkdir(parents=True)
    (ddinter / "normalized").mkdir()
    for code in ("A", "B", "D", "H", "L", "P", "R", "V"):
        (ddinter / "raw" / f"ddinter_downloads_code_{code}.csv").write_text(
            "drug1,drug2\n", encoding="utf-8"
        )
    (ddinter / "normalized" / "ddinter_interactions_normalized.csv").write_text(
        "drug1,drug2\n", encoding="utf-8"
    )
    (ddinter / "normalized" / "ddinter_manifest.json").write_text(
        "{}", encoding="utf-8"
    )

    manifest = verify(tmp_path, ["ddinter"])

    assert manifest["sources"]["ddinter"]["status"] == "verified"
    assert manifest["sources"]["ddinter"]["missing_required_files"] == []
