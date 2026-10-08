"""Verify already-present AuditDDI source folders without downloading data.

The verifier is intentionally read-only with respect to source files. It
computes file metadata and SHA-256 digests, checks expected files where known,
and writes one verification manifest under the selected data root.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "twosides": {
        "required": {"drug_drug_edges.csv", "event_profile.csv", "side_effects.csv", "twosides_drugs.csv"},
        "license_status": "verify source terms before redistribution",
        "role": "primary reported DDI associations; absence is unknown",
    },
    "ddinter": {
        "required": {
            "raw/ddinter_downloads_code_A.csv", "raw/ddinter_downloads_code_B.csv",
            "raw/ddinter_downloads_code_D.csv", "raw/ddinter_downloads_code_H.csv",
            "raw/ddinter_downloads_code_L.csv", "raw/ddinter_downloads_code_P.csv",
            "raw/ddinter_downloads_code_R.csv", "raw/ddinter_downloads_code_V.csv",
            "normalized/ddinter_interactions_normalized.csv", "normalized/ddinter_manifest.json",
        },
        "license_status": "CC BY-NC-SA 4.0; verify current terms",
        "role": "curated interaction evidence; absence is unknown",
    },
    "faers": {"required": set(), "license_status": "FDA public data; verify release terms", "role": "pharmacovigilance evidence"},
    "PubChem": {"required": set(), "license_status": "verify NCBI terms", "role": "chemical identity and structure evidence"},
    "uniprot": {"required": {"uniprot_download_manifest.json"}, "license_status": "verify UniProt terms", "role": "protein sequence evidence"},
    "BindingDB": {"required": set(), "license_status": "verify permitted release terms", "role": "binding evidence"},
    "chembl": {"required": set(), "license_status": "verify ChEMBL release terms", "role": "chemical and target evidence"},
    "pharmgkb": {"required": set(), "license_status": "verify authorization and redistribution terms", "role": "pharmacogenomic evidence"},
    "PDB": {"required": set(), "license_status": "verify wwPDB terms", "role": "optional protein structure evidence"},
    "GEO": {"required": set(), "license_status": "verify GEO accession terms", "role": "additional expression evidence; not required by core source list"},
}

EXPECTED_HASHES = {
    "twosides/drug_drug_edges.csv": "b05f1c17a8c26966f468ae96a775c8621d6824613aba811f669dc62c8a3bba69",
    "ddinter/normalized/ddinter_interactions_normalized.csv": "ed580f51bbb02b645c86ad0f9ea2bed727e8391df30dc69bc8b207c81a69e21d",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(data_root: Path, source: str, *, skip_hash: bool = False) -> dict[str, Any]:
    spec = SOURCE_SPECS[source]
    folder = data_root / source
    files = sorted(path for path in folder.rglob("*") if path.is_file()) if folder.is_dir() else []
    file_records = []
    for path in files:
        relative = path.relative_to(data_root).as_posix()
        record = {"path": relative, "bytes": path.stat().st_size}
        if not skip_hash:
            record["sha256"] = sha256_file(path)
        file_records.append(record)
    present = {record["path"] for record in file_records}
    missing = sorted(set(spec["required"]) - present)
    expected_hash_results = []
    for relative, expected in EXPECTED_HASHES.items():
        if relative.startswith(f"{source}/"):
            actual = next((record.get("sha256") for record in file_records if record["path"] == relative), None)
            expected_hash_results.append({"path": relative, "expected": expected, "actual": actual, "matches": actual == expected})
    return {
        "folder": str(folder),
        "status": "verified" if folder.is_dir() and files and not missing else "needs_review",
        "file_count": len(file_records),
        "total_bytes": sum(record["bytes"] for record in file_records),
        "required_files": sorted(spec["required"]),
        "missing_required_files": missing,
        "files": file_records,
        "expected_hashes": expected_hash_results,
        "role": spec["role"],
        "license_status": spec["license_status"],
        "manual_review_required": not bool(expected_hash_results) or any(
            not result["matches"] for result in expected_hash_results
        ),
    }


def verify(data_root: str | Path, sources: list[str] | None = None, *, skip_hash: bool = False) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    selected = sources or list(SOURCE_SPECS)
    unknown = sorted(set(selected) - set(SOURCE_SPECS))
    if unknown:
        raise ValueError(f"Unknown source(s): {', '.join(unknown)}")
    manifest = {
        "dataset_root": str(root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify_existing_datasets.py",
        "read_only_source_verification": True,
        "hashes_skipped": skip_hash,
        "sources": {source: verify_source(root, source, skip_hash=skip_hash) for source in selected},
        "excluded_sources": {
            "drugbank": "excluded: written permission is required",
            "medispan": "excluded: commercial procurement is required",
        },
    }
    output = root / "verification_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", action="append", choices=sorted(SOURCE_SPECS))
    parser.add_argument("--skip-hash", action="store_true", help="Only inspect files; do not compute SHA-256.")
    args = parser.parse_args(argv)
    try:
        manifest = verify(args.data_root, args.source, skip_hash=args.skip_hash)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "dataset_root": manifest["dataset_root"],
        "manifest": str(Path(manifest["dataset_root"]) / "verification_manifest.json"),
        "sources": {name: details["status"] for name, details in manifest["sources"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
