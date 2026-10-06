"""Import official DDInter 2.0 exports without inventing binary labels.

DDInter's published files are identifier/name/severity evidence.  This loader
keeps the source fields and writes a normalized interaction table; it does not
turn unlisted pairs into negatives or claim that the output is SMILES-ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


DOWNLOAD_BASE = "https://ddinter2.scbdd.com/static/media/download"
EXPORT_CODES = ("A", "B", "D", "H", "L", "P", "R", "V")
ALIASES = {
    "drug1_id": ("drug1_id", "drug_a_id", "drugbank_id1", "drugbank_id_1", "drugid1", "ddinter_id1", "drug_a_ddinter_id"),
    "drug1_name": ("drug1_name", "drug_a_name", "drugname1", "drug_name_1", "drug1", "drug_a", "drug_a_drug_name"),
    "drug2_id": ("drug2_id", "drug_b_id", "drugbank_id2", "drugbank_id_2", "drugid2", "ddinter_id2", "drug_b_ddinter_id"),
    "drug2_name": ("drug2_name", "drug_b_name", "drugname2", "drug_name_2", "drug2", "drug_b", "drug_b_drug_name"),
    "severity": ("severity", "level", "risk_level", "interaction_level", "severity_level", "risk"),
    "description": ("description", "interaction_description", "mechanism", "effect", "interaction"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_official_exports(destination: str | Path, *, timeout: int = 90) -> list[Path]:
    """Download the eight CSV exports linked from DDInter's official page."""
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for code in EXPORT_CODES:
        name = f"ddinter_downloads_code_{code}.csv"
        response = requests.get(f"{DOWNLOAD_BASE}/{name}", timeout=timeout)
        response.raise_for_status()
        if len(response.content) < 100 or b"\x00" in response.content[:1024]:
            raise ValueError(f"DDInter download did not look like a CSV: {name}")
        path = folder / name
        path.write_bytes(response.content)
        # Fail early on changed/HTML response before proceeding with the other files.
        _read_table(path).head(0)
        downloaded.append(path)
    return downloaded


def _read_table(path: Path) -> pd.DataFrame:
    # DDInter mirrors may use CSV or TSV; infer the delimiter from the header.
    frame = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip().casefold().replace(" ", "_") for column in frame.columns]
    return frame


def normalize_export(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"DDInter export not found: {source}")
    frame = _read_table(source)
    selected: dict[str, str] = {}
    for output, candidates in ALIASES.items():
        match = next((name for name in candidates if name in frame.columns), None)
        if match:
            selected[output] = match
    if not ({"drug1_id", "drug1_name"} & selected.keys()) or not ({"drug2_id", "drug2_name"} & selected.keys()):
        raise ValueError(
            f"Unrecognized DDInter columns in {source.name}: {list(frame.columns)}. "
            "Inspect the official export and extend ALIASES without discarding its raw file."
        )
    result = pd.DataFrame(index=frame.index)
    for field in ALIASES:
        result[field] = frame[selected[field]] if field in selected else ""
    result["source_file"] = source.name
    result["source_row"] = range(2, len(result) + 2)
    result["source_label_semantics"] = "curated_DDInter_interaction_record; unlisted_pairs_are_unknown"
    return result


def import_directory(input_dir: str | Path, output_dir: str | Path) -> dict:
    source_dir, destination = Path(input_dir), Path(output_dir)
    files = sorted(source_dir.glob("ddinter_downloads_code_*.csv"))
    if not files:
        files = sorted(source_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No DDInter CSV exports found in {source_dir}")
    frames = [normalize_export(path) for path in files]
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["drug1_id", "drug2_id", "drug1_name", "drug2_name", "severity", "description"]
    )
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "ddinter_interactions_normalized.csv"
    combined.to_csv(output, index=False)
    manifest = {
        "dataset_name": "DDInter 2.0",
        "source_url_or_doi": "https://ddinter2.scbdd.com/download/",
        "license_terms": "CC BY-NC-SA 4.0; review official terms before redistribution/use",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_files": [{"path": str(p), "sha256": sha256_file(p), "rows": len(_read_table(p))} for p in files],
        "normalized_file": str(output),
        "normalized_sha256": sha256_file(output),
        "rows": len(combined),
        "label_definition": "Curated reported interaction records with source severity/description; absence is unknown, not a negative.",
        "split_definition": "Not split; source evidence import only.",
        "external_evaluation_ready": False,
        "reason_not_evaluation_ready": "Drug identity-to-SMILES mapping, overlap audit, and a defensible negative-label source are still required.",
    }
    (destination / "ddinter_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, nargs="?", help="Folder with official DDInter CSV exports")
    parser.add_argument("output_dir", type=Path, help="Output folder (prefer auditddi-results/external/DDInter)")
    parser.add_argument("--download", action="store_true", help="Download the official eight CSV exports into input_dir first")
    args = parser.parse_args()
    if args.download:
        if args.input_dir is None:
            parser.error("input_dir is required with --download")
        download_official_exports(args.input_dir)
    if args.input_dir is None:
        parser.error("input_dir is required")
    print(json.dumps(import_directory(args.input_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
