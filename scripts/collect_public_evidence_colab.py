"""Collect public identity and regulatory evidence for the project drug list.

This script performs bounded, cached API retrieval for RxNorm, openFDA, and
DailyMed. It uses the existing TWOSIDES drug catalog as input and stores one
raw JSON response per drug plus a provenance manifest. It does not create DDI
labels: regulatory text and identity matches are source evidence only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


USER_AGENT = "AuditDDI-public-evidence/1.0 (research-only)"
ENDPOINTS = {
    "rxnorm": "https://rxnav.nlm.nih.gov/REST/rxcui.json?name={name}",
    "openfda": "https://api.fda.gov/drug/label.json?search=openfda.generic_name:{name}&limit=100",
    "dailymed": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name={name}&pagesize=100",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned[:100] or "empty"


def load_drug_names(path: str | Path, column: str | None = None) -> tuple[list[str], str]:
    """Read unique query identifiers from a CSV and return them with the column used."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Drug catalog has no header: {source}")
        fields = {field.casefold().strip(): field for field in reader.fieldnames}
        candidates = ("drug_name", "drugname", "name", "drug", "source", "drug_id")
        selected_column = fields.get(column.casefold().strip()) if column else None
        if selected_column is None:
            selected_column = next((fields[name] for name in candidates if name in fields), None)
        if selected_column is None:
            raise ValueError(
                f"Could not find a drug-name column in {source}; "
                f"observed columns: {reader.fieldnames}"
            )
        names = {
            str(row.get(selected_column, "")).strip()
            for row in reader
            if str(row.get(selected_column, "")).strip()
        }
    return sorted(names, key=lambda value: (value.casefold(), value)), selected_column


def fetch_json(url: str, timeout: int) -> tuple[int, bytes]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except HTTPError as exc:
        return int(exc.code), exc.read()
    except URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc}") from exc


def collect_source(
    source: str,
    names: list[str],
    output_root: Path,
    *,
    query_column: str,
    delay: float,
    timeout: int,
    overwrite: bool = False,
) -> dict:
    folder = output_root / source
    folder.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for index, name in enumerate(names, start=1):
        destination = folder / f"{_safe_name(name)}.json"
        url = ENDPOINTS[source].format(name=quote(name, safe=""))
        if destination.is_file() and not overwrite:
            status = "cached"
        else:
            http_status, payload = fetch_json(url, timeout)
            destination.write_bytes(payload)
            status = "downloaded"
            if http_status >= 400:
                status = f"http_{http_status}"
        records.append({
            "drug_name": name,
            "url": url,
            "path": destination.relative_to(output_root).as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "status": status,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        if index < len(names) and delay:
            time.sleep(delay)
    manifest = {
        "source": source,
        "source_role": {
            "rxnorm": "identity resolution; not DDI labels",
            "openfda": "regulatory label evidence; not pairwise ground truth",
            "dailymed": "structured labeling evidence; absence is unknown",
        }[source],
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "drug_count": len(names),
        "query_column": query_column,
        "records": records,
    }
    manifest_path = folder / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--drug-file", type=Path, required=True)
    parser.add_argument(
        "--drug-column",
        help="Column containing query identifiers; defaults to common names, then drug_id.",
    )
    parser.add_argument("--source", action="append", choices=sorted(ENDPOINTS), default=list(ENDPOINTS))
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.delay < 0 or args.timeout <= 0:
        parser.error("--delay must be non-negative and --timeout must be positive")
    try:
        names, query_column = load_drug_names(args.drug_file, args.drug_column)
        manifests = {
            source: collect_source(
                source, names, args.data_root, query_column=query_column, delay=args.delay,
                timeout=args.timeout, overwrite=args.overwrite,
            )
            for source in args.source
        }
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "data_root": str(args.data_root.resolve()),
        "drug_count": len(names),
        "query_column": query_column,
        "sources": {source: manifest["drug_count"] for source, manifest in manifests.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
