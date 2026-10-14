"""Collect source datasets into a project-local ``auditddi-data`` directory.

This is an acquisition/bootstrap tool for Google Colab.  It deliberately does
not guess unstable download URLs, scrape licensed sources, or turn a missing
source into an empty dataset.  Every downloaded file is written atomically and
listed with its URL, retrieval time, byte count, and SHA-256 digest.

Example:
    python scripts/collect_datasets_colab.py \
        --project-root /content/drive/MyDrive/AuditDDI \
        --source all \
        --url twosides=https://example.org/twosides.zip \
        --url faers=https://example.org/faers.zip \
        --url chembl=https://example.org/chembl.tsv.gz

Use one ``--url SOURCE=URL`` per public or separately authorized source.  The
URL is intentionally supplied by the operator because releases and licenses
change.  DrugBank and Medi-Span are blocked unless explicitly enabled.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SOURCE_NAMES = (
    "rxnorm",
    "pubchem",
    "twosides",
    "ddinter",
    "kegg",
    "openfda",
    "dailymed",
    "drugcentral",
    "chembl",
    "bindingdb",
    "pharmgkb",
    "uniprot",
    "pdb",
    "faers",
)
BLOCKED_SOURCES = {"drugbank", "medispan"}
DDINTER_CODES = ("A", "B", "D", "H", "L", "P", "R", "V")
DEFAULT_USER_AGENT = "AuditDDI-dataset-collector/1.0 (research-only)"


def _safe_source(value: str) -> str:
    source = value.strip().casefold()
    if source == "all":
        return source
    if source not in SOURCE_NAMES and source not in BLOCKED_SOURCES:
        allowed = ", ".join((*SOURCE_NAMES, *sorted(BLOCKED_SOURCES)))
        raise ValueError(f"Unknown source {value!r}; choose one of: {allowed}.")
    return source


def _parse_url(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected SOURCE=URL, got {value!r}.")
    source, url = value.split("=", 1)
    source = _safe_source(source)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"URL for {source} must be an absolute HTTP(S) URL.")
    return source, url


def _filename_from_url(url: str, source: str, ordinal: int) -> str:
    name = Path(urlparse(url).path).name
    if not name or name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        name = f"{source}_{ordinal:03d}.download"
    return name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, timeout: int, user_agent: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{destination.name}.", suffix=".part",
        dir=destination.parent, delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            request = Request(url, headers={"User-Agent": user_agent})
            with urlopen(request, timeout=timeout) as response:
                shutil.copyfileobj(response, temporary)
            temporary.flush()
        except (HTTPError, URLError, TimeoutError) as exc:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"Download failed for {url}: {exc}") from exc
    temporary_path.replace(destination)
    return {
        "url": url,
        "path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def collect(
    project_root: str | Path,
    urls: dict[str, list[str]],
    *,
    timeout: int = 120,
    user_agent: str = DEFAULT_USER_AGENT,
    allow_licensed: bool = False,
    allow_drugbank: bool = False,
) -> dict[str, Any]:
    """Create source folders, download explicit URLs, and write a manifest."""
    root = Path(project_root).expanduser().resolve()
    data_root = root / "auditddi-data"
    data_root.mkdir(parents=True, exist_ok=True)
    for source in (*SOURCE_NAMES, *sorted(BLOCKED_SOURCES)):
        (data_root / source).mkdir(exist_ok=True)

    if not allow_licensed:
        restricted = {"kegg", "pharmgkb", "drugcentral", "bindingdb", "chembl"}
        requested = {source for source in restricted if urls.get(source)}
        if requested:
            raise PermissionError(
                "Refusing restricted sources without --allow-licensed: "
                + ", ".join(sorted(requested))
            )
    if urls.get("drugbank") and not allow_drugbank:
        raise PermissionError(
            "DrugBank is legally gated and blocked by default. Do not redistribute "
            "it; use --allow-drugbank only when written permission covers this run."
        )
    if urls.get("medispan"):
        raise PermissionError("Medi-Span collection is not supported by this tool.")

    entries: list[dict[str, Any]] = []
    for source in (*SOURCE_NAMES, *sorted(BLOCKED_SOURCES)):
        for ordinal, url in enumerate(urls.get(source, []), start=1):
            filename = _filename_from_url(url, source, ordinal)
            target = data_root / source / filename
            file_entry = _download(url, target, timeout, user_agent)
            file_entry.update({"source": source, "retrieved_at_utc": datetime.now(timezone.utc).isoformat()})
            entries.append(file_entry)

    manifest = {
        "dataset_root": str(data_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collector": "scripts/collect_datasets_colab.py",
        "research_only": True,
        "sources": {
            source: {
                "folder": str(data_root / source),
                "requested_urls": urls.get(source, []),
                "downloaded_files": [entry for entry in entries if entry["source"] == source],
                "status": "collected" if any(entry["source"] == source for entry in entries) else "not_requested",
            }
            for source in (*SOURCE_NAMES, *sorted(BLOCKED_SOURCES))
        },
        "label_semantics": {
            "twosides": "reported associations; absence is unknown, not a negative",
            "faers": "pharmacovigilance reports; disproportionality is not causal proof",
            "other_sources": "source-specific evidence; not automatic DDI labels",
        },
    }
    manifest_path = data_root / "collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], help="Source to initialize (default: all).")
    parser.add_argument("--url", action="append", default=[], metavar="SOURCE=URL")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--allow-licensed", action="store_true")
    parser.add_argument("--allow-drugbank", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    try:
        parsed = [_parse_url(value) for value in args.url]
        urls: dict[str, list[str]] = {}
        for source, url in parsed:
            urls.setdefault(source, []).append(url)
        requested_values = args.source if args.source else ["all"]
        requested: list[str] = []
        for value in requested_values:
            source = _safe_source(value)
            requested.extend((*SOURCE_NAMES, *sorted(BLOCKED_SOURCES)) if source == "all" else [source])
        for source in requested:
            urls.setdefault(source, [])
        manifest = collect(
            args.project_root,
            urls,
            timeout=args.timeout,
            allow_licensed=args.allow_licensed,
            allow_drugbank=args.allow_drugbank,
        )
    except (OSError, PermissionError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "dataset_root": manifest["dataset_root"],
        "downloaded_files": sum(len(item["downloaded_files"]) for item in manifest["sources"].values()),
        "manifest": str(Path(manifest["dataset_root"]) / "collection_manifest.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
