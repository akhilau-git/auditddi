"""Download an authorized DrugBank full-database release into auditddi-data.

Credentials are read only from AUDITDDI_DRUGBANK_EMAIL and
AUDITDDI_DRUGBANK_PASSWORD. Never pass them as command-line arguments or log
them. This script does not grant DrugBank access or permission; use it only
when the account and intended use are authorized by DrugBank.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

import requests


PORTAL = "https://portal.drugbank.com"
RELEASE_LIST_URL = f"{PORTAL}/releases.json"
TIMEOUT = (30, 120)
CHUNK_SIZE = 1024 * 1024
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def parse_release_list(payload: object) -> list[dict]:
    """Accept the documented JSON array and common wrapped-array responses."""
    if isinstance(payload, list):
        releases = payload
    elif isinstance(payload, dict):
        releases = next(
            (payload[key] for key in ("releases", "data", "items") if isinstance(payload.get(key), list)),
            None,
        )
        if releases is None:
            raise ValueError("DrugBank release JSON did not contain a release list.")
    else:
        raise ValueError("DrugBank release JSON was not a list or object.")

    valid = [item for item in releases if isinstance(item, dict) and VERSION_PATTERN.fullmatch(str(item.get("version", "")))]
    if not valid:
        raise ValueError("No version entries were found in DrugBank release JSON.")
    return valid


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_release(version: str, destination_root: Path, *, force: bool = False) -> Path:
    email = os.environ.get("AUDITDDI_DRUGBANK_EMAIL", "").strip()
    password = os.environ.get("AUDITDDI_DRUGBANK_PASSWORD", "")
    if not email or not password:
        raise RuntimeError(
            "Set AUDITDDI_DRUGBANK_EMAIL and AUDITDDI_DRUGBANK_PASSWORD from a secret manager first."
        )

    session = requests.Session()
    session.auth = (email, password)
    session.headers.update({"User-Agent": "AuditDDI authorized dataset downloader"})

    releases_response = session.get(RELEASE_LIST_URL, timeout=TIMEOUT, allow_redirects=True)
    releases_response.raise_for_status()
    releases = parse_release_list(releases_response.json())
    selected_version = releases[0]["version"] if version == "latest" else version
    if not VERSION_PATTERN.fullmatch(selected_version):
        raise ValueError("Version must be 'latest' or a version such as 5.1.22.")
    matching = next((item for item in releases if item["version"] == selected_version), None)
    if matching is None:
        raise ValueError(f"DrugBank did not list release {selected_version} for this request.")

    release_slug = selected_version.replace(".", "-")
    release_details_url = f"{PORTAL}/releases/{release_slug}.json"
    details_response = session.get(release_details_url, timeout=TIMEOUT, allow_redirects=True)
    details_response.raise_for_status()
    release_details = details_response.json()

    version_dir = destination_root / "DrugBank" / "raw" / selected_version
    version_dir.mkdir(parents=True, exist_ok=True)
    archive_path = version_dir / f"drugbank-{selected_version}-full-database.xml.zip"
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    if archive_path.exists() and not force:
        raise FileExistsError(f"File already exists: {archive_path}. Use --force to replace it.")

    download_url = f"{PORTAL}/releases/{release_slug}/downloads/all-full-database"
    response = session.get(download_url, stream=True, timeout=TIMEOUT, allow_redirects=True)
    try:
        response.raise_for_status()
        with partial_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    output.write(chunk)

        if not zipfile.is_zipfile(partial_path):
            content_type = response.headers.get("Content-Type", "unknown")
            raise ValueError(
                "The response is not a valid ZIP archive. Access may be unavailable "
                f"for this account/release (Content-Type: {content_type})."
            )
        with zipfile.ZipFile(partial_path) as archive:
            corrupt_member = archive.testzip()
            if corrupt_member:
                raise ValueError(f"Downloaded ZIP failed integrity check at member: {corrupt_member}")

        if archive_path.exists():
            archive_path.unlink()
        partial_path.replace(archive_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    finally:
        response.close()

    manifest = {
        "dataset": "DrugBank full database XML archive",
        "version": selected_version,
        "released_on": matching.get("released_on"),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "release_list_url": RELEASE_LIST_URL,
        "release_details_url": release_details_url,
        "download_route": f"/releases/{release_slug}/downloads/all-full-database",
        "release_details_saved": False,
        "release_details_type": type(release_details).__name__,
        "license_note": "Use/retention/redistribution must follow the account license and written permission.",
    }
    manifest_path = version_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="latest", help="Release version (for example 5.1.22) or latest.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("AUDITDDI_DATA_BASE", "/content/drive/MyDrive/auditddi-data")),
        help="Google Drive auditddi-data directory.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing archive for this release.")
    args = parser.parse_args()

    try:
        result = download_release(args.version, args.data_root, force=args.force)
    except requests.HTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        print(
            f"DrugBank request failed with HTTP {status}. Check account access, release availability, "
            "and the written license permissions. Credentials were not printed.",
            file=sys.stderr,
        )
        return 1
    except Exception as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1

    print(f"Verified DrugBank archive saved to: {result}")
    print(f"Manifest saved to: {result.parent / 'download_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
