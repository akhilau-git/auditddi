"""Universal Dataset and Path Resolver for AuditDDI.

Handles transparent resolution across local development, Windows environments,
and Google Colab runtimes (both primary accounts and multi-account rotations
using shared Google Drive shortcuts).

Folder Conventions:
- Google Drive Datasets: ``/content/drive/MyDrive/auditddi-data``
  (or shortcut at ``/content/drive/.shortcut-targets-by-id/*/auditddi-data``)
- Google Drive Outputs : ``/content/drive/MyDrive/auditddi-results``
- Datasets inside ``auditddi-data`` include:
  TWOSIDES, FAERS, UniProt, BindingDB, PharmGKB, ChEMBL, PubChem, PDB, GEO,
  STITCH, TCGA, ZINC.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_auditddi_env(key_suffix: str, default: Any = None) -> Any:
    """Retrieve an environment variable prioritizing AUDITDDI_* with fallback to PXDDI_*."""
    auditddi_key = f"AUDITDDI_{key_suffix}"
    pxddi_key = f"PXDDI_{key_suffix}"
    val = os.environ.get(auditddi_key)
    if val is not None:
        return val
    val = os.environ.get(pxddi_key)
    if val is not None:
        return val
    return default


def resolve_data_base(configured_path: str | Path | None = None) -> Path:
    """Resolve the dataset root directory containing all dataset subfolders.

    Supports:
    1. Explicit parameter or AUDITDDI_DATA_BASE / PXDDI_DATA_BASE env vars.
    2. Colab primary Drive: /content/drive/MyDrive/auditddi-data
    3. Colab shared shortcut: /content/drive/.shortcut-targets-by-id/*/auditddi-data
    4. Local paths: ./auditddi-data, Quantum_AI_Pharma/datasets, ./data
    5. Automatic repair of nested Drive shortcut paths.
    """
    candidates: list[Path] = []

    # Priority 1: Configured path or environment variable
    env_path = configured_path or get_auditddi_env("DATA_BASE")
    if env_path:
        p = Path(env_path)
        # Check if caller passed an erroneously nested shortcut path
        parts = p.parts
        marker = ".shortcut-targets-by-id"
        if marker in parts:
            idx = parts.index(marker)
            fixed = Path("/content/drive").joinpath(*parts[idx:])
            if fixed.is_dir():
                return fixed
        if p.is_dir():
            return p
        # An explicit argument or environment setting is a user-selected
        # dataset location. Falling through to an unrelated local/Drive
        # directory can silently train or evaluate on the wrong data.
        raise FileNotFoundError(
            f"Configured AUDITDDI_DATA_BASE does not exist or is not a directory: {p}"
        )

    # Priority 2: Google Colab primary path
    colab_primary = Path("/content/drive/MyDrive/auditddi-data")
    if colab_primary.is_dir():
        return colab_primary
    candidates.append(colab_primary)

    # Priority 3: Google Colab multi-account shared shortcuts
    shortcut_base = Path("/content/drive/.shortcut-targets-by-id")
    if shortcut_base.is_dir():
        for pattern in ["*/auditddi-data", "*/AuditDDI-data", "*/auditddi", "*/pxddi-data", "*"]:
            matches = glob.glob(str(shortcut_base / pattern))
            for m in matches:
                mp = Path(m)
                if mp.is_dir():
                    # Check if this folder contains known datasets or matches the name
                    if "auditddi-data" in mp.name.lower() or (mp / "TWOSIDES").is_dir() or (mp / "twosides").is_dir():
                        return mp

    # Priority 4: Legacy Drive paths
    for legacy_name in ["auditddi", "pxddi-data", "AuditDDI"]:
        lp = Path("/content/drive/MyDrive") / legacy_name
        if lp.is_dir():
            return lp

    # Priority 5: Local workspace development paths
    local_candidates = [
        PROJECT_ROOT / "auditddi-data",
        PROJECT_ROOT.parent / "Quantum_AI_Pharma" / "datasets",
        Path(r"D:\Drug-Drug Interaction\Quantum_AI_Pharma\datasets"),
        PROJECT_ROOT / "data",
        Path("auditddi-data"),
    ]
    for lc in local_candidates:
        if lc.is_dir():
            return lc
        candidates.append(lc)

    # If we reached here, no valid path was found. Provide a helpful error.
    checked_str = "\n".join(f"  - {c}" for c in candidates[:6])
    raise FileNotFoundError(
        f"AuditDDI dataset root directory could not be found.\n"
        f"Checked paths include:\n{checked_str}\n\n"
        f"Fix:\n"
        f"  1. In Google Colab, make sure Google Drive is mounted:\n"
        f"     from google.colab import drive; drive.mount('/content/drive')\n"
        f"  2. Confirm 'auditddi-data' is present in your Drive or added as a shortcut to MyDrive.\n"
        f"  3. Alternatively, export AUDITDDI_DATA_BASE=/path/to/auditddi-data"
    )


def resolve_results_base(configured_path: str | Path | None = None) -> Path:
    """Resolve the writable directory for run artifacts, logs, and checkpoints.

    Crucial for Colab multi-account rotations: shared dataset shortcuts are
    read-only, so all outputs must write to the current user's own Drive space.
    """
    env_path = configured_path or get_auditddi_env("RESULTS_BASE")
    if env_path:
        p = Path(env_path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # If running in Colab and Drive is mounted, default to user's writable Drive
    colab_drive = Path("/content/drive/MyDrive")
    if colab_drive.is_dir():
        results_dir = colab_drive / "auditddi-results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    # Fallback to local workspace results
    local_results = PROJECT_ROOT / "results"
    local_results.mkdir(parents=True, exist_ok=True)
    return local_results


def resolve_dataset_subpath(
    data_base: Path,
    subfolder_name: str,
    filename: str | None = None,
) -> Path:
    """Resolve a dataset subfolder and optional file inside auditddi-data case-insensitively.

    Handles differences between Windows (case-insensitive) and Linux/Colab
    (case-sensitive), e.g., 'twosides' vs 'TWOSIDES' vs 'TwoSides'.
    """
    data_base = Path(data_base)
    if not data_base.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_base}")

    # Case variations for the directory
    folder_candidates = [
        data_base / subfolder_name,
        data_base / subfolder_name.upper(),
        data_base / subfolder_name.lower(),
        data_base / subfolder_name.capitalize(),
    ]
    resolved_folder: Path | None = None
    for cand in folder_candidates:
        if cand.is_dir():
            resolved_folder = cand
            break

    if resolved_folder is None:
        # Scan directory case-insensitively
        try:
            target_lower = subfolder_name.lower()
            for entry in data_base.iterdir():
                if entry.is_dir() and entry.name.lower() == target_lower:
                    resolved_folder = entry
                    break
        except Exception:
            pass

    if resolved_folder is None:
        # Fallback to the canonical subfolder path even if missing yet
        resolved_folder = data_base / subfolder_name

    if filename is None:
        return resolved_folder

    # Now resolve filename inside resolved_folder case-insensitively
    file_candidates = [
        resolved_folder / filename,
        resolved_folder / filename.lower(),
        resolved_folder / filename.upper(),
    ]
    for fc in file_candidates:
        if fc.is_file():
            return fc

    if resolved_folder.is_dir():
        try:
            fn_lower = filename.lower()
            for entry in resolved_folder.iterdir():
                if entry.is_file() and entry.name.lower() == fn_lower:
                    return entry
        except Exception:
            pass
    return resolved_folder / filename


ACTIVE_DATASETS = [
    "TWOSIDES",
    "FAERS",
    "UniProt",
    "BindingDB",
    "PharmGKB",
    "ChEMBL",
    "PubChem",
    "PDB",
    "GEO",
]


def print_path_resolution_summary(data_base: Path | str | None = None) -> dict[str, bool]:
    """Inspect and print the status of the 9 active AuditDDI dataset subfolders.

    Returns a mapping of dataset name to existence boolean.
    """
    if data_base is None:
        base = resolve_data_base()
    else:
        base = Path(data_base)

    print("=" * 64)
    print(f"AuditDDI Active Dataset Path Resolution Summary: {base}")
    print("=" * 64)
    status_map: dict[str, bool] = {}
    total_found = 0
    for ds in ACTIVE_DATASETS:
        resolved = resolve_dataset_subpath(base, ds)
        exists = resolved.is_dir()
        status_map[ds] = exists
        icon = "OK" if exists else "--"
        if exists:
            total_found += 1
            try:
                files = [f for f in resolved.rglob("*") if f.is_file()]
                count = len(files)
                total_bytes = sum(f.stat().st_size for f in files)
                if total_bytes >= 1024**3:
                    sz_str = f"{total_bytes / (1024**3):.2f} GB"
                elif total_bytes >= 1024**2:
                    sz_str = f"{total_bytes / (1024**2):.2f} MB"
                elif total_bytes >= 1024:
                    sz_str = f"{total_bytes / 1024:.2f} KB"
                else:
                    sz_str = f"{total_bytes} B"
                print(f"  [{icon}] {ds:12s} -> {resolved.name}/ ({count} files, {sz_str})")
            except Exception:
                print(f"  [{icon}] {ds:12s} -> {resolved.name}/")
        else:
            print(f"  [{icon}] {ds:12s} -> NOT FOUND in {base.name}")
    print("-" * 64)
    print(f"  Located {total_found} of {len(ACTIVE_DATASETS)} active datasets.")
    print("=" * 64)
    return status_map
