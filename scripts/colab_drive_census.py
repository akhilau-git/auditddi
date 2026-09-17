"""Self-contained Google Drive dataset auditor for Google Colab.
Requires NO dependencies (uses only standard library). Run directly in a Colab cell.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} B"

def count_lines_fast(file_path: Path, max_lines: int | None = None) -> int:
    """Count lines quickly without loading whole file into memory."""
    try:
        count = 0
        with open(file_path, "rb") as f:
            buf_size = 1024 * 1024
            read_buf = f.raw.read if hasattr(f, "raw") else f.read
            buf = read_buf(buf_size)
            while buf:
                count += buf.count(b"\n")
                if max_lines and count >= max_lines:
                    break
                buf = read_buf(buf_size)
        return count
    except Exception:
        return -1

def locate_auditddi_data_dir() -> Path | None:
    """Search common mount paths in Google Colab."""
    candidates = [
        Path("/content/drive/MyDrive/auditddi-data"),
        Path("/content/drive/MyDrive/AuditDDI-data"),
        Path("/content/drive/MyDrive/auditddi_data"),
        Path("/content/drive/MyDrive/datasets"),
    ]
    for c in candidates:
        if c.is_dir():
            return c

    # Search inside /content/drive/MyDrive
    mydrive = Path("/content/drive/MyDrive")
    if mydrive.is_dir():
        for item in mydrive.iterdir():
            if item.is_dir() and "auditddi" in item.name.lower():
                return item

    # Search inside shortcuts if applicable
    shortcuts = Path("/content/drive/.shortcut-targets-by-id")
    if shortcuts.is_dir():
        for target_id in shortcuts.iterdir():
            if target_id.is_dir():
                for sub in target_id.iterdir():
                    if sub.is_dir() and "auditddi" in sub.name.lower():
                        return sub

    return None

def run_colab_drive_census(target_path: str | Path | None = None):
    print("=" * 80)
    print("           AUDITDDI GOOGLE DRIVE LIVE DATASET AUDIT & CENSUS")
    print("=" * 80)

    if target_path:
        base_dir = Path(target_path)
    else:
        base_dir = locate_auditddi_data_dir()

    if not base_dir or not base_dir.is_dir():
        print("\n[ERROR] Could not find 'auditddi-data' folder in /content/drive/MyDrive/!")
        print("\nPlease check:")
        print(" 1. Did you run: from google.colab import drive; drive.mount('/content/drive') ?")
        print(" 2. Is the folder named 'auditddi-data' visible in your Google Drive 'MyDrive'?")
        print("\nAvailable folders in /content/drive/MyDrive:")
        mydrive = Path("/content/drive/MyDrive")
        if mydrive.is_dir():
            for d in sorted(mydrive.iterdir()):
                if d.is_dir() and not d.name.startswith("."):
                    print(f"   - {d.name}/")
        return

    print(f"\n[OK] Found Target Root: {base_dir}")
    print("Scanning dataset folders and calculating live data volumes...\n")

    target_9 = [
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

    # Discover all top-level directories in auditddi-data
    existing_dirs = {d.name.lower(): d for d in base_dir.iterdir() if d.is_dir()}
    
    grand_total_bytes = 0
    grand_total_files = 0
    folder_summaries = []

    print("-" * 80)
    print("SECTION 1: THE 9 TARGET DATASETS")
    print("-" * 80)

    for name in target_9:
        matched_dir = existing_dirs.get(name.lower())
        if matched_dir is None:
            print(f"\n[-- MISSING] {name}")
            print(f"     Status: Folder '{name}' NOT found in {base_dir.name}/")
            if name == "UniProt":
                print("     Note  : UniProt target fastas can be generated via: python -m src.data_prep.download_uniprot_data")
            folder_summaries.append((name, False, 0, 0, []))
            continue

        files_info = []
        folder_bytes = 0
        for root, _, files in os.walk(matched_dir):
            for f in files:
                fp = Path(root) / f
                try:
                    sz = fp.stat().st_size
                    folder_bytes += sz
                    rel = fp.relative_to(matched_dir)
                    files_info.append((str(rel), sz, fp))
                except Exception:
                    pass

        grand_total_bytes += folder_bytes
        grand_total_files += len(files_info)
        folder_summaries.append((name, True, folder_bytes, len(files_info), files_info))

        print(f"\n[OK FOUND] {name} (Folder: {matched_dir.name}/)")
        print(f"     Total Size : {format_size(folder_bytes)} ({folder_bytes:,} bytes)")
        print(f"     File Count : {len(files_info)} file(s)")
        
        # Display files
        sorted_files = sorted(files_info, key=lambda x: x[1], reverse=True)
        print("     File Details:")
        for rel_name, sz, fp in sorted_files[:8]:
            line_info = ""
            # If CSV or TSV, report line count
            if fp.suffix.lower() in [".csv", ".tsv"] and sz < 600 * 1024 * 1024:
                lc = count_lines_fast(fp)
                if lc >= 0:
                    line_info = f" | {lc:,} lines"
            print(f"       * {rel_name:42s} : {format_size(sz):>10s}{line_info}")
        if len(sorted_files) > 8:
            print(f"       ... and {len(sorted_files) - 8} more file(s)")

    # Check for other folders in auditddi-data
    target_names_lower = {n.lower() for n in target_9}
    other_dirs = [d for k, d in existing_dirs.items() if k not in target_names_lower]
    if other_dirs:
        print("\n" + "-" * 80)
        print("SECTION 2: OTHER FOLDERS PRESENT IN auditddi-data")
        print("-" * 80)
        for od in other_dirs:
            cnt = sum(1 for _ in od.glob("**/*") if _.is_file())
            sz = sum(_.stat().st_size for _ in od.glob("**/*") if _.is_file())
            print(f"  [INFO] {od.name:20s} : {len(cnt) if isinstance(cnt, list) else cnt} files, {format_size(sz)}")

    print("\n" + "=" * 80)
    print("                      GOOGLE DRIVE LIVE CENSUS SUMMARY")
    print("=" * 80)
    found_9 = sum(1 for _, exists, _, _, _ in folder_summaries if exists)
    print(f" Target Datasets Present : {found_9} / 9")
    print(f" Total Files Counted     : {grand_total_files:,} files")
    print(f" Total Live Data Volume  : {format_size(grand_total_bytes)} ({grand_total_bytes:,} bytes)")
    print("=" * 80)

if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_colab_drive_census(path_arg)
