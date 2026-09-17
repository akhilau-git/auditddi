"""Comprehensive Google Drive and Local Dataset Auditor for AuditDDI.

Audits the 9 active datasets:
  1. TWOSIDES
  2. FAERS
  3. UniProt
  4. BindingDB
  5. PharmGKB
  6. ChEMBL
  7. PubChem
  8. PDB
  9. GEO

Computes exact file lists, file sizes (MB/GB), total folder volumes,
and verifies the presence of critical pipeline files.
Can be run directly on Google Colab or locally.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# 9 Active Datasets for the AuditDDI Study
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

# Essential files expected in each dataset for the pipeline to operate
CRITICAL_FILES = {
    "TWOSIDES": ["drug_drug_edges.csv", "twosides_drugs.csv"],
    "FAERS": ["ASCII/DRUG23Q4.txt", "ASCII/DEMO23Q4.txt", "ASCII/REAC23Q4.txt"],
    "UniProt": ["uniprot_sequences.fasta", "uniprot_targets_metadata.csv", "target_sequences.json"],
    "BindingDB": ["drug_target_edges.csv", "bindingdb_drugs.csv", "bindingdb_targets.csv"],
    "PharmGKB": ["relationships.tsv", "variants.tsv", "phenotypes.tsv"],
    "ChEMBL": ["chembl_37_chemreps.txt.gz", "chembl_uniprot_mapping.txt"],
    "PubChem": ["CID-Synonym-filtered.gz", "pubchem.csv"],
    "PDB": ["1IYT.pdb", "1T46.pdb", "1J1M.pdb"],
    "GEO": ["Cardiovascular_HF_GSE57338.txt.gz", "Brain_Alzheimers_GSE5281.txt.gz"],
}


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable B, KB, MB, or GB string."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} B"


def find_dataset_folder(data_base: Path, folder_name: str) -> Path | None:
    """Locate a dataset folder case-insensitively."""
    candidates = [
        data_base / folder_name,
        data_base / folder_name.upper(),
        data_base / folder_name.lower(),
        data_base / folder_name.capitalize(),
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand

    # Scan directory if case variation is non-standard
    if data_base.is_dir():
        target_lower = folder_name.lower()
        try:
            for entry in data_base.iterdir():
                if entry.is_dir() and entry.name.lower() == target_lower:
                    return entry
        except Exception:
            pass
    return None


def audit_datasets(data_base_path: str | Path) -> dict:
    """Audit all 9 active datasets in the specified directory."""
    base = Path(data_base_path).resolve()
    print("=" * 78)
    print("           AUDITDDI GOOGLE DRIVE / STORAGE DATASET AUDIT REPORT")
    print(f" Target Directory: {base}")
    print("=" * 78)

    if not base.is_dir():
        print(f"\n[ERROR] The target directory does not exist: {base}")
        print("Please ensure Google Drive is mounted at /content/drive and auditddi-data is present.")
        return {"status": "error", "message": f"Directory not found: {base}"}

    grand_total_bytes = 0
    grand_total_files = 0
    census_results = {}

    for ds_name in ACTIVE_DATASETS:
        folder_path = find_dataset_folder(base, ds_name)
        if folder_path is None:
            print(f"\n[-- MISSING] {ds_name}")
            print(f"     Status : Folder not found in {base.name}")
            if ds_name == "UniProt":
                print("     Note   : Run 'python -m src.data_prep.download_uniprot_data' to fetch targets.")
            census_results[ds_name] = {
                "exists": False,
                "path": None,
                "total_bytes": 0,
                "file_count": 0,
                "critical_status": {},
            }
            continue

        file_entries = []
        folder_bytes = 0
        for root, _, files in os.walk(folder_path):
            for f in files:
                fp = Path(root) / f
                try:
                    sz = fp.stat().st_size
                    folder_bytes += sz
                    rel = fp.relative_to(folder_path)
                    file_entries.append((str(rel), sz, fp))
                except Exception:
                    pass

        grand_total_bytes += folder_bytes
        grand_total_files += len(file_entries)

        print(f"\n[OK FOUND] {ds_name} -> {folder_path.name}/")
        print(f"     Total Size : {format_size(folder_bytes)} ({folder_bytes:,} bytes)")
        print(f"     File Count : {len(file_entries)} file(s)")

        # Verify critical expected files
        crit_status = {}
        expected_crits = CRITICAL_FILES.get(ds_name, [])
        if expected_crits:
            print("     Key Files  :")
            for cf in expected_crits:
                # check file existence case-insensitively
                cf_path = folder_path / cf
                cf_exists = cf_path.is_file()
                if not cf_exists:
                    # try normalized path
                    parts = cf.split("/")
                    cur = folder_path
                    for p in parts:
                        match = None
                        if cur.is_dir():
                            for child in cur.iterdir():
                                if child.name.lower() == p.lower():
                                    match = child
                                    break
                        cur = match if match else cur / p
                    cf_exists = cur.is_file()
                    if cf_exists:
                        cf_path = cur

                crit_status[cf] = cf_exists
                mark = "[OK]" if cf_exists else "[--]"
                if cf_exists:
                    sz_str = format_size(cf_path.stat().st_size)
                    print(f"       {mark} {cf:35s} ({sz_str})")
                else:
                    print(f"       {mark} {cf:35s} (NOT FOUND)")

        # Print top files by size
        if file_entries:
            print("     Top Files by Size :")
            sorted_files = sorted(file_entries, key=lambda x: x[1], reverse=True)
            for rel, sz, _ in sorted_files[:5]:
                print(f"       * {rel:40s} : {format_size(sz)}")
            if len(sorted_files) > 5:
                print(f"       ... and {len(sorted_files) - 5} more file(s)")

        census_results[ds_name] = {
            "exists": True,
            "path": str(folder_path),
            "total_bytes": folder_bytes,
            "formatted_size": format_size(folder_bytes),
            "file_count": len(file_entries),
            "critical_status": crit_status,
        }

    print("\n" + "=" * 78)
    print("                          AUDIT SUMMARY CENSUS")
    print("=" * 78)
    found_count = sum(1 for r in census_results.values() if r["exists"])
    print(f" Active Datasets Located : {found_count} of {len(ACTIVE_DATASETS)}")
    print(f" Total Files Cataloged   : {grand_total_files:,} files")
    print(f" Total Dataset Volume    : {format_size(grand_total_bytes)} ({grand_total_bytes:,} bytes)")
    print("=" * 78)

    return {
        "data_base": str(base),
        "found_count": found_count,
        "total_datasets": len(ACTIVE_DATASETS),
        "grand_total_bytes": grand_total_bytes,
        "formatted_total_size": format_size(grand_total_bytes),
        "grand_total_files": grand_total_files,
        "datasets": census_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit the 9 active AuditDDI datasets.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to auditddi-data root directory. Defaults to auto-detection.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir:
        # Auto-detect using path_resolver if available
        try:
            from src.data_prep.path_resolver import resolve_data_base
            data_dir = resolve_data_base()
        except Exception:
            candidates = [
                Path("/content/drive/MyDrive/auditddi-data"),
                Path("D:/Drug-Drug Interaction/Quantum_AI_Pharma/datasets"),
                Path("D:/Drug-Drug Interaction/AuditDDI/data"),
            ]
            for c in candidates:
                if c.is_dir():
                    data_dir = c
                    break
            if not data_dir:
                data_dir = Path("/content/drive/MyDrive/auditddi-data")

    audit_datasets(data_dir)


if __name__ == "__main__":
    main()
