"""Build a time-separated, structure-level serious-outcome FAERS cohort.

The target is the fraction of mapped FAERS case reports for a drug structure
that contain a recorded serious outcome code. This is descriptive reporting
evidence, not causal toxicity probability or clinical incidence.

The main cohort uses PS/SS drug roles. An all-reported-drug-role sensitivity
cohort (PS/SS/C/I) is written separately. Drug aliases are mapped first, then
each case ID is counted once per canonical structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SERIOUS_OUTCOME_CODES = {"DE", "HO", "LT", "DS", "CA", "RI", "OT"}
SUSPECT_ROLES = {"PS", "SS"}
ALL_ADMINISTERED_ROLES = {"PS", "SS", "C", "I"}
QUARTERS = [
    "2025q1", "2025q2", "2025q3", "2025q4", "2026q1", "2026q2",
]


def _table_member(archive: zipfile.ZipFile, prefix: str) -> str:
    matches = [
        name for name in archive.namelist()
        if Path(name).name.upper().startswith(prefix)
        and Path(name).suffix.lower() == ".txt"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {prefix} table in ZIP; found {matches}")
    return matches[0]


def _read_table(
    archive: zipfile.ZipFile,
    member: str,
    usecols: list[str],
) -> pd.DataFrame:
    with archive.open(member) as stream:
        return pd.read_csv(
            stream,
            sep="$",
            usecols=usecols,
            dtype=str,
            keep_default_na=False,
            encoding="latin-1",
            low_memory=False,
        )


def _read_deleted_case_ids(archive: zipfile.ZipFile) -> set[str]:
    member = _table_member(archive, "DELETE")
    with archive.open(member) as stream:
        return {
            line.decode("latin-1", errors="replace").strip()
            for line in stream
            if line.decode("latin-1", errors="replace").strip()
        }


def _latest_case_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one highest available case version, deterministically."""
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    data["caseversion_num"] = pd.to_numeric(data["caseversion"], errors="coerce").fillna(-1)
    data["fda_dt_num"] = pd.to_numeric(data["fda_dt"], errors="coerce").fillna(-1)
    data = data.sort_values(
        ["caseid", "caseversion_num", "fda_dt_num", "quarter", "primaryid"],
        kind="mergesort",
    )
    return data.drop_duplicates("caseid", keep="last").reset_index(drop=True)


def _load_name_structure_map(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    bridge = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"drugname", "canonical_smiles"}
    missing = required - set(bridge.columns)
    if missing:
        raise ValueError(f"Bridge is missing columns: {sorted(missing)}")
    bridge["name_key"] = bridge["drugname"].astype(str).str.strip().str.upper()
    bridge = bridge[
        (bridge["name_key"] != "")
        & (bridge["canonical_smiles"].astype(str).str.strip() != "")
    ].copy()
    counts = bridge.groupby("name_key")["canonical_smiles"].nunique()
    ambiguous = set(counts[counts > 1].index)
    bridge = bridge[~bridge["name_key"].isin(ambiguous)]
    bridge = bridge.drop_duplicates(["name_key", "canonical_smiles"])
    mapping = dict(zip(bridge["name_key"], bridge["canonical_smiles"]))
    return mapping, {
        "bridge_rows": int(len(bridge)),
        "mapped_unique_names": int(len(mapping)),
        "ambiguous_names_excluded": int(len(ambiguous)),
        "mapped_unique_structures": int(bridge["canonical_smiles"].nunique()),
    }


def build_labels(data_root: Path, archive_dir: Path, bridge_path: Path, output_dir: Path) -> dict[str, Any]:
    archives: dict[str, Path] = {
        q: archive_dir / f"faers_ascii_{q}.zip" for q in QUARTERS[:5]
    }
    archives["2026q2"] = data_root / "faers" / "faers_ascii_2026q2.zip"
    missing = [str(path) for path in archives.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing quarterly archives: " + ", ".join(missing))
    if not bridge_path.is_file():
        raise FileNotFoundError(f"Bridge file not found: {bridge_path}")

    name_to_structure, bridge_summary = _load_name_structure_map(bridge_path)

    deleted_by_quarter: dict[str, set[str]] = {}
    demos: list[pd.DataFrame] = []
    for quarter, path in archives.items():
        print(f"Reading {quarter} DEMO and DELETE tables from {path.name}...", flush=True)
        with zipfile.ZipFile(path) as archive:
            deleted_by_quarter[quarter] = _read_deleted_case_ids(archive)
            demo_member = _table_member(archive, "DEMO")
            demo = _read_table(
                archive, demo_member,
                ["primaryid", "caseid", "caseversion", "fda_dt"],
            )
        demo["primaryid"] = demo["primaryid"].str.strip()
        demo["caseid"] = demo["caseid"].str.strip()
        demo = demo[(demo["primaryid"] != "") & (demo["caseid"] != "")].copy()
        demo["quarter"] = quarter
        demos.append(demo)
        print(f"  {quarter}: {len(demo):,} nonempty report rows", flush=True)

    deleted_ids = set().union(*deleted_by_quarter.values())
    demo_all = pd.concat(demos, ignore_index=True)
    all_demo_case_ids = set(demo_all["caseid"])
    demo_all = demo_all[~demo_all["caseid"].isin(deleted_ids)].copy()

    train_source = demo_all[demo_all["quarter"].str.startswith("2025")].copy()
    test_source = demo_all[demo_all["quarter"].str.startswith("2026")].copy()
    train_demo = _latest_case_rows(train_source)
    train_case_ids = set(train_demo["caseid"])
    test_source = test_source[~test_source["caseid"].isin(train_case_ids)].copy()
    test_demo = _latest_case_rows(test_source)

    train_demo["split"] = "train_2025"
    test_demo["split"] = "test_new_cases_2026h1"
    print(
        f"Selected latest cases: 2025 train={train_demo['caseid'].nunique():,}; "
        f"2026 H1 new-case test={test_demo['caseid'].nunique():,}; "
        f"deleted CASEIDs={len(deleted_ids & all_demo_case_ids):,}.",
        flush=True,
    )
    train_test_case_overlap = set(train_demo["caseid"]) & set(test_demo["caseid"])
    if train_test_case_overlap:
        raise AssertionError(
            f"Train/test CASEID overlap: {len(train_test_case_overlap)} cases."
        )
    selected = pd.concat([train_demo, test_demo], ignore_index=True)

    primary_meta: dict[str, tuple[str, str, str]] = {}
    for row in selected[["primaryid", "caseid", "split", "quarter"]].itertuples(index=False):
        primary_meta[str(row.primaryid)] = (str(row.caseid), str(row.split), str(row.quarter))
    selected_primaryids = set(primary_meta)

    # A recorded OUTC row uses one of the serious-outcome codes. No OUTC row
    # is treated only as "no serious outcome recorded"; it is not called safe.
    serious_primaryids: set[str] = set()
    outcome_codes_seen: set[str] = set()
    outcome_primaryids_seen: set[str] = set()
    for quarter, path in archives.items():
        print(f"Reading {quarter} OUTC outcomes...", flush=True)
        with zipfile.ZipFile(path) as archive:
            outc_member = _table_member(archive, "OUTC")
            outc = _read_table(archive, outc_member, ["primaryid", "outc_cod"])
        outc["primaryid"] = outc["primaryid"].str.strip()
        outc["outc_cod"] = outc["outc_cod"].str.strip().str.upper()
        outc = outc[outc["primaryid"].isin(selected_primaryids)]
        outcome_codes_seen.update(x for x in outc["outc_cod"].unique() if x)
        outcome_primaryids_seen.update(outc["primaryid"].unique())
        serious_primaryids.update(
            outc.loc[outc["outc_cod"].isin(SERIOUS_OUTCOME_CODES), "primaryid"].unique()
        )

    unexpected_outcome_codes = sorted(outcome_codes_seen - SERIOUS_OUTCOME_CODES)
    if unexpected_outcome_codes:
        raise ValueError(
            "Unexpected OUTC codes need a reviewed label policy before aggregation: "
            + ", ".join(unexpected_outcome_codes)
        )

    # Accumulate unique (CASEID, structure) pairs so multiple name aliases in
    # one report never inflate a structure's denominator.
    stats: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "k": 0, "quarters": set()}
    )
    seen: dict[str, set[tuple[str, str, str]]] = {
        "suspect_roles": set(), "all_administered_roles": set()
    }
    unmapped_names: set[str] = set()
    selected_drug_rows = 0
    mapped_selected_drug_rows = 0
    roles_seen: dict[str, int] = defaultdict(int)

    for quarter, path in archives.items():
        print(f"Mapping {quarter} DRUG names to structures...", flush=True)
        with zipfile.ZipFile(path) as archive:
            drug_member = _table_member(archive, "DRUG")
            with archive.open(drug_member) as raw:
                import io
                with io.TextIOWrapper(raw, encoding="latin-1", errors="replace", newline="") as text:
                    for chunk_number, chunk in enumerate(pd.read_csv(
                        text,
                        sep="$",
                        usecols=["primaryid", "caseid", "drugname", "role_cod"],
                        dtype=str,
                        keep_default_na=False,
                        chunksize=250_000,
                    ), start=1):
                        chunk["primaryid"] = chunk["primaryid"].str.strip()
                        chunk["caseid"] = chunk["caseid"].str.strip()
                        chunk["role_cod"] = chunk["role_cod"].str.strip().str.upper()
                        chunk = chunk[chunk["primaryid"].isin(selected_primaryids)].copy()
                        if chunk.empty:
                            continue

                        meta = chunk["primaryid"].map(primary_meta)
                        chunk["selected_caseid"] = meta.map(lambda x: x[0] if x else "")
                        chunk["split"] = meta.map(lambda x: x[1] if x else "")
                        chunk["selected_quarter"] = meta.map(lambda x: x[2] if x else "")
                        chunk = chunk[
                            (chunk["caseid"] == chunk["selected_caseid"])
                            & (chunk["selected_quarter"] == quarter)
                        ].copy()
                        if chunk.empty:
                            continue

                        selected_drug_rows += len(chunk)
                        for role, count in chunk["role_cod"].value_counts().items():
                            roles_seen[str(role)] += int(count)
                        chunk["name_key"] = chunk["drugname"].str.strip().str.upper()
                        chunk["canonical_smiles"] = chunk["name_key"].map(name_to_structure)
                        unmapped_names.update(
                            chunk.loc[chunk["canonical_smiles"].isna(), "name_key"].unique()
                        )
                        mapped = chunk.dropna(subset=["canonical_smiles"]).copy()
                        mapped_selected_drug_rows += len(mapped)
                        if chunk_number == 1 or chunk_number % 5 == 0:
                            print(
                                f"  {quarter}: processed DRUG chunk {chunk_number}; "
                                f"selected rows={selected_drug_rows:,}, "
                                f"mapped rows={mapped_selected_drug_rows:,}.",
                                flush=True,
                            )

                        for label, allowed_roles in (
                            ("suspect_roles", SUSPECT_ROLES),
                            ("all_administered_roles", ALL_ADMINISTERED_ROLES),
                        ):
                            subset = mapped[mapped["role_cod"].isin(allowed_roles)]
                            subset = subset.drop_duplicates(["caseid", "canonical_smiles"])
                            for row in subset[["caseid", "split", "canonical_smiles", "primaryid", "selected_quarter"]].itertuples(index=False):
                                key = (str(row.split), str(row.caseid), str(row.canonical_smiles))
                                if key in seen[label]:
                                    continue
                                seen[label].add(key)
                                stat_key = (str(row.split), label, str(row.canonical_smiles))
                                item = stats[stat_key]
                                item["n"] += 1
                                item["k"] += int(str(row.primaryid) in serious_primaryids)
                                item["quarters"].add(str(row.selected_quarter))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, str] = {}
    structure_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (split, label, smiles), item in sorted(stats.items()):
        structure_sets[(split, label)].add(smiles)
    for split in ("train_2025", "test_new_cases_2026h1"):
        for label in ("suspect_roles", "all_administered_roles"):
            rows = []
            for (row_split, row_label, smiles), item in stats.items():
                if row_split != split or row_label != label:
                    continue
                n = int(item["n"])
                k = int(item["k"])
                rows.append({
                    "canonical_smiles": smiles,
                    "serious_outcome_reports": k,
                    "n_reports": n,
                    "serious_outcome_recorded_fraction": k / n if n else 0.0,
                    "source_quarters": "|".join(sorted(item["quarters"])),
                })
            frame = pd.DataFrame(
                rows,
                columns=[
                    "canonical_smiles",
                    "serious_outcome_reports",
                    "n_reports",
                    "serious_outcome_recorded_fraction",
                    "source_quarters",
                ],
            ).sort_values("canonical_smiles")
            filename = f"{split}_{label}.csv"
            frame.to_csv(output_dir / filename, index=False)
            output_files[f"{split}_{label}"] = str(output_dir / filename)
            print(
                f"{filename}: {len(frame):,} structures; "
                f"{int(frame['n_reports'].sum()) if not frame.empty else 0:,} mapped case-structure reports"
            )

    suspect_train = structure_sets[("train_2025", "suspect_roles")]
    suspect_test = structure_sets[("test_new_cases_2026h1", "suspect_roles")]
    summary = {
        "target": (
            "Fraction of unique mapped FAERS case reports for a structure with a recorded serious outcome code. "
            "Descriptive report evidence; not causation, incidence, or patient risk."
        ),
        "serious_outcome_codes": sorted(SERIOUS_OUTCOME_CODES),
        "primary_roles": sorted(SUSPECT_ROLES),
        "all_role_sensitivity_roles": sorted(ALL_ADMINISTERED_ROLES),
        "deleted_ids_read": int(len(deleted_ids)),
        "deleted_caseids_matching_six_quarter_demos": int(len(deleted_ids & all_demo_case_ids)),
        "train_unique_cases": int(train_demo["caseid"].nunique()),
        "test_new_unique_cases": int(test_demo["caseid"].nunique()),
        "train_test_case_overlap": int(len(set(train_demo["caseid"]) & set(test_demo["caseid"]))),
        "train_suspect_structures": int(len(suspect_train)),
        "test_suspect_structures": int(len(suspect_test)),
        "suspect_train_test_structure_overlap": int(len(suspect_train & suspect_test)),
        "mapped_name_structure_bridge": bridge_summary,
        "selected_drug_rows": int(selected_drug_rows),
        "selected_drug_rows_mapped": int(mapped_selected_drug_rows),
        "unique_unmapped_drug_names": int(len(unmapped_names)),
        "role_counts_selected_reports": dict(sorted(roles_seen.items())),
        "outcome_codes_seen": sorted(outcome_codes_seen),
        "serious_primaryids": int(len(serious_primaryids)),
        "primaryids_with_any_outcome": int(len(outcome_primaryids_seen)),
        "outputs": output_files,
    }
    manifest_path = output_dir / "cohort_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Cohort manifest:", manifest_path)
    print("Train/test suspect-structure overlap:", summary["suspect_train_test_structure_overlap"])
    print("Unique unmapped names:", summary["unique_unmapped_drug_names"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_labels(args.data_root, args.archive_dir, args.bridge, args.output_dir)


if __name__ == "__main__":
    main()
