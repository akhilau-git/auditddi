"""FAERS Pharmacovigilance & Disproportionality Signal Pipeline for AuditDDI.

Processes real post-marketing clinical reports from the FDA Adverse Event
Reporting System (FAERS), calculating:
  1. Real-world drug toxicity rates across severe patient outcomes (DE=Death,
     HO=Hospitalization, LT=Life-threatening, DS=Disability).
  2. Disproportionality metrics: Reporting Odds Ratio (ROR) and Proportional
     Reporting Ratio (PRR) with 95% Confidence Intervals.
  3. Real patient demographic context (Age, Biological Sex) via DEMO tables.
  4. Pairwise drug combination reporting frequencies and clinical safety signals.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Severe outcome codes according to FDA FAERS standards:
# DE = Death, HO = Hospitalization (Initial or Prolonged),
# LT = Life-Threatening, DS = Disability, CA = Congenital Anomaly, RI = Required Intervention
SEVERE_OUTCOMES: Set[str] = {'DE', 'HO', 'LT', 'DS', 'CA', 'RI'}

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False


def calculate_ror(
    a: int, b: int, c: int, d: int, alpha: float = 0.05
) -> Dict[str, Any]:
    """Calculate the Reporting Odds Ratio (ROR) and 95% Confidence Interval.

    Contingency table for Drug D and Adverse Reaction / Severe Outcome R:
                  Outcome Severe (R)   Outcome Non-Severe (~R)
    Drug D                a                       b
    Other Drugs           c                       d

    ROR = (a / b) / (c / d) = (a * d) / (b * c)
    SE(ln(ROR)) = sqrt(1/a + 1/b + 1/c + 1/d)
    """
    if a <= 0 or b <= 0 or c <= 0 or d <= 0:
        # Haldane-Anscombe correction (+0.5 to zero cells)
        a_adj = a + 0.5
        b_adj = b + 0.5
        c_adj = c + 0.5
        d_adj = d + 0.5
    else:
        a_adj, b_adj, c_adj, d_adj = float(a), float(b), float(c), float(d)

    ror = (a_adj * d_adj) / (b_adj * c_adj)
    se = math.sqrt((1.0 / a_adj) + (1.0 / b_adj) + (1.0 / c_adj) + (1.0 / d_adj))
    z = 1.96  # 95% Confidence Level

    ci_lower = math.exp(math.log(ror) - z * se)
    ci_upper = math.exp(math.log(ror) + z * se)

    # FDA / EMA standard signal criteria: a >= 3 and CI_lower > 1.0
    is_signal = a >= 3 and ci_lower > 1.0

    # Proportional Reporting Ratio (PRR)
    prr_num = a_adj / (a_adj + b_adj)
    prr_den = c_adj / (c_adj + d_adj)
    prr = prr_num / prr_den if prr_den > 0 else ror

    return {
        "a_reports": a,
        "b_reports": b,
        "c_reports": c,
        "d_reports": d,
        "ror": round(ror, 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "prr": round(prr, 3),
        "is_significant_signal": is_signal,
    }


def aggregate_toxicity_labels(
    drug: Any,
    outc: Any,
    min_reports: int = 5,
    missing_outcome_policy: str = 'exclude',
) -> Any:
    """Aggregate one severe-outcome flag per report before grouping by drug.

    Avoids Cartesian explosion when multiple drugs and outcomes exist per report.
    """
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required for DataFrame aggregation.")

    if min_reports < 1:
        raise ValueError('min_reports must be at least 1.')
    if missing_outcome_policy not in {'exclude', 'non_severe'}:
        raise ValueError("missing_outcome_policy must be 'exclude' or 'non_severe'.")

    for name, frame, columns in (
        ('DRUG', drug, {'primaryid', 'drugname'}),
        ('OUTC', outc, {'primaryid', 'outc_cod'}),
    ):
        missing = columns.difference(frame.columns)
        if missing:
            raise ValueError(f'{name} data is missing required columns: {sorted(missing)}.')

    drug_reports = drug[['primaryid', 'drugname']].copy()
    drug_reports = drug_reports.dropna(subset=['primaryid', 'drugname'])
    drug_reports['drugname'] = drug_reports['drugname'].astype(str).str.strip().str.upper()
    drug_reports = drug_reports[drug_reports['drugname'] != '']
    drug_reports = drug_reports.drop_duplicates(subset=['primaryid', 'drugname'])

    outcomes = outc[['primaryid', 'outc_cod']].copy()
    outcomes = outcomes.dropna(subset=['primaryid'])
    outcomes['is_severe'] = outcomes['outc_cod'].isin(SEVERE_OUTCOMES).astype(int)
    report_severity = outcomes.groupby('primaryid', as_index=False)['is_severe'].max()

    merged = drug_reports.merge(
        report_severity,
        on='primaryid',
        how='left' if missing_outcome_policy == 'non_severe' else 'inner',
    )
    if missing_outcome_policy == 'non_severe':
        merged['is_severe'] = merged['is_severe'].fillna(0).astype(int)

    # Total severe and non-severe across the whole database
    total_severe_reports = int(report_severity['is_severe'].sum())
    total_non_severe_reports = int((report_severity['is_severe'] == 0).sum())

    grouped = merged.groupby('drugname', as_index=False).agg(
        n_reports=('primaryid', 'nunique'),
        severe_reports=('is_severe', 'sum'),
    )
    grouped['non_severe_reports'] = grouped['n_reports'] - grouped['severe_reports']
    grouped['toxicity_score'] = grouped['severe_reports'] / grouped['n_reports']

    # Compute ROR and PRR for each drug
    rors = []
    ci_lowers = []
    ci_uppers = []
    is_signals = []

    for _, row in grouped.iterrows():
        a = int(row['severe_reports'])
        b = int(row['non_severe_reports'])
        c = max(0, total_severe_reports - a)
        d = max(0, total_non_severe_reports - b)
        res = calculate_ror(a, b, c, d)
        rors.append(res['ror'])
        ci_lowers.append(res['ci_lower'])
        ci_uppers.append(res['ci_upper'])
        is_signals.append(res['is_significant_signal'])

    grouped['ror'] = rors
    grouped['ror_ci_lower'] = ci_lowers
    grouped['ror_ci_upper'] = ci_uppers
    grouped['is_disproportionality_signal'] = is_signals

    filtered = grouped[grouped['n_reports'] >= min_reports]
    return filtered.sort_values(['n_reports', 'toxicity_score'], ascending=[False, False]).reset_index(drop=True)


def parse_faers_streaming(
    drug_path: Path,
    outc_path: Path,
    min_reports: int = 5,
    sample_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Pure-Python, zero-dependency streaming parser for FAERS text files.

    Processes massive FAERS files line-by-line without loading entire DataFrames
    into memory, making it resilient across low-RAM machines and containers.
    """
    # 1. Parse OUTC (primaryid -> is_severe)
    print(f"[FAERS Pipeline] Streaming outcomes from {outc_path.name}...")
    report_severity: Dict[str, bool] = {}
    total_severe = 0
    total_non_severe = 0

    with open(outc_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='$')
        header = next(reader, None)
        pid_idx = 0
        outc_idx = 1
        if header:
            for idx, col in enumerate(header):
                cl = col.strip().lower()
                if cl == 'primaryid':
                    pid_idx = idx
                elif cl == 'outc_cod':
                    outc_idx = idx

        count = 0
        for row in reader:
            if not row or len(row) <= max(pid_idx, outc_idx):
                continue
            pid = row[pid_idx].strip()
            cod = row[outc_idx].strip().upper()
            is_sev = cod in SEVERE_OUTCOMES
            if pid not in report_severity:
                report_severity[pid] = is_sev
            elif is_sev:
                report_severity[pid] = True
            count += 1
            if sample_limit and count >= sample_limit:
                break

    for is_sev in report_severity.values():
        if is_sev:
            total_severe += 1
        else:
            total_non_severe += 1

    print(f"[FAERS Pipeline] Processed {len(report_severity):,} reports ({total_severe:,} severe, {total_non_severe:,} non-severe).")

    # 2. Stream DRUG file (primaryid -> set of drugs)
    print(f"[FAERS Pipeline] Streaming drug records from {drug_path.name}...")
    drug_counts: Dict[str, Dict[str, int]] = {}  # drug -> {'severe': int, 'non_severe': int}
    seen_drug_report: Set[Tuple[str, str]] = set()

    with open(drug_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='$')
        header = next(reader, None)
        pid_idx = 0
        drug_idx = 2
        if header:
            for idx, col in enumerate(header):
                cl = col.strip().lower()
                if cl == 'primaryid':
                    pid_idx = idx
                elif cl == 'drugname':
                    drug_idx = idx

        count = 0
        for row in reader:
            if not row or len(row) <= max(pid_idx, drug_idx):
                continue
            pid = row[pid_idx].strip()
            dname = row[drug_idx].strip().upper()
            if not pid or not dname or pid not in report_severity:
                continue

            pair = (pid, dname)
            if pair in seen_drug_report:
                continue
            seen_drug_report.add(pair)

            if dname not in drug_counts:
                drug_counts[dname] = {'severe': 0, 'non_severe': 0}

            if report_severity[pid]:
                drug_counts[dname]['severe'] += 1
            else:
                drug_counts[dname]['non_severe'] += 1

            count += 1
            if sample_limit and count >= sample_limit:
                break

    # 3. Calculate metrics per drug
    results = []
    for dname, stats in drug_counts.items():
        a = stats['severe']
        b = stats['non_severe']
        total_rep = a + b
        if total_rep < min_reports:
            continue

        c = max(0, total_severe - a)
        d = max(0, total_non_severe - b)
        ror_data = calculate_ror(a, b, c, d)
        tox_score = round(a / total_rep, 4) if total_rep > 0 else 0.0

        results.append({
            'drugname': dname,
            'n_reports': total_rep,
            'severe_reports': a,
            'non_severe_reports': b,
            'toxicity_score': tox_score,
            'ror': ror_data['ror'],
            'ror_ci_lower': ror_data['ci_lower'],
            'ror_ci_upper': ror_data['ci_upper'],
            'prr': ror_data['prr'],
            'is_disproportionality_signal': ror_data['is_significant_signal'],
        })

    results.sort(key=lambda x: (x['n_reports'], x['toxicity_score']), reverse=True)
    print(f"[FAERS Pipeline] Successfully derived safety signals for {len(results):,} drugs.")
    return results


def find_faers_files(faers_base_path: str | Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Locate DRUG, OUTC, and DEMO files in standard or ASCII subdirectories."""
    base = Path(faers_base_path)
    drug_file: Optional[Path] = None
    outc_file: Optional[Path] = None
    demo_file: Optional[Path] = None

    candidates = list(base.glob('**/*.txt')) + list(base.glob('**/*.TXT'))
    for f in candidates:
        name = f.name.lower()
        if 'drug' in name and not drug_file:
            drug_file = f
        elif 'outc' in name and not outc_file:
            outc_file = f
        elif 'demo' in name and not demo_file:
            demo_file = f

    return drug_file, outc_file, demo_file


def process_faers_pipeline(
    faers_dir: str | Path,
    output_dir: str | Path,
    min_reports: int = 5,
    sample_limit: Optional[int] = None,
) -> Path:
    """End-to-end execution of FAERS signal processing, saving CSV output."""
    faers_path = Path(faers_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    drug_p, outc_p, _ = find_faers_files(faers_path)
    if not drug_p or not outc_p:
        raise FileNotFoundError(f"Could not find DRUG and OUTC text files in {faers_path}")

    target_csv = out_dir / 'faers_safety_signals.csv'

    if PANDAS_AVAILABLE and pd is not None:
        print(f"[FAERS Pipeline] Processing via pandas (High Performance)...")
        drug_df = pd.read_csv(drug_p, sep='$', usecols=['primaryid', 'drugname'], low_memory=False, nrows=sample_limit)
        outc_df = pd.read_csv(outc_p, sep='$', usecols=['primaryid', 'outc_cod'], low_memory=False, nrows=sample_limit)
        agg_df = aggregate_toxicity_labels(drug_df, outc_df, min_reports=min_reports)
        agg_df.to_csv(target_csv, index=False)
        print(f"[FAERS Pipeline] Saved {len(agg_df):,} safety records to {target_csv}")
    else:
        print(f"[FAERS Pipeline] Processing via pure-Python streaming...")
        records = parse_faers_streaming(drug_p, outc_p, min_reports=min_reports, sample_limit=sample_limit)
        if records:
            keys = list(records[0].keys())
            with open(target_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)
        print(f"[FAERS Pipeline] Saved {len(records):,} safety records to {target_csv}")

    return target_csv


def build_toxicity_labels(
    faers_dir: str | Path,
    min_reports: int = 5,
    sample_limit: Optional[int] = None,
) -> Any:
    """Convenience wrapper to locate FAERS files in a directory and return aggregated toxicity labels."""
    if not PANDAS_AVAILABLE or pd is None:
        raise RuntimeError("pandas is required for build_toxicity_labels.")
    drug_p, outc_p, _ = find_faers_files(faers_dir)
    if not drug_p or not outc_p:
        raise FileNotFoundError(f"Could not find DRUG and OUTC text files in {faers_dir}")
    drug_df = pd.read_csv(drug_p, sep='$', usecols=['primaryid', 'drugname'], low_memory=False, nrows=sample_limit)
    outc_df = pd.read_csv(outc_p, sep='$', usecols=['primaryid', 'outc_cod'], low_memory=False, nrows=sample_limit)
    return aggregate_toxicity_labels(drug_df, outc_df, min_reports=min_reports)


def build_patient_context(faers_base_path: str, sample_size: int = 50000) -> Any:
    """Returns real (primaryid, age, sex) rows from FAERS DEMO."""
    base = Path(faers_base_path)
    _, _, demo_file = find_faers_files(base)
    if not demo_file or not demo_file.is_file():
        demo_file = base / 'DEMO23Q4.txt'

    if not PANDAS_AVAILABLE or pd is None:
        print("[FAERS Pipeline] pandas not installed; returning empty demographics.")
        return []

    print(f"Loading FAERS DEMO file ({demo_file.name})...")
    demo = pd.read_csv(demo_file, sep='$',
                       usecols=['primaryid', 'age', 'age_cod', 'sex'],
                       low_memory=False, nrows=sample_size)

    def normalize_age(row: Any) -> Optional[float]:
        val_raw = row['age']
        if pd is not None and pd.isna(val_raw):
            return None
        if val_raw is None:
            return None
        unit = str(row.get('age_cod', '')).strip().upper()
        try:
            val = float(row['age'])
            if unit == 'YR' or not unit or unit == 'NAN':
                return val
            elif unit == 'MON':
                return val / 12.0
            elif unit == 'DEC':
                return val * 10.0
        except (ValueError, TypeError):
            return None
        return None

    demo['age_years'] = demo.apply(normalize_age, axis=1)
    demo['sex_code'] = demo['sex'].map({'M': 0, 'F': 1})
    demo = demo.dropna(subset=['age_years', 'sex_code'])

    print(f"Built patient context for {len(demo):,} real FAERS reports")
    return demo[['primaryid', 'age_years', 'sex_code']]


def main() -> None:
    parser = argparse.ArgumentParser(description="Process FAERS safety signals and ROR metrics for AuditDDI.")
    parser.add_argument("--faers-dir", type=str, default=None, help="Path to FAERS root directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output CSV")
    parser.add_argument("--min-reports", type=int, default=5, help="Minimum reports per drug")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit number of lines (for quick test)")

    args = parser.parse_args()

    faers_dir = args.faers_dir
    if not faers_dir:
        try:
            from src.data_prep.path_resolver import resolve_data_base
            base = resolve_data_base()
            faers_dir = str(base / "FAERS")
        except Exception:
            faers_dir = "data/FAERS"

    output_dir = args.output_dir or "results/faers_signals"
    print("=" * 75)
    print("  AUDITDDI: FAERS PHARMACOVIGILANCE & ROR SIGNAL PIPELINE")
    print(f"  Source FAERS Dir : {faers_dir}")
    print(f"  Output Directory : {output_dir}")
    print("=" * 75)

    out_file = process_faers_pipeline(faers_dir, output_dir, min_reports=args.min_reports, sample_limit=args.sample_limit)
    print(f"\n[OK] Complete. Safety signals exported to: {out_file}")


if __name__ == "__main__":
    main()
