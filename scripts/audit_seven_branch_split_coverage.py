"""Audit per-branch drug coverage on saved AuditDDI split tables.

The audit joins feature rows back to split drug indices through the audited
``drug_features_by_cid.csv`` identity table. It reports drug-level coverage
and pair coverage (both endpoints observed, or at least one endpoint observed)
for every saved split manifest found under ``results/artifacts``.

Example (Colab):
    python scripts/audit_seven_branch_split_coverage.py \
      --data-root /content/drive/MyDrive/Projects/auditddi

This is an audit only: it does not alter splits, impute features, or retrain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


BRANCH_FILES = {
    'drug_profile': 'drug_profile.csv',
    'adverse_event': 'adverse_event.csv',
    'protein_target': 'protein_target.csv',
    'pdb_structure': 'pdb_structure.csv',
    'gene_expression': 'gene_expression.csv',
}


def _drug_key(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def _feature_smiles(path: Path, known_smiles: set[str]) -> set[str]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path, low_memory=False)
    if 'canonical_smiles' not in frame:
        raise ValueError(f'{path} must have a canonical_smiles column.')
    if frame['canonical_smiles'].duplicated().any():
        raise ValueError(f'{path} contains duplicate canonical_smiles rows.')
    # Some feature tables intentionally contain one row for every drug and
    # encode missingness with per-feature observed_* bits. Row presence alone
    # therefore overstates coverage (notably for protein_target.csv).
    observed_columns = [column for column in frame if column.startswith('observed_')]
    if observed_columns:
        observed = frame[observed_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        has_features = observed.gt(0).any(axis=1)
    else:
        # Other current branch tables contain only drugs with observed data.
        has_features = pd.Series(True, index=frame.index)
    return {
        smiles
        for smiles, has_data in zip(frame['canonical_smiles'], has_features)
        if has_data and pd.notna(smiles) and str(smiles) in known_smiles
    }


def _pair_columns(frame: pd.DataFrame, split_path: Path) -> tuple[str, str]:
    for left, right in (('source', 'target'), ('drug_a', 'drug_b'), ('pair_a', 'pair_b')):
        if {left, right}.issubset(frame.columns):
            return left, right
    raise ValueError(
        f'{split_path} has no recognized drug-pair columns. '
        f'Columns: {frame.columns.tolist()}'
    )


def audit(data_root: Path, output: Path | None = None) -> pd.DataFrame:
    processed = data_root / 'processed' / 'auditddi_v2'
    identities_path = processed / 'drug_features_by_cid.csv'
    if not identities_path.is_file():
        raise FileNotFoundError(f'Audited drug identity table not found: {identities_path}')
    identities = pd.read_csv(identities_path, low_memory=False)
    required = {'drug_idx', 'canonical_smiles'}
    if not required.issubset(identities.columns):
        raise ValueError(f'{identities_path} needs columns {sorted(required)}.')
    identities = identities.dropna(subset=['drug_idx', 'canonical_smiles']).copy()
    identities['drug_idx'] = identities['drug_idx'].map(_drug_key)
    identities['canonical_smiles'] = identities['canonical_smiles'].astype(str).str.strip()
    if identities['drug_idx'].duplicated().any():
        raise ValueError('Identity table has duplicate drug_idx values.')
    if identities['canonical_smiles'].duplicated().any():
        raise ValueError('Identity table has duplicate canonical_smiles values.')
    known_smiles = set(identities['canonical_smiles'])
    index_to_smiles = dict(zip(identities['drug_idx'], identities['canonical_smiles']))

    def resolve_split_drug(value: Any) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        # The trainer stores split endpoints as canonical SMILES. Also accept
        # integer IDs for older/raw splits, resolving them through the same
        # audited catalogue rather than comparing unlike key types.
        if text in known_smiles:
            return text
        return index_to_smiles.get(_drug_key(value))

    feature_dir = data_root / 'processed' / 'seven_branch'
    branch_drugs = {
        name: _feature_smiles(feature_dir / filename, known_smiles)
        for name, filename in BRANCH_FILES.items()
    }
    manifests = sorted(
        (data_root / 'results' / 'artifacts').rglob('split_manifest.json'),
        key=lambda p: p.stat().st_mtime,
    )
    if not manifests:
        raise FileNotFoundError(
            f'No split_manifest.json found under {data_root / "results" / "artifacts"}.'
        )

    records: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        split_dir = manifest_path.parent
        for split_name, details in manifest.items():
            relative = details.get('path') if isinstance(details, dict) else None
            split_path = Path(relative) if relative else split_dir / f'{split_name}.csv'
            if not split_path.is_absolute():
                split_path = split_dir / split_path
            if not split_path.is_file():
                continue
            split = pd.read_csv(split_path, low_memory=False)
            left_col, right_col = _pair_columns(split, split_path)
            left = split[left_col].map(resolve_split_drug)
            right = split[right_col].map(resolve_split_drug)
            valid = left.notna() & right.notna()
            left, right = left[valid], right[valid]
            all_drugs = set(left) | set(right)
            for branch, covered in branch_drugs.items():
                left_observed = left.isin(covered)
                right_observed = right.isin(covered)
                records.append({
                    'run': split_dir.parent.name,
                    'split': split_name,
                    'branch': branch,
                    'rows': int(len(left)),
                    'unique_drugs': len(all_drugs),
                    'covered_unique_drugs': len(all_drugs & covered),
                    'drug_coverage': len(all_drugs & covered) / len(all_drugs) if all_drugs else 0.0,
                    'pairs_both_drugs_covered': int((left_observed & right_observed).sum()),
                    'pair_coverage_both': float((left_observed & right_observed).mean()) if len(left) else 0.0,
                    'pairs_any_drug_covered': int((left_observed | right_observed).sum()),
                    'pair_coverage_any': float((left_observed | right_observed).mean()) if len(left) else 0.0,
                    'feature_file_present': (feature_dir / BRANCH_FILES[branch]).is_file(),
                    'split_file': str(split_path),
                })

    report = pd.DataFrame(records)
    if output is None:
        output = feature_dir / 'split_branch_coverage.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(f'Feature coverage rows: {len(records)}')
    print(f'Split manifests inspected: {len(manifests)}')
    print(f'Coverage report: {output}')
    if not report.empty:
        recent = report[report['run'] == manifests[-1].parent.parent.name]
        if not recent.empty:
            print('\nMost recently modified split run:')
            print(recent[['split', 'branch', 'unique_drugs', 'drug_coverage',
                          'pair_coverage_both']].to_string(index=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    audit(args.data_root, args.output)


if __name__ == '__main__':
    main()
