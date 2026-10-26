"""Load and validate the audited frozen DDI pair-split tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


FROZEN_SPLIT_SCHEMA = 'auditddi_frozen_pair_splits_v1'
FROZEN_SPLIT_NAMES = (
    'transductive_train',
    'validation',
    'transductive_test',
    's1_dev',
    's2_dev',
    's1_test',
    's2_test',
)
PAIR_COLUMNS = ('source', 'target', 'label', 'label_evidence')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_pair_splits(
    split_dir: str | Path,
    *,
    data_root: str | Path,
    expected_cohort: str,
    expected_excluded_drug_ids: list[int] | tuple[int, ...],
    expected_split_seed: int,
    expected_positive_pair_rows: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load saved pair tables and fail closed if their provenance differs."""
    directory = Path(split_dir).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    manifest_path = directory / 'split_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Frozen split manifest not found: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != FROZEN_SPLIT_SCHEMA:
        raise ValueError(f'Unsupported frozen split schema: {manifest.get("schema")!r}.')
    if manifest.get('cohort') != expected_cohort:
        raise ValueError(
            f'Frozen split cohort is {manifest.get("cohort")!r}; '
            f'expected {expected_cohort!r}.'
        )
    if int(manifest.get('split_seed', -1)) != int(expected_split_seed):
        raise ValueError('Frozen split seed does not match AUDITDDI_SPLIT_SEED.')
    if sorted(map(int, manifest.get('excluded_drug_ids', []))) != sorted(
        map(int, expected_excluded_drug_ids)
    ):
        raise ValueError('Frozen split identity exclusions do not match the requested cohort.')
    if int(manifest.get('input_positive_pair_rows', -1)) != int(expected_positive_pair_rows):
        raise ValueError(
            'Frozen split positive-pair count does not match the cleaned training input.'
        )

    if set(manifest.get('known_pair_sources_for_negative_exclusion', [])) != {
        'TWOSIDES', 'DDInter v2'
    }:
        raise ValueError('Frozen splits do not document both TWOSIDES and DDInter negative exclusions.')
    if 'not known safe' not in str(manifest.get('negative_label_meaning', '')).lower():
        raise ValueError('Frozen split manifest must identify sampled negatives as not known safe.')

    input_hashes = manifest.get('input_files', {})
    required_input_paths = {
        'processed/auditddi_v1/twosides_drug_cid_resolution.csv',
        'processed/auditddi_v2/twosides_pairs.csv',
        'processed/auditddi_v2/ddinter_pairs.csv',
    }
    if not required_input_paths.issubset(input_hashes):
        raise ValueError('Frozen split manifest does not hash all required identity and pair inputs.')
    for relative_path, expected_hash in input_hashes.items():
        input_path = root / relative_path
        if not input_path.is_file():
            raise FileNotFoundError(f'Frozen split source input is missing: {input_path}')
        if _sha256(input_path) != expected_hash:
            raise ValueError(f'Frozen split source hash mismatch: {relative_path}')

    per_split = manifest.get('split_audit', {}).get('per_split', {})
    result: dict[str, pd.DataFrame] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for name in FROZEN_SPLIT_NAMES:
        path = directory / f'{name}.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Frozen split table not found: {path}')
        frame = pd.read_csv(path)
        missing_columns = set(PAIR_COLUMNS).difference(frame.columns)
        if missing_columns:
            raise ValueError(f'{path.name} lacks required columns: {sorted(missing_columns)}')
        frame = frame.loc[:, PAIR_COLUMNS].copy()
        if frame[list(PAIR_COLUMNS)].isna().any().any():
            raise ValueError(f'{path.name} contains missing pair data.')
        frame['source'] = frame['source'].astype(str)
        frame['target'] = frame['target'].astype(str)
        frame['label'] = pd.to_numeric(frame['label'], errors='raise').astype('float32')
        if not set(frame['label'].unique()).issubset({0.0, 1.0}):
            raise ValueError(f'{path.name} contains labels outside binary 0/1.')
        if frame.duplicated(['source', 'target']).any():
            raise ValueError(f'{path.name} contains duplicate directed pair rows.')

        counts = per_split.get(name)
        if not isinstance(counts, dict):
            raise ValueError(f'Frozen manifest has no audit counts for {name}.')
        positives = int((frame['label'] == 1.0).sum())
        negatives = int((frame['label'] == 0.0).sum())
        if (
            len(frame) != int(counts.get('total_rows', -1))
            or positives != int(counts.get('reported_positive_pairs', -1))
            or negatives != int(counts.get('sampled_negatives', -1))
        ):
            raise ValueError(f'{path.name} row or label counts do not match its split manifest.')
        if positives != negatives:
            raise ValueError(f'{path.name} is not balanced 1:1 as declared for this cohort.')

        pair_keys = [tuple(sorted((a, b))) for a, b in zip(frame['source'], frame['target'])]
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError(f'{path.name} contains duplicate unordered pairs.')
        overlap = seen_pairs.intersection(pair_keys)
        if overlap:
            raise ValueError(f'{path.name} repeats pairs assigned to an earlier split.')
        seen_pairs.update(pair_keys)
        result[name] = frame

    summary = {
        'protocol': 'frozen_audited_pair_splits_v1',
        'cohort': expected_cohort,
        'frozen_split_dir': str(directory),
        'frozen_split_manifest': manifest,
        'frozen_split_manifest_sha256': _sha256(manifest_path),
        'frozen_split_files_sha256': {
            f'{name}.csv': _sha256(directory / f'{name}.csv')
            for name in FROZEN_SPLIT_NAMES
        },
        'effective_positive_pairs': int(sum(
            int((frame['label'] == 1.0).sum()) for frame in result.values()
        )),
        'sampled_unreported_negative_pairs': int(sum(
            int((frame['label'] == 0.0).sum()) for frame in result.values()
        )),
        'total_pair_rows_before_validation_partition': int(sum(map(len, result.values()))),
        'negative_label_meaning': manifest.get('negative_label_meaning'),
        'known_pair_sources_for_negative_exclusion': manifest.get(
            'known_pair_sources_for_negative_exclusion'
        ),
    }
    return result, summary
