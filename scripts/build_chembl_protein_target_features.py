#!/usr/bin/env python
"""Build audited drug-level human protein-target features from ChEMBL activities.

The script exact-matches the project's audited canonical SMILES to the local
ChEMBL chemreps export, then retrieves paginated activity records for those
ChEMBL molecule IDs from the official REST API. It does not infer a target
from target metadata alone. Run in Colab with Drive mounted so API responses
and the compact output remain in Drive.

Example:
  python scripts/build_chembl_protein_target_features.py \
    --data-root /content/drive/MyDrive/Projects/auditddi \
    --chemreps /content/drive/MyDrive/Projects/auditddi/chembl/chembl_37_chemreps.txt.gz \
    --output-dir /content/drive/MyDrive/Projects/auditddi/processed/seven_branch
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, rdBase


API_ROOT = 'https://www.ebi.ac.uk/chembl/api/data/'
RESOLUTION_RELATIVE = Path('processed/auditddi_v1/twosides_drug_cid_resolution.csv')
ALLOWED_STANDARD_TYPES = {'IC50', 'KI', 'KD', 'EC50', 'AC50'}
ALLOWED_ASSAY_TYPES = {'B', 'F'}
API_FIELDS = ','.join((
    'molecule_chembl_id', 'target_chembl_id', 'target_organism',
    'target_tax_id', 'target_type', 'assay_type', 'confidence_score',
    'standard_type', 'standard_relation', 'pchembl_value',
    'assay_chembl_id', 'document_chembl_id',
))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonicalize(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value.strip())
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def load_resolution(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    cid_col = next((name for name in ('selected_pubchem_cid', 'cid', 'pubchem_cid') if name in frame), None)
    required = {'drug_idx', 'canonical_smiles'}
    if required.difference(frame.columns) or cid_col is None:
        raise ValueError(
            f'{path} must contain drug_idx, canonical_smiles, and a selected PubChem CID column; '
            f'found {frame.columns.tolist()}'
        )
    frame['drug_idx'] = pd.to_numeric(frame['drug_idx'], errors='coerce')
    frame['canonical_smiles'] = frame['canonical_smiles'].map(canonicalize)
    frame['pubchem_cid'] = frame[cid_col].astype(str).str.strip()
    frame = frame.dropna(subset=['drug_idx', 'canonical_smiles'])
    frame['drug_idx'] = frame['drug_idx'].astype(int)
    frame = frame[frame['pubchem_cid'].str.fullmatch(r'\d+')]
    if frame['drug_idx'].duplicated().any():
        raise ValueError('The audited identity table contains duplicate drug_idx rows.')
    if frame['canonical_smiles'].duplicated().any():
        raise ValueError('The audited identity table maps multiple drug IDs to the same SMILES.')
    if frame.empty:
        raise ValueError('The audited identity table has no valid CID/SMILES records.')
    return frame[['drug_idx', 'canonical_smiles', 'pubchem_cid']].sort_values('drug_idx')


def load_structure_matches(chemreps: Path, smiles_to_idx: dict[str, int], chunksize: int) -> tuple[dict[str, set[str]], int]:
    matched: dict[str, set[str]] = defaultdict(set)
    source_rows = 0
    reader = pd.read_csv(
        chemreps, sep='\t', compression='infer', dtype=str, keep_default_na=False,
        usecols=lambda col: col in {'chembl_id', 'canonical_smiles'},
        chunksize=chunksize, low_memory=False,
    )
    saw_rows = False
    for chunk in reader:
        saw_rows = True
        source_rows += len(chunk)
        for row in chunk.itertuples(index=False, name=None):
            chembl_id, raw_smiles = row
            smiles = canonicalize(raw_smiles)
            if smiles in smiles_to_idx and chembl_id:
                matched[smiles].add(chembl_id)
    if not saw_rows:
        raise ValueError(f'ChEMBL chemreps file is empty: {chemreps}')
    return matched, source_rows


def get_json(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = session.get(url, params=params, timeout=(20, 180))
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == 5:
                break
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f'ChEMBL API request failed after retries: {url}') from last_error


def api_version(session: requests.Session) -> str:
    result = get_json(session, urljoin(API_ROOT, 'status.json'))
    status = result.get('status', result)
    return str(status.get('chembl_db_version') or status.get('database_version') or status)


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def fetch_activity_batch(
    session: requests.Session,
    molecule_ids: list[str],
    cache_path: Path,
    page_limit: int,
    delay: float,
) -> list[dict[str, Any]]:
    if cache_path.is_file():
        cached: list[dict[str, Any]] = []
        with cache_path.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    cached.append(json.loads(line))
        return cached

    endpoint = urljoin(API_ROOT, 'activity.json')
    params: dict[str, Any] | None = {
        'molecule_chembl_id__in': ','.join(molecule_ids),
        'pchembl_value__isnull': 'false',
        'only': API_FIELDS,
        'limit': page_limit,
        'offset': 0,
    }
    records: list[dict[str, Any]] = []
    while endpoint:
        payload = get_json(session, endpoint, params)
        params = None
        block = payload.get('activities', [])
        records.extend(block)
        next_page = payload.get('page_meta', {}).get('next')
        endpoint = urljoin('https://www.ebi.ac.uk', next_page) if next_page else ''
        if endpoint:
            time.sleep(delay)
    temp = cache_path.with_suffix(cache_path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(',', ':')) + '\n')
    temp.replace(cache_path)
    return records


def keep_activity(record: dict[str, Any]) -> tuple[bool, str | None]:
    organism = str(record.get('target_organism') or '').strip().lower()
    tax_id = str(record.get('target_tax_id') or '').strip()
    if organism != 'homo sapiens' and tax_id != '9606':
        return False, None
    if str(record.get('assay_type') or '').strip().upper() not in ALLOWED_ASSAY_TYPES:
        return False, None
    if str(record.get('standard_type') or '').strip().upper() not in ALLOWED_STANDARD_TYPES:
        return False, None
    if str(record.get('standard_relation') or '').strip() not in {'=', '=='}:
        return False, None
    try:
        confidence = int(record.get('confidence_score'))
        pchembl = float(record.get('pchembl_value'))
    except (TypeError, ValueError):
        return False, None
    target = str(record.get('target_chembl_id') or '').strip()
    if confidence < 8 or not target or not math.isfinite(pchembl) or not 0.0 <= pchembl <= 14.0:
        return False, None
    return True, target


def build_features(
    resolution: pd.DataFrame,
    chembl_ids_by_smiles: dict[str, set[str]],
    records: list[dict[str, Any]],
    top_targets: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    smiles_to_idx = dict(zip(resolution['canonical_smiles'], resolution['drug_idx']))
    chembl_to_smiles: dict[str, set[str]] = defaultdict(set)
    for smiles, ids in chembl_ids_by_smiles.items():
        for chembl_id in ids:
            chembl_to_smiles[chembl_id].add(smiles)

    # Collapse assay replicates to one median pChEMBL value per drug/target.
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    rejection_counts: dict[str, int] = defaultdict(int)
    eligible_rows = 0
    ambiguous_molecule_links = 0
    for record in records:
        keep, target = keep_activity(record)
        if not keep or target is None:
            rejection_counts['filtered_out'] += 1
            continue
        eligible_rows += 1
        molecule_id = str(record.get('molecule_chembl_id') or '').strip()
        smiles_candidates = chembl_to_smiles.get(molecule_id, set())
        if len(smiles_candidates) != 1:
            if len(smiles_candidates) > 1:
                ambiguous_molecule_links += 1
            rejection_counts['unmapped_or_ambiguous_molecule'] += 1
            continue
        smiles = next(iter(smiles_candidates))
        try:
            value = float(record['pchembl_value'])
        except (TypeError, ValueError):
            continue
        values[(smiles, target)].append(value)

    target_drug_counts: dict[str, int] = defaultdict(int)
    collapsed: dict[tuple[str, str], float] = {}
    for key, measurements in values.items():
        collapsed[key] = float(np.median(np.asarray(measurements, dtype=np.float64)))
        target_drug_counts[key[1]] += 1
    selected_targets = sorted(
        target_drug_counts,
        key=lambda target: (-target_drug_counts[target], target),
    )[:top_targets]
    if not selected_targets:
        raise RuntimeError(
            'No ChEMBL activities passed the documented human/high-confidence filters. '
            'Review API coverage and filters; do not create placeholder target features.'
        )

    # Per-target centered/scaled affinity and a companion observed bit retain
    # missingness without making absence look like measured zero affinity.
    targets_by_smiles: dict[str, dict[str, float]] = defaultdict(dict)
    for (smiles, target), value in collapsed.items():
        if target in selected_targets:
            targets_by_smiles[smiles][target] = value

    columns: dict[str, np.ndarray] = {}
    for target in selected_targets:
        observed_values = np.asarray(
            [targets_by_smiles[s][target] for s in targets_by_smiles if target in targets_by_smiles[s]],
            dtype=np.float64,
        )
        mean = float(observed_values.mean())
        std = float(observed_values.std())
        scale = std if std > 1e-8 else 1.0
        activity_col = f'pchembl_z_{target}'
        observed_col = f'observed_{target}'
        columns[activity_col] = np.zeros(len(resolution), dtype=np.float32)
        columns[observed_col] = np.zeros(len(resolution), dtype=np.float32)
        for row_index, smiles in enumerate(resolution['canonical_smiles']):
            raw = targets_by_smiles.get(smiles, {}).get(target)
            if raw is not None:
                columns[activity_col][row_index] = np.float32(np.clip((raw - mean) / scale, -3.0, 3.0) / 3.0)
                columns[observed_col][row_index] = 1.0

    table = pd.DataFrame({'canonical_smiles': resolution['canonical_smiles'].tolist()})
    for name, vector in columns.items():
        table[name] = vector
    drugs_with_features = int((table.drop(columns='canonical_smiles').to_numpy().sum(axis=1) > 0).sum())
    summary = {
        'total_api_activity_records': len(records),
        'eligible_human_activity_records_before_exact_molecule_mapping': eligible_rows,
        'filtered_out_activity_records': rejection_counts['filtered_out'],
        'unmapped_or_ambiguous_molecule_records': rejection_counts['unmapped_or_ambiguous_molecule'],
        'ambiguous_molecule_link_records': ambiguous_molecule_links,
        'mapped_drug_target_pairs_before_top_target_selection': len(collapsed),
        'selected_target_count': len(selected_targets),
        'selected_targets': selected_targets,
        'drugs_with_selected_target_features': drugs_with_features,
        'total_drugs': len(resolution),
        'drug_coverage': drugs_with_features / len(resolution),
        'target_pair_aggregation': 'median pChEMBL across eligible records per exact canonical drug and target',
        'feature_encoding': (
            'for each selected target: mean/std-standardized median pChEMBL clipped to [-3,3] '
            'and divided by 3, plus a separate observed bit; missing affinity stored as 0 '
            'with observed bit 0'
        ),
        'activity_filters': {
            'organism_or_tax_id': 'Homo sapiens or NCBI tax ID 9606',
            'assay_type': sorted(ALLOWED_ASSAY_TYPES),
            'standard_type': sorted(ALLOWED_STANDARD_TYPES),
            'standard_relation': ['=', '=='],
            'minimum_confidence_score': 8,
            'pchembl_value': 'finite, between 0 and 14',
        },
    }
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--chemreps', type=Path)
    parser.add_argument('--resolution', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--top-targets', type=int, default=512)
    parser.add_argument('--molecule-batch-size', type=int, default=20)
    parser.add_argument('--page-limit', type=int, default=1000)
    parser.add_argument('--chunksize', type=int, default=100_000)
    parser.add_argument('--request-delay-seconds', type=float, default=0.15)
    parser.add_argument('--user-agent', default='AuditDDI research feature preparation (contact: project maintainer)')
    args = parser.parse_args()
    data_root = args.data_root
    chemreps = args.chemreps or data_root / 'chembl' / 'chembl_37_chemreps.txt.gz'
    resolution_path = args.resolution or data_root / RESOLUTION_RELATIVE
    output_dir = args.output_dir or data_root / 'processed' / 'seven_branch'
    if args.top_targets < 1 or args.molecule_batch_size < 1 or args.page_limit < 1:
        raise ValueError('top-targets, molecule-batch-size, and page-limit must be positive.')
    for label, path in [('ChEMBL chemreps', chemreps), ('TWOSIDES identity resolution', resolution_path)]:
        if not path.is_file():
            raise FileNotFoundError(f'{label} not found: {path}')
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = load_resolution(resolution_path)
    smiles_to_idx = dict(zip(resolution['canonical_smiles'], resolution['drug_idx']))
    print(f'Identity rows: {len(resolution)}; canonical drugs: {len(smiles_to_idx)}')
    print(f'Scanning ChEMBL structures: {chemreps}')
    chembl_ids_by_smiles, structure_rows = load_structure_matches(chemreps, smiles_to_idx, args.chunksize)
    matched_smiles = set(chembl_ids_by_smiles)
    molecule_ids = sorted(set().union(*chembl_ids_by_smiles.values())) if matched_smiles else []
    print(
        f'ChEMBL structure scan: {structure_rows:,} rows; exact drug structure matches '
        f'{len(matched_smiles)}/{len(resolution)}; molecule IDs: {len(molecule_ids)}'
    )
    if not molecule_ids:
        raise RuntimeError('No exact canonical-SMILES matches to ChEMBL; stopping before activity retrieval.')

    session = requests.Session()
    session.headers.update({'User-Agent': args.user_agent, 'Accept': 'application/json'})
    version = api_version(session)
    print('ChEMBL API database version:', version)
    cache_version = re.sub(r'[^A-Za-z0-9_.-]+', '_', version)
    cache_dir = output_dir / 'chembl_activity_cache' / f'chembl_{cache_version}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    batches = list(chunks(molecule_ids, args.molecule_batch_size))
    for batch_index, molecule_batch in enumerate(batches):
        cache_path = cache_dir / f'activity_batch_{batch_index:05d}.jsonl'
        records = fetch_activity_batch(
            session, molecule_batch, cache_path, args.page_limit, args.request_delay_seconds
        )
        all_records.extend(records)
        print(f'Activity batch {batch_index + 1}/{len(batches)}: {len(records):,} records (cached in Drive)')
        if batch_index + 1 < len(batches):
            time.sleep(args.request_delay_seconds)

    table, audit = build_features(resolution, chembl_ids_by_smiles, all_records, args.top_targets)
    table_path = output_dir / 'protein_target.csv'
    table.to_csv(table_path, index=False, float_format='%.7g')
    manifest = {
        'schema': 'auditddi_chembl_protein_target_features_v1',
        'dataset': 'ChEMBL activity records',
        'release': version,
        'source': 'ChEMBL REST activity endpoint',
        'source_api': urljoin(API_ROOT, 'activity.json'),
        'activity_encoding': audit['feature_encoding'],
        'drug_identity_mapping': (
            'exact RDKit canonical-SMILES match between audited TWOSIDES PubChem identity table '
            'and local ChEMBL 37 chemreps; ambiguous ChEMBL molecule-to-SMILES links excluded'
        ),
        'chemreps_path': str(chemreps),
        'chemreps_sha256': sha256_file(chemreps),
        'resolution_path': str(resolution_path),
        'resolution_sha256': sha256_file(resolution_path),
        'api_filters': audit['activity_filters'],
        'aggregation_rule': audit['target_pair_aggregation'],
        'target_selection': f'top {args.top_targets} by number of mapped drugs, tie-broken by target ChEMBL ID',
        'coverage': {
            'resolution_drugs': len(resolution),
            'chembl_structure_matched_drugs': len(matched_smiles),
            'activity_feature_drugs': audit['drugs_with_selected_target_features'],
            'fraction_of_resolution_drugs': audit['drug_coverage'],
        },
        'audit': audit,
        'cache_dir': str(cache_dir),
        'output_csv': str(table_path),
        'output_sha256': sha256_file(table_path),
        'limitations': [
            'ChEMBL activity evidence reflects published in-vitro assays and is not a complete targetome.',
            'Only exact canonical-SMILES mappings and high-confidence human protein assays are retained.',
            'Feature coverage and label/source overlap must be audited on the final DDI splits.',
            'Target features do not establish DDI mechanism or direction.',
        ],
    }
    manifest_path = output_dir / 'protein_target.manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Feature table:', table_path)
    print('Manifest:', manifest_path)
    print(f"Feature coverage: {audit['drugs_with_selected_target_features']}/{len(resolution)} drugs")
    print(f"Target features: {len(table.columns) - 1} ({audit['selected_target_count']} targets × value/presence)")


if __name__ == '__main__':
    main()
