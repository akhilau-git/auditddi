"""Join compact, per-drug modality tables into the seven-branch model bundle.

Each input CSV must contain one row per canonicalizable `canonical_smiles`
and numeric feature columns. Aggregate context-specific observations (for
example GEO cell/treatment signatures) before calling this script. Missing
drug rows remain masked; they are never treated as observed zero vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

SRC_PATH = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_prep.pubchem_bridge import canonicalize


BRANCHES = (
    'drug_profile',
    'adverse_event',
    'protein_target',
    'pdb_structure',
    'gene_expression',
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_table(path: Path, branch: str) -> tuple[dict[str, np.ndarray], list[str]]:
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            required = {'canonical_smiles', 'features', 'feature_names'}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f'{path} is missing arrays: {sorted(missing)}.')
            smiles = archive['canonical_smiles'].astype(str)
            matrix = archive['features'].astype(np.float64)
            feature_columns = archive['feature_names'].astype(str).tolist()
        if matrix.ndim != 2 or matrix.shape != (len(smiles), len(feature_columns)):
            raise ValueError(f'{path} has inconsistent matrix and feature metadata shapes.')
        records = zip(smiles, matrix)
    else:
        frame = pd.read_csv(path)
        if 'canonical_smiles' not in frame:
            raise ValueError(f'{path} must have a canonical_smiles column.')
        feature_columns = [column for column in frame.columns if column != 'canonical_smiles']
        if not feature_columns:
            raise ValueError(f'{path} has no feature columns for {branch}.')
        numeric = frame[feature_columns].apply(pd.to_numeric, errors='coerce')
        bad_text = frame[feature_columns].notna() & numeric.isna()
        if bad_text.any().any():
            columns = bad_text.columns[bad_text.any()].tolist()
            raise ValueError(f'{path} has non-numeric values in feature columns {columns}.')
        records = zip(frame['canonical_smiles'], numeric.to_numpy(dtype=np.float64))
    if not feature_columns:
        raise ValueError(f'{path} has no feature columns for {branch}.')
    vectors: dict[str, np.ndarray] = {}
    for raw_smiles, row in records:
        key = canonicalize(str(raw_smiles)) if pd.notna(raw_smiles) else None
        if key is None:
            continue
        if key in vectors:
            raise ValueError(
                f'{path} contains multiple rows for the same canonical drug ({key}). '
                'Aggregate the source observations with an explicit context policy first.'
            )
        observed = np.isfinite(row)
        if observed.any() and not observed.all():
            raise ValueError(
                f'{path} has partially missing values for {key}. Impute them using a '
                'documented source-specific rule before bundling; the row mask only '
                'represents whether the modality is observed for that drug.'
            )
        vector = np.where(observed, row, 0.0).astype(np.float32)
        if not np.isfinite(vector).all():
            raise ValueError(f'{path} contains non-finite feature values.')
        # A row with every value missing is not evidence of feature availability.
        if observed.any():
            vectors[key] = vector
    if not vectors:
        raise ValueError(f'{path} has no valid, non-empty mapped rows for {branch}.')
    return vectors, [str(name) for name in feature_columns]


def build(
    tables: dict[str, Path], output: Path, provenance_path: Path
) -> dict[str, Any]:
    missing = set(BRANCHES).difference(tables)
    if missing:
        raise ValueError(f'Provide a feature table for every branch: {sorted(missing)}.')
    loaded = {name: _load_table(tables[name], name) for name in BRANCHES}
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    branch_provenance = provenance.get('branches', {})
    for branch in BRANCHES:
        details = branch_provenance.get(branch, {})
        for field in ('dataset', 'release', 'feature_definition', 'drug_identity_mapping'):
            if not str(details.get(field, '')).strip():
                raise ValueError(f'Provenance for {branch} must declare {field}.')
    if branch_provenance['adverse_event'].get('pairwise_ddi_signals_excluded') is not True:
        raise ValueError(
            'Adverse-event provenance must confirm pairwise DDI-derived signals were excluded.'
        )
    if not str(branch_provenance['adverse_event'].get('cutoff_date', '')).strip():
        raise ValueError('Adverse-event provenance must declare a reporting cutoff date.')
    for field in ('cell_or_tissue_context', 'perturbation_type', 'aggregation_rule'):
        if not str(branch_provenance['gene_expression'].get(field, '')).strip():
            raise ValueError(f'Gene-expression provenance must declare {field}.')
    if not str(branch_provenance['protein_target'].get('activity_encoding', '')).strip():
        raise ValueError('Protein-target provenance must declare activity_encoding.')
    if not str(branch_provenance['pdb_structure'].get('structure_to_drug_mapping', '')).strip():
        raise ValueError('PDB provenance must declare structure_to_drug_mapping.')
    smiles = sorted(set().union(*(set(values) for values, _ in loaded.values())))
    arrays: dict[str, np.ndarray] = {'canonical_smiles': np.asarray(smiles, dtype=str)}
    coverage: dict[str, Any] = {}
    for branch in BRANCHES:
        vectors, names = loaded[branch]
        dimension = len(names)
        matrix = np.zeros((len(smiles), dimension), dtype=np.float32)
        mask = np.zeros((len(smiles),), dtype=np.float32)
        for index, key in enumerate(smiles):
            if key in vectors:
                if vectors[key].shape != (dimension,):
                    raise ValueError(f'Inconsistent vector shape in {branch} table for {key}.')
                matrix[index] = vectors[key]
                mask[index] = 1.0
        arrays[f'{branch}_features'] = matrix
        arrays[f'{branch}_mask'] = mask
        arrays[f'{branch}_feature_names'] = np.asarray(names, dtype=str)
        coverage[branch] = {
            'dimension': dimension,
            'available_drugs': int(mask.sum()),
            'drug_coverage': float(mask.mean()) if len(mask) else 0.0,
            'source_csv': str(tables[branch]),
            'source_sha256': _sha256(tables[branch]),
            'feature_names': names,
        }
        if mask.sum() == 0:
            raise ValueError(f'{branch} has no mapped drug features; refusing an empty branch.')

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    manifest = {
        'schema': 'auditddi_seven_branch_feature_bundle_v1',
        'drug_key': 'canonical_smiles',
        'row_count': len(smiles),
        'branches': coverage,
        'source_provenance': branch_provenance,
        'source_provenance_file': str(provenance_path),
        'source_provenance_sha256': _sha256(provenance_path),
        'missingness_policy': 'zero storage plus explicit observed-row mask; mask is model input',
        'context_aggregation_policy': (
            'performed upstream; this builder rejects duplicate drug rows'
        ),
        'bundle_npz': str(output),
        'bundle_sha256': _sha256(output),
    }
    manifest_path = output.with_suffix('.manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for branch in BRANCHES:
        parser.add_argument(f'--{branch.replace("_", "-")}', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provenance-json', type=Path, required=True)
    args = parser.parse_args()
    tables = {branch: getattr(args, branch) for branch in BRANCHES}
    print(json.dumps(build(tables, args.output, args.provenance_json), indent=2))


if __name__ == '__main__':
    main()

