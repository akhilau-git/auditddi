"""Build a variable-width, drug-mapped PharmGKB gene profile.

This consumes the audited long annotations emitted by
``prepare_auditddi_source_bundle.py``. Drug identity is joined only through
the selected PubChem CID crosswalk; drug-name matching is not performed here.

Example (Colab):
    python scripts/build_pharmacogenomic_drug_profile.py \
      --data-root /content/drive/MyDrive/Projects/auditddi

Outputs ``processed/seven_branch/drug_profile.csv`` and a JSON manifest.
The feature width is selected from observed PharmGKB genes rather than being
fixed at 51. Coverage remains explicit: only drugs with at least one mapped
gene annotation appear as rows, so the seven-branch bundler can create the
availability mask without mistaking missing records for true zero profiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _cid_key(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'null'}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number <= 0 or not number.is_integer():
        return None
    return str(int(number))


def build(
    annotations_path: Path,
    identities_path: Path,
    output_dir: Path,
    max_genes: int,
    min_drug_frequency: int,
) -> dict[str, object]:
    annotations = pd.read_csv(annotations_path, low_memory=False)
    identities = pd.read_csv(identities_path, low_memory=False)
    required_annotations = {'cid', 'gene_name'}
    required_identities = {'cid', 'canonical_smiles'}
    if not required_annotations.issubset(annotations.columns):
        raise ValueError(
            f'{annotations_path} needs columns {sorted(required_annotations)}; '
            f'found {annotations.columns.tolist()}'
        )
    if not required_identities.issubset(identities.columns):
        raise ValueError(
            f'{identities_path} needs columns {sorted(required_identities)}; '
            f'found {identities.columns.tolist()}'
        )
    if max_genes <= 0 or min_drug_frequency <= 0:
        raise ValueError('max_genes and min_drug_frequency must be positive integers.')

    identities = identities[['cid', 'canonical_smiles']].copy()
    identities['cid'] = identities['cid'].map(_cid_key)
    identities['canonical_smiles'] = (
        identities['canonical_smiles'].fillna('').astype(str).str.strip()
    )
    identities = identities.dropna(subset=['cid'])
    identities = identities[identities['canonical_smiles'].ne('')]
    if identities['cid'].duplicated().any():
        raise ValueError('Drug identity table contains duplicate PubChem CIDs.')
    if identities['canonical_smiles'].duplicated().any():
        raise ValueError('Drug identity table contains duplicate canonical SMILES.')

    annotations = annotations[['cid', 'gene_name']].copy()
    annotations['cid'] = annotations['cid'].map(_cid_key)
    annotations['gene_name'] = (
        annotations['gene_name'].fillna('').astype(str).str.strip()
    )
    annotations = annotations.dropna(subset=['cid'])
    annotations = annotations[annotations['gene_name'].ne('')]
    mapped = annotations.merge(identities, on='cid', how='inner', validate='many_to_one')
    mapped = mapped.drop_duplicates(['canonical_smiles', 'gene_name'])
    if mapped.empty:
        raise ValueError('No PharmGKB gene annotations matched the audited PubChem identities.')

    drug_frequency = mapped.groupby('gene_name')['canonical_smiles'].nunique()
    eligible = drug_frequency[drug_frequency >= min_drug_frequency]
    vocabulary = sorted(
        eligible.index,
        key=lambda gene: (-int(eligible.loc[gene]), gene.casefold(), gene),
    )[:max_genes]
    if not vocabulary:
        raise ValueError(
            'No genes pass min_drug_frequency. Lower that threshold after reviewing coverage.'
        )

    # Percent-encoding yields deterministic, unambiguous column names even
    # when a source gene label contains punctuation.
    feature_columns = [f'pharmgkb_gene:{quote(gene, safe="")}' for gene in vocabulary]
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError('PharmGKB feature-name encoding produced a collision.')
    column_for_gene = dict(zip(vocabulary, feature_columns))
    selected = mapped[mapped['gene_name'].isin(vocabulary)]
    rows = sorted(selected['canonical_smiles'].unique())
    row_index = {smiles: index for index, smiles in enumerate(rows)}
    col_index = {gene: index for index, gene in enumerate(vocabulary)}
    matrix = np.zeros((len(rows), len(vocabulary)), dtype=np.float32)
    for record in selected[['canonical_smiles', 'gene_name']].itertuples(index=False):
        matrix[row_index[record.canonical_smiles], col_index[record.gene_name]] = 1.0
    if not np.isfinite(matrix).all():
        raise ValueError('Generated profile has non-finite values.')

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / 'drug_profile.csv'
    frame = pd.DataFrame(matrix, columns=feature_columns)
    frame.insert(0, 'canonical_smiles', rows)
    frame.to_csv(output_csv, index=False)
    manifest: dict[str, object] = {
        'schema': 'auditddi_pharmacogenomic_drug_profile_v1',
        'dataset': 'PharmGKB chemical-gene annotations supplied with the project',
        'feature_definition': (
            'Binary presence of an audited PharmGKB drug-gene annotation, '
            'joined by selected PubChem CID and represented as a canonical-SMILES keyed row.'
        ),
        'drug_identity_mapping': 'exact selected PubChem CID crosswalk; no name or similarity fallback',
        'source_annotations': str(annotations_path),
        'source_annotations_sha256': _sha256(annotations_path),
        'identity_crosswalk': str(identities_path),
        'identity_crosswalk_sha256': _sha256(identities_path),
        'output_csv': str(output_csv),
        'output_csv_sha256': _sha256(output_csv),
        'drug_rows_with_selected_gene_features': len(rows),
        'mapped_drugs_available_for_profile': int(identities['canonical_smiles'].nunique()),
        'coverage_of_mapped_drugs': float(len(rows) / identities['canonical_smiles'].nunique()),
        'unique_mapped_genes_before_filtering': int(mapped['gene_name'].nunique()),
        'genes_passing_minimum_frequency': int(len(eligible)),
        'selected_gene_features': len(vocabulary),
        'max_genes': max_genes,
        'min_drug_frequency': min_drug_frequency,
        'feature_names': feature_columns,
        'selected_gene_vocabulary': vocabulary,
        'observed_missingness_policy': (
            'Only drugs with at least one selected gene annotation are emitted. '
            'The bundle builder supplies the observed-row mask; it does not infer absent genes.'
        ),
        'scope_limitations': [
            'This table contains mapped gene annotations only; it does not claim measured expression, variants, or pathways.',
            'The 51-column legacy source matrix remains a separate backward-compatible output.',
        ],
    }
    manifest_path = output_dir / 'drug_profile.manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({
        'output': str(output_csv),
        'manifest': str(manifest_path),
        'profile_rows': len(rows),
        'mapped_drugs': int(identities['canonical_smiles'].nunique()),
        'coverage': manifest['coverage_of_mapped_drugs'],
        'features': len(vocabulary),
        'unique_genes': manifest['unique_mapped_genes_before_filtering'],
    }, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, default=None)
    parser.add_argument('--identities', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--max-genes', type=int, default=2048)
    parser.add_argument('--min-drug-frequency', type=int, default=1)
    args = parser.parse_args()
    root = args.data_root
    annotations = args.annotations or (
        root / 'processed' / 'auditddi_v2' / 'pharmgkb_drug_gene_annotations.csv'
    )
    identities = args.identities or (
        root / 'processed' / 'auditddi_v2' / 'drug_features_by_cid.csv'
    )
    output_dir = args.output_dir or root / 'processed' / 'seven_branch'
    build(annotations, identities, output_dir, args.max_genes, args.min_drug_frequency)


if __name__ == '__main__':
    main()
