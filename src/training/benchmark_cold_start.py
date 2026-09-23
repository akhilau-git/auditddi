"""Cold-Start (S1) Benchmarking Suite for AuditDDI.

Compares:
1. Baseline Model: Molecular Graph GNN (edge_aware_gat_v2).
2. Multimodal Model: Molecular Graph + ECFP + PharmGKB Genes + FAERS Toxicity (auditddi_multimodal_v1).

Tracks:
- Transductive Test AUROC / AUPRC / F1 / MCC
- S2 Semi-Inductive Test AUROC / AUPRC
- S1 Cold-Start (Unseen Drugs) AUROC / AUPRC
- Training Runtime per Epoch (seconds)
- Peak GPU/CPU Memory Footprint (MB)
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

# Ensure repository root is on sys.path when run directly as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data_prep.cached_graph_loader import (
    MolecularCache,
    build_cached_multimodal_dataloader,
)
from src.data_prep.splits import create_split_aware_binary_splits
from src.models.ddi_model import (
    MODEL_ARCHITECTURE_EDGE_AWARE,
    MODEL_ARCHITECTURE_MULTIMODAL,
    AuditDDIModel,
)

REQUIRED_SPLIT_FILES = [
    'transductive_train.csv',
    'validation.csv',
    'transductive_test.csv',
    's1_dev.csv',
    's2_dev.csv',
    's1_test.csv',
    's2_test.csv',
]


def _collect_aliases(value: Any) -> set[str]:
    """Collect stable scalar aliases, including values nested in JSON fields."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    if isinstance(value, dict):
        aliases: set[str] = set()
        for nested in value.values():
            aliases.update(_collect_aliases(nested))
        return aliases
    if isinstance(value, (list, tuple, set)):
        aliases: set[str] = set()
        for nested in value:
            aliases.update(_collect_aliases(nested))
        return aliases
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'null'}:
        return set()
    if text[:1] in {'{', '['}:
        try:
            return _collect_aliases(json.loads(text))
        except (TypeError, ValueError):
            pass
    return {text}


def _map_edges_to_master_ids(
    pairs: pd.DataFrame, nodes: pd.DataFrame, source_col: str, target_col: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map source identifiers to the exact node keys used by MolecularCache.

    Ambiguous aliases are discarded rather than guessed. This prevents a
    source ID/name from being treated as a molecular graph key by accident.
    """
    node_id_col = 'drug_id' if 'drug_id' in nodes.columns else (
        'canonical_smiles' if 'canonical_smiles' in nodes.columns else nodes.columns[0]
    )
    alias_cols = [
        col for col in (
            node_id_col, 'drug_id', 'canonical_smiles', 'inchikey', 'pubchem_cid',
            'display_name', 'drug_name', 'source_ids', 'source_ids_json',
            'synonyms', 'synonyms_json',
        ) if col in nodes.columns
    ]
    alias_to_ids: dict[str, set[str]] = {}
    folded_alias_to_ids: dict[str, set[str]] = {}

    def external_alias_keys(alias: str) -> set[str]:
        keys = {alias.casefold()}
        # TWOSIDES/STITCH commonly prefixes PubChem CIDs with zero padding.
        # Normalize only the unambiguous CID form; leave CIDm stereochemical
        # identifiers untouched instead of collapsing distinct compounds.
        cid_match = re.fullmatch(r'CID0*(\d+)', alias, flags=re.IGNORECASE)
        if cid_match:
            keys.add(cid_match.group(1).lstrip('0') or '0')
        numeric_match = re.fullmatch(r'(\d+)(?:\.0+)?', alias)
        if numeric_match:
            keys.add(numeric_match.group(1).lstrip('0') or '0')
        return keys

    for _, row in nodes.iterrows():
        node_id = str(row[node_id_col]).strip()
        if not node_id or node_id.lower() == 'nan':
            continue
        aliases = {node_id}
        if 'canonical_smiles' in nodes.columns:
            aliases.update(_collect_aliases(row['canonical_smiles']))
        for alias in aliases:
            alias_to_ids.setdefault(alias, set()).add(node_id)
        for col in alias_cols:
            if col in {node_id_col, 'canonical_smiles'}:
                continue
            for alias in _collect_aliases(row[col]):
                for key in external_alias_keys(alias):
                    folded_alias_to_ids.setdefault(key, set()).add(node_id)

    unambiguous = {
        alias: next(iter(node_ids))
        for alias, node_ids in alias_to_ids.items()
        if len(node_ids) == 1
    }
    folded_unambiguous = {
        alias: next(iter(node_ids))
        for alias, node_ids in folded_alias_to_ids.items()
        if len(node_ids) == 1
    }
    def resolve_endpoint(value: Any) -> str | None:
        text = str(value).strip()
        if text in unambiguous:
            return unambiguous[text]
        matched_ids = {
            folded_unambiguous[key]
            for key in external_alias_keys(text)
            if key in folded_unambiguous
        }
        return next(iter(matched_ids)) if len(matched_ids) == 1 else None

    mapped_a = pairs[source_col].map(resolve_endpoint)
    mapped_b = pairs[target_col].map(resolve_endpoint)
    matched = mapped_a.notna() & mapped_b.notna()
    result = pd.DataFrame({
        'drug_a_id': mapped_a[matched].values,
        'drug_b_id': mapped_b[matched].values,
    })
    audit = {
        'master_node_rows': int(len(nodes)),
        'unambiguous_aliases': int(len(unambiguous) + len(folded_unambiguous)),
        'ambiguous_aliases_excluded': int(
            sum(len(ids) > 1 for ids in alias_to_ids.values())
            + sum(len(ids) > 1 for ids in folded_alias_to_ids.values())
        ),
        'raw_edge_rows': int(len(pairs)),
        'edge_rows_with_both_drugs_mapped': int(matched.sum()),
        'unmapped_edge_rows': int((~matched).sum()),
        'unique_mapped_positive_pairs': int(result.shape[0]),
    }
    return result, audit


def _check_minimum_split_support(
    splits_path: Path, min_examples_per_class: int
) -> dict[str, dict[str, int]]:
    """Reject empty/tiny partitions before model training or score reporting."""
    counts: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    for filename in REQUIRED_SPLIT_FILES:
        path = splits_path / filename
        if not path.is_file():
            failures.append(f'{filename}: missing')
            continue
        frame = pd.read_csv(path)
        if 'label' not in frame.columns:
            failures.append(f'{filename}: missing label column')
            continue
        labels = pd.to_numeric(frame['label'], errors='coerce')
        positive = int((labels == 1).sum())
        negative = int((labels == 0).sum())
        counts[filename] = {'rows': int(len(frame)), 'positive': positive, 'negative': negative}
        if positive < min_examples_per_class or negative < min_examples_per_class:
            failures.append(
                f'{filename}: positive={positive}, negative={negative} '
                f'(need at least {min_examples_per_class} of each)'
            )
    if failures:
        formatted = '\n  - '.join(failures)
        raise ValueError(
            'The DDI source or split is too small for a meaningful leakage-safe '
            'benchmark; training was stopped before starting.\n  - ' + formatted +
            '\nUse the full TWOSIDES pair file, confirm its drug IDs map to the '
            'master-node identifiers, then generate a fresh split/output folder. '
            'Do not lower this threshold merely to make the run proceed.'
        )
    return counts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_benchmark_splits(
    splits_dir: str | Path | None = None,
    master_nodes_path: str | Path | None = None,
    master_edges_path: str | Path | None = None,
    seed: int = 42,
    holdout_fraction: float = 0.15,
    min_examples_per_class: int = 2,
    **kwargs: Any,
) -> Path:
    """Ensure benchmark splits exist; generate them automatically if missing.

    If splits_dir is None, attempts to resolve a 'splits' folder next to or
    above master_nodes_path. If split CSV files are missing, loads positive pairs
    from master_edges_path (or inferred paths) and generates leakage-safe
    transductive, S1, and S2 splits.
    """
    if master_edges_path is None:
        master_edges_path = kwargs.get('edges_path') or kwargs.get('master_edges')
    splits_path: Path | None = None
    if splits_dir is not None and str(splits_dir).strip().lower() not in ('none', ''):
        splits_path = Path(splits_dir)
    elif master_nodes_path is not None:
        nodes_p = Path(master_nodes_path).resolve()
        candidate_parent = nodes_p.parent.parent / 'splits'
        candidate_sibling = nodes_p.parent / 'splits'
        if candidate_parent.is_dir() and all((candidate_parent / f).is_file() for f in REQUIRED_SPLIT_FILES):
            splits_path = candidate_parent
        elif candidate_sibling.is_dir() and all((candidate_sibling / f).is_file() for f in REQUIRED_SPLIT_FILES):
            splits_path = candidate_sibling
        else:
            splits_path = candidate_parent
    else:
        splits_path = Path('splits')

    splits_path.mkdir(parents=True, exist_ok=True)

    # Check if all required files exist and are non-empty
    existing_files = [
        f for f in REQUIRED_SPLIT_FILES
        if (splits_path / f).is_file() and (splits_path / f).stat().st_size > 0
    ]
    if min_examples_per_class < 2:
        raise ValueError('min_examples_per_class must be at least 2.')
    if len(existing_files) == len(REQUIRED_SPLIT_FILES):
        print(f"Using existing benchmark splits from: {splits_path}")
        _check_minimum_split_support(splits_path, min_examples_per_class)
        return splits_path

    print(f"Benchmark split files missing in {splits_path}. Auto-generating leakage-safe splits...")

    # Locate edges file
    edge_candidates: list[Path] = []
    if master_edges_path is not None and str(master_edges_path).strip().lower() not in ('none', ''):
        edge_candidates.append(Path(master_edges_path))

    if master_nodes_path is not None:
        nodes_p = Path(master_nodes_path).resolve()
        edge_candidates.extend([
            nodes_p.parent / 'master_ddi_edges.csv',
            nodes_p.parent.parent / 'unified_graph' / 'master_ddi_edges.csv',
            nodes_p.parent.parent / 'twosides' / 'drug_drug_edges.csv',
            nodes_p.parent / 'drug_drug_edges.csv',
        ])

    edge_candidates.extend([
        Path('unified_graph') / 'master_ddi_edges.csv',
        Path('twosides') / 'drug_drug_edges.csv',
        Path('drug_drug_edges.csv'),
    ])

    resolved_edges = next((p for p in edge_candidates if p.is_file()), None)
    if resolved_edges is None:
        raise FileNotFoundError(
            f"Could not find DDI edges to generate benchmark splits. "
            f"Looked in: {[str(p) for p in edge_candidates[:4]]}. "
            f"Please specify master_edges_path or pre-generate splits into '{splits_path}'."
        )

    print(f"Loading positive interaction pairs from: {resolved_edges}")
    header_sample = pd.read_csv(resolved_edges, nrows=2)

    def choose_endpoint(candidates: tuple[str, ...]) -> str | None:
        return next((name for name in candidates if name in header_sample.columns), None)

    src_col = choose_endpoint((
        'drug_a_id', 'source', 'drug1_id', 'drug_a', 'drug1',
        'stitch_id1', 'STITCH 1', 'stitch1', 'drugbank_id1', 'Drug1',
    ))
    dst_col = choose_endpoint((
        'drug_b_id', 'target', 'drug2_id', 'drug_b', 'drug2',
        'stitch_id2', 'STITCH 2', 'stitch2', 'drugbank_id2', 'Drug2',
    ))
    if src_col is None or dst_col is None:
        raise ValueError(
            f'Cannot identify the two drug columns in {resolved_edges}. '
            f'Columns found: {header_sample.columns.tolist()}. Pass a pair-level '
            'TWOSIDES CSV with identifiable drug endpoint columns.'
        )

    df_raw = pd.read_csv(resolved_edges, usecols=[src_col, dst_col], low_memory=False)

    # Unique pairs (ignore redundant polypharmacy side-effect rows for split generation)
    pairs_df = df_raw[[src_col, dst_col]].drop_duplicates().rename(
        columns={src_col: 'drug_a_id', dst_col: 'drug_b_id'}
    )

    # Map TWOSIDES/source aliases onto the exact drug_id used as MolecularCache's
    # graph key. Merely checking that an alias exists in the node table is not
    # enough: the model would otherwise look up a DrugBank ID as if it were SMILES.
    mapping_audit: dict[str, int] = {}
    if master_nodes_path is not None and Path(master_nodes_path).is_file():
        nodes_df = pd.read_csv(master_nodes_path)
        pairs_df, mapping_audit = _map_edges_to_master_ids(
            pairs_df, nodes_df, 'drug_a_id', 'drug_b_id'
        )
        pairs_df = pairs_df.drop_duplicates().reset_index(drop=True)
        print(
            'Mapped DDI edge endpoints to master drug keys: '
            f"{mapping_audit['edge_rows_with_both_drugs_mapped']:,}/"
            f"{mapping_audit['raw_edge_rows']:,} rows; "
            f"{len(pairs_df):,} unique pairs."
        )

    print(f"Constructing leakage-safe splits across {len(pairs_df):,} unique positive drug pairs...")
    if len(pairs_df) < min_examples_per_class * 2:
        raise ValueError(
            f'Only {len(pairs_df):,} unique positive DDI pairs map to the master '
            f'drug catalog; at least {min_examples_per_class * 2:,} are required '
            'even for the smallest meaningful partition. Check that --edges points '
            'to the full TWOSIDES pair file and that its IDs can map to master nodes.'
        )
    splits, audit = create_split_aware_binary_splits(
        positive_pairs=pairs_df,
        known_reported_positive_pairs=pairs_df,
        source_col='drug_a_id',
        target_col='drug_b_id',
        holdout_fraction=holdout_fraction,
        seed=seed,
        negative_sampling_strategy='uniform',
        allow_zero_negatives=kwargs.get('allow_zero_negatives', False),
    )

    audit['source'] = {
        'edge_file': str(Path(resolved_edges).resolve()),
        'edge_file_sha256': _sha256_file(Path(resolved_edges)),
        'master_nodes_file': str(Path(master_nodes_path).resolve()) if master_nodes_path else None,
        'master_nodes_file_sha256': (
            _sha256_file(Path(master_nodes_path))
            if master_nodes_path and Path(master_nodes_path).is_file() else None
        ),
        'identifier_mapping': mapping_audit,
    }
    audit['minimum_examples_per_class'] = min_examples_per_class
    audit['split_counts'] = {
        f'{name}.csv': {
            'rows': int(len(frame)),
            'positive': int((frame['label'] == 1).sum()),
            'negative': int((frame['label'] == 0).sum()),
        }
        for name, frame in splits.items()
    }
    failures = [
        f'{name}: positive={counts["positive"]}, negative={counts["negative"]}'
        for name, counts in audit['split_counts'].items()
        if counts['positive'] < min_examples_per_class or counts['negative'] < min_examples_per_class
    ]
    if failures:
        raise ValueError(
            'Split construction produced partitions too small for evaluation; '
            'no split files were written and training must not proceed.\n  - ' +
            '\n  - '.join(failures) +
            '\nUse the full interaction dataset and verify drug identity mappings.'
        )

    for name, df in splits.items():
        out_file = splits_path / f'{name}.csv'
        df.to_csv(out_file, index=False)
        print(f"  Generated {name}.csv: {len(df):,} pairs")

    audit_path = splits_path / 'split_audit.json'
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    _check_minimum_split_support(splits_path, min_examples_per_class)
    print(f"All benchmark splits successfully written to: {splits_path}")
    return splits_path


def safe_forward_multimodal(
    model: AuditDDIModel,
    batch: dict[str, Any],
    da: Any,
    db: Any,
    device: torch.device,
) -> tuple[torch.Tensor, Any, Any]:
    """Safely invoke multimodal forward with signature inspection to guard against stale modules."""
    kwargs: dict[str, Any] = {
        'drug_a': da,
        'drug_b': db,
        'fp_a': batch['fp_a'].to(device),
        'fp_b': batch['fp_b'].to(device),
        'gene_a': batch['gene_a'].to(device),
        'gene_b': batch['gene_b'].to(device),
        'gene_mask_a': batch['gene_mask_a'].to(device),
        'gene_mask_b': batch['gene_mask_b'].to(device),
        'clinical_tox_a': batch['tox_a'].to(device),
        'clinical_tox_b': batch['tox_b'].to(device),
    }
    try:
        import inspect
        sig = inspect.signature(model.forward).parameters
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
        if ('target_a' in sig or has_var_kwargs) and 'target_a' in batch:
            kwargs['target_a'] = batch['target_a'].to(device)
            kwargs['target_b'] = batch['target_b'].to(device)
            kwargs['target_mask_a'] = batch['target_mask_a'].to(device)
            kwargs['target_mask_b'] = batch['target_mask_b'].to(device)
        if ('memory_features' in sig or has_var_kwargs) and 'memory_features' in batch and batch['memory_features'] is not None:
            kwargs['memory_features'] = batch['memory_features'].to(device)
        if ('geo_a' in sig or has_var_kwargs) and 'geo_a' in batch and batch['geo_a'] is not None:
            kwargs['geo_a'] = batch['geo_a'].to(device)
            kwargs['geo_b'] = batch['geo_b'].to(device)
            kwargs['geo_mask_a'] = batch['geo_mask_a'].to(device)
            kwargs['geo_mask_b'] = batch['geo_mask_b'].to(device)
        if ('pdb_a' in sig or has_var_kwargs) and 'pdb_a' in batch and batch['pdb_a'] is not None:
            kwargs['pdb_a'] = batch['pdb_a'].to(device)
            kwargs['pdb_b'] = batch['pdb_b'].to(device)
            kwargs['pdb_mask_a'] = batch['pdb_mask_a'].to(device)
            kwargs['pdb_mask_b'] = batch['pdb_mask_b'].to(device)
        if ('target_seq_a' in sig or has_var_kwargs) and 'target_seq_a' in batch and batch['target_seq_a'] is not None:
            kwargs['target_seq_a'] = batch['target_seq_a']
            kwargs['target_seq_b'] = batch['target_seq_b']
        if ('biophysical_a' in sig or has_var_kwargs) and 'biophysical_a' in batch and batch['biophysical_a'] is not None:
            kwargs['biophysical_a'] = batch['biophysical_a'].to(device)
            kwargs['biophysical_b'] = batch['biophysical_b'].to(device)
    except Exception:
        pass
    return model(**kwargs)


def evaluate_loader(
    model: AuditDDIModel,
    loader: Any,
    device: torch.device,
    is_multimodal: bool = False,
) -> dict[str, float]:
    """Compute comprehensive performance metrics on an evaluation split."""
    model.eval()
    all_targets: list[float] = []
    all_scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            da = batch['drug_a'].to(device)
            db = batch['drug_b'].to(device)
            lbls = batch['labels'].cpu().numpy()

            if is_multimodal:
                risk_logits, _, _ = safe_forward_multimodal(model, batch, da, db, device)
            else:
                risk_logits, _, _ = model(drug_a=da, drug_b=db)

            probs = torch.sigmoid(risk_logits).cpu().numpy().ravel()
            all_scores.extend(probs.tolist())
            all_targets.extend(lbls.ravel().tolist())

    targets = np.array(all_targets)
    scores = np.array(all_scores)

    if len(targets) == 0:
        return {'auroc': 0.5, 'auprc': 0.0, 'accuracy': 0.5, 'f1': 0.0, 'mcc': 0.0, 'brier': 0.25, 'optimal_threshold': 0.5, 'false_negatives': 0}

    if len(np.unique(targets)) < 2:
        return {'auroc': 0.5, 'auprc': float(np.mean(targets)) if len(targets) else 0.0, 'accuracy': 0.5, 'f1': 0.0, 'mcc': 0.0, 'brier': 0.25, 'optimal_threshold': 0.5, 'false_negatives': 0}

    auroc = float(roc_auc_score(targets, scores))
    auprc = float(average_precision_score(targets, scores))
    brier = float(brier_score_loss(targets, scores))

    try:
        fpr_arr, tpr_arr, thresh_arr = roc_curve(targets, scores)
        j_scores = tpr_arr - fpr_arr
        best_idx = int(np.argmax(j_scores)) if len(j_scores) else 0
        opt_thresh = float(thresh_arr[best_idx]) if len(thresh_arr) > best_idx else 0.35
        opt_thresh = max(min(opt_thresh, 0.50), 0.20)
    except Exception:
        opt_thresh = 0.35

    preds = (scores >= opt_thresh).astype(int)
    pos_mask = (targets == 1)
    neg_mask = (targets == 0)
    fn = int(np.sum((preds == 0) & pos_mask))
    fp = int(np.sum((preds == 1) & neg_mask))
    tp = int(np.sum((preds == 1) & pos_mask))
    tn = int(np.sum((preds == 0) & neg_mask))
    fnr = float(fn / max(pos_mask.sum(), 1))
    fpr = float(fp / max(neg_mask.sum(), 1))
    recall = float(tp / max(pos_mask.sum(), 1))
    sensitivity = recall
    specificity = float(tn / max(neg_mask.sum(), 1))

    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': float(accuracy_score(targets, preds)),
        'f1': float(f1_score(targets, preds, zero_division=0)),
        'mcc': matthews_corrcoef(targets, preds),
        'brier': brier,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'optimal_threshold': opt_thresh,
        'false_negatives': fn,
        'false_positives': fp,
        'true_positives': tp,
        'true_negatives': tn,
        'fnr': fnr,
        'fpr': fpr,
    }


def train_benchmark_model(
    architecture_version: str,
    cache: MolecularCache,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_splits: dict[str, pd.DataFrame],
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train and evaluate one model architecture on the benchmark splits."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_multimodal = (architecture_version == MODEL_ARCHITECTURE_MULTIMODAL)
    print(f"\n--- Training: {architecture_version} (Multimodal={is_multimodal}) on {device} ---")

    train_loader = build_cached_multimodal_dataloader(train_df, cache, batch_size=batch_size, shuffle=True)
    val_loader = build_cached_multimodal_dataloader(val_df, cache, batch_size=batch_size, shuffle=False)

    sample_batch = next(iter(train_loader))
    in_channels = sample_batch['drug_a'].x.size(1)
    edge_dim = sample_batch['drug_a'].edge_attr.size(1)

    model = AuditDDIModel(
        in_channels=in_channels,
        hidden_channels=64,
        edge_feature_dim=edge_dim,
        architecture_version=architecture_version,
        gene_feature_dim=cache.gene_dim,
        gene_hidden_channels=64,
        use_clinical_toxicity=is_multimodal,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    epoch_times = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            da = batch['drug_a'].to(device)
            db = batch['drug_b'].to(device)
            y = batch['labels'].to(device)

            if is_multimodal:
                risk_logits, _, _ = model(
                    drug_a=da,
                    drug_b=db,
                    fp_a=batch['fp_a'].to(device),
                    fp_b=batch['fp_b'].to(device),
                    gene_a=batch['gene_a'].to(device),
                    gene_b=batch['gene_b'].to(device),
                    gene_mask_a=batch['gene_mask_a'].to(device),
                    gene_mask_b=batch['gene_mask_b'].to(device),
                    clinical_tox_a=batch['tox_a'].to(device),
                    clinical_tox_b=batch['tox_b'].to(device),
                )
            else:
                risk_logits, _, _ = model(drug_a=da, drug_b=db)

            loss = criterion(risk_logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        ep_duration = time.perf_counter() - ep_start
        epoch_times.append(ep_duration)
        val_metrics = evaluate_loader(model, val_loader, device, is_multimodal=is_multimodal)
        print(f"  Epoch {epoch}/{epochs} ({ep_duration:.2f}s) - Loss: {total_loss/max(n_batches,1):.4f} - Val AUROC: {val_metrics['auroc']:.4f}")

    total_training_time = time.perf_counter() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == 'cuda' else 0.0

    # Evaluate across all test splits
    results = {
        'architecture': architecture_version,
        'avg_epoch_time_sec': float(np.mean(epoch_times)),
        'total_training_time_sec': total_training_time,
        'peak_memory_mb': peak_memory_mb,
    }

    for split_name, split_df in test_splits.items():
        loader = build_cached_multimodal_dataloader(split_df, cache, batch_size=batch_size, shuffle=False)
        metrics = evaluate_loader(model, loader, device, is_multimodal=is_multimodal)
        for k, v in metrics.items():
            results[f'{split_name}_{k}'] = v

    return results


def run_benchmark(
    master_nodes_path: str | Path,
    splits_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    master_edges_path: str | Path | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    device: torch.device | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Execute complete cold-start benchmark comparing baseline vs multimodal GNN."""
    if master_edges_path is None:
        master_edges_path = kwargs.get('edges_path') or kwargs.get('master_edges')
    if output_dir is None:
        out_dir = Path(master_nodes_path).resolve().parent.parent / 'benchmark_results'
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_path = ensure_benchmark_splits(
        splits_dir=splits_dir,
        master_nodes_path=master_nodes_path,
        master_edges_path=master_edges_path,
        **kwargs,
    )

    print("=" * 80)
    print("STARTING AUDITDDI COLD-START BENCHMARK")
    print(f"Master Nodes : {master_nodes_path}")
    print(f"Splits Dir   : {splits_path}")
    print(f"Output Dir   : {out_dir}")
    print("=" * 80)

    # 1. Populate Cache
    cache = MolecularCache(gene_dim=50)
    cache.populate_from_master_nodes(master_nodes_path)

    # 2. Load Splits
    train_df = pd.read_csv(splits_path / 'transductive_train.csv')
    val_df = pd.read_csv(splits_path / 'validation.csv')

    test_splits = {
        'transductive': pd.read_csv(splits_path / 'transductive_test.csv'),
        's1_cold': pd.read_csv(splits_path / 's1_test.csv'),
        's2_semi': pd.read_csv(splits_path / 's2_test.csv'),
    }

    print(f"Loaded splits: Train={len(train_df):,}, Val={len(val_df):,}, "
          f"Transductive={len(test_splits['transductive']):,}, S1_Cold={len(test_splits['s1_cold']):,}, S2_Semi={len(test_splits['s2_semi']):,}")

    # 3. Train Baseline
    baseline_res = train_benchmark_model(
        architecture_version=MODEL_ARCHITECTURE_EDGE_AWARE,
        cache=cache,
        train_df=train_df,
        val_df=val_df,
        test_splits=test_splits,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    # 4. Train Multi-Modal
    multimodal_res = train_benchmark_model(
        architecture_version=MODEL_ARCHITECTURE_MULTIMODAL,
        cache=cache,
        train_df=train_df,
        val_df=val_df,
        test_splits=test_splits,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    # 5. Export Summary
    comparison_df = pd.DataFrame([baseline_res, multimodal_res])
    csv_out = out_dir / 'benchmark_results.csv'
    json_out = out_dir / 'benchmark_summary.json'

    comparison_df.to_csv(csv_out, index=False)
    json_out.write_text(json.dumps([baseline_res, multimodal_res], indent=2), encoding='utf-8')

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS COMPARISON:")
    print("=" * 80)
    cols = ['architecture', 's1_cold_auroc', 's1_cold_auprc', 's2_semi_auroc', 'transductive_auroc', 'avg_epoch_time_sec', 'peak_memory_mb']
    print(comparison_df[[c for c in cols if c in comparison_df.columns]].to_string(index=False))
    print(f"\nSaved benchmark outputs to: {out_dir}")
    return comparison_df
