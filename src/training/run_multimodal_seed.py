"""Run one leakage-audited multimodal seed in Colab.

Each invocation writes to ``<output-dir>/seed_<seed>``. Keep ``split-seed``
fixed across accounts and change only ``seed`` to add matched model runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.path_resolver import resolve_data_base, resolve_results_base
from src.training.benchmark_cold_start import ensure_benchmark_splits
from src.training.train_multimodal_study import run_full_multimodal_study


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run one multimodal DDI training seed with fixed, auditable splits.'
    )
    parser.add_argument('--master-nodes', type=Path, help='Enriched master drug-node CSV from the Drive data/results folders.')
    parser.add_argument('--edges', type=Path, help='TWOSIDES pair CSV or unified master DDI edge CSV.')
    parser.add_argument('--splits-dir', type=Path, help='Shared split directory reused for every seed/account.')
    parser.add_argument('--output-dir', type=Path, help='Writable shared Drive folder for seed outputs.')
    parser.add_argument('--seed', type=int, default=11, help='Model initialization seed for this Colab run.')
    parser.add_argument('--split-seed', type=int, default=42, help='Keep fixed across all model seeds.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--holdout-fraction', type=float, default=0.30,
                        help='Total fraction of DDI drugs assigned to separate S1-dev/S1-test identity groups.')
    parser.add_argument('--minimum-per-class', type=int, default=50,
                        help='Require at least this many positive and negative pairs in every split.')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Resolve inputs/build and validate splits, then exit without training.')
    args = parser.parse_args()

    if args.seed <= 0 or args.split_seed <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error('seed, split-seed, epochs, and batch-size must be positive integers.')
    if args.minimum_per_class < 2 or not 0.05 <= args.holdout_fraction <= 0.60:
        parser.error('minimum-per-class must be at least 2 and holdout-fraction must be between 0.05 and 0.60.')
    if not args.prepare_only and not __import__('torch').cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this Colab training entry point; select a GPU runtime and rerun.')

    data_base = resolve_data_base()
    results_base = resolve_results_base()
    nodes = args.master_nodes or _first_existing([
        results_base / 'unified_graph' / 'master_drug_nodes_verified_targets.csv',
        data_base / 'unified_graph' / 'master_drug_nodes_verified_targets.csv',
        results_base / 'unified_graph' / 'master_drug_nodes.csv',
        data_base / 'unified_graph' / 'master_drug_nodes.csv',
    ])
    if nodes is None or not nodes.is_file():
        raise FileNotFoundError(
            'Could not find the enriched master drug-node CSV. Pass --master-nodes '
            'with its Google Drive path.'
        )

    # Prefer the raw TWOSIDES pair table. A pre-filtered unified edge file may
    # contain only a tiny subset and must not silently become the benchmark.
    twosides_dirs = [
        path for path in (data_base / 'twosides', data_base / 'TWOSIDES', data_base / 'TwoSides')
        if path.is_dir()
    ]
    recursive_twosides_csvs = sorted(
        (
            path for folder in twosides_dirs for path in folder.rglob('*.csv')
            if 'twosides' in path.name.lower() or path.name.lower() == 'drug_drug_edges.csv'
        ),
        key=lambda path: (
            0 if 'twosides' in path.name.lower() else 1,
            0 if path.name.lower() == 'twosides.csv' else 1,
            str(path).lower(),
        ),
    )
    edges = args.edges or _first_existing([
        data_base / 'twosides' / 'twosides.csv',
        data_base / 'TWOSIDES' / 'twosides.csv',
        *recursive_twosides_csvs,
        data_base / 'twosides' / 'drug_drug_edges.csv',
        data_base / 'TWOSIDES' / 'drug_drug_edges.csv',
        data_base / 'TwoSides' / 'drug_drug_edges.csv',
        data_base / 'unified_graph' / 'master_ddi_edges.csv',
        results_base / 'unified_graph' / 'master_ddi_edges.csv',
    ])
    if edges is None or not edges.is_file():
        raise FileNotFoundError(
            'Could not find TWOSIDES or unified DDI edges. Pass --edges with its Google Drive path.'
        )

    output_root = args.output_dir or (results_base / 'multimodal_seed_study_v2')
    output_root.mkdir(parents=True, exist_ok=True)
    split_dir = args.splits_dir or (output_root / f'fixed_splits_seed_{args.split_seed}')
    split_dir = Path(split_dir)
    seed_output = output_root / f'seed_{args.seed}'
    completion = seed_output / 'seed_execution_status.json'
    if completion.is_file():
        status = json.loads(completion.read_text(encoding='utf-8'))
        if status.get('status') == 'complete':
            print(f"Seed {args.seed} is already complete at {seed_output}; leaving its artifacts unchanged.")
            return
    if seed_output.exists():
        raise FileExistsError(
            f'{seed_output} already exists but has no completed status marker. '
            'Use a new seed/output folder or review the partial run before removing it.'
        )

    # Preserve the source CSV: enrichment writers in the multimodal workflow
    # operate on their input master-node file.
    snapshot = output_root / 'input_snapshot' / 'master_drug_nodes.csv'
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        shutil.copy2(nodes, snapshot)

    split_dir.mkdir(parents=True, exist_ok=True)
    ensure_benchmark_splits(
        splits_dir=split_dir,
        master_nodes_path=snapshot,
        master_edges_path=edges,
        seed=args.split_seed,
        holdout_fraction=args.holdout_fraction,
        min_examples_per_class=args.minimum_per_class,
    )

    if args.prepare_only:
        audit_path = split_dir / 'split_audit.json'
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        print(f"Input preflight succeeded. TWOSIDES edge file: {edges}")
        print(f"Validated splits: {split_dir}")
        print(json.dumps(audit.get('split_counts', {}), indent=2))
        return

    os.environ['AUDITDDI_MODEL_SEED'] = str(args.seed)
    os.environ['AUDITDDI_SPLIT_SEED'] = str(args.split_seed)
    print(f'Data root: {data_base}')
    print(f'Model seed: {args.seed}; fixed split seed: {args.split_seed}; epochs: {args.epochs}')
    print(f'Outputs: {seed_output}')

    result = run_full_multimodal_study(
        master_nodes_path=snapshot,
        master_edges_path=edges,
        splits_dir=split_dir,
        output_dir=seed_output,
        data_dir=data_base,
        seed=args.seed,
        split_seed=args.split_seed,
        min_examples_per_class=args.minimum_per_class,
        extended_epochs=args.epochs,
        batch_size=args.batch_size,
        run_ablation=False,
        run_error_analysis=True,
        calibrate=True,
        patience=0,
        use_ssl=False,
        use_target_encoder=True,
        use_protein_sequence_encoder=True,
        select_best_by='val',
    )
    (seed_output / 'seed_metrics.json').write_text(
        json.dumps(result, indent=2, sort_keys=True, default=lambda value: value.item() if hasattr(value, 'item') else str(value)),
        encoding='utf-8',
    )
    history_path = seed_output / 'auditddi_multimodal_v1_training_history.csv'
    if not history_path.is_file():
        raise FileNotFoundError(f'Expected training history was not written: {history_path}')
    with history_path.open('r', encoding='utf-8') as stream:
        completed_epochs = max(sum(1 for _ in stream) - 1, 0)
    if completed_epochs != args.epochs:
        raise RuntimeError(
            f'Requested {args.epochs} epochs but training recorded {completed_epochs}; '
            'leaving this run without a completion marker.'
        )
    completion.parent.mkdir(parents=True, exist_ok=True)
    completion.write_text(json.dumps({
        'status': 'complete',
        'model_seed': args.seed,
        'split_seed': args.split_seed,
        'epochs': completed_epochs,
        'requested_epochs': args.epochs,
        'completed_epochs': completed_epochs,
        'seed_output': str(seed_output.resolve()),
    }, indent=2, sort_keys=True), encoding='utf-8')


if __name__ == '__main__':
    main()
