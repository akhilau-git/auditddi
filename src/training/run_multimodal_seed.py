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
    parser.add_argument('--seed', type=int, required=True, help='Model initialization seed for this Colab run.')
    parser.add_argument('--split-seed', type=int, default=42, help='Keep fixed across all model seeds.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    if args.seed <= 0 or args.split_seed <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error('seed, split-seed, epochs, and batch-size must be positive integers.')
    if not __import__('torch').cuda.is_available():
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

    edges = args.edges or _first_existing([
        results_base / 'unified_graph' / 'master_ddi_edges.csv',
        data_base / 'unified_graph' / 'master_ddi_edges.csv',
        data_base / 'twosides' / 'drug_drug_edges.csv',
        data_base / 'TWOSIDES' / 'drug_drug_edges.csv',
    ])
    if edges is None or not edges.is_file():
        raise FileNotFoundError(
            'Could not find TWOSIDES or unified DDI edges. Pass --edges with its Google Drive path.'
        )

    output_root = args.output_dir or (results_base / 'multimodal_seed_study')
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
    )

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
        extended_epochs=args.epochs,
        batch_size=args.batch_size,
        run_ablation=False,
        run_error_analysis=True,
        calibrate=True,
        use_ssl=False,
        use_target_encoder=True,
        use_protein_sequence_encoder=True,
        select_best_by='val',
    )
    (seed_output / 'seed_metrics.json').write_text(
        json.dumps(result, indent=2, sort_keys=True, default=lambda value: value.item() if hasattr(value, 'item') else str(value)),
        encoding='utf-8',
    )
    completion.parent.mkdir(parents=True, exist_ok=True)
    completion.write_text(json.dumps({
        'status': 'complete',
        'model_seed': args.seed,
        'split_seed': args.split_seed,
        'epochs': args.epochs,
        'seed_output': str(seed_output.resolve()),
    }, indent=2, sort_keys=True), encoding='utf-8')


if __name__ == '__main__':
    main()
