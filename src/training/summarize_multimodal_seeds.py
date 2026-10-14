"""Aggregate independently run multimodal seeds after verifying shared inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ('transductive', 's1_cold', 's2_semi')
METRICS = (
    'auroc', 'auprc', 'mcc', 'accuracy', 'balanced_accuracy', 'f1',
    'recall', 'specificity', 'brier',
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize completed multimodal seed runs.')
    parser.add_argument('--study-dir', type=Path, required=True, help='Shared output folder containing seed_<n> subfolders.')
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--skip-missing', action='store_true',
                        help='Skip seeds that are incomplete or missing, summarizing only completed seeds.')
    parser.add_argument('--allow-hash-mismatch', action='store_true',
                        help='Warn instead of raising error if input manifest hashes differ across seeds.')
    args = parser.parse_args()

    manifests: list[dict] = []
    metric_rows: list[dict] = []
    for seed in args.seeds:
        seed_dir = args.study_dir / f'seed_{seed}'
        status_path = seed_dir / 'seed_execution_status.json'
        input_path = seed_dir / 'run_input_manifest.json'
        metrics_path = seed_dir / 'seed_metrics.json'
        best_weights_path = seed_dir / 'auditddi_multimodal_v1_best.pt'

        if not seed_dir.is_dir():
            msg = (
                f"\n❌ Seed {seed} directory was not found at:\n   {seed_dir}\n"
                f"   [Multi-Account Notice] If you ran Seed {seed} on a different Google Colab email/account, "
                f"the output folder was saved to that account's Google Drive. "
                f"Please share or copy the 'seed_{seed}' folder from that Google Drive into:\n   {args.study_dir}\n"
            )
            if args.skip_missing:
                print(msg)
                print(f"Skipping Seed {seed} (--skip-missing is active)...")
                continue
            raise FileNotFoundError(msg)

        missing = [p.name for p in (status_path, input_path, metrics_path) if not p.is_file()]
        if missing:
            msg = f"\n❌ Seed {seed} is incomplete under {seed_dir}. Missing required files: {missing}\n"
            if best_weights_path.is_file():
                msg += (
                    f"   💡 Found saved weights ({best_weights_path.name}). You can finalize Seed {seed} in ~30s by running:\n"
                    f"      !python src/training/run_multimodal_seed.py --seed {seed} --split-seed 42 --epochs 200 --evaluate-only --output-dir {args.study_dir}\n"
                )
            if args.skip_missing:
                print(msg)
                print(f"Skipping Seed {seed} (--skip-missing is active)...")
                continue
            raise FileNotFoundError(msg)

        status = json.loads(status_path.read_text(encoding='utf-8'))
        manifest = json.loads(input_path.read_text(encoding='utf-8'))
        result = json.loads(metrics_path.read_text(encoding='utf-8'))
        if status.get('status') != 'complete' or int(manifest.get('model_seed', -1)) != seed:
            raise ValueError(f'Seed {seed} status or model-seed metadata is inconsistent.')
        manifests.append(manifest)
        metrics = result.get('extended_metrics', {})
        row = {'seed': seed}
        for split in SPLITS:
            for metric in METRICS:
                key = f'{split}_{metric}'
                if metrics.get(key) is not None:
                    row[key] = float(metrics[key])
        metric_rows.append(row)

    if not manifests:
        raise ValueError("No completed seeds found to summarize. Check that seed folders exist and have been evaluated.")

    reference = manifests[0]
    for manifest in manifests[1:]:
        for key in ('split_seed', 'master_nodes_sha256', 'split_sha256', 'split_audit_sha256'):
            if manifest.get(key) != reference.get(key):
                msg = f'Seed runs do not share identical {key} (Seed {manifest.get("model_seed")}: {manifest.get(key)} vs Ref Seed {reference.get("model_seed")}: {reference.get(key)}).'
                if args.allow_hash_mismatch:
                    print(f"⚠️ Warning: {msg} Continuing because --allow-hash-mismatch was specified.")
                else:
                    raise ValueError(f'{msg} Refusing to aggregate without --allow-hash-mismatch.')

    table = pd.DataFrame(metric_rows).sort_values('seed')
    args.study_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.study_dir / 'multimodal_seed_metrics.csv', index=False)
    summary: dict[str, dict] = {
        'seeds': table['seed'].astype(int).tolist(),
        'split_seed': reference['split_seed'],
        'master_nodes_sha256': reference['master_nodes_sha256'],
        'split_sha256': reference['split_sha256'],
        'aggregation': 'seed-level mean and bootstrap 95% percentile interval; descriptive with five seeds',
        'metrics': {},
    }
    rng = np.random.default_rng(0)
    for column in table.columns:
        if column == 'seed':
            continue
        values = table[column].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        sampled_means = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
        summary['metrics'][column] = {
            'n': int(len(values)),
            'mean': float(values.mean()),
            'standard_deviation': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'ci_95_lower': float(np.percentile(sampled_means, 2.5)),
            'ci_95_upper': float(np.percentile(sampled_means, 97.5)),
        }
    (args.study_dir / 'multimodal_seed_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8'
    )
    print(f"Aggregated {len(table)} matched seeds after confirming identical master-node and split hashes.")
    print(args.study_dir / 'multimodal_seed_metrics.csv')
    print(args.study_dir / 'multimodal_seed_summary.json')


if __name__ == '__main__':
    main()
