"""Aggregate independently run multimodal seeds after verifying shared inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ('transductive', 's1_cold', 's2_semi')
METRICS = (
    'auroc', 'auprc', 'mcc', 'accuracy', 'f1', 'recall', 'specificity', 'brier',
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize completed multimodal seed runs.')
    parser.add_argument('--study-dir', type=Path, required=True, help='Shared output folder containing seed_<n> subfolders.')
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args()

    manifests: list[dict] = []
    metric_rows: list[dict] = []
    for seed in args.seeds:
        seed_dir = args.study_dir / f'seed_{seed}'
        status_path = seed_dir / 'seed_execution_status.json'
        input_path = seed_dir / 'run_input_manifest.json'
        metrics_path = seed_dir / 'seed_metrics.json'
        if not all(path.is_file() for path in (status_path, input_path, metrics_path)):
            raise FileNotFoundError(f'Seed {seed} is incomplete; expected status, input manifest, and metrics under {seed_dir}.')
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

    reference = manifests[0]
    for manifest in manifests[1:]:
        for key in ('split_seed', 'master_nodes_sha256', 'split_sha256'):
            if manifest.get(key) != reference.get(key):
                raise ValueError(f'Seed runs do not share identical {key}; refusing to aggregate.')

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
