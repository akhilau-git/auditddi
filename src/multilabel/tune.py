"""Validation-only hyper-parameter search for the multi-label model, then one final fit.

    python -m src.multilabel.tune \
        --table .../event_table_v1 --splits .../splits_v1 --name cold_drug_seed0 \
        --feature-store .../features_v3/drug_features.npz --features ecfp esm cyp \
        --out .../multilabel_tune_acct1 --n-configs 12

Each configuration is trained with --val-only: the test pairs are never scored, so nothing here can be
selected on test performance.  The configuration with the best VALIDATION macro AUPRC is then refitted
once with test scoring, scores and a checkpoint saved under <out>/best.  tuning.csv lists every
configuration with its validation score, so the search effort is reportable.  Resumable.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

GRID = {"bio_dropout": [0.0, 0.15, 0.3], "dropout": [0.1, 0.3, 0.5], "weight_decay": [1e-5, 1e-3], "emb": [32, 64], "lr": [1e-3, 3e-4]}


def sample_configs(n: int, seed: int = 0, grid: Optional[dict] = None) -> List[dict]:
    grid = grid or GRID
    keys = sorted(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(combos))[: min(n, len(combos))]
    return [dict(zip(keys, combos[i])) for i in sorted(pick)]


def cfg_args(cfg: dict) -> List[str]:
    out: List[str] = []
    for k, v in cfg.items():
        out += [f"--{k.replace('_', '-')}", str(v)]
    return out


def main(argv: Optional[Sequence[str]] = None) -> dict:
    from src.multilabel.train import main as train_main                  # imported here: needs torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--feature-store", required=True, type=Path)
    ap.add_argument("--features", nargs="+", default=["ecfp"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-configs", type=int, default=12)
    ap.add_argument("--tune-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-final", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)

    a.out.mkdir(parents=True, exist_ok=True)
    cfgs = sample_configs(a.n_configs, a.tune_seed)
    log_path = a.out / "tuning.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=["config_id", "val_auprc", "val_auroc", "epochs", "config"])
    done = set(log.config_id.astype(int))
    base = ["--table", str(a.table), "--splits", str(a.splits), "--name", a.name, "--feature-store", str(a.feature_store), "--features", *a.features,
            "--epochs", str(a.epochs), "--patience", str(a.patience), "--min-pos", str(a.min_pos)] + (["--cpu"] if a.cpu else [])
    for i, cfg in enumerate(cfgs):
        if i in done:
            continue
        res = train_main(base + cfg_args(cfg) + ["--val-only", "--no-checkpoint", "--out", str(a.out / f"cfg{i:02d}")])
        log = pd.concat([log, pd.DataFrame([{"config_id": i, "val_auprc": res["best_val_auprc"], "val_auroc": res["val_macro_auroc"], "epochs": res["epochs"],
                                             "config": json.dumps(cfg, sort_keys=True)}])], ignore_index=True)
        log.to_csv(log_path, index=False)
        print(f"config {i:02d} {cfg} -> val macro AUPRC {res['best_val_auprc']:.4f}")
    best = log.sort_values("val_auprc", ascending=False).iloc[0]
    best_cfg = json.loads(best.config)
    print(f"BEST by validation: config {int(best.config_id)} {best_cfg}  (val AUPRC {best.val_auprc:.4f}; searched {len(log)} configs)")
    (a.out / "best_config.json").write_text(json.dumps({"config": best_cfg, "val_auprc": float(best.val_auprc), "n_configs_searched": int(len(log))}, indent=2))
    if not a.no_final:
        train_main(base + cfg_args(best_cfg) + ["--force", "--save-scores", "--out", str(a.out / "best")])
    return {"best_config": best_cfg, "n_configs": int(len(log))}


if __name__ == "__main__":
    main()
