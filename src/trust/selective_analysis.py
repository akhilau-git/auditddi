"""Run the abstention analysis on saved baseline scores.

    python -m src.trust.selective_analysis \
        --table /content/drive/MyDrive/auditddi-data/event_table_v1 \
        --splits /content/drive/MyDrive/auditddi-data/splits_v1 \
        --baselines /content/drive/MyDrive/auditddi-data/baselines_v1 \
        --names cold_drug_seed0 scaffold_seed0 --models logreg mlp512

Needs the score files written by run_baselines.py --save-scores.  Writes
risk_coverage.csv, abstention_gain.csv and deployed_abstention.csv next to the scores.
The abstention threshold tau is fixed on VALIDATION pairs (AD scores need no model
scores; model-confidence thresholds need val scores, saved by newer runs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.baselines.run_baselines import append_results, load_fingerprints
from src.data_prep.build_event_table import labels_for, load_event_table
from src.trust.selective import (
    abstention_gain,
    deployed_summary,
    drug_domain_score,
    pair_reliability,
    random_abstention_curve,
    risk_coverage,
    threshold_for_coverage,
    top_k_mean_score,
)

COVERAGES = (1.0, 0.9, 0.75, 0.5, 0.25)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--baselines", required=True, type=Path)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["logreg", "mlp512"])
    ap.add_argument("--pair-mode", default="symmetric")
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--gain-coverage", type=float, default=0.5)
    ap.add_argument("--min-pos", type=int, default=5)
    a = ap.parse_args(list(argv) if argv is not None else None)

    table = load_event_table(a.table)
    F = load_fingerprints(table.drugs, a.baselines / "ecfp6_r3_1024.npz")
    A, B = table.pairs["drug_a"].to_numpy(), table.pairs["drug_b"].to_numpy()
    rc_rows, gain_rows, dep_rows = [], [], []

    for name in a.names:
        split = {k: v for k, v in np.load(a.splits / f"{name}.npz").items()}
        vocab = pd.read_csv(a.splits / f"{name}_vocab.csv", keep_default_na=False)
        train_drugs = np.unique(np.concatenate([A[split["train"]], B[split["train"]]]))
        dscore = drug_domain_score(F, train_drugs)
        val_rows = split["val_all"]
        val_rel = pair_reliability(dscore, A[val_rows], B[val_rows])

        for model in a.models:
            for part in [k for k in split if k.startswith("test_")]:
                f = a.baselines / f"scores_{name}_{model}_{a.pair_mode}_{part}.npz"
                if not f.exists():
                    print(f"missing {f.name} (run run_baselines.py with --save-scores)")
                    continue
                z = np.load(f)
                S, rows = z["S"].astype(np.float32), z["rows"]
                Y = labels_for(table.Y, rows, vocab).astype(np.int8)
                rel = pair_reliability(dscore, A[rows], B[rows])
                rel["top20_mean"] = top_k_mean_score(S, 20)
                rel["random"] = np.random.default_rng(0).random(len(rows))
                null = {r["coverage"]: r for r in random_abstention_curve(Y, S, COVERAGES, n_rep=8, min_pos=a.min_pos)}
                for rname, r in rel.items():
                    for row in risk_coverage(Y, S, r, COVERAGES, a.min_pos):
                        row.update({"split": name, "model": model, "partition": part, "reliability": rname,
                                    "null_mean": null[row["coverage"]]["null_mean"], "null_sd": null[row["coverage"]]["null_sd"]})
                        rc_rows.append(row)
                for rname in ("ad_min", "ad_mean"):
                    g = abstention_gain(Y, S, rel[rname], a.gain_coverage, a.n_boot, a.min_pos)
                    g.update({"split": name, "model": model, "partition": part, "reliability": rname})
                    gain_rows.append(g)
                    for cov in (0.9, 0.75, 0.5):
                        tau = threshold_for_coverage(val_rel[rname], cov)
                        d = deployed_summary(Y, S, rel[rname], tau, a.min_pos)
                        d.update({"split": name, "model": model, "partition": part, "reliability": rname, "target_coverage_on_val": cov})
                        dep_rows.append(d)
                last = [r for r in rc_rows if r["split"] == name and r["model"] == model and r["partition"] == part and r["reliability"] == "ad_min"]
                print(f"{name:20s} {model:8s} {part:8s} ad_min macro AUROC by coverage: " +
                      " ".join(f"{r['coverage']:.2f}:{r['macro_auroc']:.3f}" for r in last) +
                      f" | random null @0.5: {null[0.5]['null_mean']:.3f}")

    for fn, rows in (("risk_coverage.csv", rc_rows), ("abstention_gain.csv", gain_rows), ("deployed_abstention.csv", dep_rows)):
        if rows:
            append_results(a.baselines / fn, rows)


if __name__ == "__main__":
    main()
