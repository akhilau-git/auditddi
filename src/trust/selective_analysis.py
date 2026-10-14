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
import sys
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _find_score_file(baselines_dir: Path, name: str, model: str, mode: str, part: str, feature_tag: str = "") -> Optional[Path]:
    """Find the score file for a given run, trying explicit feature tags, auto-tags, or untagged."""
    candidates = []
    if feature_tag:
        clean_tag = feature_tag if feature_tag.startswith("@") else f"@{feature_tag}"
        candidates.append(baselines_dir / f"scores_{name}_{model}{clean_tag}_{mode}_{part}.npz")
    candidates.append(baselines_dir / f"scores_{name}_{model}_{mode}_{part}.npz")
    # Also auto-search any file matching pattern scores_{name}_{model}*_{mode}_{part}.npz
    globbed = sorted(baselines_dir.glob(f"scores_{name}_{model}*_{mode}_{part}.npz"))
    candidates.extend(globbed)
    for c in candidates:
        if c.exists():
            return c
    return None


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--baselines", required=True, type=Path)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["logreg", "mlp512"])
    ap.add_argument("--feature-tag", default="", help="feature tag used in run_baselines.py, e.g. ecfp+target+gene")
    ap.add_argument("--fingerprints", type=Path, default=None, help="ECFP6 cache path; default <baselines>/ecfp6_r3_1024.npz")
    ap.add_argument("--pair-mode", default="symmetric")
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--gain-coverage", type=float, default=0.5)
    ap.add_argument("--min-pos", type=int, default=5)
    a = ap.parse_args(list(argv) if argv is not None else None)

    table = load_event_table(a.table)
    fp_path = a.fingerprints or (a.baselines / "ecfp6_r3_1024.npz")
    F = load_fingerprints(table.drugs, fp_path)
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
                f = _find_score_file(a.baselines, name, model, a.pair_mode, part, a.feature_tag)
                if f is None or not f.exists():
                    print(f"missing score file for {name} {model} {part} in {a.baselines} (run run_baselines.py with --save-scores)")
                    continue
                tag_label = a.feature_tag or (f.stem.split("@")[1].split("_")[0] if "@" in f.stem else "ecfp")
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
                                    "features": tag_label, "score_file": f.name,
                                    "null_mean": null[row["coverage"]]["null_mean"], "null_sd": null[row["coverage"]]["null_sd"]})
                        rc_rows.append(row)
                for rname in ("ad_min", "ad_mean"):
                    g = abstention_gain(Y, S, rel[rname], a.gain_coverage, a.n_boot, a.min_pos)
                    g.update({"split": name, "model": model, "partition": part, "reliability": rname,
                              "features": tag_label, "score_file": f.name})
                    gain_rows.append(g)
                    for cov in (0.9, 0.75, 0.5):
                        tau = threshold_for_coverage(val_rel[rname], cov)
                        d = deployed_summary(Y, S, rel[rname], tau, a.min_pos)
                        d.update({"split": name, "model": model, "partition": part, "reliability": rname,
                                  "features": tag_label, "score_file": f.name, "target_coverage_on_val": cov})
                        dep_rows.append(d)
                last = [r for r in rc_rows if r["split"] == name and r["model"] == model and r["partition"] == part and r["reliability"] == "ad_min"]
                print(f"{name:20s} {model:8s} {part:8s} [{tag_label}] ad_min macro AUROC by coverage: " +
                      " ".join(f"{r['coverage']:.2f}:{r['macro_auroc']:.3f}" for r in last) +
                      f" | random null @0.5: {null[0.5]['null_mean']:.3f}")

    for fn, rows in (("risk_coverage.csv", rc_rows), ("abstention_gain.csv", gain_rows), ("deployed_abstention.csv", dep_rows)):
        if rows:
            append_results(a.baselines / fn, rows)


if __name__ == "__main__":
    main()
