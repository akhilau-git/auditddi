"""Does the model's pair-level risk agree with independent curated evidence, beyond drug popularity?

    python -m src.evidence.external_agreement \
        --table /content/drive/MyDrive/auditddi-data/event_table_v1 \
        --splits /content/drive/MyDrive/auditddi-data/splits_v1 \
        --baselines /content/drive/MyDrive/auditddi-data/baselines_v1 \
        --ddinter /content/drive/MyDrive/auditddi-data/evidence_v1/ddinter_pairs.csv \
        --label-pairs /content/drive/MyDrive/auditddi-data/evidence_v1/label_pairs.csv \
        --names cold_drug_seed0 scaffold_seed0 --models logreg mlp512

For every TEST partition saved by run_baselines.py --save-scores, each test pair gets
  * external flag  : DDInter pair (any severity), DDInter moderate/major, or label-mention pair
  * model score    : mean_prob (mean predicted probability over events), top20_mean, mean_rank
                     (average per-event percentile rank; insensitive to the weighted-loss scale)
  * popularity baseline : log1p(training degree of drug A) + log1p(training degree of drug B),
                     where degree counts a drug's partners among TRAINING pairs only.  A drug unseen
                     in training has degree 0, so in S1 the baseline is constant by construction.
and we report AUROC (+ pair-bootstrap CI) for each score against each flag.

Neither DDInter nor the labels were used to train anything, so this is genuine external agreement.
CAVEATS stated with the output:
  * DDInter and TWOSIDES both over-represent widely used drugs; the popularity baseline is there to expose that.
  * Pairs share drugs, so the pair bootstrap understates uncertainty; treat intervals as optimistic.
  * 'agreement with a curated list' is not 'clinical truth': DDInter is not complete and 57% of its
    records have unknown severity.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.baselines.run_baselines import append_results
from src.data_prep.build_event_table import load_event_table


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    p, q = int(y.sum()), int((~y).sum())
    if p == 0 or q == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - p * (p + 1) / 2.0) / (p * q))


def pair_scores(S: np.ndarray) -> Dict[str, np.ndarray]:
    k = min(20, S.shape[1])
    ranks = np.apply_along_axis(rankdata, 0, S) / len(S)           # per-event percentile of each pair
    return {"mean_prob": S.mean(axis=1), "top20_mean": np.partition(S, -k, axis=1)[:, -k:].mean(axis=1), "mean_rank": ranks.mean(axis=1)}


def degree_baseline(n_drugs: int, a: np.ndarray, b: np.ndarray, train_rows: np.ndarray, rows: np.ndarray) -> np.ndarray:
    deg = np.bincount(np.concatenate([a[train_rows], b[train_rows]]), minlength=n_drugs).astype(float)
    return np.log1p(deg[a[rows]]) + np.log1p(deg[b[rows]])


def flags_for_pairs(a: np.ndarray, b: np.ndarray, ddinter: pd.DataFrame, label_pairs: Optional[pd.DataFrame]) -> Dict[str, np.ndarray]:
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * 100000 + hi
    out: Dict[str, np.ndarray] = {}
    dd_key = ddinter.drug_a.astype(np.int64) * 100000 + ddinter.drug_b
    out["ddinter_any"] = np.isin(key, dd_key.to_numpy())
    sev = ddinter[ddinter.severity.isin(["moderate", "major"])]
    out["ddinter_moderate_major"] = np.isin(key, (sev.drug_a.astype(np.int64) * 100000 + sev.drug_b).to_numpy())
    if label_pairs is not None and len(label_pairs):
        la, lb = label_pairs.drug_label.to_numpy(), label_pairs.drug_mentioned.to_numpy()
        lk = np.minimum(la, lb).astype(np.int64) * 100000 + np.maximum(la, lb)
        out["label_mention"] = np.isin(key, lk)
    return out


def boot_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(y)
    v = np.array([auroc(y[i], s[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    v = v[~np.isnan(v)]
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (float("nan"), float("nan"))


def evaluate_partition(S: np.ndarray, rows: np.ndarray, a_all: np.ndarray, b_all: np.ndarray, train_rows: np.ndarray, n_drugs: int,
                       ddinter: pd.DataFrame, label_pairs: Optional[pd.DataFrame], n_boot: int = 200, min_pos: int = 20) -> pd.DataFrame:
    a, b = a_all[rows], b_all[rows]
    scores = pair_scores(S)
    scores["popularity_baseline"] = degree_baseline(n_drugs, a_all, b_all, train_rows, rows)
    out = []
    for fname, y in flags_for_pairs(a, b, ddinter, label_pairs).items():
        n_pos = int(y.sum())
        for sname, s in scores.items():
            const = float(np.ptp(s)) == 0.0
            row = {"flag": fname, "score": sname, "n_pairs": int(len(y)), "n_flag_pos": n_pos, "flag_prevalence": float(y.mean()),
                   "score_is_constant": const}
            if n_pos < min_pos or n_pos > len(y) - min_pos or const:
                row.update({"auroc": 0.5 if const and n_pos >= min_pos else float("nan"), "ci_low": float("nan"), "ci_high": float("nan")})
            else:
                lo, hi = boot_ci(y, s, n_boot)
                row.update({"auroc": auroc(y, s), "ci_low": lo, "ci_high": hi})
            out.append(row)
    return pd.DataFrame(out)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--baselines", required=True, type=Path)
    ap.add_argument("--ddinter", required=True, type=Path)
    ap.add_argument("--label-pairs", type=Path, default=None)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["logreg"])
    ap.add_argument("--feature-tag", default="")
    ap.add_argument("--pair-mode", default="symmetric")
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args(list(argv) if argv is not None else None)

    table = load_event_table(a.table)
    A, B = table.pairs["drug_a"].to_numpy(), table.pairs["drug_b"].to_numpy()
    dd = pd.read_csv(a.ddinter)
    lp = pd.read_csv(a.label_pairs) if a.label_pairs and a.label_pairs.exists() else None
    tag = "" if a.feature_tag in ("", "ecfp") else "@" + a.feature_tag
    rows_out = []
    for name in a.names:
        split = np.load(a.splits / f"{name}.npz")
        for model in a.models:
            for part in [k for k in split.files if k.startswith("test_")]:
                f = a.baselines / f"scores_{name}_{model}{tag}_{a.pair_mode}_{part}.npz"
                if not f.exists():
                    print(f"missing {f.name}")
                    continue
                z = np.load(f)
                df = evaluate_partition(z["S"].astype(np.float32), z["rows"], A, B, split["train"], len(table.drugs), dd, lp, a.n_boot)
                df.insert(0, "partition", part); df.insert(0, "model", model); df.insert(0, "split", name)
                df["features"] = a.feature_tag or "ecfp"
                rows_out.append(df)
                k = df[(df.flag == "ddinter_any")].set_index("score")
                print(f"{name:18s} {model:8s} {part:8s} DDInter AUROC: model(mean_rank) {k.loc['mean_rank', 'auroc']:.3f} "
                      f"[{k.loc['mean_rank', 'ci_low']:.3f},{k.loc['mean_rank', 'ci_high']:.3f}] | popularity baseline {k.loc['popularity_baseline', 'auroc']:.3f} "
                      f"| prevalence {k.loc['mean_rank', 'flag_prevalence']:.2f}")
    if rows_out:
        append_results(a.baselines / "external_agreement.csv", pd.concat(rows_out, ignore_index=True).to_dict("records"))


if __name__ == "__main__":
    main()
