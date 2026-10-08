"""Across-seed comparison of runs from results.csv (Guide 1, sections 5.8-5.9).

    python -m src.evaluation.compare_runs \
        --results /content/drive/MyDrive/auditddi-data/baselines_v1/results.csv \
        --model logreg --reference ecfp --metric macro_auroc

Seeds are the unit of replication: for one split family (cold_drug, scaffold, ...)
and one partition, the same split (same seed) is evaluated with two feature sets,
so differences are paired by seed.  Reports mean +- sd per feature set, the paired
mean difference with a t-interval, the paired t-test, the exact Wilcoxon test, how
many seeds improved, and a Benjamini-Hochberg q-value across all comparisons.

HONEST LIMIT: with n = 5 seeds the smallest possible two-sided exact Wilcoxon p is
0.0625, so it can never reach 0.05.  The t-test is therefore the primary test, with
the sign count shown beside it; both are weak evidence at n = 5.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.multilabel_metrics import benjamini_hochberg


def load_runs(path: Path, model: str, stratified: Optional[bool] = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "features" not in df:
        df["features"] = "ecfp"
    df["features"] = df["features"].fillna("ecfp")
    if "stratified" not in df:
        df["stratified"] = False
    df["stratified"] = df["stratified"].fillna(False).astype(bool)
    df = df[df["model"] == model]
    if stratified is not None:
        df = df[df["stratified"] == stratified]
    return df.drop_duplicates(["split", "features", "partition"], keep="last")


def paired_seed_comparison(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> dict:
    """b - a over seeds (paired).  a, b: arrays of equal length."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = b - a
    n = len(d)
    out = {"n_seeds": int(n), "mean_diff": float(d.mean()) if n else float("nan"), "sd_diff": float(d.std(ddof=1)) if n > 1 else float("nan"),
           "n_improved": int((d > 0).sum()), "ci_low": float("nan"), "ci_high": float("nan"), "p_ttest": float("nan"), "p_wilcoxon": float("nan")}
    if n >= 2 and out["sd_diff"] > 0:
        se = out["sd_diff"] / np.sqrt(n)
        h = stats.t.ppf(1 - (1 - level) / 2, n - 1) * se
        out["ci_low"], out["ci_high"] = out["mean_diff"] - h, out["mean_diff"] + h
        out["p_ttest"] = float(stats.ttest_rel(b, a).pvalue)
    elif n >= 2:
        out["ci_low"] = out["ci_high"] = out["mean_diff"]
        out["p_ttest"] = 1.0 if out["mean_diff"] == 0 else 0.0
    if n >= 2 and np.any(d != 0):
        out["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue)
    elif n >= 2:
        out["p_wilcoxon"] = 1.0
    return out


def compare(df: pd.DataFrame, reference: str, metric: str) -> pd.DataFrame:
    df = df.copy()
    df["kind_seed"] = df["kind"].astype(str) + "|" + df["seed"].astype(str)
    rows: List[dict] = []
    for (kind, part), g in df.groupby(["kind", "partition"]):
        wide = g.pivot_table(index="seed", columns="features", values=metric, aggfunc="last")
        if reference not in wide:
            continue
        for feats in wide.columns:
            r = {"kind": kind, "partition": part, "features": feats, "n_seeds_with_value": int(wide[feats].notna().sum()),
                 f"mean_{metric}": float(wide[feats].mean()), f"sd_{metric}": float(wide[feats].std(ddof=1)) if wide[feats].notna().sum() > 1 else float("nan")}
            if feats != reference:
                r.update(paired_seed_comparison(wide[reference].to_numpy(), wide[feats].to_numpy()))
            rows.append(r)
    out = pd.DataFrame(rows)
    if "p_ttest" in out:
        m = out["p_ttest"].notna()
        if m.any():
            q, _ = benjamini_hochberg(out.loc[m, "p_ttest"].to_numpy())
            out.loc[m, "q_ttest_BH"] = q
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--model", default="logreg")
    ap.add_argument("--reference", default="ecfp")
    ap.add_argument("--metric", default="macro_auroc")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(list(argv) if argv is not None else None)
    df = load_runs(a.results, a.model)
    res = compare(df, a.reference, a.metric)
    pd.set_option("display.width", 220)
    show = [c for c in res.columns if c in ("kind", "partition", "features", "n_seeds_with_value", f"mean_{a.metric}", f"sd_{a.metric}", "mean_diff", "ci_low", "ci_high", "n_improved", "p_ttest", "p_wilcoxon", "q_ttest_BH")]
    print(res[show].round(4).to_string(index=False))
    if a.out:
        res.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
