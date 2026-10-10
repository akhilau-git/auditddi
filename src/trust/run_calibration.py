"""Calibrate saved baseline scores and measure conformal coverage.

    python -m src.trust.run_calibration \
        --table /content/drive/MyDrive/auditddi-data/event_table_v1 \
        --splits /content/drive/MyDrive/auditddi-data/splits_v1 \
        --baselines /content/drive/MyDrive/auditddi-data/baselines_v1 \
        --names cold_drug_seed0 scaffold_seed0 --models logreg mlp512 [--feature-tag ecfp+target+gene]

Needs score files from run_baselines.py --save-scores (validation scores are saved by
current runs).  For each split/model:
  1. validation pairs are split in two disjoint halves: Platt-FIT and conformal-CAL
  2. per-event Platt maps (shrunk toward the pooled map) are fitted on Platt-FIT only
  3. conformal thresholds come from conformal-CAL only (calibrated probabilities)
  4. the decision threshold for F1 is chosen on conformal-CAL, never on test
  5. every test partition is scored raw, after Platt, and with conformal sets
Scores are stored as float16, so very small probabilities are rounded; the calibrators
operate on the rounded values and logit(.) clips at 1e-6.
Writes calibration_results.csv next to the scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.baselines.run_baselines import append_results
from src.data_prep.build_event_table import labels_for, load_event_table
from src.evaluation.multilabel_metrics import best_global_threshold, calibration_macro, evaluate
from src.trust.calibration import EventCalibrator, conformal_report, conformal_sets, conformal_thresholds, split_calibration
from src.trust.regime import RegimeAwareTrust, regime_of_pairs, seen_mask


def _load(path: Path):
    z = np.load(path)
    return z["S"].astype(np.float32), z["rows"]


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--baselines", required=True, type=Path)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["logreg"])
    ap.add_argument("--feature-tag", default="", help="e.g. ecfp+target+gene (empty = plain ecfp runs)")
    ap.add_argument("--pair-mode", default="symmetric")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.05])
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior-mode", choices=["pooled", "shared_slope"], default="pooled",
                    help="shared_slope is recommended when validation pairs are scarce (e.g. S1)")
    ap.add_argument("--regime-aware", action="store_true",
                    help="also fit calibrators/conformal thresholds separately for regimes 0/1/2 (number of drugs unseen in training)")
    a = ap.parse_args(list(argv) if argv is not None else None)

    table = load_event_table(a.table)
    PA, PB = table.pairs["drug_a"].to_numpy(), table.pairs["drug_b"].to_numpy()
    tag = "" if a.feature_tag in ("", "ecfp") else "@" + a.feature_tag
    rows = []
    for name in a.names:
        vocab = pd.read_csv(a.splits / f"{name}_vocab.csv", keep_default_na=False)
        split = np.load(a.splits / f"{name}.npz")
        tests = [k for k in split.files if k.startswith("test_")]
        for model in a.models:
            base = f"scores_{name}_{model}{tag}_{a.pair_mode}"
            vf = a.baselines / f"{base}_val.npz"
            if not vf.exists():
                print(f"missing {vf.name}: rerun run_baselines.py with --save-scores")
                continue
            Sv, rv = _load(vf)
            Yv = labels_for(table.Y, rv, vocab).astype(np.int8)
            fit, cal = split_calibration(len(rv), a.seed)
            calib = EventCalibrator(lam=a.lam, prior_mode=a.prior_mode).fit(Sv[fit], Yv[fit])
            Pv = calib.transform(Sv)
            thr_raw = best_global_threshold(Yv[cal], Sv[cal])
            thr_cal = best_global_threshold(Yv[cal], Pv[cal])
            qs = {al: conformal_thresholds(Pv[cal], Yv[cal], al) for al in a.alphas}
            trust = None
            if a.regime_aware:
                seen = seen_mask(len(table.drugs), PA, PB, split["train"])
                trust = RegimeAwareTrust(a.lam, alphas=a.alphas, seed=a.seed, prior_mode=a.prior_mode).fit(Sv, Yv, regime_of_pairs(seen, PA[rv], PB[rv]))
                thr_reg = best_global_threshold(Yv, trust.transform(Sv, regime_of_pairs(seen, PA[rv], PB[rv])))
            ident = {"split": name, "model": model, "features": a.feature_tag or "ecfp", "n_fit": int(len(fit)), "n_cal": int(len(cal)),
                     "pooled_a": calib.a0_, "pooled_b": calib.b0_, "prior_mode": a.prior_mode}
            for part in tests:
                f = a.baselines / f"{base}_{part}.npz"
                if not f.exists():
                    continue
                S, rt = _load(f)
                Y = labels_for(table.Y, rt, vocab).astype(np.int8)
                P = calib.transform(S)
                for stage, X, thr in (("raw", S, thr_raw), ("platt", P, thr_cal)):
                    m = evaluate(Y, X, threshold=thr, min_pos=a.min_pos)
                    rows.append({**ident, "partition": part, "stage": stage, **m})
                for al in a.alphas:
                    q1, q0, _, _ = qs[al]
                    in1, in0 = conformal_sets(P, q1, q0)
                    rows.append({**ident, "partition": part, "stage": f"conformal_alpha{al}", "alpha": al,
                                 **conformal_report(Y, in1, in0, a.min_pos, al)})
                n_new = 2 + len(a.alphas)
                if trust is not None:
                    reg = regime_of_pairs(seen, PA[rt], PB[rt])
                    Pr = trust.transform(S, reg)
                    rows.append({**ident, "partition": part, "stage": "platt_regime", "regime_info": str(trust.info_),
                                 **evaluate(Y, Pr, threshold=thr_reg, min_pos=a.min_pos)})
                    for al in a.alphas:
                        i1, i0 = trust.sets(Pr, reg, al)
                        rows.append({**ident, "partition": part, "stage": f"conformal_regime_alpha{al}", "alpha": al,
                                     **conformal_report(Y, i1, i0, a.min_pos, al)})
                    n_new += 1 + len(a.alphas)
                r = {x["stage"]: x for x in rows[-n_new:]}
                if trust is not None:
                    rr, cc = r["platt_regime"], r[f"conformal_regime_alpha{a.alphas[0]}"]
                    print(f"{'':18s} {'':8s} {'  +regime':8s} ECE {rr['macro_ece']:.3f} | median slope {rr['median_cal_slope']:.2f} | "
                          f"conformal(regime) pos-cov {cc['coverage_positive_macro']:.2f} set-size {cc['mean_set_size']:.2f}")
                print(f"{name:18s} {model:8s} {part:8s} macro ECE {r['raw']['macro_ece']:.3f}->{r['platt']['macro_ece']:.3f} | "
                      f"median slope {r['raw']['median_cal_slope']:.2f}->{r['platt']['median_cal_slope']:.2f} | "
                      f"conformal(0.1) pos-cov {r[f'conformal_alpha{a.alphas[0]}']['coverage_positive_macro']:.2f} "
                      f"set-size {r[f'conformal_alpha{a.alphas[0]}']['mean_set_size']:.2f}")
    if rows:
        append_results(a.baselines / "calibration_results.csv", rows)


if __name__ == "__main__":
    main()
