"""Regime-aware calibration and conformal sets (extends Guide 1, sections 4.11-4.12).

Why: a calibrator fitted on a validation set dominated by one-new-drug (S2) pairs is
too confident for both-new-drug (S1) pairs, where the model has almost no signal.
A pair's regime is knowable at prediction time, because it only depends on whether
each drug occurred in the TRAINING pairs:

    regime 0 : both drugs seen in training      (transductive)
    regime 1 : exactly one drug unseen          (S2)
    regime 2 : both drugs unseen                (S1)

Calibrators and conformal thresholds are fitted per regime on validation pairs of that
regime only (Mondrian conformal), so within each regime the usual exchangeability
argument applies.  A regime with fewer than ``min_pairs`` validation pairs falls back
to the pooled calibrator and is flagged; a regime absent from validation also falls
back.  Fewer calibration positives automatically give larger (less informative) sets.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from src.trust.calibration import EventCalibrator, conformal_sets, conformal_thresholds, split_calibration


def seen_mask(n_drugs: int, a: np.ndarray, b: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    m = np.zeros(n_drugs, dtype=bool)
    m[a[train_rows]] = True
    m[b[train_rows]] = True
    return m


def regime_of_pairs(seen: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (~seen[a]).astype(np.int8) + (~seen[b]).astype(np.int8)


class RegimeAwareTrust:
    def __init__(self, lam: float = 3.0, min_pairs: int = 200, alphas: Sequence[float] = (0.1,), seed: int = 0, prior_mode: str = "pooled"):
        self.lam, self.min_pairs, self.alphas, self.seed, self.prior_mode = lam, min_pairs, tuple(alphas), seed, prior_mode

    def fit(self, S: np.ndarray, Y: np.ndarray, regime: np.ndarray) -> "RegimeAwareTrust":
        regime = np.asarray(regime)
        # pooled fallback: all validation pairs, disjoint fit / conformal halves
        fit, cal = split_calibration(len(S), self.seed)
        self.pooled_ = EventCalibrator(self.lam, prior_mode=self.prior_mode).fit(S[fit], Y[fit])
        Pc = self.pooled_.transform(S[cal])
        self.pooled_q_ = {al: conformal_thresholds(Pc, Y[cal], al)[:2] for al in self.alphas}
        self.cal_, self.q_, self.info_ = {}, {}, {}
        for r in (0, 1, 2):
            idx = np.where(regime == r)[0]
            self.info_[r] = {"n_val": int(len(idx)), "fallback_to_pooled": bool(len(idx) < self.min_pairs)}
            if len(idx) < self.min_pairs:
                continue
            f, c = split_calibration(len(idx), self.seed + 1 + r)
            cal = EventCalibrator(self.lam, prior_mode=self.prior_mode).fit(S[idx[f]], Y[idx[f]])
            Pr = cal.transform(S[idx[c]])
            self.cal_[r] = cal
            self.q_[r] = {al: conformal_thresholds(Pr, Y[idx[c]], al)[:2] for al in self.alphas}
        return self

    def transform(self, S: np.ndarray, regime: np.ndarray) -> np.ndarray:
        regime = np.asarray(regime)
        P = np.empty(S.shape, dtype=np.float32)
        for r in np.unique(regime):
            m = regime == r
            P[m] = (self.cal_.get(int(r), self.pooled_)).transform(S[m])
        return P

    def sets(self, P: np.ndarray, regime: np.ndarray, alpha: float):
        regime = np.asarray(regime)
        in1 = np.zeros(P.shape, dtype=bool)
        in0 = np.zeros(P.shape, dtype=bool)
        for r in np.unique(regime):
            m = regime == r
            q1, q0 = (self.q_[int(r)][alpha] if int(r) in self.q_ else self.pooled_q_[alpha])
            a1, a0 = conformal_sets(P[m], q1, q0)
            in1[m], in0[m] = a1, a0
        return in1, in0
