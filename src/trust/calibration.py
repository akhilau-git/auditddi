"""Per-event calibration and conformal sets (Guide 1, sections 4.11-4.12, 5.5).

Platt scaling
    p_cal = sigmoid(a_e + b_e * logit(s))      one (a_e, b_e) per event e
Each event's parameters are fitted on calibration-FIT pairs by penalised maximum
likelihood, with a ridge pull toward the pooled (all-events) calibrator:
    minimise  NLL_e(a, b) + lam * ((a - a0)^2 + (b - b0)^2)
so an event with 3 positives stays near the pooled map instead of overfitting.
b_e is constrained to be > 0 (a positive slope keeps the model's ranking unchanged,
so calibration can never change per-event AUROC).

Conformal sets  (class-conditional / Mondrian split conformal, per event)
    nonconformity of label y for a pair = 1 - p_cal(y)
    q_{e,y}  = ceil((n_{e,y}+1)(1-alpha))-th smallest nonconformity among CALIBRATION-CONF
               pairs whose true label for event e is y     (inf if there are too few)
    label y is IN the set of a new pair iff 1 - p_cal(y) <= q_{e,y}
Guarantee: per event and per class, coverage >= 1-alpha IF the new pairs are
exchangeable with the calibration-conf pairs.  Under cold-start shift this is NOT
guaranteed; ``conformal_report`` measures the shortfall instead of assuming it.

The calibration pairs must be disjoint from the pairs used to fit Platt scaling
(``split_calibration``): reusing them would make coverage look better than it is.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

EPS = 1e-12
LOGIT_CLIP = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


# --------------------------------------------------------------------------- #
# Platt scaling with shrinkage
# --------------------------------------------------------------------------- #
def fit_platt(x: np.ndarray, y: np.ndarray, prior: Tuple[float, float] = (0.0, 1.0), lam: float = 0.0,
              iters: int = 100, min_slope: float = 1e-4) -> Tuple[float, float]:
    """Penalised logistic fit of y on x = logit(score).  Convex; damped Newton with
    backtracking.  Returns (a, b) with b >= min_slope."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a0, b0 = prior

    def obj(a, b):
        q = np.clip(sigmoid(a + b * x), EPS, 1 - EPS)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)) + lam * ((a - a0) ** 2 + (b - b0) ** 2)

    a, b = a0, b0
    cur = obj(a, b)
    for _ in range(iters):
        q = sigmoid(a + b * x)
        w = np.maximum(q * (1 - q), 1e-9)
        g = np.array([np.sum(q - y) + 2 * lam * (a - a0), np.sum((q - y) * x) + 2 * lam * (b - b0)])
        H = np.array([[w.sum() + 2 * lam, (w * x).sum()], [(w * x).sum(), (w * x * x).sum() + 2 * lam]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-8:
            na, nb = a - t * step[0], max(b - t * step[1], min_slope)
            new = obj(na, nb)
            if new <= cur + 1e-12:
                break
            t *= 0.5
        else:
            break
        gain = cur - new
        a, b, cur = na, nb, new
        if gain < 1e-10:
            break
    return float(a), float(max(b, min_slope))


class EventCalibrator:
    """Per-event Platt maps with shrinkage toward the pooled map."""

    def __init__(self, lam: float = 3.0, min_pos_own_fit: int = 1):
        self.lam, self.min_pos = lam, min_pos_own_fit

    def fit(self, S: np.ndarray, Y: np.ndarray) -> "EventCalibrator":
        X = logit(S)
        Yf = np.asarray(Y, dtype=np.float64)
        # pooled prior: one (a0, b0) over every (pair, event) cell, subsampled for speed
        rng = np.random.default_rng(0)
        flat = rng.choice(X.size, size=min(X.size, 2_000_000), replace=False)
        self.a0_, self.b0_ = fit_platt(X.ravel()[flat], Yf.ravel()[flat])
        E = S.shape[1]
        self.a_ = np.empty(E)
        self.b_ = np.empty(E)
        self.n_pos_fit_ = Yf.sum(axis=0).astype(int)
        for e in range(E):
            if self.n_pos_fit_[e] < self.min_pos or self.n_pos_fit_[e] == len(Yf):
                self.a_[e], self.b_[e] = self.a0_, self.b0_
            else:
                self.a_[e], self.b_[e] = fit_platt(X[:, e], Yf[:, e], prior=(self.a0_, self.b0_), lam=self.lam)
        return self

    def transform(self, S: np.ndarray) -> np.ndarray:
        return sigmoid(self.a_[None, :] + self.b_[None, :] * logit(S)).astype(np.float32)


def split_calibration(n: int, seed: int = 0, frac_fit: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Disjoint index sets inside the validation partition: (platt_fit, conformal_cal)."""
    perm = np.random.default_rng(seed).permutation(n)
    k = int(round(n * frac_fit))
    return np.sort(perm[:k]), np.sort(perm[k:])


# --------------------------------------------------------------------------- #
# conformal
# --------------------------------------------------------------------------- #
def conformal_thresholds(P_cal: np.ndarray, Y_cal: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (q1, q0, n1, n0) per event; q = +inf where there are too few
    calibration examples of that class to certify 1-alpha coverage."""
    E = P_cal.shape[1]
    q1 = np.full(E, np.inf)
    q0 = np.full(E, np.inf)
    n1 = Y_cal.sum(axis=0).astype(int)
    n0 = (len(Y_cal) - n1).astype(int)
    for e in range(E):
        for y, q, n in ((1, q1, n1[e]), (0, q0, n0[e])):
            if n == 0:
                continue
            k = int(np.ceil((n + 1) * (1 - alpha)))
            if k > n:
                continue                                            # too few examples: stay at +inf (always include)
            col = P_cal[:, e][Y_cal[:, e] == y]
            nonconf = np.sort(1.0 - (col if y == 1 else 1.0 - col))
            q[e] = nonconf[k - 1]
    return q1, q0, n1, n0


def conformal_sets(P: np.ndarray, q1: np.ndarray, q0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """in1[i,e] True if label 1 is in the set; in0[i,e] True if label 0 is."""
    in1 = (1.0 - P) <= q1[None, :]
    in0 = P <= q0[None, :]
    return in1, in0


def conformal_report(Y: np.ndarray, in1: np.ndarray, in0: np.ndarray, min_pos: int = 5, alpha: float = 0.1) -> Dict[str, float]:
    Y = np.asarray(Y).astype(bool)
    pos, neg = Y.sum(axis=0), (~Y).sum(axis=0)
    ev = (pos >= min_pos) & (neg >= min_pos)
    cov1 = np.where(pos > 0, (in1 & Y).sum(axis=0) / np.maximum(pos, 1), np.nan)
    cov0 = np.where(neg > 0, (in0 & ~Y).sum(axis=0) / np.maximum(neg, 1), np.nan)
    size = in1.astype(int) + in0.astype(int)
    return {
        "n_events_evaluated": int(ev.sum()),
        "coverage_positive_macro": float(np.nanmean(cov1[ev])) if ev.any() else float("nan"),
        "coverage_negative_macro": float(np.nanmean(cov0[ev])) if ev.any() else float("nan"),
        "coverage_positive_min_event": float(np.nanmin(cov1[ev])) if ev.any() else float("nan"),
        "frac_events_pos_cov_within_5pts_of_nominal": float(np.nanmean(cov1[ev] >= 1 - alpha - 0.05)) if ev.any() else float("nan"),
        "mean_set_size": float(size.mean()),
        "frac_singleton_1": float(((size == 1) & in1).mean()),
        "frac_singleton_0": float(((size == 1) & in0).mean()),
        "frac_both": float((size == 2).mean()),
        "frac_empty": float((size == 0).mean()),
    }
