"""Evaluation metrics for the multi-label DDI task (Guide 1, sections 5.4-5.8).

Conventions
-----------
Y : (n_pairs, n_events) array of {0,1}
S : (n_pairs, n_events) array of scores / probabilities in [0, 1]
An event is *evaluable* in a partition only if it has at least ``min_pos``
positives AND ``min_pos`` negatives there; every macro average is taken over
evaluable events only and reports how many there were.  Unreported pairs are
treated as negatives for the event head, which is the benchmark convention, not
proof of safety.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

EPS = 1e-12


# --------------------------------------------------------------------------- #
# support
# --------------------------------------------------------------------------- #
def evaluable_mask(Y: np.ndarray, min_pos: int = 5) -> np.ndarray:
    pos = Y.sum(axis=0)
    return (pos >= min_pos) & ((Y.shape[0] - pos) >= min_pos)


# --------------------------------------------------------------------------- #
# ranking metrics
# --------------------------------------------------------------------------- #
def auroc_per_event(Y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Tie-aware rank AUROC per column; NaN when a column has one class only."""
    n, m = Y.shape
    out = np.full(m, np.nan)
    pos = Y.sum(axis=0)
    for j in range(m):
        p = pos[j]
        q = n - p
        if p == 0 or q == 0:
            continue
        r = rankdata(S[:, j])
        out[j] = (r[Y[:, j] == 1].sum() - p * (p + 1) / 2.0) / (p * q)
    return out


def auprc_per_event(Y: np.ndarray, S: np.ndarray) -> np.ndarray:
    out = np.full(Y.shape[1], np.nan)
    pos = Y.sum(axis=0)
    for j in range(Y.shape[1]):
        if pos[j] > 0:
            out[j] = average_precision_score(Y[:, j], S[:, j])
    return out


def micro_auroc(Y: np.ndarray, S: np.ndarray) -> float:
    y = Y.ravel().astype(np.int8)
    p = y.sum()
    q = y.size - p
    if p == 0 or q == 0:
        return float("nan")
    r = rankdata(S.ravel())
    return float((r[y == 1].sum() - p * (p + 1) / 2.0) / (p * q))


def micro_auprc(Y: np.ndarray, S: np.ndarray) -> float:
    return float(average_precision_score(Y.ravel(), S.ravel()))


def precision_recall_at_k(Y: np.ndarray, S: np.ndarray, ks: Sequence[int] = (1, 5, 10, 20)) -> Dict[str, float]:
    """Per-pair ranked event lists, averaged over pairs that have >= 1 true event."""
    out: Dict[str, float] = {}
    n, m = Y.shape
    order = np.argsort(-S, axis=1)
    has = Y.sum(axis=1) > 0
    for k in ks:
        k = min(k, m)
        top = order[:, :k]
        hit = np.take_along_axis(Y, top, axis=1).sum(axis=1)
        out[f"precision@{k}"] = float((hit[has] / k).mean())
        out[f"recall@{k}"] = float((hit[has] / Y[has].sum(axis=1)).mean())
    return out


# --------------------------------------------------------------------------- #
# thresholded (set-valued) metrics
# --------------------------------------------------------------------------- #
def set_metrics(Y: np.ndarray, S: np.ndarray, threshold: float) -> Dict[str, float]:
    P = (S >= threshold).astype(np.int8)
    Yi = Y.astype(np.int8)
    tp = (P & Yi).sum(axis=1).astype(np.float64)
    fp = (P & (1 - Yi)).sum(axis=1)
    fn = ((1 - P) & Yi).sum(axis=1)
    denom = 2 * tp + fp + fn
    ex_f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 1.0)
    union = tp + fp + fn
    jacc = np.where(union > 0, tp / np.maximum(union, 1), 1.0)

    TP = (P & Yi).sum(axis=0).astype(np.float64)
    FP = (P & (1 - Yi)).sum(axis=0)
    FN = ((1 - P) & Yi).sum(axis=0)
    micro = 2 * TP.sum() / max(2 * TP.sum() + FP.sum() + FN.sum(), 1)
    d = 2 * TP + FP + FN
    f1_e = np.where(d > 0, 2 * TP / np.maximum(d, 1), np.nan)
    return {
        "threshold": float(threshold),
        "example_f1": float(ex_f1.mean()),
        "micro_f1": float(micro),
        "macro_f1": float(np.nanmean(f1_e)) if np.isfinite(f1_e).any() else float("nan"),
        "jaccard": float(jacc.mean()),
        "hamming_loss": float((P != Yi).mean()),
    }


def best_global_threshold(Y: np.ndarray, S: np.ndarray, grid: Optional[Sequence[float]] = None) -> float:
    """Single threshold maximising micro-F1; choose it on VALIDATION data only."""
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    best, bt = -1.0, 0.5
    for t in grid:
        f = set_metrics(Y, S, t)["micro_f1"]
        if f > best:
            best, bt = f, float(t)
    return bt


# --------------------------------------------------------------------------- #
# calibration (Guide 1 section 5.5)
# --------------------------------------------------------------------------- #
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece_mce(y: np.ndarray, p: np.ndarray, n_bins: int = 15, equal_mass: bool = True):
    """Expected / maximum calibration error.  Equal-mass bins by default because
    most event probabilities are tiny and equal-width bins would be empty."""
    if equal_mass:
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        edges = np.unique(edges)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ece, mce = 0.0, 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            gap = abs(float(y[m].mean()) - float(p[m].mean()))
            ece += m.mean() * gap
            mce = max(mce, gap)
    return float(ece), float(mce)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray, iters: int = 100):
    """Logistic recalibration  y ~ sigmoid(a + b * logit(p)).  Ideal: a=0, b=1.
    Damped Newton with backtracking line search (convex problem, always converges)."""
    x = _logit(p)

    def nll_ab(a_, b_):
        q_ = np.clip(_sigmoid(a_ + b_ * x), EPS, 1 - EPS)
        return -np.sum(y * np.log(q_) + (1 - y) * np.log(1 - q_))

    a, b = 0.0, 1.0
    cur = nll_ab(a, b)
    for _ in range(iters):
        q = _sigmoid(a + b * x)
        w = np.maximum(q * (1 - q), 1e-9)
        g = np.array([np.sum(q - y), np.sum((q - y) * x)])
        H = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x * x).sum()]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-6:
            new = nll_ab(a - t * step[0], b - t * step[1])
            if new <= cur + 1e-12:
                break
            t *= 0.5
        else:
            break
        a, b, improved = a - t * step[0], b - t * step[1], cur - new
        cur = new
        if improved < 1e-10:
            break
    return float(b), float(a)       # slope, intercept


def calibration_summary(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    ece, mce = ece_mce(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    return {"brier": brier(y, p), "nll": nll(y, p), "ece": ece, "mce": mce, "cal_slope": slope, "cal_intercept": intercept}


def calibration_macro(Y: np.ndarray, S: np.ndarray, min_pos: int = 5) -> Dict[str, float]:
    ev = np.where(evaluable_mask(Y, min_pos))[0]
    rows = [calibration_summary(Y[:, j].astype(np.float64), S[:, j].astype(np.float64)) for j in ev]
    out = {f"macro_{k}": float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}
    if rows:
        # the mean slope is dominated by outlier events when probabilities barely vary (low-signal data);
        # the median is the robust summary and the one to quote
        out["median_cal_slope"] = float(np.median([r["cal_slope"] for r in rows]))
        out["median_cal_intercept"] = float(np.median([r["cal_intercept"] for r in rows]))
    out["n_events_evaluated"] = int(len(ev))
    micro = calibration_summary(Y.ravel().astype(np.float64), S.ravel().astype(np.float64))
    out.update({f"micro_{k}": v for k, v in micro.items()})
    return out


# --------------------------------------------------------------------------- #
# one-call evaluation
# --------------------------------------------------------------------------- #
def evaluate(Y: np.ndarray, S: np.ndarray, threshold: Optional[float] = None, min_pos: int = 5, ks: Sequence[int] = (1, 5, 10, 20), with_calibration: bool = True) -> Dict[str, float]:
    ev = evaluable_mask(Y, min_pos)
    au = auroc_per_event(Y, S)
    ap = auprc_per_event(Y, S)
    res: Dict[str, float] = {
        "n_pairs": int(Y.shape[0]),
        "n_events": int(Y.shape[1]),
        "n_events_evaluable": int(ev.sum()),
        "macro_auroc": float(np.nanmean(au[ev])) if ev.any() else float("nan"),
        "macro_auprc": float(np.nanmean(ap[ev])) if ev.any() else float("nan"),
        "micro_auroc": micro_auroc(Y, S),
        "micro_auprc": micro_auprc(Y, S),
    }
    res.update(precision_recall_at_k(Y, S, ks))
    if threshold is not None:
        res.update(set_metrics(Y, S, threshold))
    if with_calibration:
        res.update(calibration_macro(Y, S, min_pos))
    return res


# --------------------------------------------------------------------------- #
# resampling statistics (Guide 1 section 5.8): unit of resampling = PAIR
# --------------------------------------------------------------------------- #
def _macro_auroc(Y, S, min_pos):
    ev = evaluable_mask(Y, min_pos)
    a = auroc_per_event(Y[:, ev], S[:, ev])
    return float(np.nanmean(a))


def bootstrap_ci(metric: Callable[[np.ndarray, np.ndarray], float], Y: np.ndarray, S: np.ndarray, n_boot: int = 200, seed: int = 0, level: float = 0.95) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = metric(Y[i], S[i])
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"estimate": float(metric(Y, S)), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot, "seed": seed, "level": level, "unit": "pair"}


def paired_bootstrap(metric: Callable[[np.ndarray, np.ndarray], float], Y: np.ndarray, S_a: np.ndarray, S_b: np.ndarray, n_boot: int = 200, seed: int = 0, level: float = 0.95) -> Dict[str, float]:
    """Difference metric(A) - metric(B) on the SAME resampled pairs."""
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        d[b] = metric(Y[i], S_a[i]) - metric(Y[i], S_b[i])
    lo, hi = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    p = max(p, 1.0 / (n_boot + 1))                 # a bootstrap cannot certify p below 1/(B+1)
    return {"diff": float(metric(Y, S_a) - metric(Y, S_b)), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(p, 1.0)), "n_boot": n_boot, "seed": seed, "level": level, "unit": "pair"}


def benjamini_hochberg(p: Sequence[float], alpha: float = 0.05):
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(adj, 1.0)
    return q, q <= alpha


def macro_auroc_metric(min_pos: int = 5) -> Callable[[np.ndarray, np.ndarray], float]:
    return lambda Y, S: _macro_auroc(Y, S, min_pos)
