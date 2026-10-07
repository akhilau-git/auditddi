"""Selective prediction / abstention analysis (Guide 1, sections 4.11-4.12, 5.7).

Question answered: if the system abstains on the pairs it considers unreliable, does
performance on the pairs it KEEPS improve beyond what random abstention gives?

Reliability scores (higher = more reliable), one per pair:
    ad_min      min over the two drugs of (max Tanimoto similarity to any TRAINING drug)
                -> the less familiar drug limits trust.  1.0 for a drug seen in training.
    ad_mean     mean of the two drugs' max-Tanimoto values
    top20_mean  mean of the model's 20 highest event probabilities for the pair
    random      seeded random ordering (the null every other score must beat)

Metric on retained pairs: macro AUROC over events that still have >= min_pos
positives AND negatives among the retained pairs; the number of such events is
reported because it shrinks with coverage.  The abstention threshold is chosen on
VALIDATION pairs and then applied unchanged to test pairs.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from src.evaluation.multilabel_metrics import auroc_per_event, evaluable_mask, micro_auprc


# --------------------------------------------------------------------------- #
# applicability domain
# --------------------------------------------------------------------------- #
def max_tanimoto_to_set(bits: np.ndarray, query_idx: Sequence[int], ref_idx: Sequence[int], exclude_self: bool = True) -> np.ndarray:
    """For each query drug, the maximum Tanimoto similarity to the reference drugs.
    ``bits`` is [n_drugs, n_bits] in {0,1}.  If a query drug is itself in the
    reference set its self-similarity is excluded only when ``exclude_self`` is True
    (use False for 'is this drug known?': a training drug then gets 1.0)."""
    B = bits.astype(np.float32)
    q = B[np.asarray(query_idx)]
    r = B[np.asarray(ref_idx)]
    inter = q @ r.T
    union = q.sum(1, keepdims=True) + r.sum(1)[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    if exclude_self:
        ref = np.asarray(ref_idx)
        for i, d in enumerate(np.asarray(query_idx)):
            sim[i, ref == d] = -1.0
    return sim.max(axis=1)


def drug_domain_score(bits: np.ndarray, train_drugs: np.ndarray) -> np.ndarray:
    """Per-drug AD score for EVERY drug: 1.0 if the drug occurs in training pairs,
    otherwise its max Tanimoto to the training drugs."""
    n = bits.shape[0]
    score = np.zeros(n, dtype=np.float32)
    train_drugs = np.unique(train_drugs)
    is_train = np.zeros(n, dtype=bool)
    is_train[train_drugs] = True
    score[is_train] = 1.0
    rest = np.where(~is_train)[0]
    if len(rest):
        score[rest] = max_tanimoto_to_set(bits, rest, train_drugs, exclude_self=False)
    return score


def pair_reliability(drug_score: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    sa, sb = drug_score[a], drug_score[b]
    return {"ad_min": np.minimum(sa, sb), "ad_mean": (sa + sb) / 2.0}


def top_k_mean_score(S: np.ndarray, k: int = 20) -> np.ndarray:
    k = min(k, S.shape[1])
    return np.partition(S, -k, axis=1)[:, -k:].mean(axis=1)


# --------------------------------------------------------------------------- #
# risk-coverage
# --------------------------------------------------------------------------- #
def _retained(reliability: np.ndarray, coverage: float, rng: np.random.Generator) -> np.ndarray:
    """Indices of the top-`coverage` fraction.  Ties are broken randomly so that a
    constant score (e.g. ad_min in a transductive split) behaves like random
    abstention instead of silently favouring low row indices."""
    n = len(reliability)
    k = max(1, int(round(coverage * n)))
    jitter = rng.random(n) * 1e-9
    return np.argsort(-(reliability + jitter))[:k]


def macro_auroc_subset(Y: np.ndarray, S: np.ndarray, idx: np.ndarray, min_pos: int = 5):
    Ys, Ss = Y[idx], S[idx]
    ev = evaluable_mask(Ys, min_pos)
    if not ev.any():
        return float("nan"), 0
    return float(np.nanmean(auroc_per_event(Ys[:, ev], Ss[:, ev]))), int(ev.sum())


def risk_coverage(Y: np.ndarray, S: np.ndarray, reliability: np.ndarray, coverages: Sequence[float] = (1.0, 0.9, 0.75, 0.5, 0.25), min_pos: int = 5, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    rows = []
    for c in coverages:
        idx = _retained(reliability, c, rng)
        au, n_ev = macro_auroc_subset(Y, S, idx, min_pos)
        rows.append({"coverage": float(c), "n_pairs": int(len(idx)), "macro_auroc": au, "n_events_evaluable": n_ev,
                     "min_reliability": float(reliability[idx].min())})
    return rows


def random_abstention_curve(Y: np.ndarray, S: np.ndarray, coverages: Sequence[float] = (1.0, 0.9, 0.75, 0.5, 0.25), n_rep: int = 10, min_pos: int = 5, seed: int = 0) -> list:
    """Null: mean and sd of macro AUROC when the kept pairs are chosen at random."""
    rng = np.random.default_rng(seed)
    out = []
    for c in coverages:
        vals = []
        for _ in range(n_rep):
            idx = _retained(rng.random(len(Y)), c, rng)
            vals.append(macro_auroc_subset(Y, S, idx, min_pos)[0])
        out.append({"coverage": float(c), "null_mean": float(np.nanmean(vals)), "null_sd": float(np.nanstd(vals))})
    return out


def abstention_gain(Y: np.ndarray, S: np.ndarray, reliability: np.ndarray, coverage: float, n_boot: int = 100, min_pos: int = 5, seed: int = 0) -> Dict[str, float]:
    """macro AUROC(kept top-`coverage`) - macro AUROC(all), bootstrapped over PAIRS.
    The selection is redone inside every resample."""
    rng = np.random.default_rng(seed)
    n = len(Y)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        Yb, Sb, rb = Y[i], S[i], reliability[i]
        kept = _retained(rb, coverage, rng)
        d[b] = macro_auroc_subset(Yb, Sb, kept, min_pos)[0] - macro_auroc_subset(Yb, Sb, np.arange(n), min_pos)[0]
    lo, hi = np.nanquantile(d, [0.025, 0.975])
    est = macro_auroc_subset(Y, S, _retained(reliability, coverage, rng), min_pos)[0] - macro_auroc_subset(Y, S, np.arange(n), min_pos)[0]
    return {"coverage": float(coverage), "gain": float(est), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# threshold selection on validation, applied to test
# --------------------------------------------------------------------------- #
def threshold_for_coverage(val_reliability: np.ndarray, target_coverage: float) -> float:
    """tau such that ~target_coverage of VALIDATION pairs have reliability >= tau."""
    return float(np.quantile(val_reliability, 1.0 - target_coverage))


def apply_abstention(test_reliability: np.ndarray, tau: float) -> np.ndarray:
    """Boolean mask: True = answer, False = ABSTAIN."""
    return test_reliability >= tau


def deployed_summary(Y_test, S_test, rel_test, tau, min_pos: int = 5) -> Dict[str, float]:
    keep = apply_abstention(rel_test, tau)
    idx = np.where(keep)[0]
    au, n_ev = macro_auroc_subset(Y_test, S_test, idx, min_pos) if len(idx) else (float("nan"), 0)
    all_au, _ = macro_auroc_subset(Y_test, S_test, np.arange(len(Y_test)), min_pos)
    return {"tau": float(tau), "realised_coverage": float(keep.mean()), "n_answered": int(keep.sum()),
            "macro_auroc_answered": au, "macro_auroc_all": all_au, "n_events_evaluable": n_ev}
