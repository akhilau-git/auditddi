"""Multi-label baselines in pure numpy (Guide 1, section 4.13).

Why numpy: the maths is fully testable without a GPU stack (the finite-difference
gradient test lives in tests/test_baselines.py) and the results are bit-for-bit
reproducible from a seed.

All models share
    fit(X_train, Y_train, X_val, Y_val) -> self
    predict_proba(X) -> (n, n_events) float32 in [0, 1]

Loss  (Guide 1, section 4.8), per event e and pair i:
    L = -pw_e * y * log(sigma(z)) - (1 - y) * log(1 - sigma(z)),   pw_e = n_neg_e / n_pos_e
computed from TRAINING labels only.  Optimiser: AdamW (decoupled weight decay),
global gradient clipping 1.0, ReduceLROnPlateau on the validation monitor, early
stopping with patience.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from src.evaluation.multilabel_metrics import auprc_per_event, auroc_per_event, evaluable_mask


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def monitor_score(Y: np.ndarray, S: np.ndarray, kind: str = "auprc", min_pos: int = 5) -> float:
    ev = evaluable_mask(Y, min_pos)
    if not ev.any():
        return float("nan")
    f = auprc_per_event if kind == "auprc" else auroc_per_event
    return float(np.nanmean(f(Y[:, ev], S[:, ev])))


# --------------------------------------------------------------------------- #
# trivial reference
# --------------------------------------------------------------------------- #
class FrequencyPrior:
    """Same score for every pair: the (smoothed) training prevalence of each event.
    Per-event AUROC is exactly 0.5, so any real model must beat it."""

    name = "prior"

    def fit(self, X, Y, Xv=None, Yv=None):
        self.p_ = ((Y.sum(axis=0) + 1.0) / (len(Y) + 2.0)).astype(np.float32)
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (len(X), 1))


# --------------------------------------------------------------------------- #
# logistic regression (hidden=0) and one-hidden-layer MLP
# --------------------------------------------------------------------------- #
class NumpyMultiLabelNet:
    def __init__(
        self,
        hidden: int = 0,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        epochs: int = 60,
        patience: int = 10,
        lr_patience: int = 3,
        clip: float = 1.0,
        pos_weight_clip: Optional[float] = None,
        monitor: str = "auprc",
        seed: int = 0,
        verbose: bool = False,
    ):
        self.hidden, self.dropout, self.lr, self.wd = hidden, dropout, lr, weight_decay
        self.batch_size, self.epochs, self.patience, self.lr_patience = batch_size, epochs, patience, lr_patience
        self.clip, self.pw_clip, self.monitor, self.seed, self.verbose = clip, pos_weight_clip, monitor, seed, verbose
        self.name = "logreg" if hidden == 0 else f"mlp{hidden}"
        self.history_: list = []

    # ---- parameters ------------------------------------------------------
    def _init(self, d: int, m: int, Y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        prev = np.clip((Y.sum(axis=0) + 1.0) / (len(Y) + 2.0), 1e-4, 1 - 1e-4)
        b_out = np.log(prev / (1 - prev)).astype(np.float32)          # start at the prior
        if self.hidden == 0:
            self.P = {"W": (rng.normal(0, 0.01, (d, m))).astype(np.float32), "b": b_out}
        else:
            h = self.hidden
            self.P = {
                "W1": (rng.normal(0, np.sqrt(2.0 / d), (d, h))).astype(np.float32),
                "b1": np.zeros(h, dtype=np.float32),
                "W2": (rng.normal(0, np.sqrt(1.0 / h), (h, m)) * 0.1).astype(np.float32),
                "b2": b_out,
            }
        self.M = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.V = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    # ---- forward / backward ----------------------------------------------
    def _forward(self, X, train: bool, rng=None):
        if self.hidden == 0:
            return X @ self.P["W"] + self.P["b"], None
        hp = X @ self.P["W1"] + self.P["b1"]
        h = np.maximum(hp, 0)
        mask = None
        if train and self.dropout > 0:
            mask = (rng.random(h.shape) >= self.dropout).astype(np.float32) / (1.0 - self.dropout)
            h = h * mask
        z = h @ self.P["W2"] + self.P["b2"]
        return z, (hp, h, mask)

    def loss_and_grads(self, X, Y, pw, train: bool = False, rng=None):
        """Mean over pairs AND events of the weighted BCE; returns (loss, grads)."""
        B, M = Y.shape
        z, cache = self._forward(X, train, rng)
        s = _sigmoid(z)
        zc = np.clip(z, -35, 35)
        # stable: log s = -softplus(-z) ; log(1-s) = -softplus(z)
        sp_neg = np.logaddexp(0, -zc)
        sp_pos = np.logaddexp(0, zc)
        loss = float(np.mean(pw * Y * sp_neg + (1 - Y) * sp_pos))
        dz = (-(pw * Y) * (1 - s) + (1 - Y) * s) / (B * M)
        dz = dz.astype(X.dtype)
        if self.hidden == 0:
            return loss, {"W": X.T @ dz, "b": dz.sum(0)}
        hp, h, mask = cache
        dh = dz @ self.P["W2"].T
        if mask is not None:
            dh = dh * mask
        dh = dh * (hp > 0)
        return loss, {"W2": h.T @ dz, "b2": dz.sum(0), "W1": X.T @ dh, "b1": dh.sum(0)}

    # ---- optimiser -------------------------------------------------------
    def _step(self, G, lr):
        gn = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in G.values()))
        scale = min(1.0, self.clip / (gn + 1e-12))
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in G.items():
            g = g * scale
            self.M[k] = b1 * self.M[k] + (1 - b1) * g
            self.V[k] = b2 * self.V[k] + (1 - b2) * g * g
            mh = self.M[k] / (1 - b1 ** self.t)
            vh = self.V[k] / (1 - b2 ** self.t)
            upd = mh / (np.sqrt(vh) + eps)
            if k.startswith("W"):
                upd = upd + self.wd * self.P[k]               # decoupled decay, weights only
            self.P[k] = (self.P[k] - lr * upd).astype(np.float32)
        return gn

    # ---- training --------------------------------------------------------
    def fit(self, X, Y, Xv=None, Yv=None):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        n, d = X.shape
        m = Y.shape[1]
        pos = Y.sum(axis=0)
        pw = ((n - pos) / np.maximum(pos, 1.0)).astype(np.float32)
        if self.pw_clip is not None:
            pw = np.minimum(pw, self.pw_clip)
        self.pos_weight_ = pw
        self._init(d, m, Y)
        rng = np.random.default_rng(self.seed + 1)
        lr = self.lr
        best, best_P, bad, plateau = -np.inf, None, 0, 0
        for ep in range(self.epochs):
            order = rng.permutation(n)
            tot, nb = 0.0, 0
            for i in range(0, n, self.batch_size):
                idx = order[i:i + self.batch_size]
                loss, G = self.loss_and_grads(X[idx], Y[idx], pw, train=True, rng=rng)
                self._step(G, lr)
                tot += loss
                nb += 1
            rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "lr": lr}
            if Xv is not None and len(Xv):
                score = monitor_score(Yv, self.predict_proba(Xv), self.monitor)
                rec["val_" + self.monitor] = score
                if score > best + 1e-5:
                    best, best_P, bad, plateau = score, copy.deepcopy(self.P), 0, 0
                else:
                    bad += 1
                    plateau += 1
                    if plateau >= self.lr_patience:
                        lr *= 0.5
                        plateau = 0
                if bad >= self.patience:
                    self.history_.append(rec)
                    break
            self.history_.append(rec)
            if self.verbose:
                print(rec)
        if best_P is not None:
            self.P = best_P
        self.best_val_ = best if best > -np.inf else float("nan")
        return self

    def predict_proba(self, X, batch: int = 4096):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty((len(X), self.P["b"].shape[0] if self.hidden == 0 else self.P["b2"].shape[0]), dtype=np.float32)
        for i in range(0, len(X), batch):
            z, _ = self._forward(X[i:i + batch], train=False)
            out[i:i + batch] = _sigmoid(z)
        return out


# --------------------------------------------------------------------------- #
# random forest on PCA-compressed targets
# --------------------------------------------------------------------------- #
class RandomForestPCA:
    """Random forest for ~860 correlated outputs.

    A forest with 860 regression outputs stores 860 values in every node and is far
    too slow/large.  We regress the top ``n_components`` principal components of the
    training label matrix and reconstruct event scores from them.  This is an
    ADAPTATION of 'random forest on ECFP6 pair features' and is reported as such.
    Reconstructed scores are clipped to [0, 1]; they are rankings, not calibrated
    probabilities."""

    name = "rf_pca"

    def __init__(self, n_components: int = 48, n_estimators: int = 100, max_leaf_nodes: int = 2000,
                 min_samples_leaf: int = 3, max_features="sqrt", max_samples: float = 0.5, seed: int = 0, n_jobs: int = -1):
        self.kw = dict(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                       max_features=max_features, max_samples=max_samples, random_state=seed, n_jobs=n_jobs)
        self.k = n_components

    def fit(self, X, Y, Xv=None, Yv=None):
        from sklearn.ensemble import RandomForestRegressor

        Y = np.asarray(Y, dtype=np.float32)
        self.mu_ = Y.mean(axis=0)
        U, S, Vt = np.linalg.svd(Y - self.mu_, full_matrices=False)
        k = min(self.k, Vt.shape[0])
        self.comp_ = Vt[:k].astype(np.float32)
        self.explained_ = float((S[:k] ** 2).sum() / max((S ** 2).sum(), 1e-12))
        T = (Y - self.mu_) @ self.comp_.T
        self.rf_ = RandomForestRegressor(**self.kw).fit(np.asarray(X, dtype=np.float32), T)
        return self

    def predict_proba(self, X):
        T = self.rf_.predict(np.asarray(X, dtype=np.float32))
        return np.clip(T @ self.comp_ + self.mu_, 0.0, 1.0).astype(np.float32)


def make_model(name: str, seed: int = 0, **kw):
    if name == "prior":
        return FrequencyPrior()
    if name == "logreg":
        return NumpyMultiLabelNet(hidden=0, seed=seed, **kw)
    if name.startswith("mlp"):
        h = int(name[3:] or 512)
        return NumpyMultiLabelNet(hidden=h, seed=seed, **kw)
    if name == "rf_pca":
        return RandomForestPCA(seed=seed)
    raise ValueError(name)
