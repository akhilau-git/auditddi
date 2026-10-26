"""Temporal, count-aware evaluation of single-drug FAERS report fractions.

This estimates the fraction of mapped FAERS cases for a drug with a recorded
serious outcome. It is a pharmacovigilance report statistic, not clinical
incidence, causal toxicity, or patient risk.

Use 2025 labels for fitting and the separately built 2026 H1 new-case table as
the untouched temporal test. Internal scaffold validation on 2025 selects the
beta-binomial MLP epoch. Morgan ExtraTrees and two training-only mean baselines
are included as comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.special import betaln, gammaln
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BetaBinomialMLP(nn.Module):
    def __init__(self, input_bits: int = 1024) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_bits, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.log_concentration = nn.Parameter(torch.tensor(3.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.network(x).squeeze(-1)
        concentration = torch.nn.functional.softplus(self.log_concentration) + 0.1
        return mean, concentration


def beta_binomial_nll(
    mean: torch.Tensor,
    concentration: torch.Tensor,
    severe: torch.Tensor,
    total: torch.Tensor,
) -> torch.Tensor:
    mean = mean.to(torch.float64).clamp(1e-6, 1.0 - 1e-6)
    concentration = concentration.to(torch.float64).clamp(0.101, 1e5)
    severe = severe.to(torch.float64)
    total = total.to(torch.float64)
    nonsevere = total - severe
    alpha = (mean * concentration).clamp_min(1e-8)
    beta = ((1.0 - mean) * concentration).clamp_min(1e-8)
    log_choose = (
        torch.lgamma(total + 1.0)
        - torch.lgamma(severe + 1.0)
        - torch.lgamma(nonsevere + 1.0)
    )
    log_beta_post = (
        torch.lgamma(severe + alpha)
        + torch.lgamma(nonsevere + beta)
        - torch.lgamma(total + alpha + beta)
    )
    log_beta_prior = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return -(log_choose + log_beta_post - log_beta_prior).mean()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "canonical_smiles", "serious_outcome_reports", "n_reports",
        "serious_outcome_recorded_fraction",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["canonical_smiles"] = frame["canonical_smiles"].astype(str).str.strip()
    frame["serious_outcome_reports"] = pd.to_numeric(
        frame["serious_outcome_reports"], errors="raise"
    ).astype(int)
    frame["n_reports"] = pd.to_numeric(frame["n_reports"], errors="raise").astype(int)
    if frame["canonical_smiles"].eq("").any():
        raise ValueError(f"{path} contains empty SMILES")
    if (frame["n_reports"] <= 0).any():
        raise ValueError(f"{path} contains nonpositive report denominators")
    if (
        (frame["serious_outcome_reports"] < 0)
        | (frame["serious_outcome_reports"] > frame["n_reports"])
    ).any():
        raise ValueError(f"{path} contains impossible severe/total report counts")
    frame["target"] = frame["serious_outcome_reports"] / frame["n_reports"]
    if not np.allclose(
        frame["target"], frame["serious_outcome_recorded_fraction"], atol=1e-10
    ):
        raise ValueError(f"{path} fraction column does not equal severe_count / n_reports")

    canonical: list[str] = []
    for value in frame["canonical_smiles"]:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"Invalid SMILES in {path}: {value}")
        canonical.append(Chem.MolToSmiles(mol, canonical=True))
    frame["canonical_smiles"] = canonical
    if frame["canonical_smiles"].duplicated().any():
        duplicates = int(frame["canonical_smiles"].duplicated().sum())
        raise ValueError(f"{path} has {duplicates} duplicate canonical structures")
    frame["scaffold"] = [
        MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles), includeChirality=False)
        or f"ACYCLIC::{smiles}"
        for smiles in canonical
    ]
    return frame


def _fingerprints(
    train: pd.DataFrame,
    test: pd.DataFrame,
    bits: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    by_smiles: dict[str, np.ndarray] = {}

    def encode(smiles: str) -> np.ndarray:
        if smiles not in by_smiles:
            mol = Chem.MolFromSmiles(smiles)
            arr = np.zeros(bits, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
            by_smiles[smiles] = arr
        return by_smiles[smiles]

    x_train = np.stack([encode(s) for s in train["canonical_smiles"]]).astype(np.float32)
    x_test = np.stack([encode(s) for s in test["canonical_smiles"]]).astype(np.float32)
    return x_train, x_test


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    totals: np.ndarray,
    beta_nll: float | None = None,
) -> dict[str, float | None]:
    pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    truth = np.asarray(y_true, dtype=float)
    abs_error = np.abs(truth - pred)
    sq_error = (truth - pred) ** 2
    weights = np.sqrt(np.maximum(np.asarray(totals, dtype=float), 1.0))
    rho = (
        float(spearmanr(truth, pred).statistic)
        if len(truth) > 1 and np.ptp(truth) > 0 and np.ptp(pred) > 0 else None
    )
    denom = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(sq_error.mean())),
        "r2": float(1.0 - sq_error.sum() / denom) if denom > 1e-12 else None,
        "spearman": rho,
        "sqrt_report_weighted_mae": float(np.average(abs_error, weights=weights)),
        "beta_binomial_nll_per_drug": beta_nll,
    }


def _beta_binomial_nll_numpy(
    severe: np.ndarray,
    total: np.ndarray,
    mean: np.ndarray,
    concentration: float,
) -> float:
    mu = np.clip(np.asarray(mean, dtype=float), 1e-6, 1.0 - 1e-6)
    k = np.asarray(severe, dtype=float)
    n = np.asarray(total, dtype=float)
    c = max(float(concentration), 0.101)
    alpha = np.maximum(mu * c, 1e-8)
    beta = np.maximum((1.0 - mu) * c, 1e-8)
    log_choose = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    log_probability = log_choose + betaln(k + alpha, n - k + beta) - betaln(alpha, beta)
    return float(-np.mean(log_probability))


def _fit_mlp(
    x: np.ndarray,
    severe: np.ndarray,
    total: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
) -> tuple[dict[str, torch.Tensor], int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = BetaBinomialMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(
        torch.from_numpy(x[train_idx]),
        torch.from_numpy(severe[train_idx]),
        torch.from_numpy(total[train_idx]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, kb, nb in loader:
            xb, kb, nb = xb.to(device), kb.to(device), nb.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, concentration = model(xb)
            loss = beta_binomial_nll(mean, concentration, kb, nb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite beta-binomial loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_mean, _ = model(torch.from_numpy(x[val_idx]).to(device))
            val_pred = val_mean.cpu().numpy()
        val_mae = float(np.mean(np.abs(y[val_idx] - val_pred)))
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"seed={seed} epoch={epoch}/{epochs} val_scaffold_MAE={val_mae:.5f} "
            f"best={best_mae:.5f}@{best_epoch}", flush=True
        )
        if patience > 0 and stale >= patience:
            print(f"Early stopping at epoch {epoch}; selected epoch {best_epoch}.", flush=True)
            break
    if best_state is None:
        raise RuntimeError("No beta-binomial checkpoint was selected")
    return best_state, best_epoch


def _refit_mlp(
    x: np.ndarray,
    severe: np.ndarray,
    total: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> tuple[BetaBinomialMLP, float]:
    random.seed(seed + 10_000)
    np.random.seed(seed + 10_000)
    torch.manual_seed(seed + 10_000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 10_000)
    model = BetaBinomialMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(severe), torch.from_numpy(total))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1, max(epochs, 1) + 1):
        model.train()
        for xb, kb, nb in loader:
            xb, kb, nb = xb.to(device), kb.to(device), nb.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, concentration = model(xb)
            loss = beta_binomial_nll(mean, concentration, kb, nb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        _, concentration = model(torch.from_numpy(x[:1]).to(device))
    return model, float(concentration.cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[203, 204, 205, 206, 207])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        raise ValueError("epochs/batch-size must be positive and patience nonnegative")
    for path in (args.train_labels, args.test_labels):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    train = _load_labels(args.train_labels)
    test = _load_labels(args.test_labels)
    train_smiles = set(train["canonical_smiles"])
    test["is_new_structure"] = ~test["canonical_smiles"].isin(train_smiles)
    print(
        f"Train structures={len(train):,}; test structures={len(test):,}; "
        f"test-only structures={int(test['is_new_structure'].sum()):,}; "
        f"shared={int((~test['is_new_structure']).sum()):,}.", flush=True
    )

    x_train, x_test = _fingerprints(train, test, args.fingerprint_bits, args.radius)
    y_train = train["target"].to_numpy(dtype=np.float32)
    k_train = train["serious_outcome_reports"].to_numpy(dtype=np.float32)
    n_train = train["n_reports"].to_numpy(dtype=np.float32)
    y_test = test["target"].to_numpy(dtype=np.float64)
    k_test = test["serious_outcome_reports"].to_numpy(dtype=np.float64)
    n_test = test["n_reports"].to_numpy(dtype=np.float64)

    # Two transparent baselines, both estimated from 2025 only.
    mean_drug = float(y_train.mean())
    pooled = float(k_train.sum() / n_train.sum())
    baseline_predictions = {
        "train_mean_per_drug": np.full(len(test), mean_drug),
        "train_pooled_report_fraction": np.full(len(test), pooled),
    }
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for name, pred in baseline_predictions.items():
        metrics_rows.append({
            "seed": 0, "model": name, "cohort": "all_2026h1_test",
            **_metrics(y_test, pred, n_test),
        })
        prediction_rows.append(pd.DataFrame({
            "seed": 0, "model": name,
            "canonical_smiles": test["canonical_smiles"], "y_true": y_test,
            "y_pred": pred, "serious_outcome_reports": k_test,
            "n_reports": n_test.astype(int), "is_new_structure": test["is_new_structure"],
        }))

    tree_predictions: list[np.ndarray] = []
    for seed in args.seeds:
        print(f"Fitting Morgan ExtraTrees for seed {seed}...", flush=True)
        tree = ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=5, max_features=0.8,
            random_state=seed, n_jobs=-1,
        )
        tree.fit(x_train, y_train)
        tree_pred = np.clip(tree.predict(x_test), 0.0, 1.0)
        tree_predictions.append(tree_pred)
        metrics_rows.append({
            "seed": seed, "model": "ecfp6_extratrees", "cohort": "all_2026h1_test",
            **_metrics(y_test, tree_pred, n_test),
        })
        joblib.dump(tree, args.output_dir / f"ecfp6_extratrees_seed_{seed}.joblib")
        prediction_rows.append(pd.DataFrame({
            "seed": seed, "model": "ecfp6_extratrees",
            "canonical_smiles": test["canonical_smiles"], "y_true": y_test,
            "y_pred": tree_pred, "serious_outcome_reports": k_test,
            "n_reports": n_test.astype(int), "is_new_structure": test["is_new_structure"],
        }))

    tree_ensemble_pred = np.mean(np.stack(tree_predictions), axis=0)
    metrics_rows.append({
        "seed": -1, "model": "ecfp6_extratrees_seed_mean",
        "cohort": "all_2026h1_test",
        **_metrics(y_test, tree_ensemble_pred, n_test),
    })

    groups = train["scaffold"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seeds[0])
    fit_idx, val_idx = next(splitter.split(x_train, y_train, groups))
    if not set(groups[fit_idx]).isdisjoint(set(groups[val_idx])):
        raise RuntimeError("Scaffold overlap in internal 2025 validation split")

    mlp_predictions: list[np.ndarray] = []
    for seed in args.seeds:
        print(f"Selecting beta-binomial MLP duration for seed {seed}...", flush=True)
        _, best_epoch = _fit_mlp(
            x_train, k_train, n_train, y_train, fit_idx, val_idx,
            seed=seed, device=device, epochs=args.epochs,
            batch_size=args.batch_size, patience=args.patience,
        )
        model, concentration = _refit_mlp(
            x_train, k_train, n_train, seed=seed, device=device,
            epochs=best_epoch, batch_size=args.batch_size,
        )
        with torch.no_grad():
            pred, _ = model(torch.from_numpy(x_test).to(device))
        predictions = np.clip(pred.cpu().numpy(), 0.0, 1.0)
        mlp_predictions.append(predictions)
        nll = _beta_binomial_nll_numpy(k_test, n_test, predictions, concentration)
        metrics_rows.append({
            "seed": seed, "model": "ecfp6_beta_binomial_mlp", "cohort": "all_2026h1_test",
            "best_epoch": best_epoch, "concentration": concentration,
            **_metrics(y_test, predictions, n_test, beta_nll=nll),
        })
        torch.save({
            "model_state_dict": model.state_dict(),
            "fingerprint_bits": args.fingerprint_bits,
            "radius": args.radius,
            "seed": seed,
            "best_epoch": best_epoch,
            "concentration": concentration,
        }, args.output_dir / f"beta_binomial_mlp_seed_{seed}.pt")
        prediction_rows.append(pd.DataFrame({
            "seed": seed, "model": "ecfp6_beta_binomial_mlp",
            "canonical_smiles": test["canonical_smiles"], "y_true": y_test,
            "y_pred": predictions, "serious_outcome_reports": k_test,
            "n_reports": n_test.astype(int), "is_new_structure": test["is_new_structure"],
        }))

    mlp_ensemble_pred = np.mean(np.stack(mlp_predictions), axis=0)
    mlp_ensemble_nll = _beta_binomial_nll_numpy(
        k_test,
        n_test,
        mlp_ensemble_pred,
        float(np.mean([
            row["concentration"]
            for row in metrics_rows
            if row.get("model") == "ecfp6_beta_binomial_mlp"
        ])),
    )
    metrics_rows.append({
        "seed": -1, "model": "ecfp6_beta_binomial_mlp_seed_mean",
        "cohort": "all_2026h1_test",
        **_metrics(y_test, mlp_ensemble_pred, n_test, beta_nll=mlp_ensemble_nll),
    })

    new_mask = test["is_new_structure"].to_numpy(dtype=bool)
    if int(new_mask.sum()) >= 2:
        for name, pred in [
            ("train_mean_per_drug", baseline_predictions["train_mean_per_drug"]),
            ("train_pooled_report_fraction", baseline_predictions["train_pooled_report_fraction"]),
            ("ecfp6_extratrees_seed_mean", tree_ensemble_pred),
            ("ecfp6_beta_binomial_mlp_seed_mean", mlp_ensemble_pred),
        ]:
            metrics_rows.append({
                "seed": -1, "model": name, "cohort": "new_structures_only_exploratory",
                **_metrics(y_test[new_mask], pred[new_mask], n_test[new_mask]),
            })

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.output_dir / "temporal_metrics.csv", index=False)
    repeat_metrics = metrics.loc[
        metrics["seed"].isin(args.seeds)
        & metrics["model"].isin(["ecfp6_extratrees", "ecfp6_beta_binomial_mlp"])
        & metrics["cohort"].eq("all_2026h1_test")
    ]
    summary_columns = [
        column for column in (
            "mae", "rmse", "r2", "spearman",
            "sqrt_report_weighted_mae", "beta_binomial_nll_per_drug",
        ) if column in repeat_metrics.columns
    ]
    if not repeat_metrics.empty:
        seed_summary = repeat_metrics.groupby("model")[summary_columns].agg(["mean", "std"])
        seed_summary.columns = [f"{metric}_{stat}" for metric, stat in seed_summary.columns]
        seed_summary.reset_index().to_csv(
            args.output_dir / "temporal_seed_summary.csv", index=False
        )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output_dir / "temporal_predictions.csv", index=False
    )
    config = {
        "task": "FAERS serious-outcome report fraction regression",
        "interpretation": "descriptive report statistic; not clinical incidence, causality, or patient risk",
        "train_labels": str(args.train_labels.resolve()),
        "test_labels": str(args.test_labels.resolve()),
        "train_sha256": _sha256(args.train_labels),
        "test_sha256": _sha256(args.test_labels),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "new_test_structures": int(test["is_new_structure"].sum()),
        "shared_test_structures": int((~test["is_new_structure"]).sum()),
        "seeds": args.seeds,
        "epochs_max": args.epochs,
        "internal_validation": "2025 train-only scaffold-disjoint split; 2026 H1 untouched until evaluation",
        "fingerprint": {"type": "Morgan/ECFP6", "radius": args.radius, "bits": args.fingerprint_bits},
        "device": str(device),
        "metrics_file": str(args.output_dir / "temporal_metrics.csv"),
    }
    (args.output_dir / "temporal_run_manifest.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print("\nTemporal comparison complete.")
    print(metrics.to_string(index=False))
    print("Outputs:", args.output_dir)


if __name__ == "__main__":
    main()
