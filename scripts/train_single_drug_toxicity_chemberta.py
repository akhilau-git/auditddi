"""Evaluate frozen ChemBERTa embeddings for temporal FAERS report-fraction prediction.

The target is a descriptive fraction of mapped FAERS reports with a serious
outcome, not clinical incidence or causal toxicity risk. Model settings are
selected on a scaffold-disjoint validation subset from 2025 only. The 2026 H1
cohort is used only for final exploratory evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from train_single_drug_toxicity_temporal import _load_labels, _metrics, _sha256


MODEL_ID = "DeepChem/ChemBERTa-77M-MLM"
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def _encode_smiles(
    smiles: list[str], *, model_id: str, device: torch.device, batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    configured_max = getattr(tokenizer, "model_max_length", 512)
    max_length = configured_max if isinstance(configured_max, int) and configured_max < 10000 else 512
    embeddings: list[np.ndarray] = []
    truncated_count = 0
    for start in range(0, len(smiles), batch_size):
        batch = smiles[start:start + batch_size]
        tokens = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt", return_length=True,
        )
        truncated_count += sum(int(length >= max_length) for length in tokens.pop("length"))
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        embeddings.append(pooled.cpu().numpy().astype(np.float32))

    metadata = {
        "model_id": model_id,
        "pooling": "attention-mask mean of final hidden states, including special tokens",
        "max_tokens": max_length,
        "possibly_truncated_sequences": truncated_count,
        "embedding_dimension": int(embeddings[0].shape[1]),
    }
    return np.concatenate(embeddings, axis=0), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-predictions", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    for path in (args.train_labels, args.test_labels):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.benchmark_predictions and not args.benchmark_predictions.is_file():
        raise FileNotFoundError(args.benchmark_predictions)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Device:", device, flush=True)
    print("Pretrained encoder:", args.model_id, flush=True)

    train = _load_labels(args.train_labels)
    test = _load_labels(args.test_labels)
    train_smiles = set(train["canonical_smiles"])
    test["is_new_structure"] = ~test["canonical_smiles"].isin(train_smiles)
    all_smiles = list(dict.fromkeys(
        train["canonical_smiles"].tolist() + test["canonical_smiles"].tolist()
    ))
    print(
        f"Train structures={len(train):,}; test structures={len(test):,}; "
        f"test-only structures={int(test['is_new_structure'].sum()):,}.", flush=True
    )
    x_all, encoder_metadata = _encode_smiles(
        all_smiles, model_id=args.model_id, device=device, batch_size=args.batch_size
    )
    embedding_by_smiles = {smiles: x_all[i] for i, smiles in enumerate(all_smiles)}
    x_train = np.stack([embedding_by_smiles[s] for s in train["canonical_smiles"]])
    x_test = np.stack([embedding_by_smiles[s] for s in test["canonical_smiles"]])
    np.savez_compressed(
        args.output_dir / "frozen_chemberta_embeddings.npz",
        train_smiles=train["canonical_smiles"].to_numpy(),
        test_smiles=test["canonical_smiles"].to_numpy(),
        x_train=x_train, x_test=x_test,
    )

    y_train = train["target"].to_numpy(dtype=np.float64)
    y_test = test["target"].to_numpy(dtype=np.float64)
    n_test = test["n_reports"].to_numpy(dtype=np.float64)
    k_test = test["serious_outcome_reports"].to_numpy(dtype=np.int64)
    groups = train["scaffold"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=203)
    fit_idx, val_idx = next(splitter.split(x_train, y_train, groups))
    if not set(groups[fit_idx]).isdisjoint(set(groups[val_idx])):
        raise RuntimeError("Scaffold overlap in internal 2025 validation split")

    validation_rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        candidate.fit(x_train[fit_idx], y_train[fit_idx])
        val_prediction = np.clip(candidate.predict(x_train[val_idx]), 0.0, 1.0)
        score = float(np.mean(np.abs(y_train[val_idx] - val_prediction)))
        validation_rows.append({"alpha": alpha, "scaffold_validation_mae": score})
        print(f"alpha={alpha:g}; 2025 scaffold-validation MAE={score:.5f}", flush=True)
    validation = pd.DataFrame(validation_rows).sort_values(
        ["scaffold_validation_mae", "alpha"], ascending=[True, False]
    )
    best_alpha = float(validation.iloc[0]["alpha"])
    validation.to_csv(args.output_dir / "scaffold_validation_alpha_search.csv", index=False)
    print("Selected alpha using 2025 scaffold validation:", best_alpha, flush=True)

    model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    model.fit(x_train, y_train)
    chemberta_prediction = np.clip(model.predict(x_test), 0.0, 1.0)
    train_mean = float(y_train.mean())
    mean_prediction = np.full(len(test), train_mean)

    metrics_rows: list[dict[str, Any]] = [
        {
            "model": "train_mean_per_drug", "cohort": "all_2026h1_test",
            **_metrics(y_test, mean_prediction, n_test),
        },
        {
            "model": "chemberta_frozen_ridge", "cohort": "all_2026h1_test",
            "selected_alpha": best_alpha,
            **_metrics(y_test, chemberta_prediction, n_test),
        },
    ]
    prediction_rows = [
        pd.DataFrame({
            "model": name, "canonical_smiles": test["canonical_smiles"],
            "y_true": y_test, "y_pred": prediction,
            "serious_outcome_reports": k_test, "n_reports": n_test.astype(int),
            "is_new_structure": test["is_new_structure"],
        })
        for name, prediction in [
            ("train_mean_per_drug", mean_prediction),
            ("chemberta_frozen_ridge", chemberta_prediction),
        ]
    ]

    if args.benchmark_predictions:
        benchmark = pd.read_csv(args.benchmark_predictions)
        benchmark = benchmark[
            (benchmark["model"] == "ecfp6_extratrees")
            & benchmark["seed"].isin([203, 204, 205, 206, 207])
        ]
        if benchmark.empty:
            raise ValueError("Benchmark predictions have no ExtraTrees rows for seeds 203-207")
        tree = benchmark.groupby("canonical_smiles", as_index=False).agg(
            y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
        )
        expected = pd.DataFrame({
            "canonical_smiles": test["canonical_smiles"], "y_true": y_test,
        })
        tree = expected.merge(tree, on="canonical_smiles", how="left", suffixes=("_expected", ""))
        if tree["y_pred"].isna().any() or not np.allclose(
            tree["y_true_expected"], tree["y_true"], atol=1e-10
        ):
            raise ValueError("ExtraTrees benchmark predictions do not align with these test labels")
        tree_prediction = tree["y_pred"].to_numpy()
        metrics_rows.append({
            "model": "ecfp6_extratrees_seed_mean", "cohort": "all_2026h1_test",
            **_metrics(y_test, tree_prediction, n_test),
        })
        prediction_rows.append(pd.DataFrame({
            "model": "ecfp6_extratrees_seed_mean",
            "canonical_smiles": test["canonical_smiles"], "y_true": y_test,
            "y_pred": tree_prediction, "serious_outcome_reports": k_test,
            "n_reports": n_test.astype(int), "is_new_structure": test["is_new_structure"],
        }))

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.output_dir / "pretrained_temporal_metrics.csv", index=False)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(args.output_dir / "pretrained_temporal_predictions.csv", index=False)
    new_mask = test["is_new_structure"].to_numpy(dtype=bool)
    if int(new_mask.sum()) >= 2:
        new_rows = []
        for name, subset in predictions.groupby("model"):
            subset = subset.set_index("canonical_smiles").loc[test.loc[new_mask, "canonical_smiles"]]
            new_rows.append({
                "model": name, "cohort": "new_structures_only_exploratory",
                **_metrics(y_test[new_mask], subset["y_pred"].to_numpy(), n_test[new_mask]),
            })
        pd.DataFrame(new_rows).to_csv(
            args.output_dir / "new_structures_only_exploratory_metrics.csv", index=False
        )

    manifest = {
        "task": "FAERS serious-outcome report fraction regression",
        "interpretation": "descriptive report statistic; not clinical incidence, causality, or patient risk",
        "model": encoder_metadata,
        "regression_head": "StandardScaler + Ridge",
        "ridge_alpha_grid": ALPHAS,
        "selected_alpha": best_alpha,
        "selection": "single scaffold-disjoint 2025 validation split; 2026 H1 test is exploratory and was not used for tuning",
        "train_labels": str(args.train_labels.resolve()),
        "test_labels": str(args.test_labels.resolve()),
        "train_sha256": _sha256(args.train_labels),
        "test_sha256": _sha256(args.test_labels),
        "train_structures": int(len(train)),
        "test_structures": int(len(test)),
        "new_test_structures": int(test["is_new_structure"].sum()),
        "device": str(device),
        "metrics_file": str(args.output_dir / "pretrained_temporal_metrics.csv"),
        "benchmark_predictions": str(args.benchmark_predictions.resolve()) if args.benchmark_predictions else None,
    }
    (args.output_dir / "pretrained_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Pretrained temporal comparison complete.", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print("Outputs:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
