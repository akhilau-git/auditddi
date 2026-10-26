"""Compare ExtraTrees, bootstrap Random Forest, and XGBoost for FAERS fractions.

Hyperparameters are selected by five-fold scaffold-grouped validation using
2025 training labels only. The 2026 H1 temporal cohort is evaluated after model
selection and is exploratory because those results have already been reviewed.
The target is a descriptive FAERS report fraction, not clinical toxicity risk.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from train_single_drug_toxicity_temporal import (
    _fingerprints, _load_labels, _metrics, _sha256,
)


SEEDS = [203, 204, 205, 206, 207]


def _model(name: str, params: dict[str, Any], seed: int):
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=500, min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"], bootstrap=True,
            random_state=seed, n_jobs=-1,
        )
    if name == "xgboost":
        return XGBRegressor(
            n_estimators=400, objective="reg:squarederror", eval_metric="mae",
            tree_method="hist", learning_rate=params["learning_rate"],
            max_depth=params["max_depth"], min_child_weight=params["min_child_weight"],
            subsample=0.8, colsample_bytree=0.7, reg_lambda=10.0,
            random_state=seed, n_jobs=-1, verbosity=0,
        )
    if name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=5, max_features=0.8,
            bootstrap=False, random_state=seed, n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-predictions", type=Path, required=True)
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=3)
    args = parser.parse_args()
    if args.fingerprint_bits < 1 or args.radius < 0:
        raise ValueError("fingerprint-bits must be positive and radius nonnegative")
    for path in (args.train_labels, args.test_labels, args.benchmark_predictions):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = _load_labels(args.train_labels)
    test = _load_labels(args.test_labels)
    test["is_new_structure"] = ~test["canonical_smiles"].isin(set(train["canonical_smiles"]))
    x_train, x_test = _fingerprints(train, test, args.fingerprint_bits, args.radius)
    y_train = train["target"].to_numpy(dtype=np.float64)
    y_test = test["target"].to_numpy(dtype=np.float64)
    n_test = test["n_reports"].to_numpy(dtype=np.float64)
    groups = train["scaffold"].to_numpy()
    cv = GroupKFold(n_splits=5)
    folds = list(cv.split(x_train, y_train, groups))
    for fit_idx, val_idx in folds:
        if not set(groups[fit_idx]).isdisjoint(set(groups[val_idx])):
            raise RuntimeError("Scaffold overlap in a 2025 cross-validation fold")
    print(
        f"2025 train structures={len(train):,}; 2026 H1 test structures={len(test):,}; "
        f"new test structures={int(test['is_new_structure'].sum()):,}; "
        "all tuning uses 2025 scaffold folds only.", flush=True
    )

    configurations: dict[str, list[dict[str, Any]]] = {
        "extratrees": [{"min_samples_leaf": 5, "max_features": 0.8}],
        "random_forest": [
            {"min_samples_leaf": leaf, "max_features": features}
            for leaf, features in itertools.product([3, 5, 10], ["sqrt", 0.5, 0.8])
        ],
        "xgboost": [
            {"learning_rate": rate, "max_depth": depth, "min_child_weight": child}
            for rate, depth, child in itertools.product([0.03, 0.05], [2, 3], [3, 8])
        ],
    }
    selected: dict[str, dict[str, Any]] = {}
    cv_rows: list[dict[str, Any]] = []
    for name, configs in configurations.items():
        print(f"Scaffold-CV tuning {name}: {len(configs)} configurations", flush=True)
        for config_index, params in enumerate(configs):
            print(
                f"  {name} configuration {config_index + 1}/{len(configs)}: {params}",
                flush=True,
            )
            fold_maes = []
            for fold_index, (fit_idx, val_idx) in enumerate(folds):
                estimator = _model(name, params, seed=203 + fold_index)
                estimator.fit(x_train[fit_idx], y_train[fit_idx])
                prediction = np.clip(estimator.predict(x_train[val_idx]), 0.0, 1.0)
                fold_maes.append(float(np.mean(np.abs(y_train[val_idx] - prediction))))
            cv_rows.append({
                "model": name, "configuration": config_index,
                "parameters": json.dumps(params, sort_keys=True),
                "cv_mae_mean": float(np.mean(fold_maes)),
                "cv_mae_std": float(np.std(fold_maes, ddof=1)),
                "fold_maes": json.dumps(fold_maes),
            })
        best = min(
            (row for row in cv_rows if row["model"] == name),
            key=lambda row: (row["cv_mae_mean"], row["cv_mae_std"]),
        )
        selected[name] = json.loads(best["parameters"])
        print(
            f"Selected {name}: CV MAE={best['cv_mae_mean']:.5f} "
            f"(+/- {best['cv_mae_std']:.5f}); params={selected[name]}", flush=True
        )
    cv_frame = pd.DataFrame(cv_rows).sort_values(["model", "cv_mae_mean"])
    cv_frame.to_csv(args.output_dir / "scaffold_cv_model_search.csv", index=False)

    test_predictions: dict[str, list[np.ndarray]] = {name: [] for name in configurations}
    test_metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for name, params in selected.items():
        for seed in SEEDS:
            print(f"Final fit {name}, seed={seed}...", flush=True)
            estimator = _model(name, params, seed=seed)
            estimator.fit(x_train, y_train)
            prediction = np.clip(estimator.predict(x_test), 0.0, 1.0)
            test_predictions[name].append(prediction)
            test_metrics.append({
                "seed": seed, "model": name, "cohort": "all_2026h1_test",
                **_metrics(y_test, prediction, n_test),
            })
            joblib.dump(estimator, args.output_dir / f"{name}_seed_{seed}.joblib")
            prediction_frames.append(pd.DataFrame({
                "seed": seed, "model": name,
                "canonical_smiles": test["canonical_smiles"],
                "y_true": y_test, "y_pred": prediction,
                "n_reports": test["n_reports"].to_numpy(dtype=int),
                "is_new_structure": test["is_new_structure"].to_numpy(dtype=bool),
            }))

    mean_predictions = {}
    for name, values in test_predictions.items():
        mean_prediction = np.mean(np.stack(values), axis=0)
        mean_predictions[name] = mean_prediction
        test_metrics.append({
            "seed": -1, "model": f"{name}_seed_mean", "cohort": "all_2026h1_test",
            **_metrics(y_test, mean_prediction, n_test),
        })
        prediction_frames.append(pd.DataFrame({
            "seed": -1, "model": f"{name}_seed_mean",
            "canonical_smiles": test["canonical_smiles"],
            "y_true": y_test, "y_pred": mean_prediction,
            "n_reports": test["n_reports"].to_numpy(dtype=int),
            "is_new_structure": test["is_new_structure"].to_numpy(dtype=bool),
        }))

    baseline_predictions = np.full(len(test), float(y_train.mean()))
    test_metrics.append({
        "seed": 0, "model": "train_mean_per_drug", "cohort": "all_2026h1_test",
        **_metrics(y_test, baseline_predictions, n_test),
    })
    prediction_frames.append(pd.DataFrame({
        "seed": 0, "model": "train_mean_per_drug",
        "canonical_smiles": test["canonical_smiles"],
        "y_true": y_test, "y_pred": baseline_predictions,
        "n_reports": test["n_reports"].to_numpy(dtype=int),
        "is_new_structure": test["is_new_structure"].to_numpy(dtype=bool),
    }))

    benchmark = pd.read_csv(args.benchmark_predictions)
    benchmark = benchmark[
        (benchmark["model"] == "ecfp6_extratrees")
        & benchmark["seed"].isin(SEEDS)
    ]
    tree = benchmark.groupby("canonical_smiles", as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
    )
    expected = pd.DataFrame({"canonical_smiles": test["canonical_smiles"], "y_true": y_test})
    aligned = expected.merge(tree, on="canonical_smiles", how="left", suffixes=("_expected", ""))
    if aligned["y_pred"].isna().any() or not np.allclose(
        aligned["y_true_expected"], aligned["y_true"], atol=1e-10
    ):
        raise ValueError("Saved ExtraTrees benchmark predictions do not match the current test labels")
    et_prediction = aligned["y_pred"].to_numpy()
    test_metrics.append({
        "seed": -1, "model": "ecfp6_extratrees_seed_mean", "cohort": "all_2026h1_test",
        **_metrics(y_test, et_prediction, n_test),
    })
    prediction_frames.append(pd.DataFrame({
        "seed": -1, "model": "ecfp6_extratrees_seed_mean",
        "canonical_smiles": test["canonical_smiles"], "y_true": y_test,
        "y_pred": et_prediction, "n_reports": test["n_reports"].to_numpy(dtype=int),
        "is_new_structure": test["is_new_structure"].to_numpy(dtype=bool),
    }))

    metrics = pd.DataFrame(test_metrics)
    metrics.to_csv(args.output_dir / "tree_ensemble_temporal_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.output_dir / "tree_ensemble_temporal_predictions.csv", index=False
    )
    seed_rows = metrics[
        metrics["seed"].isin(SEEDS)
        & metrics["model"].isin(["random_forest", "xgboost", "extratrees"])
    ]
    metric_columns = ["mae", "rmse", "r2", "spearman", "sqrt_report_weighted_mae"]
    seed_summary = seed_rows.groupby("model")[metric_columns].agg(["mean", "std"])
    seed_summary.columns = [f"{metric}_{stat}" for metric, stat in seed_summary.columns]
    seed_summary.reset_index().to_csv(args.output_dir / "tree_ensemble_seed_summary.csv", index=False)

    manifest = {
        "task": "FAERS serious-outcome report fraction regression",
        "interpretation": "descriptive report statistic; not clinical incidence, causality, or patient risk",
        "selection": "five-fold scaffold-grouped validation on 2025 train labels only",
        "2026_test_status": "exploratory; previous 2026 H1 model comparisons have already been reviewed",
        "fingerprint": {"type": "Morgan/ECFP6", "radius": args.radius, "bits": args.fingerprint_bits},
        "seeds": SEEDS,
        "selected_parameters": selected,
        "train_labels": str(args.train_labels.resolve()),
        "test_labels": str(args.test_labels.resolve()),
        "train_sha256": _sha256(args.train_labels),
        "test_sha256": _sha256(args.test_labels),
        "benchmark_predictions": str(args.benchmark_predictions.resolve()),
        "outputs": {
            "cv_search": str(args.output_dir / "scaffold_cv_model_search.csv"),
            "test_metrics": str(args.output_dir / "tree_ensemble_temporal_metrics.csv"),
            "seed_summary": str(args.output_dir / "tree_ensemble_seed_summary.csv"),
        },
    }
    (args.output_dir / "tree_ensemble_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Tree ensemble comparison complete.", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print("Across-seed summary:")
    print(pd.read_csv(args.output_dir / "tree_ensemble_seed_summary.csv").to_string(index=False))
    print("Outputs:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
