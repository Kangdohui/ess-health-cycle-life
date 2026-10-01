"""Small-sample regression with cell/protocol-safe validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PAPER_MAPE = 0.091


def regression_metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "outputs/eda/cell_features.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.features)
    batch1_all = data[data.batch == "Batch 1"].reset_index(drop=True)
    batch2 = data[data.batch == "Batch 2"].reset_index(drop=True)
    batch3 = data[data.batch == "Batch 3"].reset_index(drop=True)
    batch1 = batch1_all.dropna(subset=["cycle_life"]).reset_index(drop=True)
    batch2_eval = batch2.dropna(subset=["cycle_life"]).reset_index(drop=True)
    batch3_eval = batch3.dropna(subset=["cycle_life"]).reset_index(drop=True)
    if len(batch1) < 12 or batch2_eval.empty:
        raise ValueError("Need at least 12 Batch 1 cells and Batch 2 cells for the required evaluation")

    # Cycle-life forecast uses only measurements from the first 100 cycles.
    candidate_cols = [c for c in data.columns if (c.startswith("early_") and not c.startswith("early_cycle_")) or c.startswith("dq_")]
    if not candidate_cols:
        raise ValueError("No first-100-cycle or ΔQ(V) features found")

    # Hold out whole charging policies so the validation set contains unseen protocols.
    groups = batch1.charging_policy.astype(str).to_numpy()
    if pd.Series(groups).nunique() < 3:
        raise ValueError("Too few distinct charging policies for grouped hold-out and CV")
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
    train_idx, valid_idx = next(split.split(batch1[candidate_cols], batch1.cycle_life, groups))
    train, valid = batch1.iloc[train_idx], batch1.iloc[valid_idx]
    # Exclude features with no observed training values before fitting transformers.
    feature_cols = [c for c in candidate_cols if train[c].notna().any()]
    if not feature_cols:
        raise ValueError("No observed candidate features in the Batch 1 training subset")

    cv_splits = min(5, train.charging_policy.nunique())
    if cv_splits < 2:
        raise ValueError("Too few training policies for grouped cross-validation")
    cv = GroupKFold(n_splits=cv_splits)
    baseline_cv_scores = cross_val_score(
        DummyRegressor(strategy="median"), train[feature_cols], train.cycle_life,
        groups=train.charging_policy.astype(str), cv=cv,
        scoring="neg_mean_absolute_percentage_error", n_jobs=1,
    )
    baseline_holdout = DummyRegressor(strategy="median").fit(train[feature_cols], train.cycle_life)
    baseline_valid_metrics = regression_metrics(valid.cycle_life, baseline_holdout.predict(valid[feature_cols]))
    k_features = min(8, len(feature_cols))
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_regression, k=k_features)),
        ("scale", StandardScaler()),
        ("model", Ridge()),
    ])
    grid = [
        {"model": [Ridge()], "model__alpha": [0.1, 1.0, 10.0, 100.0]},
        {"model": [RandomForestRegressor(random_state=args.seed, n_estimators=300, min_samples_leaf=2)],
         "model__max_depth": [2, 3, None], "model__max_features": [0.6, 1.0]},
    ]
    search = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_percentage_error", cv=cv, refit=True, n_jobs=1, return_train_score=True)
    search.fit(train[feature_cols], train.cycle_life, groups=train.charging_policy.astype(str))
    best = search.best_estimator_
    valid_pred = best.predict(valid[feature_cols])
    valid_metrics = regression_metrics(valid.cycle_life, valid_pred)

    # Freeze model choice before using Batch 2, then fit on all Batch 1 cells.
    selected_params = search.best_params_
    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_regression, k=k_features)),
        ("scale", StandardScaler()),
        ("model", selected_params["model"]),
    ])
    final_params = {k.replace("model__", "model__", 1): v for k, v in selected_params.items() if k.startswith("model__")}
    final_pipe.set_params(**final_params)
    final_pipe.fit(batch1[feature_cols], batch1.cycle_life)
    test_pred = final_pipe.predict(batch2[feature_cols])
    test_pred_eval = final_pipe.predict(batch2_eval[feature_cols])
    test_metrics = regression_metrics(batch2_eval.cycle_life, test_pred_eval)
    batch3_pred = final_pipe.predict(batch3[feature_cols])
    batch3_pred_eval = final_pipe.predict(batch3_eval[feature_cols])
    batch3_metrics = regression_metrics(batch3_eval.cycle_life, batch3_pred_eval)
    final_baseline = DummyRegressor(strategy="median").fit(batch1[feature_cols], batch1.cycle_life)
    baseline_test_metrics = regression_metrics(batch2_eval.cycle_life, final_baseline.predict(batch2_eval[feature_cols]))
    baseline_batch3_metrics = regression_metrics(batch3_eval.cycle_life, final_baseline.predict(batch3_eval[feature_cols]))
    train_cv_metrics = {
        "MAPE_mean": float(-search.best_score_),
        "MAPE_std": float(search.cv_results_["std_test_score"][search.best_index_]),
    }

    feature_mask = final_pipe.named_steps["select"].get_support()
    selected_features = [name for name, keep in zip(feature_cols, feature_mask) if keep]
    report = {
        "target": "cycle_life",
        "input_window": "first 100 cycles",
        "unlabeled_target_cells": {"Batch 1": int(batch1_all.cycle_life.isna().sum()), "Batch 2": int(batch2.cycle_life.isna().sum()), "Batch 3": int(batch3.cycle_life.isna().sum())},
        "labeled_target_cells": {"Batch 1": int(len(batch1)), "Batch 2": int(len(batch2_eval)), "Batch 3": int(len(batch3_eval))},
        "holdout_sizes": {"Batch 1 train subset": int(len(train)), "Batch 1 validation": int(len(valid))},
        "seed": args.seed,
        "package_versions": {"python": __import__("platform").python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "split": "Batch 1 grouped by charging_policy (80/20); GroupKFold on train subset; final test on Batch 2",
        "selected_model": type(final_pipe.named_steps["model"]).__name__,
        "selected_hyperparameters": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else type(v).__name__) for k, v in selected_params.items()},
        "selected_features": selected_features,
        "train_cv": train_cv_metrics,
        "baseline_median": {
            "train_cv": {"MAPE_mean": float(-baseline_cv_scores.mean()), "MAPE_std": float(baseline_cv_scores.std(ddof=1))},
            "valid_batch1_holdout": baseline_valid_metrics,
            "test_batch2": baseline_test_metrics,
            "additional_batch3": baseline_batch3_metrics,
        },
        "valid_batch1_holdout": valid_metrics,
        "test_batch2": test_metrics,
        "paper_test_mape_target": PAPER_MAPE,
        "mape_gaps_percentage_points": {
            "valid_minus_train_cv": (valid_metrics["MAPE"] - train_cv_metrics["MAPE_mean"]) * 100,
            "batch2_test_minus_valid": (test_metrics["MAPE"] - valid_metrics["MAPE"]) * 100,
            "batch2_test_minus_paper_target": (test_metrics["MAPE"] - PAPER_MAPE) * 100,
            "batch2_test_minus_batch3": (test_metrics["MAPE"] - batch3_metrics["MAPE"]) * 100,
            "batch3_test_minus_paper_target": (batch3_metrics["MAPE"] - PAPER_MAPE) * 100,
        },
        "additional_batch3": batch3_metrics,
    }
    with (args.out_dir / "metrics.json").open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    evaluation_rows = [
        {"split": "Train CV: median baseline", **{"MAPE": -baseline_cv_scores.mean()}},
        {"split": "Train CV: selected model", **{"MAPE": train_cv_metrics["MAPE_mean"]}},
        baseline_valid_metrics | {"split": "Valid: median baseline"},
        valid_metrics | {"split": "Valid: selected model"},
        baseline_test_metrics | {"split": "Test: Batch 2 median baseline"},
        test_metrics | {"split": "Test: Batch 2 selected model"},
        baseline_batch3_metrics | {"split": "Additional: Batch 3 median baseline"},
        batch3_metrics | {"split": "Additional: Batch 3 selected model"},
    ]
    pd.DataFrame(evaluation_rows).to_csv(args.out_dir / "evaluation.csv", index=False)
    pd.DataFrame({"cell_id": batch2.cell_id, "actual": batch2.cycle_life, "predicted": test_pred}).to_csv(args.out_dir / "batch2_predictions.csv", index=False)
    pd.DataFrame({"cell_id": batch3.cell_id, "actual": batch3.cycle_life, "predicted": batch3_pred}).to_csv(args.out_dir / "batch3_predictions.csv", index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
