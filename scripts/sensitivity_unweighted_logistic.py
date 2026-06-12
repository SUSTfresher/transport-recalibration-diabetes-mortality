from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "sensitivity_unweighted_logistic"
EPS = 1e-6


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nm = load_module(ROOT / "scripts" / "nhanes_mimic_oneyear_mortality_transport.py", "nhanes_mimic_transport")
icu = load_module(ROOT / "scripts" / "mimic_icu_eicu_transport_recalibration.py", "icu_transport")


def inv_logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expected_calibration_error(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> float:
    order = np.argsort(pred)
    y_sorted = y[order]
    pred_sorted = pred[order]
    bins = np.array_split(np.arange(len(pred_sorted)), min(n_bins, len(pred_sorted)))
    ece = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        ece += (len(idx) / len(pred_sorted)) * abs(float(y_sorted[idx].mean()) - float(pred_sorted[idx].mean()))
    return float(ece)


def expected_calibration_error_equal_width(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), EPS, 1 - EPS)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(pred, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(pred[mask].mean()))
    return float(ece)


def calibration_regression(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    x = logit(pred)
    if not np.all(np.isfinite(x)) or float(np.std(x)) < 1e-8:
        return np.nan, np.nan
    x = x.reshape(-1, 1)

    # Use sklearn's numerically guarded optimizer here because the NHANES
    # unweighted sensitivity can produce a narrow low-probability range.
    candidates = [
        {"C": 1e12},
        {"C": 1e6},
    ]
    for kwargs in candidates:
        try:
            model = LogisticRegression(max_iter=5000, solver="lbfgs", fit_intercept=True, **kwargs)
            model.fit(x, y.astype(int))
            return float(model.coef_[0, 0]), float(model.intercept_[0])
        except Exception:
            continue
    return np.nan, np.nan


def point_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    pred = np.clip(np.asarray(pred, dtype=float), EPS, 1 - EPS)
    slope, intercept = calibration_regression(y, pred)
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "mean_prediction": float(pred.mean()),
        "roc_auc": float(roc_auc_score(y, pred)),
        "pr_auc": float(average_precision_score(y, pred)),
        "brier_score": float(brier_score_loss(y, pred)),
        "ece_10bin": expected_calibration_error(y, pred),
        "ece_10bin_equal_frequency": expected_calibration_error(y, pred),
        "ece_10bin_equal_width": expected_calibration_error_equal_width(y, pred),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def make_unweighted_nhanes_model(features: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ]
    )
    return Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=2000, solver="lbfgs"))])


def make_unweighted_icu_model(numeric: list[str], binary: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("binary", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), binary),
        ]
    )
    return Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=2000, solver="lbfgs"))])


def logistic_nhanes_to_mimic(class_weight: str) -> dict[str, object]:
    nhanes = nm.load_nhanes()
    mimic = nm.load_mimic()
    nh_train, _ = nm.temporal_nhanes_split(nhanes)
    _, mimic_test = nm.mimic_split(mimic)
    features = nm.COMMON_BASE
    model = nm.make_model("logistic_regression", features) if class_weight == "balanced" else make_unweighted_nhanes_model(features)
    train = nh_train.dropna(subset=["outcome"]).copy()
    target = mimic_test.dropna(subset=["outcome"]).copy()
    model.fit(train[features], train["outcome"].astype(int))
    pred = model.predict_proba(target[features])[:, 1]
    metrics = point_metrics(target["outcome"].astype(int).to_numpy(), pred)
    return {
        "scenario": "Extreme cross-setting stress test",
        "direction": "NHANES -> MIMIC-IV",
        "endpoint": "1-year mortality",
        "feature_set": "NHANES-compatible base",
        "model": "logistic_regression_weighted" if class_weight == "balanced" else "logistic_regression_unweighted",
        "class_weight": class_weight,
        **metrics,
    }


def logistic_icu_direction(train_source: str, target_source: str, class_weight: str) -> dict[str, object]:
    sources = {
        "MIMIC-IV ICU": icu.load_source("MIMIC-IV ICU"),
        "eICU": icu.load_source("eICU"),
    }
    splits: dict[str, dict[str, pd.DataFrame]] = {}
    for i, source in enumerate(["MIMIC-IV ICU", "eICU"]):
        development, holdout = icu.split_by_patient(sources[source], icu.RANDOM_SEED + i)
        splits[source] = {"development": development, "holdout": holdout}

    spec = icu.FEATURE_SETS["icu_native_primary"]
    numeric = spec["numeric"]
    binary = spec["binary"]
    features = numeric + binary
    model = icu.make_model("logistic_regression", numeric, binary) if class_weight == "balanced" else make_unweighted_icu_model(numeric, binary)
    train = splits[train_source]["development"]
    target = splits[target_source]["holdout"]
    model.fit(train[features], train["outcome"].astype(int))
    pred = model.predict_proba(target[features])[:, 1]
    metrics = point_metrics(target["outcome"].astype(int).to_numpy(), pred)
    return {
        "scenario": "Realistic ICU deployment",
        "direction": f"{train_source} -> {target_source}",
        "endpoint": "Hospital mortality",
        "feature_set": "ICU-native primary",
        "model": "logistic_regression_weighted" if class_weight == "balanced" else "logistic_regression_unweighted",
        "class_weight": class_weight,
        **metrics,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for class_weight in ["balanced", "none"]:
        rows.extend(
            [
                logistic_nhanes_to_mimic(class_weight),
                logistic_icu_direction("MIMIC-IV ICU", "eICU", class_weight),
                logistic_icu_direction("eICU", "MIMIC-IV ICU", class_weight),
            ]
        )
    combined = pd.DataFrame(rows)
    combined["prediction_minus_event_rate"] = combined["mean_prediction"] - combined["event_rate"]
    combined = combined[
        [
            "scenario",
            "direction",
            "endpoint",
            "feature_set",
            "model",
            "class_weight",
            "n",
            "events",
            "event_rate",
            "mean_prediction",
            "prediction_minus_event_rate",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "ece_10bin",
            "ece_10bin_equal_frequency",
            "ece_10bin_equal_width",
            "calibration_slope",
            "calibration_intercept",
        ]
    ]
    combined.to_csv(OUT_DIR / "primary_transport_weighted_vs_unweighted.csv", index=False)

    wide = combined.pivot_table(
        index=["scenario", "direction", "endpoint", "feature_set"],
        columns="class_weight",
        values=[
            "mean_prediction",
            "prediction_minus_event_rate",
            "roc_auc",
            "brier_score",
            "ece_10bin",
            "ece_10bin_equal_frequency",
            "ece_10bin_equal_width",
            "calibration_slope",
            "calibration_intercept",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{weight}" for metric, weight in wide.columns]
    wide = wide.reset_index()
    for metric in [
        "mean_prediction",
        "prediction_minus_event_rate",
        "roc_auc",
        "brier_score",
        "ece_10bin",
        "ece_10bin_equal_frequency",
        "ece_10bin_equal_width",
        "calibration_slope",
        "calibration_intercept",
    ]:
        if f"{metric}_none" in wide.columns and f"{metric}_balanced" in wide.columns:
            wide[f"{metric}_delta_unweighted_minus_weighted"] = wide[f"{metric}_none"] - wide[f"{metric}_balanced"]
    wide.to_csv(OUT_DIR / "primary_transport_weighted_vs_unweighted_wide.csv", index=False)

    metadata = {
        "purpose": "Sensitivity analysis comparing primary class-balanced logistic transport results with unweighted logistic regression.",
        "class_weight_primary": "balanced",
        "class_weight_sensitivity": "none",
        "note": "Uses the same cohort loaders, feature sets, deterministic splits, and primary logistic model settings as the primary analyses, except for class weighting. Also reports 10-bin equal-frequency and equal-width ECE as a binning robustness check; no recalibration or bootstrap intervals are run here.",
    }
    (OUT_DIR / "sensitivity_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
