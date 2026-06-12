from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "transportability_predictions.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "local_recalibration_simulation"


FRACTIONS = [0.01, 0.05, 0.10, 0.20]
N_REPEATS = 30
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def inv_logit(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def expected_calibration_error(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> float:
    order = np.argsort(pred)
    y_sorted = y[order]
    pred_sorted = pred[order]
    bins = np.array_split(np.arange(len(pred_sorted)), n_bins)
    ece = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        obs = y_sorted[idx].mean()
        exp = pred_sorted[idx].mean()
        ece += (len(idx) / len(pred_sorted)) * abs(obs - exp)
    return float(ece)


def calibration_slope_intercept(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    x = logit(pred).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    model = LogisticRegression(max_iter=1000, C=1e12)
    model.fit(x, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    slope, intercept = calibration_slope_intercept(y, pred)
    return {
        "roc_auc": float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
        "brier_score": float(brier_score_loss(y, pred)),
        "ece_10bin": expected_calibration_error(y, pred, n_bins=10),
        "mean_prediction": float(np.mean(pred)),
        "event_rate": float(np.mean(y)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def intercept_only_fit(y_cal: np.ndarray, pred_cal: np.ndarray) -> float:
    # Fixed slope 1; solve intercept offset so mean calibrated probability
    # equals observed prevalence in the local calibration sample.
    target = float(np.clip(np.mean(y_cal), EPS, 1 - EPS))
    base = logit(pred_cal)
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = (lo + hi) / 2
        current = float(np.mean(inv_logit(base + mid)))
        if current < target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def apply_method(method: str, y_cal: np.ndarray, pred_cal: np.ndarray, pred_eval: np.ndarray) -> np.ndarray:
    if method == "raw":
        return pred_eval
    if method == "intercept_only":
        offset = intercept_only_fit(y_cal, pred_cal)
        return inv_logit(logit(pred_eval) + offset)
    if method == "platt":
        model = LogisticRegression(max_iter=1000)
        model.fit(logit(pred_cal).reshape(-1, 1), y_cal)
        return model.predict_proba(logit(pred_eval).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        model.fit(pred_cal, y_cal)
        return model.predict(pred_eval)
    raise ValueError(method)


def run_one_prediction_set(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260604)
    y = df["outcome"].astype(int).to_numpy()
    pred = df["prediction"].to_numpy()
    rows = []
    prediction_rows = []

    raw_metrics = metrics(y, pred)
    raw_metrics.update(
        {
            "prediction_set": label,
            "method": "raw",
            "fraction": 0.0,
            "repeat": 0,
            "calibration_n": 0,
            "calibration_events": 0,
            "evaluation_n": int(len(y)),
            "evaluation_events": int(y.sum()),
        }
    )
    rows.append(raw_metrics)

    indices = np.arange(len(df))
    for fraction in FRACTIONS:
        cal_n = max(50, int(round(len(df) * fraction)))
        for repeat in range(N_REPEATS):
            # Keep repeats valid for rare outcomes by retrying a few times until both classes appear.
            for _ in range(100):
                cal_idx = rng.choice(indices, size=cal_n, replace=False)
                if len(np.unique(y[cal_idx])) == 2:
                    break
            eval_mask = np.ones(len(df), dtype=bool)
            eval_mask[cal_idx] = False
            eval_idx = indices[eval_mask]
            y_cal, pred_cal = y[cal_idx], pred[cal_idx]
            y_eval, pred_eval_raw = y[eval_idx], pred[eval_idx]
            for method in ["intercept_only", "platt", "isotonic"]:
                pred_eval = apply_method(method, y_cal, pred_cal, pred_eval_raw)
                m = metrics(y_eval, pred_eval)
                m.update(
                    {
                        "prediction_set": label,
                        "method": method,
                        "fraction": fraction,
                        "repeat": repeat,
                        "calibration_n": int(len(cal_idx)),
                        "calibration_events": int(y_cal.sum()),
                        "evaluation_n": int(len(eval_idx)),
                        "evaluation_events": int(y_eval.sum()),
                    }
                )
                rows.append(m)
                if repeat == 0 and fraction in {0.01, 0.10, 0.20}:
                    prediction_rows.append(
                        pd.DataFrame(
                            {
                                "prediction_set": label,
                                "method": method,
                                "fraction": fraction,
                                "outcome": y_eval,
                                "prediction": pred_eval,
                            }
                        )
                    )
    return pd.DataFrame(rows), pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "ece_10bin",
        "mean_prediction",
        "event_rate",
        "calibration_slope",
        "calibration_intercept",
        "calibration_n",
        "calibration_events",
    ]
    summary = (
        results.groupby(["prediction_set", "method", "fraction"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns.to_flat_index()]
    return summary.reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(PRED_PATH)
    target = preds[
        (preds["train_source"].eq("NHANES"))
        & (preds["test_target"].eq("MIMIC-IV"))
        & (preds["feature_set"].isin(["base", "base_labs"]))
        & (preds["model"].isin(["logistic_regression", "hist_gradient_boosting"]))
    ].copy()
    all_results = []
    all_predictions = []
    for (feature_set, model), group in target.groupby(["feature_set", "model"]):
        label = f"NHANES_to_MIMIC_1y_{feature_set}_{model}"
        print(f"Running recalibration simulation for {label}...")
        res, pred_rows = run_one_prediction_set(group.reset_index(drop=True), label)
        all_results.append(res)
        if not pred_rows.empty:
            all_predictions.append(pred_rows)
    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(OUT_DIR / "local_recalibration_results_long.csv", index=False)
    summarize(results).to_csv(OUT_DIR / "local_recalibration_summary.csv", index=False)
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(OUT_DIR / "local_recalibration_example_predictions.csv", index=False)
    metadata = {
        "prediction_path": str(PRED_PATH),
        "scenario": "NHANES-trained one-year mortality models transported to MIMIC-IV one-year mortality, then recalibrated using small local MIMIC calibration samples.",
        "fractions": FRACTIONS,
        "n_repeats": N_REPEATS,
        "methods": ["raw", "intercept_only", "platt", "isotonic"],
        "notes": [
            "Recalibration is evaluated on the remaining MIMIC transport set after holding out each calibration sample.",
            "This simulates future Chinese-hospital local recalibration but uses MIMIC as the local target site.",
        ],
    }
    (OUT_DIR / "local_recalibration_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(results)
    display_cols = [
        "prediction_set",
        "method",
        "fraction",
        "brier_score_mean",
        "ece_10bin_mean",
        "mean_prediction_mean",
        "event_rate_mean",
        "calibration_slope_mean",
        "roc_auc_mean",
        "calibration_n_mean",
        "calibration_events_mean",
    ]
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
