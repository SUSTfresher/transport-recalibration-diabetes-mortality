from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "transportability_predictions.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport"

N_BOOT = 1000
EPS = 1e-6


def expected_calibration_error(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> float:
    order = np.argsort(pred)
    y_sorted = y[order]
    pred_sorted = pred[order]
    bins = np.array_split(np.arange(len(pred_sorted)), n_bins)
    ece = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        ece += (len(idx) / len(pred_sorted)) * abs(float(y_sorted[idx].mean()) - float(pred_sorted[idx].mean()))
    return float(ece)


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    has_two_classes = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, pred)) if has_two_classes else np.nan,
        "pr_auc": float(average_precision_score(y, pred)) if has_two_classes else np.nan,
        "brier_score": float(brier_score_loss(y, pred)),
        "ece_10bin": expected_calibration_error(y, pred),
        "mean_prediction": float(np.mean(pred)),
        "event_rate": float(np.mean(y)),
    }


def bootstrap_group(group: pd.DataFrame, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    y = group["outcome"].astype(int).to_numpy()
    pred = group["prediction"].to_numpy()
    n = len(group)
    point = point_metrics(y, pred)
    id_cols = {
        "train_source": group["train_source"].iloc[0],
        "test_target": group["test_target"].iloc[0],
        "feature_set": group["feature_set"].iloc[0],
        "model": group["model"].iloc[0],
        "n": int(n),
        "events": int(y.sum()),
    }

    long_rows = []
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        pred_b = pred[idx]
        metrics = point_metrics(y_b, pred_b)
        metrics.update(id_cols)
        metrics["bootstrap"] = i
        long_rows.append(metrics)

    boot = pd.DataFrame(long_rows)
    summary_rows = []
    for metric in ["roc_auc", "pr_auc", "brier_score", "ece_10bin", "mean_prediction", "event_rate"]:
        vals = boot[metric].dropna().to_numpy()
        summary_rows.append(
            {
                **id_cols,
                "metric": metric,
                "point": point[metric],
                "ci_lower": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                "ci_upper": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                "bootstrap_n_valid": int(len(vals)),
            }
        )
    return long_rows, summary_rows


def main() -> None:
    preds = pd.read_csv(PRED_PATH)
    rng = np.random.default_rng(20260608)
    all_long = []
    all_summary = []
    group_cols = ["train_source", "test_target", "feature_set", "model"]
    for _, group in preds.groupby(group_cols, sort=False):
        long_rows, summary_rows = bootstrap_group(group.reset_index(drop=True), rng)
        all_long.extend(long_rows)
        all_summary.extend(summary_rows)

    long_df = pd.DataFrame(all_long)
    summary_df = pd.DataFrame(all_summary)
    long_df.to_csv(OUT_DIR / "transportability_bootstrap_long.csv", index=False)
    summary_df.to_csv(OUT_DIR / "transportability_bootstrap_ci.csv", index=False)
    metadata = {
        "prediction_path": str(PRED_PATH),
        "n_bootstrap": N_BOOT,
        "random_seed": 20260608,
        "metrics": ["roc_auc", "pr_auc", "brier_score", "ece_10bin", "mean_prediction", "event_rate"],
        "ci": "Percentile 2.5% and 97.5% bootstrap intervals.",
    }
    (OUT_DIR / "transportability_bootstrap_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
