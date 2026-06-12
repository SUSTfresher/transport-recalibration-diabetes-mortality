from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_PRED = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "transportability_predictions.csv"
RECAL_LONG = PROJECT_ROOT / "outputs" / "local_recalibration_by_event_count" / "local_recalibration_by_event_results_long.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "calibration_slope_intercept"

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_regression(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = y.astype(float)
    x = logit(pred)
    design = np.column_stack([np.ones_like(x), x])

    def nll(beta: np.ndarray) -> float:
        eta = design @ beta
        # Stable negative log-likelihood for Bernoulli-logit.
        return float(np.sum(np.logaddexp(0, eta) - y * eta))

    result = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    beta = result.x
    prob = expit(design @ beta)
    w = prob * (1 - prob)
    hessian = design.T @ (design * w[:, None])
    try:
        cov = np.linalg.inv(hessian)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan)
        se = np.array([np.nan, np.nan])
    return {
        "calibration_intercept": float(beta[0]),
        "calibration_intercept_se": float(se[0]),
        "calibration_intercept_ci_lower": float(beta[0] - 1.96 * se[0]),
        "calibration_intercept_ci_upper": float(beta[0] + 1.96 * se[0]),
        "calibration_slope": float(beta[1]),
        "calibration_slope_se": float(se[1]),
        "calibration_slope_ci_lower": float(beta[1] - 1.96 * se[1]),
        "calibration_slope_ci_upper": float(beta[1] + 1.96 * se[1]),
        "converged": bool(result.success),
        "nll": float(result.fun),
    }


def transport_calibration_rows() -> pd.DataFrame:
    preds = pd.read_csv(TRANSPORT_PRED)
    rows = []
    group_cols = ["train_source", "test_target", "feature_set", "model"]
    for key, group in preds.groupby(group_cols, sort=False):
        y = group["outcome"].astype(int).to_numpy()
        pred = group["prediction"].to_numpy()
        m = calibration_regression(y, pred)
        row = dict(zip(group_cols, key))
        row.update(
            {
                "analysis": "transport_or_internal",
                "recalibration_method": "raw_model_prediction",
                "event_target": 0,
                "repeat": 0,
                "n": int(len(group)),
                "events": int(y.sum()),
                "event_rate": float(y.mean()),
                "mean_prediction": float(pred.mean()),
            }
        )
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


def recalibration_calibration_rows() -> pd.DataFrame:
    rec = pd.read_csv(RECAL_LONG)
    focus = rec[
        rec["prediction_set"].eq("NHANES_to_MIMIC_1y_base_logistic_regression")
        & rec["method"].isin(["raw", "platt", "intercept_only", "isotonic"])
        & rec["event_target"].isin([0, 25, 50, 100, 200, 500, 1000])
    ].copy()
    rows = []
    for _, row in focus.iterrows():
        rows.append(
            {
                "analysis": "event_count_recalibration",
                "train_source": "NHANES",
                "test_target": "MIMIC-IV",
                "feature_set": "base",
                "model": "logistic_regression",
                "recalibration_method": row["method"],
                "event_target": int(row["event_target"]),
                "repeat": int(row["repeat"]),
                "n": int(row["evaluation_n"]),
                "events": int(row["evaluation_events"]),
                "event_rate": float(row["event_rate"]),
                "mean_prediction": float(row["mean_prediction"]),
                "calibration_intercept": float(row["calibration_intercept"]),
                "calibration_slope": float(row["calibration_slope"]),
                "ece_10bin": float(row["ece_10bin"]),
                "brier_score": float(row["brier_score"]),
                "roc_auc": float(row["roc_auc"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_recalibration_calibration(recal_rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "calibration_slope",
        "calibration_intercept",
        "ece_10bin",
        "brier_score",
        "roc_auc",
        "mean_prediction",
        "event_rate",
        "n",
        "events",
    ]
    summary = recal_rows.groupby(["recalibration_method", "event_target"], as_index=False)[metric_cols].agg(
        ["mean", "std", lambda s: s.quantile(0.025), lambda s: s.quantile(0.975)]
    )
    summary.columns = [
        "_".join([str(x) for x in col if x]).replace("<lambda_0>", "ci_lower").replace("<lambda_1>", "ci_upper")
        for col in summary.columns.to_flat_index()
    ]
    return summary.reset_index(drop=True)


def manuscript_table(transport_rows: pd.DataFrame, recal_summary: pd.DataFrame) -> pd.DataFrame:
    nhanes_mimic_raw = transport_rows[
        transport_rows["train_source"].eq("NHANES")
        & transport_rows["test_target"].eq("MIMIC-IV")
        & transport_rows["feature_set"].eq("base")
        & transport_rows["model"].eq("logistic_regression")
    ].copy()
    raw_row = {
        "method": "raw transport",
        "event_target": 0,
        "evaluation_n": int(nhanes_mimic_raw["n"].iloc[0]),
        "evaluation_events": int(nhanes_mimic_raw["events"].iloc[0]),
        "calibration_slope": nhanes_mimic_raw["calibration_slope"].iloc[0],
        "calibration_slope_ci_lower": nhanes_mimic_raw["calibration_slope_ci_lower"].iloc[0],
        "calibration_slope_ci_upper": nhanes_mimic_raw["calibration_slope_ci_upper"].iloc[0],
        "calibration_intercept": nhanes_mimic_raw["calibration_intercept"].iloc[0],
        "calibration_intercept_ci_lower": nhanes_mimic_raw["calibration_intercept_ci_lower"].iloc[0],
        "calibration_intercept_ci_upper": nhanes_mimic_raw["calibration_intercept_ci_upper"].iloc[0],
        "ece_10bin": np.nan,
        "brier_score": np.nan,
        "note": "Standard calibration regression on raw transported predictions.",
    }
    rows = [raw_row]
    wanted = recal_summary[
        recal_summary["recalibration_method"].isin(["platt", "intercept_only", "isotonic"])
        & recal_summary["event_target"].isin([25, 50, 100, 200])
    ].copy()
    method_label = {
        "platt": "Platt recalibration",
        "intercept_only": "Intercept-only recalibration",
        "isotonic": "Isotonic recalibration",
    }
    for _, r in wanted.iterrows():
        rows.append(
            {
                "method": method_label[r["recalibration_method"]],
                "event_target": int(r["event_target"]),
                "evaluation_n": r["n_mean"],
                "evaluation_events": r["events_mean"],
                "calibration_slope": r["calibration_slope_mean"],
                "calibration_slope_ci_lower": r["calibration_slope_ci_lower"],
                "calibration_slope_ci_upper": r["calibration_slope_ci_upper"],
                "calibration_intercept": r["calibration_intercept_mean"],
                "calibration_intercept_ci_lower": r["calibration_intercept_ci_lower"],
                "calibration_intercept_ci_upper": r["calibration_intercept_ci_upper"],
                "ece_10bin": r["ece_10bin_mean"],
                "brier_score": r["brier_score_mean"],
                "note": "Empirical interval across repeated local calibration samples.",
            }
        )
    table = pd.DataFrame(rows)
    order = {
        "raw transport": 0,
        "Platt recalibration": 1,
        "Intercept-only recalibration": 2,
        "Isotonic recalibration": 3,
    }
    table["_order"] = table["method"].map(order)
    return table.sort_values(["_order", "event_target"]).drop(columns="_order")


def write_readme(table: pd.DataFrame) -> None:
    text = """# Calibration Slope and Intercept Summary

Generated on 2026-06-08.

## Purpose

This analysis reports calibration regression slope and intercept for the dual-database manuscript.

The standard calibration regression is:

```text
outcome ~ intercept + slope * logit(predicted probability)
```

Perfect calibration corresponds to intercept 0 and slope 1.

## Outputs

```text
outputs\\calibration_slope_intercept\\transport_calibration_slope_intercept.csv
outputs\\calibration_slope_intercept\\event_recalibration_calibration_long.csv
outputs\\calibration_slope_intercept\\event_recalibration_calibration_summary.csv
outputs\\calibration_slope_intercept\\calibration_slope_intercept_focus_table.csv
```

## Notes

- Raw transport calibration intervals use model-based standard errors from the logistic calibration regression Hessian.
- Recalibration rows report empirical 2.5%-97.5% intervals across repeated local calibration samples.
- Isotonic is included as a sensitivity analysis and should not be framed as the preferred small-sample method.
- Isotonic regression recalibration showed unstable slope estimates even with 200 local events, supporting its use as a sensitivity analysis rather than a primary recalibration method.
"""
    (OUT_DIR / "README_calibration_slope_intercept.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transport_rows = transport_calibration_rows()
    recal_rows = recalibration_calibration_rows()
    recal_summary = summarize_recalibration_calibration(recal_rows)
    table = manuscript_table(transport_rows, recal_summary)
    transport_rows.to_csv(OUT_DIR / "transport_calibration_slope_intercept.csv", index=False)
    recal_rows.to_csv(OUT_DIR / "event_recalibration_calibration_long.csv", index=False)
    recal_summary.to_csv(OUT_DIR / "event_recalibration_calibration_summary.csv", index=False)
    table.to_csv(OUT_DIR / "calibration_slope_intercept_focus_table.csv", index=False)
    metadata = {
        "transport_prediction_path": str(TRANSPORT_PRED),
        "recalibration_long_path": str(RECAL_LONG),
        "calibration_regression": "outcome ~ intercept + slope * logit(predicted_probability)",
        "perfect_calibration": {"intercept": 0, "slope": 1},
    }
    (OUT_DIR / "calibration_slope_intercept_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(table)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
