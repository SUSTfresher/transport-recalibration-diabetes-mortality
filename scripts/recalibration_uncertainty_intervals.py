from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from local_recalibration_by_event_count import run_one_prediction_set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "transportability_predictions.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "local_recalibration_uncertainty"
TABLE_DIR = PROJECT_ROOT / "outputs" / "manuscript_tables"


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
    grouped = results.groupby(["prediction_set", "method", "event_target"], as_index=False)
    rows = []
    for key, group in grouped:
        row = dict(zip(["prediction_set", "method", "event_target"], key))
        for col in metric_cols:
            values = group[col].dropna()
            row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{col}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{col}_ci_lower"] = float(values.quantile(0.025)) if len(values) else np.nan
            row[f"{col}_ci_upper"] = float(values.quantile(0.975)) if len(values) else np.nan
        row["repeat_n"] = int(group["repeat"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def manuscript_table(summary: pd.DataFrame) -> pd.DataFrame:
    focus = summary[
        summary["prediction_set"].eq("NHANES_to_MIMIC_1y_base_logistic_regression")
        & summary["method"].isin(["raw", "platt", "intercept_only", "isotonic"])
        & summary["event_target"].isin([0, 25, 50, 100, 200, 500, 1000])
    ].copy()
    cols = [
        "prediction_set",
        "method",
        "event_target",
        "repeat_n",
        "calibration_n_mean",
        "calibration_events_mean",
        "brier_score_mean",
        "brier_score_sd",
        "brier_score_ci_lower",
        "brier_score_ci_upper",
        "ece_10bin_mean",
        "ece_10bin_sd",
        "ece_10bin_ci_lower",
        "ece_10bin_ci_upper",
        "calibration_slope_mean",
        "calibration_slope_sd",
        "calibration_slope_ci_lower",
        "calibration_slope_ci_upper",
        "calibration_intercept_mean",
        "calibration_intercept_sd",
        "calibration_intercept_ci_lower",
        "calibration_intercept_ci_upper",
        "mean_prediction_mean",
        "event_rate_mean",
        "roc_auc_mean",
    ]
    return focus[cols].sort_values(["method", "event_target"])


def write_readme(summary: pd.DataFrame) -> None:
    focus = summary[
        summary["prediction_set"].eq("NHANES_to_MIMIC_1y_base_logistic_regression")
        & summary["method"].eq("platt")
        & summary["event_target"].isin([25, 50, 100, 200])
    ][
        [
            "event_target",
            "calibration_n_mean",
            "ece_10bin_mean",
            "ece_10bin_ci_lower",
            "ece_10bin_ci_upper",
            "brier_score_mean",
            "brier_score_ci_lower",
            "brier_score_ci_upper",
            "calibration_slope_mean",
            "calibration_slope_ci_lower",
            "calibration_slope_ci_upper",
            "calibration_intercept_mean",
            "calibration_intercept_ci_lower",
            "calibration_intercept_ci_upper",
        ]
    ]
    lines = [
        "| events | local n | ECE mean | ECE 95% interval | slope mean | slope 95% interval | intercept mean | intercept 95% interval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in focus.iterrows():
        lines.append(
            f"| {int(r['event_target'])} | {r['calibration_n_mean']:.0f} | {r['ece_10bin_mean']:.4f} | {r['ece_10bin_ci_lower']:.4f}-{r['ece_10bin_ci_upper']:.4f} | {r['calibration_slope_mean']:.4f} | {r['calibration_slope_ci_lower']:.4f}-{r['calibration_slope_ci_upper']:.4f} | {r['calibration_intercept_mean']:.4f} | {r['calibration_intercept_ci_lower']:.4f}-{r['calibration_intercept_ci_upper']:.4f} |"
        )
    text = f"""# Recalibration Uncertainty Intervals

Generated on 2026-06-08.

## Purpose

This analysis summarizes uncertainty in event-count local recalibration using repeated local calibration samples.

The current run uses the event-count recalibration result file. If `local_recalibration_by_event_count.py` is configured with 200 repeats, these intervals reflect 200 repeats; otherwise they reflect the current available repeats.

## Focus Result: NHANES to MIMIC-IV Base Logistic, Platt Recalibration

{chr(10).join(lines)}

## Outputs

```text
outputs\\local_recalibration_uncertainty\\recalibration_uncertainty_summary.csv
outputs\\manuscript_tables\\Table_4b_recalibration_uncertainty_intervals.csv
```
"""
    (OUT_DIR / "README_recalibration_uncertainty_intervals.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    # Reuse the current event-count recalibration result file. This keeps this
    # summarizer independent from long-running simulation code.
    existing = PROJECT_ROOT / "outputs" / "local_recalibration_by_event_count" / "local_recalibration_by_event_results_long.csv"
    results = pd.read_csv(existing)
    summary = summarize(results)
    table = manuscript_table(summary)
    summary.to_csv(OUT_DIR / "recalibration_uncertainty_summary.csv", index=False)
    table.to_csv(TABLE_DIR / "Table_4b_recalibration_uncertainty_intervals.csv", index=False)
    metadata = {
        "source_results": str(existing),
        "interval": "Empirical 2.5%-97.5% interval across repeated local calibration samples.",
        "note": "To update from 50 to 200 repeats, set N_REPEATS=200 in local_recalibration_by_event_count.py and rerun that script before this summarizer.",
    }
    (OUT_DIR / "recalibration_uncertainty_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(summary)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
