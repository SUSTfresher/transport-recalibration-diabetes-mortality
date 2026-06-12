from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from local_recalibration_simulation import apply_method, metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "transportability_predictions.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "local_recalibration_by_event_count"

EVENT_TARGETS = [25, 50, 100, 200, 500, 1000]
N_REPEATS = 200


def sample_by_event_count(y: np.ndarray, event_target: int, rng: np.random.Generator) -> np.ndarray:
    event_idx = np.flatnonzero(y == 1)
    nonevent_idx = np.flatnonzero(y == 0)
    if event_target >= len(event_idx):
        raise ValueError(f"event_target={event_target} exceeds available events={len(event_idx)}")
    event_rate = len(event_idx) / len(y)
    total_n = int(round(event_target / event_rate))
    nonevent_target = max(1, total_n - event_target)
    if nonevent_target >= len(nonevent_idx):
        nonevent_target = len(nonevent_idx) - 1
    chosen_events = rng.choice(event_idx, size=event_target, replace=False)
    chosen_nonevents = rng.choice(nonevent_idx, size=nonevent_target, replace=False)
    return np.concatenate([chosen_events, chosen_nonevents])


def run_one_prediction_set(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rng = np.random.default_rng(20260608)
    y = df["outcome"].astype(int).to_numpy()
    pred = df["prediction"].to_numpy()
    indices = np.arange(len(df))
    rows = []

    raw = metrics(y, pred)
    raw.update(
        {
            "prediction_set": label,
            "method": "raw",
            "event_target": 0,
            "repeat": 0,
            "calibration_n": 0,
            "calibration_events": 0,
            "evaluation_n": int(len(y)),
            "evaluation_events": int(y.sum()),
        }
    )
    rows.append(raw)

    max_events = int(y.sum())
    for event_target in EVENT_TARGETS:
        if event_target >= max_events:
            continue
        for repeat in range(N_REPEATS):
            cal_idx = sample_by_event_count(y, event_target, rng)
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
                        "event_target": event_target,
                        "repeat": repeat,
                        "calibration_n": int(len(cal_idx)),
                        "calibration_events": int(y_cal.sum()),
                        "evaluation_n": int(len(eval_idx)),
                        "evaluation_events": int(y_eval.sum()),
                    }
                )
                rows.append(m)
    return pd.DataFrame(rows)


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
    summary = results.groupby(["prediction_set", "method", "event_target"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns.to_flat_index()]
    return summary.reset_index(drop=True)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    headers = cols
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_readme(summary: pd.DataFrame) -> None:
    focus = summary[
        summary["prediction_set"].eq("NHANES_to_MIMIC_1y_base_logistic_regression")
        & (
            summary["method"].eq("raw")
            | ((summary["method"].isin(["platt", "isotonic", "intercept_only"])) & summary["event_target"].isin([25, 50, 100, 200]))
        )
    ].copy()
    cols = [
        "prediction_set",
        "method",
        "event_target",
        "calibration_n_mean",
        "calibration_events_mean",
        "brier_score_mean",
        "ece_10bin_mean",
        "mean_prediction_mean",
        "event_rate_mean",
        "roc_auc_mean",
    ]
    table = markdown_table(focus, cols)
    text = f"""# Local Recalibration by Event Count

Generated on 2026-06-08.

## Purpose

This analysis repeats the local recalibration simulation using the number of local outcome events rather than only the local sample fraction.

This is more clinically interpretable because calibration reliability depends strongly on the number of observed events available at the target site.

## Design

- Source predictions: NHANES-trained one-year mortality models transported to MIMIC-IV.
- Target site for simulation: MIMIC-IV.
- Local calibration event targets: {EVENT_TARGETS}.
- Repeats per event target: {N_REPEATS}.
- Local calibration samples are stratified to preserve the approximate MIMIC event rate.
- Methods: raw, intercept-only, Platt, isotonic.

## Focus Result

NHANES to MIMIC-IV, base logistic regression:

{table}

## Interpretation

Small numbers of local events already improve calibration substantially, but very small event counts can make flexible recalibration unstable. Platt scaling is the preferred default for small local target-site samples; isotonic is better treated as a larger-sample sensitivity analysis.

## Outputs

```text
outputs\\local_recalibration_by_event_count\\local_recalibration_by_event_results_long.csv
outputs\\local_recalibration_by_event_count\\local_recalibration_by_event_summary.csv
outputs\\local_recalibration_by_event_count\\local_recalibration_by_event_metadata.json
outputs\\local_recalibration_by_event_count\\README_local_recalibration_by_event_count.md
```
"""
    (OUT_DIR / "README_local_recalibration_by_event_count.md").write_text(text, encoding="utf-8")


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
    for (feature_set, model), group in target.groupby(["feature_set", "model"]):
        label = f"NHANES_to_MIMIC_1y_{feature_set}_{model}"
        print(f"Running event-count recalibration for {label}...")
        all_results.append(run_one_prediction_set(group.reset_index(drop=True), label))
    results = pd.concat(all_results, ignore_index=True)
    summary = summarize(results)
    results.to_csv(OUT_DIR / "local_recalibration_by_event_results_long.csv", index=False)
    summary.to_csv(OUT_DIR / "local_recalibration_by_event_summary.csv", index=False)
    metadata = {
        "prediction_path": str(PRED_PATH),
        "event_targets": EVENT_TARGETS,
        "n_repeats": N_REPEATS,
        "sampling": "Stratified local calibration samples with fixed event counts and approximate target-site event prevalence.",
        "methods": ["raw", "intercept_only", "platt", "isotonic"],
    }
    (OUT_DIR / "local_recalibration_by_event_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
