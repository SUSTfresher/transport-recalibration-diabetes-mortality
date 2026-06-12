from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "transportability_predictions.csv"
FEATURE_PATH = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "nhanes_mimic_common_transport_table.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "subgroup_transportability"


MIN_N = 100
MIN_EVENTS_FOR_AUC = 5


def normalize_id(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    return out.str.replace(r"\.0$", "", regex=True)


def expected_calibration_error(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> float:
    if len(y) == 0:
        return np.nan
    order = np.argsort(pred)
    y_sorted = y[order]
    pred_sorted = pred[order]
    bins = np.array_split(np.arange(len(y_sorted)), min(n_bins, len(y_sorted)))
    ece = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        ece += (len(idx) / len(y_sorted)) * abs(float(y_sorted[idx].mean()) - float(pred_sorted[idx].mean()))
    return float(ece)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    events = int(y.sum())
    nonevents = int(len(y) - events)
    has_auc = events >= MIN_EVENTS_FOR_AUC and nonevents >= MIN_EVENTS_FOR_AUC
    return {
        "n": int(len(y)),
        "events": events,
        "event_rate": float(np.mean(y)) if len(y) else np.nan,
        "mean_prediction": float(np.mean(pred)) if len(pred) else np.nan,
        "roc_auc": float(roc_auc_score(y, pred)) if has_auc else np.nan,
        "pr_auc": float(average_precision_score(y, pred)) if has_auc else np.nan,
        "brier_score": float(brier_score_loss(y, pred)) if len(y) else np.nan,
        "ece_10bin": expected_calibration_error(y, pred),
        "eligible_for_auc": bool(has_auc),
    }


def add_subgroup_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_group"] = np.where(out["age"].ge(65), "Age >=65", "Age <65")
    out.loc[out["age"].isna(), "age_group"] = np.nan
    out["sex_group"] = out["female"].map({0: "Male", 1: "Female"})
    out["ckd_group"] = out["ckd_history"].map({0: "No CKD", 1: "CKD"})
    out["cvd_group"] = out["cvd_history"].map({0: "No CVD", 1: "CVD"})
    out["hypertension_group"] = out["hypertension_history"].map({0: "No hypertension", 1: "Hypertension"})
    out["obesity_group"] = np.where(out["bmi"].ge(30), "BMI >=30", "BMI <30")
    out.loc[out["bmi"].isna(), "obesity_group"] = np.nan
    return out


def load_merged() -> pd.DataFrame:
    preds = pd.read_csv(PRED_PATH, dtype={"id": str})
    features = pd.read_csv(FEATURE_PATH, dtype={"id": str})
    preds["id_norm"] = normalize_id(preds["id"])
    features["id_norm"] = normalize_id(features["id"])
    feature_cols = [
        "source",
        "id_norm",
        "age",
        "female",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "hypertension_history",
        "ckd_history",
        "cvd_history",
    ]
    features = features[feature_cols].rename(columns={"source": "test_target"})
    merged = preds.merge(features, on=["test_target", "id_norm"], how="left", validate="many_to_one")
    return add_subgroup_columns(merged)


def subgroup_rows(df: pd.DataFrame) -> pd.DataFrame:
    subgroup_map = {
        "age_group": "Age",
        "sex_group": "Sex",
        "ckd_group": "CKD history",
        "cvd_group": "CVD history",
        "hypertension_group": "Hypertension history",
        "obesity_group": "Obesity",
    }
    group_cols = ["train_source", "test_target", "feature_set", "model"]
    rows = []
    for model_key, group in df.groupby(group_cols, sort=False):
        y = group["outcome"].astype(int).to_numpy()
        pred = group["prediction"].to_numpy()
        base = dict(zip(group_cols, model_key))
        m = metrics(y, pred)
        m.update(base)
        m.update({"subgroup_type": "Overall", "subgroup": "Overall"})
        rows.append(m)
        for col, subgroup_type in subgroup_map.items():
            for subgroup, sub in group.dropna(subset=[col]).groupby(col, sort=False):
                if len(sub) < MIN_N:
                    continue
                y_sub = sub["outcome"].astype(int).to_numpy()
                pred_sub = sub["prediction"].to_numpy()
                m = metrics(y_sub, pred_sub)
                m.update(base)
                m.update({"subgroup_type": subgroup_type, "subgroup": subgroup})
                rows.append(m)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_readme(summary: pd.DataFrame) -> None:
    focus = summary[
        summary["train_source"].eq("NHANES")
        & summary["test_target"].eq("MIMIC-IV")
        & summary["feature_set"].eq("base")
        & summary["model"].eq("logistic_regression")
        & ~summary["subgroup_type"].eq("Overall")
    ].copy()
    focus = focus.sort_values(["subgroup_type", "subgroup"])
    cols = [
        "subgroup_type",
        "subgroup",
        "n",
        "events",
        "event_rate",
        "mean_prediction",
        "roc_auc",
        "brier_score",
        "ece_10bin",
    ]
    table = markdown_table(focus[cols], cols)
    text = f"""# Subgroup Transportability Analysis

Generated on 2026-06-08.

## Purpose

This analysis evaluates whether one-year mortality transportability differs across clinically relevant subgroups.

Subgroups:

- age <65 vs >=65
- sex
- CKD history
- CVD history
- hypertension history
- BMI <30 vs >=30

Metrics are reported for all train/test/model combinations. The main focus is NHANES-trained base logistic regression transported to MIMIC-IV.

## Focus Result

NHANES to MIMIC-IV, base logistic regression:

{table}

## Interpretation

Subgroup results help identify where transported models are more poorly calibrated or less discriminative. For high-impact review, these results are important because average AUC alone can hide clinically relevant failures in older adults, CKD, CVD, or obese subgroups.

## Outputs

```text
outputs\\subgroup_transportability\\subgroup_transportability_metrics.csv
outputs\\subgroup_transportability\\subgroup_transportability_metadata.json
outputs\\subgroup_transportability\\README_subgroup_transportability.md
```
"""
    (OUT_DIR / "README_subgroup_transportability.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_merged()
    summary = subgroup_rows(merged)
    summary.to_csv(OUT_DIR / "subgroup_transportability_metrics.csv", index=False)
    metadata = {
        "prediction_path": str(PRED_PATH),
        "feature_path": str(FEATURE_PATH),
        "min_n": MIN_N,
        "min_events_for_auc": MIN_EVENTS_FOR_AUC,
        "subgroups": ["age", "sex", "ckd_history", "cvd_history", "hypertension_history", "bmi_ge_30"],
        "notes": [
            "AUC and PR AUC are set missing when a subgroup has fewer than the minimum event or non-event count.",
            "Subgroup metrics are descriptive; multiplicity-adjusted inferential testing is not performed here.",
        ],
    }
    (OUT_DIR / "subgroup_transportability_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
