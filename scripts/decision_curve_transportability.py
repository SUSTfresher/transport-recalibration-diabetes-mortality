from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from local_recalibration_simulation import logit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "transportability_predictions.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "decision_curve"

EVENT_TARGET = 100
RANDOM_SEED = 20260608
THRESHOLDS = np.round(np.arange(0.01, 0.51, 0.01), 2)


def normalize_id(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    return out.str.replace(r"\.0$", "", regex=True)


def sample_calibration_indices(y: np.ndarray, event_target: int, rng: np.random.Generator) -> np.ndarray:
    event_idx = np.flatnonzero(y == 1)
    nonevent_idx = np.flatnonzero(y == 0)
    event_rate = len(event_idx) / len(y)
    total_n = int(round(event_target / event_rate))
    nonevent_target = max(1, total_n - event_target)
    chosen_events = rng.choice(event_idx, size=event_target, replace=False)
    chosen_nonevents = rng.choice(nonevent_idx, size=nonevent_target, replace=False)
    return np.concatenate([chosen_events, chosen_nonevents])


def net_benefit(y: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    treated = pred >= threshold
    n = len(y)
    tp = int(((y == 1) & treated).sum())
    fp = int(((y == 0) & treated).sum())
    return float(tp / n - fp / n * (threshold / (1 - threshold)))


def build_eval_predictions() -> tuple[pd.DataFrame, dict]:
    preds = pd.read_csv(PRED_PATH, dtype={"id": str})
    preds["id_norm"] = normalize_id(preds["id"])
    raw = preds[
        preds["train_source"].eq("NHANES")
        & preds["test_target"].eq("MIMIC-IV")
        & preds["feature_set"].eq("base")
        & preds["model"].eq("logistic_regression")
    ].copy()
    internal_hgb = preds[
        preds["train_source"].eq("MIMIC-IV")
        & preds["test_target"].eq("MIMIC-IV")
        & preds["feature_set"].eq("base")
        & preds["model"].eq("hist_gradient_boosting")
    ][["id_norm", "prediction"]].rename(columns={"prediction": "mimic_internal_hgb"})
    internal_logit = preds[
        preds["train_source"].eq("MIMIC-IV")
        & preds["test_target"].eq("MIMIC-IV")
        & preds["feature_set"].eq("base")
        & preds["model"].eq("logistic_regression")
    ][["id_norm", "prediction"]].rename(columns={"prediction": "mimic_internal_logistic"})

    raw = raw.rename(columns={"prediction": "nhanes_raw"})
    merged = raw[["id_norm", "outcome", "nhanes_raw"]].merge(internal_hgb, on="id_norm", how="inner").merge(internal_logit, on="id_norm", how="inner")
    y = merged["outcome"].astype(int).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    cal_idx = sample_calibration_indices(y, EVENT_TARGET, rng)
    eval_mask = np.ones(len(merged), dtype=bool)
    eval_mask[cal_idx] = False

    platt = LogisticRegression(max_iter=1000)
    platt.fit(logit(merged.loc[cal_idx, "nhanes_raw"].to_numpy()).reshape(-1, 1), y[cal_idx])
    eval_df = merged.loc[eval_mask].copy()
    eval_df["nhanes_platt_100_events"] = platt.predict_proba(logit(eval_df["nhanes_raw"].to_numpy()).reshape(-1, 1))[:, 1]
    metadata = {
        "calibration_n": int(len(cal_idx)),
        "calibration_events": int(y[cal_idx].sum()),
        "evaluation_n": int(len(eval_df)),
        "evaluation_events": int(eval_df["outcome"].sum()),
        "evaluation_event_rate": float(eval_df["outcome"].mean()),
        "event_target": EVENT_TARGET,
    }
    return eval_df, metadata


def decision_curve(eval_df: pd.DataFrame) -> pd.DataFrame:
    y = eval_df["outcome"].astype(int).to_numpy()
    event_rate = float(np.mean(y))
    models = {
        "NHANES raw logistic": eval_df["nhanes_raw"].to_numpy(),
        "NHANES Platt 100 events": eval_df["nhanes_platt_100_events"].to_numpy(),
        "MIMIC internal HGB": eval_df["mimic_internal_hgb"].to_numpy(),
        "MIMIC internal logistic": eval_df["mimic_internal_logistic"].to_numpy(),
    }
    rows = []
    for threshold in THRESHOLDS:
        rows.append({"model": "Treat none", "threshold": threshold, "net_benefit": 0.0})
        rows.append({"model": "Treat all", "threshold": threshold, "net_benefit": event_rate - (1 - event_rate) * threshold / (1 - threshold)})
        for model, pred in models.items():
            rows.append({"model": model, "threshold": threshold, "net_benefit": net_benefit(y, pred, threshold)})
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    cols = ["model", "threshold", "net_benefit"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df[cols].iterrows():
        lines.append(f"| {row['model']} | {row['threshold']:.2f} | {row['net_benefit']:.4f} |")
    return "\n".join(lines)


def write_readme(curve: pd.DataFrame, metadata: dict) -> None:
    focus = curve[curve["threshold"].isin([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])].copy()
    text = f"""# Decision Curve Analysis for Transportability

Generated on 2026-06-08.

## Purpose

This analysis evaluates clinical net benefit across risk thresholds for one-year mortality prediction in the MIMIC-IV target setting.

Compared strategies:

- Treat none
- Treat all
- NHANES raw logistic transported to MIMIC-IV
- NHANES logistic after Platt recalibration with 100 local MIMIC events
- MIMIC internal HistGradientBoosting
- MIMIC internal logistic regression

Local recalibration setup:

- Calibration n: {metadata['calibration_n']}
- Calibration events: {metadata['calibration_events']}
- Evaluation n: {metadata['evaluation_n']}
- Evaluation events: {metadata['evaluation_events']}
- Evaluation event rate: {metadata['evaluation_event_rate']:.4f}

## Selected Thresholds

{markdown_table(focus)}

## Interpretation

Decision curves translate discrimination and calibration into threshold-specific clinical utility. The recalibrated NHANES model should be interpreted as a deployment simulation, not as an independent external validation result.

## Outputs

```text
outputs\\decision_curve\\decision_curve_transportability.csv
outputs\\decision_curve\\decision_curve_eval_predictions.csv
outputs\\decision_curve\\decision_curve_metadata.json
outputs\\decision_curve\\README_decision_curve_transportability.md
```
"""
    (OUT_DIR / "README_decision_curve_transportability.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_df, metadata = build_eval_predictions()
    curve = decision_curve(eval_df)
    eval_df.to_csv(OUT_DIR / "decision_curve_eval_predictions.csv", index=False)
    curve.to_csv(OUT_DIR / "decision_curve_transportability.csv", index=False)
    metadata.update(
        {
            "prediction_path": str(PRED_PATH),
            "thresholds": [float(x) for x in THRESHOLDS],
            "models": curve["model"].drop_duplicates().tolist(),
            "notes": "Platt recalibration is fit only on the local calibration subset and evaluated on held-out MIMIC target admissions.",
        }
    )
    (OUT_DIR / "decision_curve_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(curve, metadata)
    print(curve.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
