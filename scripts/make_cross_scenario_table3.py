from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NHANES_MIMIC_PRED = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "transportability_predictions.csv"
ICU_PRED = PROJECT_ROOT / "outputs" / "mimic_icu_eicu_transport_recalibration" / "transport_predictions.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "manuscript_tables"

N_BOOTSTRAP = 1000
RANDOM_SEED = 20260611
EPS = 1e-6

METRICS = [
    "event_rate",
    "mean_prediction",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "ece_10bin",
    "calibration_slope",
    "calibration_intercept",
]


def inv_logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expected_calibration_error(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> float:
    if len(y) == 0:
        return np.nan
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


def calibration_regression(y: np.ndarray, pred: np.ndarray) -> dict[str, float | bool]:
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return {
            "calibration_slope": np.nan,
            "calibration_intercept": np.nan,
            "calibration_converged": False,
        }
    x = logit(pred)
    design = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0], dtype=float)
    converged = False
    for _ in range(50):
        eta = design @ beta
        prob = inv_logit(eta)
        weight = np.clip(prob * (1 - prob), 1e-9, None)
        gradient = design.T @ (prob - y)
        hessian = design.T @ (design * weight[:, None])
        hessian += np.eye(2) * 1e-7
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        beta -= step
        if not np.all(np.isfinite(beta)):
            break
        if float(np.max(np.abs(step))) < 1e-7:
            converged = True
            break
    return {
        "calibration_intercept": float(beta[0]) if np.all(np.isfinite(beta)) else np.nan,
        "calibration_slope": float(beta[1]) if np.all(np.isfinite(beta)) else np.nan,
        "calibration_converged": bool(converged),
    }


def point_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    pred = np.clip(np.asarray(pred, dtype=float), EPS, 1 - EPS)
    has_two_classes = len(np.unique(y)) == 2
    cal = calibration_regression(y, pred)
    return {
        "event_rate": float(y.mean()),
        "mean_prediction": float(pred.mean()),
        "roc_auc": float(roc_auc_score(y, pred)) if has_two_classes else np.nan,
        "pr_auc": float(average_precision_score(y, pred)) if has_two_classes else np.nan,
        "brier_score": float(brier_score_loss(y, pred)),
        "ece_10bin": expected_calibration_error(y, pred),
        "calibration_slope": float(cal["calibration_slope"]),
        "calibration_intercept": float(cal["calibration_intercept"]),
    }


def load_primary_transport_sets() -> list[dict[str, object]]:
    nhanes_preds = pd.read_csv(NHANES_MIMIC_PRED)
    nhanes_to_mimic = nhanes_preds[
        nhanes_preds["train_source"].eq("NHANES")
        & nhanes_preds["test_target"].eq("MIMIC-IV")
        & nhanes_preds["feature_set"].eq("base")
        & nhanes_preds["model"].eq("logistic_regression")
    ].copy()

    icu_preds = pd.read_csv(ICU_PRED)
    mimic_to_eicu = icu_preds[
        icu_preds["train_source"].eq("MIMIC-IV ICU")
        & icu_preds["test_target"].eq("eICU")
        & icu_preds["test_type"].eq("transport_test")
        & icu_preds["feature_set"].eq("icu_native_primary")
        & icu_preds["model"].eq("logistic_regression")
    ].copy()
    eicu_to_mimic = icu_preds[
        icu_preds["train_source"].eq("eICU")
        & icu_preds["test_target"].eq("MIMIC-IV ICU")
        & icu_preds["test_type"].eq("transport_test")
        & icu_preds["feature_set"].eq("icu_native_primary")
        & icu_preds["model"].eq("logistic_regression")
    ].copy()

    return [
        {
            "scenario": "Extreme cross-setting stress test",
            "direction": "NHANES -> MIMIC-IV",
            "endpoint": "1-year mortality",
            "endpoint_note": "NHANES death within 12 months; MIMIC-IV death within 1 year after discharge.",
            "feature_set": "NHANES-compatible base",
            "model": "Logistic regression",
            "data": nhanes_to_mimic,
        },
        {
            "scenario": "Realistic ICU deployment",
            "direction": "MIMIC-IV ICU -> eICU",
            "endpoint": "Hospital mortality",
            "endpoint_note": "MIMIC-IV hospital_expire_flag; eICU hospitaldischargestatus == Expired.",
            "feature_set": "ICU-native primary",
            "model": "Logistic regression",
            "data": mimic_to_eicu,
        },
        {
            "scenario": "Realistic ICU deployment",
            "direction": "eICU -> MIMIC-IV ICU",
            "endpoint": "Hospital mortality",
            "endpoint_note": "eICU hospitaldischargestatus == Expired; MIMIC-IV hospital_expire_flag.",
            "feature_set": "ICU-native primary",
            "model": "Logistic regression",
            "data": eicu_to_mimic,
        },
    ]


def bootstrap_scenario(item: dict[str, object], rng: np.random.Generator) -> tuple[dict[str, object], list[dict[str, object]]]:
    df = item["data"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Scenario data must be a DataFrame")
    y = df["outcome"].astype(int).to_numpy()
    pred = df["prediction"].to_numpy()
    n = len(df)
    point = point_metrics(y, pred)
    id_cols = {
        "scenario": item["scenario"],
        "direction": item["direction"],
        "endpoint": item["endpoint"],
        "endpoint_note": item["endpoint_note"],
        "feature_set": item["feature_set"],
        "model": item["model"],
        "n": int(n),
        "events": int(y.sum()),
    }

    long_rows = []
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        row = dict(id_cols)
        row["bootstrap"] = i
        row.update(point_metrics(y[idx], pred[idx]))
        long_rows.append(row)

    boot = pd.DataFrame(long_rows)
    summary = dict(id_cols)
    for metric in METRICS:
        vals = pd.to_numeric(boot[metric], errors="coerce").dropna().to_numpy()
        summary[f"{metric}_point"] = point[metric]
        summary[f"{metric}_ci_lower"] = float(np.quantile(vals, 0.025)) if len(vals) else np.nan
        summary[f"{metric}_ci_upper"] = float(np.quantile(vals, 0.975)) if len(vals) else np.nan
        summary[f"{metric}_bootstrap_n_valid"] = int(len(vals))
    return summary, long_rows


def fmt_estimate(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_point']:.3f} ({row[f'{metric}_ci_lower']:.3f}-{row[f'{metric}_ci_upper']:.3f})"


def make_formatted_table(numeric: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in numeric.iterrows():
        rows.append(
            {
                "Scenario": row["scenario"],
                "Direction": row["direction"],
                "Endpoint": row["endpoint"],
                "Feature set": row["feature_set"],
                "Model": row["model"],
                "N": int(row["n"]),
                "Events": int(row["events"]),
                "Event rate": fmt_estimate(row, "event_rate"),
                "Mean predicted risk": fmt_estimate(row, "mean_prediction"),
                "AUC": fmt_estimate(row, "roc_auc"),
                "PR AUC": fmt_estimate(row, "pr_auc"),
                "Brier score": fmt_estimate(row, "brier_score"),
                "ECE": fmt_estimate(row, "ece_10bin"),
                "Calibration slope": fmt_estimate(row, "calibration_slope"),
                "Calibration intercept": fmt_estimate(row, "calibration_intercept"),
            }
        )
    return pd.DataFrame(rows)


def write_readme(formatted: pd.DataFrame) -> None:
    cols = list(formatted.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")

    text = f"""# Table 3 Cross-Scenario Transport Performance

Generated on 2026-06-11.

This table fixes the manuscript's core transport-performance comparison across the two planned transport scenarios.

Important endpoint note: the NHANES -> MIMIC-IV stress-test uses a one-year mortality endpoint, whereas the MIMIC-IV ICU <-> eICU deployment analyses use harmonized hospital mortality. These rows should be compared as endpoint-specific deployment scenarios; the cross-scenario contrast is intended to compare transport failure modes and recalibration needs, not endpoint-equivalent absolute accuracy.

All displayed intervals are percentile 95% confidence intervals from {N_BOOTSTRAP} bootstrap resamples of the target/evaluation cohort.

{chr(10).join(lines)}

## Outputs

```text
outputs\\manuscript_tables\\Table_3_cross_scenario_transport_performance.csv
outputs\\manuscript_tables\\Table_3_cross_scenario_transport_performance_numeric.csv
outputs\\manuscript_tables\\Table_3_cross_scenario_transport_bootstrap_long.csv
outputs\\manuscript_tables\\Table_3_cross_scenario_transport_metadata.json
outputs\\manuscript_tables\\README_Table_3_cross_scenario_transport_performance.md
```
"""
    (OUT_DIR / "README_Table_3_cross_scenario_transport_performance.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    summaries = []
    all_long = []
    for item in load_primary_transport_sets():
        print(f"Bootstrapping {item['direction']}...")
        summary, long_rows = bootstrap_scenario(item, rng)
        summaries.append(summary)
        all_long.extend(long_rows)

    numeric = pd.DataFrame(summaries)
    formatted = make_formatted_table(numeric)
    long = pd.DataFrame(all_long)

    formatted.to_csv(OUT_DIR / "Table_3_cross_scenario_transport_performance.csv", index=False)
    numeric.to_csv(OUT_DIR / "Table_3_cross_scenario_transport_performance_numeric.csv", index=False)
    long.to_csv(OUT_DIR / "Table_3_cross_scenario_transport_bootstrap_long.csv", index=False)
    metadata = {
        "generated_on": "2026-06-11",
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "metrics": METRICS,
        "ci": "Percentile 2.5% and 97.5% bootstrap intervals.",
        "prediction_sources": {
            "NHANES_to_MIMIC": str(NHANES_MIMIC_PRED),
            "MIMIC_ICU_eICU": str(ICU_PRED),
        },
        "endpoint_note": "NHANES -> MIMIC-IV uses one-year mortality; MIMIC-IV ICU <-> eICU uses hospital mortality.",
    }
    (OUT_DIR / "Table_3_cross_scenario_transport_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(formatted)
    print(formatted.to_string(index=False))


if __name__ == "__main__":
    main()
