from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "manuscript_tables"

COMMON_TABLE = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "nhanes_mimic_common_transport_table.csv"
MIMIC_ENHANCED = PROJECT_ROOT / "data" / "mimic" / "processed" / "mimic_iv31_diabetes_lab_enhanced_admissions.csv"
SMD_PATH = PROJECT_ROOT / "outputs" / "domain_shift" / "common_feature_smd.csv"
SOURCE_SUMMARY = PROJECT_ROOT / "outputs" / "domain_shift" / "source_classifier_roc_summary.csv"
TRANSPORT_CI = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "transportability_bootstrap_ci.csv"
CAL_SLOPE_INTERCEPT = PROJECT_ROOT / "outputs" / "calibration_slope_intercept" / "transport_calibration_slope_intercept.csv"
RECAL_EVENT = PROJECT_ROOT / "outputs" / "local_recalibration_by_event_count" / "local_recalibration_by_event_summary.csv"
SUBGROUP = PROJECT_ROOT / "outputs" / "subgroup_transportability" / "subgroup_transportability_metrics.csv"
DCA = PROJECT_ROOT / "outputs" / "decision_curve" / "decision_curve_transportability.csv"


CONTINUOUS_FEATURES = [
    ("age", "Age, years"),
    ("bmi", "BMI, kg/m2"),
    ("systolic_bp", "Systolic BP, mmHg"),
    ("diastolic_bp", "Diastolic BP, mmHg"),
    ("hba1c", "HbA1c"),
    ("glucose", "Glucose"),
    ("creatinine", "Creatinine"),
]

BINARY_FEATURES = [
    ("female", "Female"),
    ("hypertension_history", "Hypertension history"),
    ("ckd_history", "CKD history"),
    ("cvd_history", "CVD history"),
    ("outcome", "One-year mortality"),
]

SMD_LABELS = {
    "age": "Age, years",
    "female": "Female",
    "bmi": "BMI, kg/m2",
    "systolic_bp": "Systolic BP, mmHg",
    "diastolic_bp": "Diastolic BP, mmHg",
    "hypertension_history": "Hypertension history",
    "ckd_history": "CKD history",
    "cvd_history": "CVD history",
}


def fmt_mean_sd(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    return f"{x.mean():.2f} ({x.std(ddof=1):.2f})"


def fmt_n_pct(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    n = int(x.sum())
    pct = 100 * float(x.mean())
    return f"{n} ({pct:.1f}%)"


def load_common_table() -> pd.DataFrame:
    df = pd.read_csv(COMMON_TABLE, low_memory=False)
    return df


def load_mimic_patient_level() -> pd.DataFrame:
    mimic = pd.read_csv(MIMIC_ENHANCED, low_memory=False)
    mimic["admittime_dt"] = pd.to_datetime(mimic["admittime"], errors="coerce")
    mimic = mimic.sort_values(["subject_id", "admittime_dt", "hadm_id"]).copy()
    first = mimic.groupby("subject_id", as_index=False).head(1).copy()
    out = pd.DataFrame(
        {
            "source": "MIMIC-IV first admission",
            "id": first["subject_id"].astype(str),
            "age": pd.to_numeric(first["age"], errors="coerce"),
            "female": pd.to_numeric(first["female"], errors="coerce"),
            "bmi": pd.to_numeric(first["bmi"], errors="coerce"),
            "systolic_bp": pd.to_numeric(first["systolic_bp"], errors="coerce"),
            "diastolic_bp": pd.to_numeric(first["diastolic_bp"], errors="coerce"),
            "hypertension_history": pd.to_numeric(first["hypertension_history"], errors="coerce"),
            "ckd_history": pd.to_numeric(first["ckd_history"], errors="coerce"),
            "cvd_history": pd.to_numeric(first["cvd_history"], errors="coerce"),
            "hba1c": pd.to_numeric(first["hba1c_first_24h"], errors="coerce"),
            "glucose": pd.to_numeric(first["glucose_first_24h"], errors="coerce"),
            "creatinine": pd.to_numeric(first["creatinine_first_24h"], errors="coerce"),
            "outcome": pd.to_numeric(first["death_within_1y_discharge"], errors="coerce"),
        }
    )
    return out


def baseline_characteristics() -> pd.DataFrame:
    common = load_common_table()
    nhanes = common[common["source"].eq("NHANES") & common["known_outcome"].astype(str).str.lower().eq("true")].copy()
    mimic_adm = common[common["source"].eq("MIMIC-IV")].copy()
    mimic_patient = load_mimic_patient_level()
    cohorts = {
        "NHANES participants": nhanes,
        "MIMIC-IV admissions": mimic_adm,
        "MIMIC-IV first admissions": mimic_patient,
    }
    rows = []
    for variable, label in CONTINUOUS_FEATURES:
        row = {"variable": label, "summary_type": "mean_sd"}
        for cohort, df in cohorts.items():
            row[cohort] = fmt_mean_sd(df[variable]) if variable in df else "NA"
            row[f"{cohort} nonmissing"] = int(pd.to_numeric(df[variable], errors="coerce").notna().sum()) if variable in df else 0
        rows.append(row)
    for variable, label in BINARY_FEATURES:
        row = {"variable": label, "summary_type": "n_percent"}
        for cohort, df in cohorts.items():
            row[cohort] = fmt_n_pct(df[variable]) if variable in df else "NA"
            row[f"{cohort} nonmissing"] = int(pd.to_numeric(df[variable], errors="coerce").notna().sum()) if variable in df else 0
        rows.append(row)
    rows.insert(
        0,
        {
            "variable": "N",
            "summary_type": "count",
            "NHANES participants": str(len(nhanes)),
            "NHANES participants nonmissing": len(nhanes),
            "MIMIC-IV admissions": str(len(mimic_adm)),
            "MIMIC-IV admissions nonmissing": len(mimic_adm),
            "MIMIC-IV first admissions": str(len(mimic_patient)),
            "MIMIC-IV first admissions nonmissing": len(mimic_patient),
        },
    )
    return pd.DataFrame(rows)


def domain_shift_table() -> pd.DataFrame:
    smd = pd.read_csv(SMD_PATH)
    source = pd.read_csv(SOURCE_SUMMARY)
    smd_table = smd.copy()
    smd_table["feature_label"] = smd_table["feature"].map(SMD_LABELS).fillna(smd_table["feature"])
    smd_table = smd_table[
        [
            "feature_label",
            "type",
            "nhanes_nonmissing",
            "mimic_nonmissing",
            "nhanes_mean_or_prop",
            "mimic_mean_or_prop",
            "smd_mimic_minus_nhanes",
        ]
    ].sort_values("smd_mimic_minus_nhanes", key=lambda s: s.abs(), ascending=False)
    source_table = source[["feature_set", "model", "roc_auc", "n_balanced", "n_per_source"]].copy()
    source_table["features"] = source["features"]
    return smd_table, source_table


def transportability_table() -> pd.DataFrame:
    ci = pd.read_csv(TRANSPORT_CI)
    wanted_metrics = ["roc_auc", "pr_auc", "brier_score", "ece_10bin", "mean_prediction", "event_rate"]
    ci = ci[ci["metric"].isin(wanted_metrics)].copy()
    ci["estimate_95ci"] = ci.apply(lambda r: f"{r['point']:.3f} ({r['ci_lower']:.3f}-{r['ci_upper']:.3f})", axis=1)
    wide = ci.pivot_table(
        index=["train_source", "test_target", "feature_set", "model", "n", "events"],
        columns="metric",
        values="estimate_95ci",
        aggfunc="first",
    ).reset_index()
    cal = pd.read_csv(CAL_SLOPE_INTERCEPT)
    cal = cal[cal["recalibration_method"].eq("raw_model_prediction")].copy()
    cal["calibration_slope"] = cal.apply(
        lambda r: f"{r['calibration_slope']:.3f} ({r['calibration_slope_ci_lower']:.3f}-{r['calibration_slope_ci_upper']:.3f})",
        axis=1,
    )
    cal["calibration_intercept"] = cal.apply(
        lambda r: f"{r['calibration_intercept']:.3f} ({r['calibration_intercept_ci_lower']:.3f}-{r['calibration_intercept_ci_upper']:.3f})",
        axis=1,
    )
    cal_cols = ["train_source", "test_target", "feature_set", "model", "calibration_slope", "calibration_intercept"]
    wide = wide.merge(cal[cal_cols], on=["train_source", "test_target", "feature_set", "model"], how="left")
    ordered = [
        "train_source",
        "test_target",
        "feature_set",
        "model",
        "n",
        "events",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "ece_10bin",
        "calibration_slope",
        "calibration_intercept",
        "mean_prediction",
        "event_rate",
    ]
    return wide[[c for c in ordered if c in wide.columns]]


def recalibration_table() -> pd.DataFrame:
    rec = pd.read_csv(RECAL_EVENT)
    focus = rec[
        rec["prediction_set"].eq("NHANES_to_MIMIC_1y_base_logistic_regression")
        & rec["method"].isin(["raw", "intercept_only", "platt", "isotonic"])
        & rec["event_target"].isin([0, 25, 50, 100, 200, 500, 1000])
    ].copy()
    cols = [
        "prediction_set",
        "method",
        "event_target",
        "calibration_n_mean",
        "calibration_events_mean",
        "roc_auc_mean",
        "pr_auc_mean",
        "brier_score_mean",
        "ece_10bin_mean",
        "mean_prediction_mean",
        "event_rate_mean",
        "calibration_slope_mean",
        "calibration_intercept_mean",
    ]
    return focus[cols].sort_values(["method", "event_target"])


def subgroup_table() -> pd.DataFrame:
    sub = pd.read_csv(SUBGROUP)
    focus = sub[
        sub["train_source"].eq("NHANES")
        & sub["test_target"].eq("MIMIC-IV")
        & sub["feature_set"].eq("base")
        & sub["model"].eq("logistic_regression")
    ].copy()
    cols = [
        "subgroup_type",
        "subgroup",
        "n",
        "events",
        "event_rate",
        "mean_prediction",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "ece_10bin",
    ]
    return focus[cols].sort_values(["subgroup_type", "subgroup"])


def dca_table() -> pd.DataFrame:
    dca = pd.read_csv(DCA)
    selected = dca[dca["threshold"].isin([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])].copy()
    wide = selected.pivot_table(index="threshold", columns="model", values="net_benefit", aggfunc="first").reset_index()
    order = ["threshold", "Treat none", "Treat all", "NHANES raw logistic", "NHANES Platt 100 events", "MIMIC internal HGB", "MIMIC internal logistic"]
    return wide[[c for c in order if c in wide.columns]]


def write_readme(tables: dict[str, pd.DataFrame]) -> None:
    text = """# Manuscript Tables

Generated on 2026-06-08.

These tables support the dual-database NHANES-MIMIC manuscript concept:

> Transported diabetes mortality models retain moderate discrimination but show severe calibration drift; local Platt recalibration with approximately 100 target-site outcome events can restore calibration and positive clinical net benefit.

## Table Files

- `Table_1_baseline_characteristics.csv`: cohort characteristics. Main MIMIC analysis is admission-level; first-admission patient-level characteristics are included to address repeated admissions.
- `Table_2a_common_feature_smd.csv`: NHANES vs MIMIC common-feature standardized mean differences.
- `Table_2b_source_classifier.csv`: source-classifier AUCs quantifying domain/measurement-process shift.
- `Table_3_transportability_metrics_ci.csv`: one-year mortality transportability metrics with bootstrap 95% confidence intervals plus raw calibration-regression slope/intercept.
- `Table_4_recalibration_by_event_count.csv`: local recalibration by number of target-site events.
- `Table_4b_recalibration_uncertainty_intervals.csv`: empirical uncertainty intervals across 200 repeated local calibration samples, including recalibration slope/intercept recovery.
- `Table_5_subgroup_transportability.csv`: subgroup transportability for NHANES-to-MIMIC base logistic transport.
- `Table_6_decision_curve_selected_thresholds.csv`: selected decision-curve net benefit thresholds.

## Analysis Unit Note

NHANES is participant-level. MIMIC-IV primary analyses are admission-level because the transportability target is hospital admissions, and one patient may contribute multiple admissions. Table 1 includes a first-admission patient-level MIMIC column as a descriptive sensitivity view.
"""
    (OUT_DIR / "README_manuscript_tables.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}
    tables["Table_1_baseline_characteristics"] = baseline_characteristics()
    smd, source = domain_shift_table()
    tables["Table_2a_common_feature_smd"] = smd
    tables["Table_2b_source_classifier"] = source
    tables["Table_3_transportability_metrics_ci"] = transportability_table()
    tables["Table_4_recalibration_by_event_count"] = recalibration_table()
    tables["Table_5_subgroup_transportability"] = subgroup_table()
    tables["Table_6_decision_curve_selected_thresholds"] = dca_table()

    for name, table in tables.items():
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)

    metadata = {
        "generated_on": "2026-06-08",
        "analysis_unit": {
            "NHANES": "participant-level",
            "MIMIC-IV primary": "admission-level",
            "MIMIC-IV sensitivity/descriptive": "first admission per subject",
        },
        "tables": list(tables.keys()),
    }
    (OUT_DIR / "manuscript_tables_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(tables)
    for name, table in tables.items():
        print(f"\n{name}")
        print(table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
