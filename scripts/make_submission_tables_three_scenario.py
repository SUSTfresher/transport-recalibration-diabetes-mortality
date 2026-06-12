from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "manuscript_tables"

NHANES_COHORT = (
    PROJECT_ROOT
    / "data"
    / "nhanes"
    / "processed"
    / "nhanes_2005_2018_diabetes_ckd_mortality_scan.csv"
)
NHANES_SOURCE_CLASSIFIER = PROJECT_ROOT / "outputs" / "domain_shift" / "source_classifier_roc_summary.csv"
NHANES_RECAL = (
    PROJECT_ROOT
    / "outputs"
    / "local_recalibration_uncertainty"
    / "recalibration_uncertainty_summary.csv"
)
NHANES_SUBGROUP = PROJECT_ROOT / "outputs" / "subgroup_transportability" / "subgroup_transportability_metrics.csv"
NHANES_DCA = PROJECT_ROOT / "outputs" / "decision_curve" / "decision_curve_transportability.csv"

MIMIC_ICU_COHORT = (
    PROJECT_ROOT
    / "data"
    / "mimic"
    / "processed"
    / "mimic_iv31_diabetes_icu_lab_vital_enhanced_cohort.csv"
)
EICU_COHORT = (
    PROJECT_ROOT
    / "data"
    / "eicu"
    / "processed"
    / "eicu_crd20_diabetes_lab_vital_enhanced_cohort.csv"
)
ICU_SOURCE_CLASSIFIER = (
    PROJECT_ROOT
    / "outputs"
    / "mimic_icu_eicu_source_classifier"
    / "mimic_icu_eicu_source_classifier_summary.csv"
)
ICU_RECAL = (
    PROJECT_ROOT
    / "outputs"
    / "mimic_icu_eicu_transport_recalibration"
    / "event_count_recalibration_summary.csv"
)
ICU_SUBGROUP = (
    PROJECT_ROOT
    / "outputs"
    / "mimic_icu_eicu_transport_recalibration"
    / "subgroup_transport_metrics.csv"
)
ICU_DCA = (
    PROJECT_ROOT
    / "outputs"
    / "mimic_icu_eicu_transport_recalibration"
    / "decision_curve_selected_thresholds.csv"
)


TABLE1_VARIABLES = [
    ("age", "Age, years", "continuous"),
    ("female", "Female", "binary"),
    ("bmi", "BMI, kg/m2", "continuous"),
    ("hypertension_history", "Hypertension history", "binary"),
    ("ckd_history", "CKD history", "binary"),
    ("cvd_history", "CVD history", "binary"),
    ("systolic_bp", "Systolic BP, mmHg", "continuous"),
    ("diastolic_bp", "Diastolic BP, mmHg", "continuous"),
    ("glucose", "Glucose, mg/dL", "continuous"),
    ("creatinine", "Creatinine, mg/dL", "continuous"),
    ("bun", "BUN, mg/dL", "continuous"),
    ("wbc", "WBC, 10^9/L", "continuous"),
    ("hemoglobin", "Hemoglobin, g/dL", "continuous"),
    ("heart_rate", "Heart rate, beats/min", "continuous"),
    ("spo2", "SpO2, %", "continuous"),
    ("albumin", "Albumin, g/dL", "continuous"),
    ("primary_outcome", "Primary endpoint event", "binary_no_smd"),
]

DIRECTION_ORDER = [
    "NHANES -> MIMIC-IV",
    "MIMIC-IV ICU -> eICU",
    "eICU -> MIMIC-IV ICU",
]


def as_bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x
    return x.astype(str).str.lower().isin(["true", "1", "yes"])


def numeric(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def fmt_mean_sd(x: pd.Series) -> str:
    x = numeric(x).dropna()
    if len(x) == 0:
        return "NA"
    return f"{x.mean():.2f} ({x.std(ddof=1):.2f})"


def fmt_n_pct(x: pd.Series) -> str:
    x = numeric(x).dropna()
    if len(x) == 0:
        return "NA"
    return f"{int(x.sum())} ({100 * x.mean():.1f}%)"


def fmt_mean_sd_with_n(x: pd.Series) -> str:
    x = numeric(x).dropna()
    if len(x) == 0:
        return "Not available"
    return f"{x.mean():.2f} ({x.std(ddof=1):.2f}); n={len(x)}"


def fmt_n_pct_with_denom(x: pd.Series) -> str:
    x = numeric(x).dropna()
    if len(x) == 0:
        return "Not available"
    return f"{int(x.sum())}/{len(x)} ({100 * x.mean():.1f}%)"


def fmt_decimal(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def fmt_ci(mean: float, lo: float | None, hi: float | None, digits: int = 3) -> str:
    if pd.isna(mean):
        return "NA"
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return fmt_decimal(mean, digits)
    if np.isclose(float(mean), float(lo)) and np.isclose(float(mean), float(hi)):
        return fmt_decimal(mean, digits)
    return f"{float(mean):.{digits}f} ({float(lo):.{digits}f}-{float(hi):.{digits}f})"


def smd_continuous(a: pd.Series, b: pd.Series) -> float:
    a = numeric(a).dropna()
    b = numeric(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled_sd == 0 or pd.isna(pooled_sd):
        return np.nan
    return abs(float((a.mean() - b.mean()) / pooled_sd))


def smd_binary(a: pd.Series, b: pd.Series) -> float:
    a = numeric(a).dropna()
    b = numeric(b).dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    p1 = float(a.mean())
    p2 = float(b.mean())
    pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    if pooled == 0 or pd.isna(pooled):
        return np.nan
    return abs((p1 - p2) / pooled)


def markdown_table(df: pd.DataFrame) -> str:
    display = df.fillna("NA").astype(str)
    cols = list(display.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(row[col] for col in cols) + " |")
    return "\n".join(lines)


def write_table(name: str, df: pd.DataFrame, notes: Iterable[str]) -> None:
    csv_path = OUT_DIR / f"{name}.csv"
    md_path = OUT_DIR / f"{name}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    note_text = "\n".join(f"- {note}" for note in notes)
    md_path.write_text(
        f"# {name}\n\n{markdown_table(df)}\n\n## Notes\n\n{note_text}\n",
        encoding="utf-8",
    )


def load_nhanes_known() -> pd.DataFrame:
    nhanes = pd.read_csv(NHANES_COHORT, low_memory=False)
    nhanes = nhanes[numeric(nhanes["diabetes"]).eq(1)].copy()
    months = numeric(nhanes["mortality_months"])
    died = as_bool_series(nhanes["all_cause_death"])
    outcome = (died & months.le(12)).astype(int)
    known_outcome = (months.ge(12) | outcome.eq(1)) & months.notna()
    nhanes = nhanes[known_outcome].copy()
    outcome = outcome[known_outcome]
    return pd.DataFrame(
        {
            "age": numeric(nhanes["RIDAGEYR"]),
            "female": numeric(nhanes["RIAGENDR"]).eq(2).astype(int),
            "bmi": numeric(nhanes["BMXBMI"]),
            "hypertension_history": numeric(nhanes["hypertension_history"]),
            "ckd_history": as_bool_series(nhanes["ckd_egfr_or_uacr"]).astype(int),
            "cvd_history": numeric(nhanes["cvd_history"]),
            "systolic_bp": numeric(nhanes["systolic_bp"]),
            "diastolic_bp": numeric(nhanes["diastolic_bp"]),
            "glucose": numeric(nhanes["LBXGLU"]),
            "creatinine": numeric(nhanes["LBXSCR"]),
            "primary_outcome": numeric(outcome),
        }
    )


def load_mimic_icu() -> pd.DataFrame:
    mimic = pd.read_csv(MIMIC_ICU_COHORT, low_memory=False)
    mimic = mimic[numeric(mimic["hospital_mortality"]).notna()].copy()
    return pd.DataFrame(
        {
            "age": numeric(mimic["age"]),
            "female": numeric(mimic["female"]),
            "bmi": numeric(mimic["bmi"]),
            "hypertension_history": numeric(mimic["hypertension_history"]),
            "ckd_history": numeric(mimic["ckd_history"]),
            "cvd_history": numeric(mimic["cvd_history"]),
            "systolic_bp": numeric(mimic["systolic_bp_first_24h"]),
            "diastolic_bp": numeric(mimic["diastolic_bp_first_24h"]),
            "glucose": numeric(mimic["glucose_first_24h"]),
            "creatinine": numeric(mimic["creatinine_first_24h"]),
            "bun": numeric(mimic["bun_first_24h"]),
            "wbc": numeric(mimic["wbc_first_24h"]),
            "hemoglobin": numeric(mimic["hemoglobin_first_24h"]),
            "heart_rate": numeric(mimic["heart_rate_first_24h"]),
            "spo2": numeric(mimic["spo2_first_24h"]),
            "albumin": numeric(mimic["albumin_first_24h"]),
            "primary_outcome": numeric(mimic["hospital_mortality"]),
        }
    )


def load_eicu_known() -> pd.DataFrame:
    eicu = pd.read_csv(EICU_COHORT, low_memory=False)
    eicu = eicu[numeric(eicu["hospital_mortality"]).notna()].copy()
    return pd.DataFrame(
        {
            "age": numeric(eicu["age"]),
            "female": numeric(eicu["female"]),
            "bmi": numeric(eicu["bmi"]),
            "hypertension_history": numeric(eicu["hypertension_history"]),
            "ckd_history": numeric(eicu["ckd_history"]),
            "cvd_history": numeric(eicu["cvd_history"]),
            "systolic_bp": numeric(eicu["systolic_bp_first_24h"]),
            "diastolic_bp": numeric(eicu["diastolic_bp_first_24h"]),
            "glucose": numeric(eicu["glucose_first_24h"]),
            "creatinine": numeric(eicu["creatinine_first_24h"]),
            "bun": numeric(eicu["bun_first_24h"]),
            "wbc": numeric(eicu["wbc_first_24h"]),
            "hemoglobin": numeric(eicu["hemoglobin_first_24h"]),
            "heart_rate": numeric(eicu["heart_rate_first_24h"]),
            "spo2": numeric(eicu["spo2_first_24h"]),
            "albumin": numeric(eicu["albumin_first_24h"]),
            "primary_outcome": numeric(eicu["hospital_mortality"]),
        }
    )


def build_table1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohorts = {
        "NHANES": load_nhanes_known(),
        "MIMIC-IV ICU": load_mimic_icu(),
        "eICU": load_eicu_known(),
    }

    detailed_rows = [
        {
            "Characteristic": "N",
            "Summary": "count",
            "NHANES": str(len(cohorts["NHANES"])),
            "NHANES nonmissing": len(cohorts["NHANES"]),
            "MIMIC-IV ICU": str(len(cohorts["MIMIC-IV ICU"])),
            "MIMIC-IV ICU nonmissing": len(cohorts["MIMIC-IV ICU"]),
            "eICU": str(len(cohorts["eICU"])),
            "eICU nonmissing": len(cohorts["eICU"]),
            "SMD NHANES vs MIMIC-IV ICU": "",
            "SMD NHANES vs eICU": "",
            "SMD MIMIC-IV ICU vs eICU": "",
            "Max absolute SMD": "",
        }
    ]
    main_rows = [
        {
            "Characteristic": "N",
            "NHANES": str(len(cohorts["NHANES"])),
            "MIMIC-IV ICU": str(len(cohorts["MIMIC-IV ICU"])),
            "eICU": str(len(cohorts["eICU"])),
            "Max absolute SMD": "",
        }
    ]
    numeric_rows = []
    pairs = [
        ("NHANES", "MIMIC-IV ICU", "SMD NHANES vs MIMIC-IV ICU"),
        ("NHANES", "eICU", "SMD NHANES vs eICU"),
        ("MIMIC-IV ICU", "eICU", "SMD MIMIC-IV ICU vs eICU"),
    ]

    for key, label, kind in TABLE1_VARIABLES:
        row = {"Characteristic": label, "Summary": "mean (SD)" if kind == "continuous" else "n (%)"}
        main_row = {"Characteristic": label}
        raw = {"Characteristic": label, "variable": key, "type": kind}
        for cohort_name, df in cohorts.items():
            values = df[key] if key in df else pd.Series(dtype=float)
            row[cohort_name] = fmt_mean_sd(values) if kind == "continuous" else fmt_n_pct(values)
            row[f"{cohort_name} nonmissing"] = int(numeric(values).notna().sum())
            main_row[cohort_name] = fmt_mean_sd_with_n(values) if kind == "continuous" else fmt_n_pct_with_denom(values)
            raw[f"{cohort_name}_mean_or_prop"] = float(numeric(values).mean()) if numeric(values).notna().any() else np.nan
            raw[f"{cohort_name}_sd"] = float(numeric(values).std(ddof=1)) if numeric(values).notna().sum() > 1 else np.nan
            raw[f"{cohort_name}_nonmissing"] = int(numeric(values).notna().sum())

        smds = []
        for left, right, col in pairs:
            if kind == "binary_no_smd":
                smd = np.nan
            elif kind == "continuous":
                smd = smd_continuous(cohorts[left].get(key, pd.Series(dtype=float)), cohorts[right].get(key, pd.Series(dtype=float)))
            else:
                smd = smd_binary(cohorts[left].get(key, pd.Series(dtype=float)), cohorts[right].get(key, pd.Series(dtype=float)))
            raw[col] = smd
            row[col] = fmt_decimal(smd, 3) if not pd.isna(smd) else ""
            if not pd.isna(smd):
                smds.append(smd)
        max_smd = max(smds) if smds else np.nan
        raw["Max absolute SMD"] = max_smd
        row["Max absolute SMD"] = fmt_decimal(max_smd, 3) if not pd.isna(max_smd) else ""
        main_row["Max absolute SMD"] = fmt_decimal(max_smd, 3) if not pd.isna(max_smd) else ""
        detailed_rows.append(row)
        main_rows.append(main_row)
        numeric_rows.append(raw)

    return pd.DataFrame(main_rows), pd.DataFrame(detailed_rows), pd.DataFrame(numeric_rows)


def clean_feature_list(value: object) -> str:
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        text = (
            text.replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
            .replace(", ", "; ")
        )
    return text


def build_table2() -> tuple[pd.DataFrame, pd.DataFrame]:
    nhanes = pd.read_csv(NHANES_SOURCE_CLASSIFIER)
    nhanes["Scenario"] = "Extreme cross-setting stress test"
    nhanes["Source contrast"] = "NHANES vs MIMIC-IV"
    nhanes["Feature block"] = nhanes["feature_set"].map(
        {
            "basic_features": "Basic",
            "lab_enhanced_features": "Basic + labs",
        }
    )
    nhanes["N balanced"] = nhanes["n_balanced"]
    nhanes["N per source"] = nhanes["n_per_source"]

    icu = pd.read_csv(ICU_SOURCE_CLASSIFIER)
    icu = icu[icu["feature_set"].ne("basic_labs_albumin_sensitivity")].copy()
    icu["Scenario"] = "Realistic ICU deployment"
    icu["Source contrast"] = "MIMIC-IV ICU vs eICU"
    icu["Feature block"] = icu["feature_set"].map(
        {
            "basic": "Basic",
            "basic_labs": "Basic + labs",
            "basic_labs_eicu_vitals_diagnostic": "Basic + labs + vitals",
        }
    )
    icu["N balanced"] = icu["n"]
    icu["N per source"] = icu["mimic_n"].astype(str) + " / " + icu["eicu_n"].astype(str)

    both = pd.concat(
        [
            nhanes[["Scenario", "Source contrast", "Feature block", "model", "roc_auc", "N balanced", "N per source", "features"]],
            icu[["Scenario", "Source contrast", "Feature block", "model", "roc_auc", "N balanced", "N per source", "features"]],
        ],
        ignore_index=True,
    )
    wide = both.pivot_table(
        index=["Scenario", "Source contrast", "Feature block", "N balanced", "N per source", "features"],
        columns="model",
        values="roc_auc",
        aggfunc="first",
    ).reset_index()
    wide = wide.rename(columns={"logistic_regression": "Logistic AUC", "random_forest": "Random forest AUC"})
    wide["Logistic AUC"] = wide["Logistic AUC"].map(lambda x: fmt_decimal(x, 3))
    wide["Random forest AUC"] = wide["Random forest AUC"].map(lambda x: fmt_decimal(x, 3))
    order_key = {
        ("Extreme cross-setting stress test", "Basic"): 1,
        ("Extreme cross-setting stress test", "Basic + labs"): 2,
        ("Realistic ICU deployment", "Basic"): 3,
        ("Realistic ICU deployment", "Basic + labs"): 4,
        ("Realistic ICU deployment", "Basic + labs + vitals"): 5,
    }
    wide["_order"] = wide.apply(lambda r: order_key.get((r["Scenario"], r["Feature block"]), 99), axis=1)
    wide = wide.sort_values("_order").drop(columns="_order")
    detail = wide[
        [
            "Scenario",
            "Source contrast",
            "Feature block",
            "Logistic AUC",
            "Random forest AUC",
            "N balanced",
            "N per source",
            "features",
        ]
    ].copy()
    detail["features"] = detail["features"].map(clean_feature_list)
    main = detail.drop(columns=["features"]).copy()
    return main, detail


def load_recalibration_rows() -> pd.DataFrame:
    nhanes = pd.read_csv(NHANES_RECAL)
    nhanes = nhanes[
        nhanes["prediction_set"].eq("NHANES_to_MIMIC_1y_base_logistic_regression")
        & nhanes["method"].isin(["raw", "intercept_only", "platt", "isotonic"])
        & nhanes["event_target"].isin([0, 25, 50, 100, 200])
    ].copy()
    nhanes["Direction"] = "NHANES -> MIMIC-IV"

    icu = pd.read_csv(ICU_RECAL)
    icu = icu[
        icu["feature_set"].eq("icu_native_primary")
        & icu["model"].eq("logistic_regression")
        & icu["method"].isin(["raw", "intercept_only", "platt", "isotonic"])
        & icu["event_target"].isin([0, 25, 50, 100, 200])
    ].copy()
    icu["Direction"] = icu["direction"].map(
        {
            "mimic_icu_to_eicu": "MIMIC-IV ICU -> eICU",
            "eicu_to_mimic_icu": "eICU -> MIMIC-IV ICU",
        }
    )
    icu = icu[icu["Direction"].notna()].copy()

    rows = []
    for _, row in nhanes.iterrows():
        rows.append(
            {
                "Direction": row["Direction"],
                "method": row["method"],
                "event_target": int(row["event_target"]),
                "ece_mean": row["ece_10bin_mean"],
                "ece_ci_lower": row["ece_10bin_ci_lower"],
                "ece_ci_upper": row["ece_10bin_ci_upper"],
                "calibration_n_mean": row["calibration_n_mean"],
                "calibration_events_mean": row["calibration_events_mean"],
                "n_repeats": row["repeat_n"],
            }
        )
    for _, row in icu.iterrows():
        rows.append(
            {
                "Direction": row["Direction"],
                "method": row["method"],
                "event_target": int(row["event_target"]),
                "ece_mean": row["ece_10bin_mean"],
                "ece_ci_lower": row["ece_10bin_ci_lower"],
                "ece_ci_upper": row["ece_10bin_ci_upper"],
                "calibration_n_mean": row["calibration_n_mean"],
                "calibration_events_mean": row["calibration_events_mean"],
                "n_repeats": row["n_repeats"],
            }
        )
    return pd.DataFrame(rows)


def build_table4() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rec = load_recalibration_rows()
    method_labels = {
        "raw": "Raw transport",
        "intercept_only": "Intercept-only",
        "platt": "Platt",
        "isotonic": "Isotonic",
    }
    method_order = {"raw": 0, "intercept_only": 1, "platt": 2, "isotonic": 3}
    rec["Method"] = rec["method"].map(method_labels)
    rec["_method_order"] = rec["method"].map(method_order)
    rec["_event_order"] = rec["event_target"]
    rec["Local outcome events"] = rec["event_target"].astype(str)
    rec["ECE (95% empirical interval)"] = rec.apply(
        lambda r: fmt_decimal(r["ece_mean"], 3)
        if r["method"] == "raw"
        else fmt_ci(r["ece_mean"], r["ece_ci_lower"], r["ece_ci_upper"], 3),
        axis=1,
    )
    numeric_long = rec.sort_values(["Direction", "_method_order", "_event_order"]).drop(
        columns=["_method_order", "_event_order"]
    )

    wide = rec.pivot_table(
        index=["_method_order", "_event_order", "Method", "Local outcome events"],
        columns="Direction",
        values="ECE (95% empirical interval)",
        aggfunc="first",
    ).reset_index()
    for direction in DIRECTION_ORDER:
        if direction not in wide.columns:
            wide[direction] = "NA"
    wide = wide.sort_values(["_method_order", "_event_order"])
    wide = wide[["Method", "Local outcome events"] + DIRECTION_ORDER]
    panel = numeric_long[
        [
            "Direction",
            "Method",
            "Local outcome events",
            "ECE (95% empirical interval)",
            "calibration_n_mean",
            "calibration_events_mean",
            "n_repeats",
        ]
    ].copy()
    panel = panel.rename(
        columns={
            "calibration_n_mean": "Mean calibration N",
            "calibration_events_mean": "Mean local events",
            "n_repeats": "Repeats",
        }
    )
    return wide, numeric_long, panel


def write_table4_panel_markdown(panel: pd.DataFrame) -> None:
    sections = ["# Table_4_recalibration_by_event_count_panel\n"]
    for direction in DIRECTION_ORDER:
        part = panel[panel["Direction"].eq(direction)].drop(columns=["Direction"]).copy()
        sections.append(f"## {direction}\n")
        sections.append(markdown_table(part))
        sections.append("")
    sections.append("## Notes\n")
    sections.append("- This panel version contains the same ECE values as Table 4 but separates the three transport directions for easier reading.")
    sections.append("- Raw transport is a point estimate before local recalibration; interval columns apply to repeated recalibration samples.")
    (OUT_DIR / "Table_4_recalibration_by_event_count_panel.md").write_text("\n".join(sections), encoding="utf-8")


def build_table5() -> pd.DataFrame:
    wanted = {"Age": ["Age <65", "Age >=65"], "CKD history": ["No CKD", "CKD"], "CVD history": ["No CVD", "CVD"]}
    rows = []

    nhanes = pd.read_csv(NHANES_SUBGROUP)
    nhanes = nhanes[
        nhanes["train_source"].eq("NHANES")
        & nhanes["test_target"].eq("MIMIC-IV")
        & nhanes["feature_set"].eq("base")
        & nhanes["model"].eq("logistic_regression")
    ].copy()
    for _, row in nhanes.iterrows():
        if row["subgroup_type"] in wanted and row["subgroup"] in wanted[row["subgroup_type"]]:
            rows.append(
                {
                    "Direction": "NHANES -> MIMIC-IV",
                    "Subgroup type": row["subgroup_type"],
                    "Subgroup": row["subgroup"],
                    "N": int(row["n"]),
                    "Events": int(row["events"]),
                    "Event rate": fmt_decimal(row["event_rate"], 3),
                    "AUC": fmt_decimal(row["roc_auc"], 3),
                    "ECE": fmt_decimal(row["ece_10bin"], 3),
                }
            )

    icu = pd.read_csv(ICU_SUBGROUP)
    icu = icu[
        icu["feature_set"].eq("icu_native_primary")
        & icu["model"].eq("logistic_regression")
        & icu["direction"].isin(["mimic_icu_to_eicu", "eicu_to_mimic_icu"])
    ].copy()
    icu["Direction"] = icu["direction"].map(
        {
            "mimic_icu_to_eicu": "MIMIC-IV ICU -> eICU",
            "eicu_to_mimic_icu": "eICU -> MIMIC-IV ICU",
        }
    )
    for _, row in icu.iterrows():
        if row["subgroup_type"] in wanted and row["subgroup"] in wanted[row["subgroup_type"]]:
            rows.append(
                {
                    "Direction": row["Direction"],
                    "Subgroup type": row["subgroup_type"],
                    "Subgroup": row["subgroup"],
                    "N": int(row["n"]),
                    "Events": int(row["events"]),
                    "Event rate": fmt_decimal(row["event_rate"], 3),
                    "AUC": fmt_decimal(row["roc_auc"], 3),
                    "ECE": fmt_decimal(row["ece_10bin"], 3),
                }
            )

    out = pd.DataFrame(rows)
    subgroup_order = {
        ("Age", "Age <65"): 1,
        ("Age", "Age >=65"): 2,
        ("CKD history", "No CKD"): 3,
        ("CKD history", "CKD"): 4,
        ("CVD history", "No CVD"): 5,
        ("CVD history", "CVD"): 6,
    }
    out["_direction_order"] = out["Direction"].map({d: i for i, d in enumerate(DIRECTION_ORDER)})
    out["_subgroup_order"] = out.apply(lambda r: subgroup_order.get((r["Subgroup type"], r["Subgroup"]), 99), axis=1)
    out = out.sort_values(["_direction_order", "_subgroup_order"]).drop(columns=["_direction_order", "_subgroup_order"])
    return out


def build_table6() -> pd.DataFrame:
    thresholds = [0.20, 0.25, 0.30]
    rows = []

    nhanes = pd.read_csv(NHANES_DCA)
    nhanes = nhanes[nhanes["threshold"].round(2).isin(thresholds)]
    strategy_map = {
        "Treat none": "Treat none",
        "Treat all": "Treat all",
        "NHANES raw logistic": "Raw transport logistic",
        "NHANES Platt 100 events": "Platt 100 events",
        "MIMIC internal HGB": "Internal HGB benchmark",
    }
    nhanes = nhanes[nhanes["model"].isin(strategy_map)].copy()
    for _, row in nhanes.iterrows():
        rows.append(
            {
                "Direction": "NHANES -> MIMIC-IV",
                "Threshold": float(row["threshold"]),
                "Strategy": strategy_map[row["model"]],
                "Net benefit": row["net_benefit"],
            }
        )

    icu = pd.read_csv(ICU_DCA)
    icu = icu[icu["threshold"].round(2).isin(thresholds)].copy()
    icu_strategy_map = {
        "Treat none": "Treat none",
        "Treat all": "Treat all",
        "Raw transport logistic": "Raw transport logistic",
        "Platt 100 events": "Platt 100 events",
        "Internal HGB benchmark": "Internal HGB benchmark",
    }
    icu = icu[icu["strategy"].isin(icu_strategy_map)].copy()
    icu["Direction"] = icu["direction"].map(
        {
            "mimic_icu_to_eicu": "MIMIC-IV ICU -> eICU",
            "eicu_to_mimic_icu": "eICU -> MIMIC-IV ICU",
        }
    )
    for _, row in icu.iterrows():
        rows.append(
            {
                "Direction": row["Direction"],
                "Threshold": float(row["threshold"]),
                "Strategy": icu_strategy_map[row["strategy"]],
                "Net benefit": row["net_benefit"],
            }
        )

    long = pd.DataFrame(rows)
    wide = long.pivot_table(
        index=["Direction", "Threshold"],
        columns="Strategy",
        values="Net benefit",
        aggfunc="first",
    ).reset_index()
    for col in ["Treat none", "Treat all", "Raw transport logistic", "Platt 100 events", "Internal HGB benchmark"]:
        if col not in wide:
            wide[col] = np.nan
        wide[col] = wide[col].map(lambda x: fmt_decimal(x, 4))
    wide["_direction_order"] = wide["Direction"].map({d: i for i, d in enumerate(DIRECTION_ORDER)})
    wide = wide.sort_values(["_direction_order", "Threshold"]).drop(columns="_direction_order")
    wide["Threshold"] = wide["Threshold"].map(lambda x: f"{x:.2f}")
    return wide[
        [
            "Direction",
            "Threshold",
            "Treat none",
            "Treat all",
            "Raw transport logistic",
            "Platt 100 events",
            "Internal HGB benchmark",
        ]
    ]


def write_readme(metadata: dict[str, object]) -> None:
    readme = f"""# Three-Scenario Submission Tables

Generated by `scripts/make_submission_tables_three_scenario.py`.

Generated files:

- `Table_1_baseline_characteristics.csv` and `.md`
- `Table_1_baseline_characteristics_detailed.csv`
- `Table_1_baseline_characteristics_numeric.csv`
- `Table_2_source_classifier_auc.csv` and `.md`
- `Table_2_source_classifier_feature_sets.csv`
- `Table_4_recalibration_by_event_count.csv` and `.md`
- `Table_4_recalibration_by_event_count_numeric_long.csv`
- `Table_4_recalibration_by_event_count_panel.csv` and `.md`
- `Table_5_subgroup_transportability.csv` and `.md`
- `Table_6_decision_curve_selected_thresholds.csv` and `.md`

## Table Notes

Table 1 reports analytic cohorts with known primary endpoint status: NHANES one-year mortality, MIMIC-IV ICU hospital mortality, and eICU hospital mortality. NHANES CKD is defined from measured eGFR and/or albuminuria, and NHANES hypertension/CVD use survey or examination-derived history variables; MIMIC-IV/eICU CKD, hypertension, and CVD are harmonized from diagnosis/history indicators. Blood pressure, glucose, and creatinine in NHANES are survey/examination measurements; MIMIC-IV ICU and eICU laboratory/vital-sign variables are first values within the first 24 hours after ICU admission. Albumin is shown for completeness but was excluded from the primary model because coverage was approximately 42% in both ICU databases. SMDs are absolute standardized mean differences; no SMD is calculated for the endpoint row because endpoints differ by scenario.

Table 2 reports source-classifier AUCs. Source classification is contrast-level, not directional; the MIMIC-IV ICU vs eICU contrast applies to both ICU transport directions.

Table 4 reports ECE with empirical 95% intervals across repeated local recalibration samples. Raw transport rows are point estimates without an event-count interval. NHANES-to-MIMIC calibration samples were drawn from target evaluation data and excluded from each replicate's evaluation set; ICU-to-ICU calibration samples were drawn from the target-site development split and evaluated on a fixed target holdout split.

Table 5 reports raw transported logistic-regression performance in clinically defined subgroups. ECE is the 10-bin equal-frequency expected calibration error.

Table 6 reports selected-threshold decision-curve net benefit for raw transport, Platt recalibration with 100 local events, and the internal target-site HGB benchmark.

## Metadata

```json
{json.dumps(metadata, indent=2)}
```
"""
    (OUT_DIR / "README_submission_tables_three_scenario.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table1, table1_detailed, table1_numeric = build_table1()
    table1.to_csv(OUT_DIR / "Table_1_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    table1_detailed.to_csv(OUT_DIR / "Table_1_baseline_characteristics_detailed.csv", index=False, encoding="utf-8-sig")
    table1_numeric.to_csv(OUT_DIR / "Table_1_baseline_characteristics_numeric.csv", index=False, encoding="utf-8-sig")
    write_table(
        "Table_1_baseline_characteristics",
        table1,
        [
            "Values are mean (SD); n=available, n/N (%), or count.",
            "NHANES uses one-year mortality as the primary endpoint; MIMIC-IV ICU and eICU use hospital mortality.",
            "Comorbidity definitions are not identical across data sources: NHANES CKD uses measured eGFR and/or albuminuria and NHANES hypertension/CVD use survey or examination-derived history variables, whereas MIMIC-IV ICU and eICU use harmonized diagnosis/history indicators.",
            "ICU laboratory and vital-sign variables are first values within 24 hours after ICU admission; NHANES measurements are survey/exam measurements.",
            "SMDs are absolute standardized mean differences; no SMD is calculated for the endpoint row.",
        ],
    )

    table2, table2_detail = build_table2()
    write_table(
        "Table_2_source_classifier_auc",
        table2,
        [
            "Source-classifier AUC quantifies feature-distribution separability between source databases.",
            "The MIMIC-IV ICU vs eICU source contrast is contrast-level and applies to both ICU transport directions.",
            "Albumin sensitivity is omitted from the main table because albumin was excluded from the primary ICU-native model.",
        ],
    )
    table2_detail.to_csv(OUT_DIR / "Table_2_source_classifier_feature_sets.csv", index=False, encoding="utf-8-sig")

    table4, table4_numeric, table4_panel = build_table4()
    write_table(
        "Table_4_recalibration_by_event_count",
        table4,
        [
            "Cells show ECE with empirical 95% intervals across 200 local recalibration samples when intervals are available.",
            "Raw transport is a point estimate before local recalibration.",
            "Event counts refer to local target-site outcome events used for recalibration.",
        ],
    )
    table4_numeric.to_csv(
        OUT_DIR / "Table_4_recalibration_by_event_count_numeric_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    table4_panel.to_csv(
        OUT_DIR / "Table_4_recalibration_by_event_count_panel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_table4_panel_markdown(table4_panel)

    table5 = build_table5()
    write_table(
        "Table_5_subgroup_transportability",
        table5,
        [
            "Rows report raw transported logistic-regression performance within target-site subgroups.",
            "AUC is ROC AUC; ECE is 10-bin equal-frequency expected calibration error.",
        ],
    )

    table6 = build_table6()
    write_table(
        "Table_6_decision_curve_selected_thresholds",
        table6,
        [
            "Net benefit is reported at thresholds 0.20, 0.25, and 0.30.",
            "Internal HGB benchmark is the target-site histogram-gradient-boosting model.",
        ],
    )

    metadata = {
        "table_1_rows": int(len(table1)),
        "table_2_rows": int(len(table2)),
        "table_4_rows": int(len(table4)),
        "table_5_rows": int(len(table5)),
        "table_6_rows": int(len(table6)),
        "directions": DIRECTION_ORDER,
        "source_files": {
            "nhanes_cohort": str(NHANES_COHORT),
            "mimic_icu_cohort": str(MIMIC_ICU_COHORT),
            "eicu_cohort": str(EICU_COHORT),
            "nhanes_source_classifier": str(NHANES_SOURCE_CLASSIFIER),
            "icu_source_classifier": str(ICU_SOURCE_CLASSIFIER),
            "nhanes_recalibration": str(NHANES_RECAL),
            "icu_recalibration": str(ICU_RECAL),
            "nhanes_subgroup": str(NHANES_SUBGROUP),
            "icu_subgroup": str(ICU_SUBGROUP),
            "nhanes_dca": str(NHANES_DCA),
            "icu_dca": str(ICU_DCA),
        },
    }
    (OUT_DIR / "submission_tables_three_scenario_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_readme(metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
