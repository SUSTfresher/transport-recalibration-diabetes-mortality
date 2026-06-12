from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_ROOT = Path(r"D:\DATABASE\mimic-iv-3.1")
OUT_DIR = PROJECT_ROOT / "data" / "mimic" / "processed"


def normalize_icd(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().str.replace(".", "", regex=False)


def condition_flags(diagnoses: pd.DataFrame) -> pd.DataFrame:
    code = normalize_icd(diagnoses["icd_code"])
    version = diagnoses["icd_version"]
    flags = pd.DataFrame(
        {
            "subject_id": diagnoses["subject_id"],
            "hadm_id": diagnoses["hadm_id"],
            "diabetes": ((version.eq(9) & code.str.startswith("250")) | (version.eq(10) & code.str.match(r"^E0[8-9]|^E1[0-3]", na=False))).astype(int),
            "hypertension_history": ((version.eq(9) & code.str.match(r"^40[1-5]", na=False)) | (version.eq(10) & code.str.match(r"^I1[0-5]", na=False))).astype(int),
            "ckd_history": ((version.eq(9) & code.str.match(r"^585", na=False)) | (version.eq(10) & code.str.match(r"^N18", na=False))).astype(int),
            "cvd_history": (
                (version.eq(9) & code.str.match(r"^41[0-4]|^428|^43[0-8]", na=False))
                | (version.eq(10) & code.str.match(r"^I2[0-5]|^I50|^I6[0-9]", na=False))
            ).astype(int),
        }
    )
    return flags.groupby(["subject_id", "hadm_id"], as_index=False).max()


def parse_float(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else np.nan


def parse_bp(value: object) -> tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan
    text = str(value)
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", text)
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2))


def summarize_omr(omr_path: Path, diabetes_subjects: set[int]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    usecols = ["subject_id", "chartdate", "result_name", "result_value"]
    wanted = {
        "BMI (kg/m2)",
        "BMI",
        "Weight (Lbs)",
        "Weight",
        "Height (Inches)",
        "Height",
        "Blood Pressure",
        "Blood Pressure Sitting",
    }
    for chunk in pd.read_csv(omr_path, usecols=usecols, chunksize=1_000_000):
        chunk = chunk[chunk["subject_id"].isin(diabetes_subjects) & chunk["result_name"].isin(wanted)].copy()
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=["subject_id", "bmi", "weight_lbs", "height_inches", "systolic_bp", "diastolic_bp"])

    omr = pd.concat(chunks, ignore_index=True)
    omr["chartdate"] = pd.to_datetime(omr["chartdate"], errors="coerce")
    omr["numeric_value"] = omr["result_value"].map(parse_float)
    bp_values = omr.loc[omr["result_name"].str.contains("Blood Pressure", na=False), "result_value"].map(parse_bp)
    if len(bp_values):
        omr.loc[omr["result_name"].str.contains("Blood Pressure", na=False), "systolic_bp"] = [x[0] for x in bp_values]
        omr.loc[omr["result_name"].str.contains("Blood Pressure", na=False), "diastolic_bp"] = [x[1] for x in bp_values]

    output = pd.DataFrame({"subject_id": sorted(diabetes_subjects)})
    mappings = {
        "bmi": ["BMI (kg/m2)", "BMI"],
        "weight_lbs": ["Weight (Lbs)", "Weight"],
        "height_inches": ["Height (Inches)", "Height"],
    }
    for out_col, names in mappings.items():
        sub = omr[omr["result_name"].isin(names)].sort_values(["subject_id", "chartdate"])
        latest = sub.groupby("subject_id", as_index=False)["numeric_value"].last().rename(columns={"numeric_value": out_col})
        output = output.merge(latest, on="subject_id", how="left")

    for out_col in ["systolic_bp", "diastolic_bp"]:
        sub = omr.dropna(subset=[out_col]).sort_values(["subject_id", "chartdate"])
        latest = sub.groupby("subject_id", as_index=False)[out_col].last()
        output = output.merge(latest, on="subject_id", how="left")

    # OMR contains occasional data-entry artifacts. Keep only clinically plausible adult ranges.
    output.loc[~output["bmi"].between(10, 80), "bmi"] = np.nan
    output.loc[~output["weight_lbs"].between(50, 700), "weight_lbs"] = np.nan
    output.loc[~output["height_inches"].between(48, 84), "height_inches"] = np.nan
    output.loc[~output["systolic_bp"].between(60, 260), "systolic_bp"] = np.nan
    output.loc[~output["diastolic_bp"].between(30, 160), "diastolic_bp"] = np.nan
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patients = pd.read_csv(MIMIC_ROOT / "hosp" / "patients.csv", usecols=["subject_id", "gender", "anchor_age", "dod"])
    admissions = pd.read_csv(
        MIMIC_ROOT / "hosp" / "admissions.csv",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "race",
            "hospital_expire_flag",
        ],
    )
    icustays = pd.read_csv(MIMIC_ROOT / "icu" / "icustays.csv", usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    diagnoses = pd.read_csv(
        MIMIC_ROOT / "hosp" / "diagnoses_icd.csv",
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": "string"},
    )

    flags = condition_flags(diagnoses)
    diabetes_hadm = flags.loc[flags["diabetes"].eq(1), ["subject_id", "hadm_id"]].drop_duplicates()
    cohort = diabetes_hadm.merge(admissions, on=["subject_id", "hadm_id"], how="left")
    cohort = cohort.merge(patients, on="subject_id", how="left")
    cohort = cohort.merge(flags.drop(columns=["diabetes"]), on=["subject_id", "hadm_id"], how="left")

    icu_counts = icustays.groupby(["subject_id", "hadm_id"], as_index=False).agg(
        icu_stay_count=("stay_id", "nunique"),
        first_icu_intime=("intime", "min"),
    )
    cohort = cohort.merge(icu_counts, on=["subject_id", "hadm_id"], how="left")
    cohort["icu_stay_count"] = cohort["icu_stay_count"].fillna(0).astype(int)
    cohort["has_icu_stay"] = cohort["icu_stay_count"].gt(0).astype(int)

    cohort["admittime"] = pd.to_datetime(cohort["admittime"], errors="coerce")
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"], errors="coerce")
    cohort["dod"] = pd.to_datetime(cohort["dod"], errors="coerce")
    cohort["length_of_stay_days"] = (cohort["dischtime"] - cohort["admittime"]).dt.total_seconds() / 86400
    cohort["postdischarge_death_within_30d"] = (
        cohort["dod"].notna()
        & cohort["dischtime"].notna()
        & ((cohort["dod"] - cohort["dischtime"]).dt.days >= 0)
        & ((cohort["dod"] - cohort["dischtime"]).dt.days <= 30)
    ).astype(int)
    cohort["postdischarge_death_within_1y"] = (
        cohort["dod"].notna()
        & cohort["dischtime"].notna()
        & ((cohort["dod"] - cohort["dischtime"]).dt.days >= 0)
        & ((cohort["dod"] - cohort["dischtime"]).dt.days <= 365)
    ).astype(int)
    cohort["death_within_30d_discharge"] = (
        cohort["hospital_expire_flag"].eq(1) | cohort["postdischarge_death_within_30d"].eq(1)
    ).astype(int)
    cohort["death_within_1y_discharge"] = (
        cohort["hospital_expire_flag"].eq(1) | cohort["postdischarge_death_within_1y"].eq(1)
    ).astype(int)
    cohort["female"] = cohort["gender"].eq("F").astype(int)
    cohort = cohort.rename(columns={"anchor_age": "age"})

    omr = summarize_omr(MIMIC_ROOT / "hosp" / "omr.csv", set(cohort["subject_id"].unique()))
    cohort = cohort.merge(omr, on="subject_id", how="left")

    keep = [
        "subject_id",
        "hadm_id",
        "age",
        "female",
        "gender",
        "race",
        "admission_type",
        "admittime",
        "dischtime",
        "length_of_stay_days",
        "hospital_expire_flag",
        "postdischarge_death_within_30d",
        "postdischarge_death_within_1y",
        "death_within_30d_discharge",
        "death_within_1y_discharge",
        "has_icu_stay",
        "icu_stay_count",
        "hypertension_history",
        "ckd_history",
        "cvd_history",
        "bmi",
        "weight_lbs",
        "height_inches",
        "systolic_bp",
        "diastolic_bp",
    ]
    cohort = cohort[keep].copy()
    for col in ["hypertension_history", "ckd_history", "cvd_history"]:
        cohort[col] = cohort[col].fillna(0).astype(int)

    out_csv = OUT_DIR / "mimic_iv31_diabetes_light_admissions.csv"
    cohort.to_csv(out_csv, index=False)

    metadata = {
        "mimic_root": str(MIMIC_ROOT),
        "output": str(out_csv),
        "rows": int(len(cohort)),
        "subjects": int(cohort["subject_id"].nunique()),
        "hospital_deaths": int(cohort["hospital_expire_flag"].sum()),
        "deaths_30d_discharge": int(cohort["death_within_30d_discharge"].sum()),
        "deaths_1y_discharge": int(cohort["death_within_1y_discharge"].sum()),
        "postdischarge_deaths_30d": int(cohort["postdischarge_death_within_30d"].sum()),
        "postdischarge_deaths_1y": int(cohort["postdischarge_death_within_1y"].sum()),
        "icu_admissions": int(cohort["has_icu_stay"].sum()),
        "columns": list(cohort.columns),
        "notes": [
            "Light cohort uses admissions, patients, diagnoses_icd, icustays, and OMR only.",
            "No labevents/chartevents variables are included in this first-pass table.",
            "Diabetes ICD definition: ICD-9 250*, ICD-10 E08-E13.",
        ],
    }
    metadata_path = OUT_DIR / "mimic_iv31_diabetes_light_admissions_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {metadata_path}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
