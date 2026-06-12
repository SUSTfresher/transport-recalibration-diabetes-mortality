from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EICU_ROOT = Path(r"D:\eicu-crd-2.0\physionet.org\files\eicu-crd\2.0")
OUT_DIR = PROJECT_ROOT / "data" / "eicu" / "processed"
TMP_DIR = PROJECT_ROOT / "outputs" / "duckdb_tmp"

DIABETES_ICD_RE = re.compile(r"(^|[^0-9A-Za-z])(250(\.|\b)|E0[89](\.|\b)|E1[0-3](\.|\b))", re.I)
DIABETES_TEXT_RE = re.compile(
    r"\b(diabetes mellitus|diabetic ketoacidosis|insulin dependent diabetes|"
    r"non-insulin dependent diabetes|noninsulin dependent diabetes)\b",
    re.I,
)
HYPERTENSION_RE = re.compile(r"(^|[^0-9A-Za-z])(40[1-5](\.|\b)|I1[0-5](\.|\b))|hypertension", re.I)
CKD_RE = re.compile(r"(^|[^0-9A-Za-z])(585(\.|\b)|N18(\.|\b))|chronic kidney|renal failure", re.I)
CVD_RE = re.compile(
    r"(^|[^0-9A-Za-z])(41[0-4](\.|\b)|428(\.|\b)|43[0-8](\.|\b)|I2[0-5](\.|\b)|I50(\.|\b)|I6[0-9](\.|\b))|"
    r"coronary|myocardial infarction|heart failure|cerebrovascular|stroke|ASHD",
    re.I,
)

LAB_PLAUSIBLE_RANGES = {
    "hba1c": (3, 20),
    "glucose": (20, 1000),
    "creatinine": (0.1, 30),
    "bun": (1, 300),
    "albumin": (0.5, 8),
    "hemoglobin": (3, 25),
    "wbc": (0.1, 200),
}

VITAL_PLAUSIBLE_RANGES = {
    "heart_rate": (20, 250),
    "respiratory_rate": (3, 80),
    "temperature": (25, 45),
    "spo2": (40, 100),
    "systolic_bp": (50, 300),
    "diastolic_bp": (20, 180),
}


def slash(path: Path) -> str:
    return str(path).replace("\\", "/")


def parse_age(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "> 89":
        return 90.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_text_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    text = pd.Series("", index=df.index, dtype="string")
    for column in columns:
        if column in df.columns:
            text = text.str.cat(df[column].fillna("").astype("string"), sep=" ")
    return text.fillna("")


def update_flag(flags: pd.DataFrame, ids: pd.Series, column: str) -> None:
    ids = pd.to_numeric(ids, errors="coerce").dropna().astype("int64").unique()
    if len(ids):
        flags.loc[flags.index.intersection(ids), column] = 1


def build_minimal_cohort(eicu_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    patient = pd.read_csv(eicu_root / "patient.csv.gz", dtype="string")
    patient["patientunitstayid"] = pd.to_numeric(patient["patientunitstayid"], errors="coerce").astype("int64")
    patient = patient.set_index("patientunitstayid", drop=False)

    flags = pd.DataFrame(index=patient.index)
    for column in [
        "diabetes_apache",
        "diabetes_diagnosis",
        "diabetes_admission_dx",
        "diabetes_past_history",
        "hypertension_history",
        "ckd_history",
        "cvd_history",
    ]:
        flags[column] = 0

    apache = pd.read_csv(eicu_root / "apachePredVar.csv.gz", usecols=["patientunitstayid", "diabetes"], dtype="string")
    apache["patientunitstayid"] = pd.to_numeric(apache["patientunitstayid"], errors="coerce")
    update_flag(flags, apache.loc[apache["diabetes"].eq("1"), "patientunitstayid"], "diabetes_apache")

    diagnosis = pd.read_csv(
        eicu_root / "diagnosis.csv.gz",
        usecols=["patientunitstayid", "diagnosisstring", "icd9code"],
        dtype="string",
    )
    diagnosis["patientunitstayid"] = pd.to_numeric(diagnosis["patientunitstayid"], errors="coerce")
    diagnosis_text = safe_text_series(diagnosis, ["icd9code", "diagnosisstring"])
    update_flag(flags, diagnosis.loc[diagnosis_text.map(lambda x: bool(DIABETES_ICD_RE.search(x) or DIABETES_TEXT_RE.search(x))), "patientunitstayid"], "diabetes_diagnosis")
    update_flag(flags, diagnosis.loc[diagnosis_text.map(lambda x: bool(HYPERTENSION_RE.search(x))), "patientunitstayid"], "hypertension_history")
    update_flag(flags, diagnosis.loc[diagnosis_text.map(lambda x: bool(CKD_RE.search(x))), "patientunitstayid"], "ckd_history")
    update_flag(flags, diagnosis.loc[diagnosis_text.map(lambda x: bool(CVD_RE.search(x))), "patientunitstayid"], "cvd_history")

    admission_dx = pd.read_csv(
        eicu_root / "admissionDx.csv.gz",
        usecols=["patientunitstayid", "admitdxpath", "admitdxname", "admitdxtext"],
        dtype="string",
    )
    admission_dx["patientunitstayid"] = pd.to_numeric(admission_dx["patientunitstayid"], errors="coerce")
    admission_dx_text = safe_text_series(admission_dx, ["admitdxpath", "admitdxname", "admitdxtext"])
    update_flag(
        flags,
        admission_dx.loc[admission_dx_text.map(lambda x: bool(DIABETES_ICD_RE.search(x) or DIABETES_TEXT_RE.search(x))), "patientunitstayid"],
        "diabetes_admission_dx",
    )

    past_history = pd.read_csv(
        eicu_root / "pastHistory.csv.gz",
        usecols=["patientunitstayid", "pasthistorypath", "pasthistoryvalue", "pasthistoryvaluetext"],
        dtype="string",
    )
    past_history["patientunitstayid"] = pd.to_numeric(past_history["patientunitstayid"], errors="coerce")
    past_text = safe_text_series(past_history, ["pasthistorypath", "pasthistoryvalue", "pasthistoryvaluetext"])
    update_flag(flags, past_history.loc[past_text.map(lambda x: bool(DIABETES_TEXT_RE.search(x))), "patientunitstayid"], "diabetes_past_history")
    update_flag(flags, past_history.loc[past_text.map(lambda x: bool(HYPERTENSION_RE.search(x))), "patientunitstayid"], "hypertension_history")
    update_flag(flags, past_history.loc[past_text.map(lambda x: bool(CKD_RE.search(x))), "patientunitstayid"], "ckd_history")
    update_flag(flags, past_history.loc[past_text.map(lambda x: bool(CVD_RE.search(x))), "patientunitstayid"], "cvd_history")

    patient = patient.join(flags)
    patient["diabetes_candidate"] = patient[
        ["diabetes_apache", "diabetes_diagnosis", "diabetes_admission_dx", "diabetes_past_history"]
    ].max(axis=1)

    cohort = patient.loc[patient["diabetes_candidate"].eq(1)].copy()
    cohort["age"] = cohort["age"].map(parse_age)
    cohort = cohort.loc[cohort["age"].ge(18, fill_value=False)].copy()
    gender_lower = cohort["gender"].str.lower()
    is_female = gender_lower.eq("female").fillna(False)
    is_male = gender_lower.eq("male").fillna(False)
    cohort["female"] = np.where(is_female, 1, np.where(is_male, 0, np.nan))
    cohort["height_cm"] = clean_numeric(cohort["admissionheight"])
    cohort["weight_kg"] = clean_numeric(cohort["admissionweight"])
    cohort["bmi"] = cohort["weight_kg"] / ((cohort["height_cm"] / 100) ** 2)
    cohort.loc[~cohort["height_cm"].between(100, 250), "height_cm"] = np.nan
    cohort.loc[~cohort["weight_kg"].between(20, 400), "weight_kg"] = np.nan
    cohort.loc[~cohort["bmi"].between(10, 80), "bmi"] = np.nan
    hospital_expired = cohort["hospitaldischargestatus"].eq("Expired").fillna(False)
    hospital_alive = cohort["hospitaldischargestatus"].eq("Alive").fillna(False)
    unit_expired = cohort["unitdischargestatus"].eq("Expired").fillna(False)
    unit_alive = cohort["unitdischargestatus"].eq("Alive").fillna(False)
    cohort["hospital_mortality"] = np.where(
        hospital_expired,
        1,
        np.where(hospital_alive, 0, np.nan),
    )
    cohort["unit_mortality"] = np.where(
        unit_expired,
        1,
        np.where(unit_alive, 0, np.nan),
    )
    cohort["hospital_los_days"] = (
        clean_numeric(cohort["hospitaldischargeoffset"]) - clean_numeric(cohort["hospitaladmitoffset"])
    ) / 1440
    cohort["unit_los_days"] = clean_numeric(cohort["unitdischargeoffset"]) / 1440

    keep_columns = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
        "hospitalid",
        "wardid",
        "age",
        "female",
        "gender",
        "ethnicity",
        "unittype",
        "unitadmitsource",
        "hospitaladmitsource",
        "hospitaldischargeyear",
        "hospitaldischargestatus",
        "unitdischargestatus",
        "hospital_mortality",
        "unit_mortality",
        "hospital_los_days",
        "unit_los_days",
        "height_cm",
        "weight_kg",
        "bmi",
        "diabetes_apache",
        "diabetes_diagnosis",
        "diabetes_admission_dx",
        "diabetes_past_history",
        "hypertension_history",
        "ckd_history",
        "cvd_history",
    ]
    cohort = cohort[keep_columns].reset_index(drop=True).sort_values("patientunitstayid").reset_index(drop=True)

    metadata = {
        "rows": int(len(cohort)),
        "unique_patients": int(cohort["uniquepid"].nunique()),
        "hospitals": int(cohort["hospitalid"].nunique()),
        "hospital_mortality_nonmissing": int(cohort["hospital_mortality"].notna().sum()),
        "hospital_mortality_events": int(cohort["hospital_mortality"].fillna(0).sum()),
        "hospital_mortality_event_rate": float(cohort["hospital_mortality"].mean()),
        "unit_mortality_nonmissing": int(cohort["unit_mortality"].notna().sum()),
        "unit_mortality_events": int(cohort["unit_mortality"].fillna(0).sum()),
        "unit_mortality_event_rate": float(cohort["unit_mortality"].mean()),
        "diabetes_source_counts": {
            column: int(cohort[column].sum())
            for column in ["diabetes_apache", "diabetes_diagnosis", "diabetes_admission_dx", "diabetes_past_history"]
        },
        "comorbidity_counts": {
            column: int(cohort[column].sum())
            for column in ["hypertension_history", "ckd_history", "cvd_history"]
        },
    }
    return cohort, metadata


def sql_range_filter(mapping: dict[str, tuple[float, float]], value_col: str = "value") -> str:
    return " OR\n          ".join(
        f"(name = '{name}' AND {value_col} BETWEEN {low} AND {high})" for name, (low, high) in mapping.items()
    )


def build_lab_vital_features(eicu_root: Path, cohort: pd.DataFrame, skip_large_tables: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(TMP_DIR / "eicu_diabetes_wide.duckdb"))
    con.execute(f"SET temp_directory='{slash(TMP_DIR)}'")
    con.execute("SET memory_limit='8GB'")
    con.execute("PRAGMA threads=4")
    cohort_ids = cohort[["patientunitstayid"]].drop_duplicates().copy()
    con.register("cohort_ids_df", cohort_ids)
    con.execute("CREATE OR REPLACE TEMP TABLE cohort_ids AS SELECT patientunitstayid::BIGINT AS patientunitstayid FROM cohort_ids_df")

    lab_path = slash(eicu_root / "lab.csv.gz")
    lab_range_sql = sql_range_filter(LAB_PLAUSIBLE_RANGES, "value")
    print("Building eICU lab features from first ICU 24h...")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE eicu_lab_events_24h AS
        WITH raw AS (
          SELECT
            l.patientunitstayid::BIGINT AS patientunitstayid,
            TRY_CAST(l.labresultoffset AS INTEGER) AS offset_min,
            lower(trim(l.labname)) AS labname_lower,
            TRY_CAST(l.labresult AS DOUBLE) AS value
          FROM read_csv_auto('{lab_path}', columns={{
            'labid': 'BIGINT',
            'patientunitstayid': 'BIGINT',
            'labresultoffset': 'INTEGER',
            'labtypeid': 'INTEGER',
            'labname': 'VARCHAR',
            'labresult': 'DOUBLE',
            'labresulttext': 'VARCHAR',
            'labmeasurenamesystem': 'VARCHAR',
            'labmeasurenameinterface': 'VARCHAR',
            'labresultrevisedoffset': 'INTEGER'
          }}, ignore_errors=true, parallel=true) l
          JOIN cohort_ids c ON l.patientunitstayid = c.patientunitstayid
          WHERE TRY_CAST(l.labresultoffset AS INTEGER) BETWEEN 0 AND 1440
        ),
        mapped AS (
          SELECT
            patientunitstayid,
            offset_min,
            value,
            CASE
              WHEN labname_lower IN ('glucose', 'bedside glucose') THEN 'glucose'
              WHEN labname_lower = 'creatinine' THEN 'creatinine'
              WHEN labname_lower = 'bun' THEN 'bun'
              WHEN labname_lower IN ('wbc', 'wbc x 1000') THEN 'wbc'
              WHEN labname_lower IN ('hgb', 'hemoglobin') THEN 'hemoglobin'
              WHEN labname_lower = 'albumin' THEN 'albumin'
              WHEN regexp_matches(labname_lower, 'a1c|glyco') THEN 'hba1c'
            END AS name
          FROM raw
          WHERE value IS NOT NULL
        )
        SELECT patientunitstayid, offset_min, name, value
        FROM mapped
        WHERE name IS NOT NULL
          AND ({lab_range_sql})
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE eicu_labs_24h_wide AS
        WITH ranked AS (
          SELECT
            patientunitstayid,
            name,
            value,
            ROW_NUMBER() OVER (PARTITION BY patientunitstayid, name ORDER BY offset_min ASC) AS rn
          FROM eicu_lab_events_24h
        ),
        agg AS (
          SELECT
            patientunitstayid,
            name,
            MAX(CASE WHEN rn = 1 THEN value END) AS first_value,
            AVG(value) AS mean_value,
            MIN(value) AS min_value,
            MAX(value) AS max_value,
            COUNT(*) AS count_value
          FROM ranked
          GROUP BY patientunitstayid, name
        )
        SELECT
          patientunitstayid,
          MAX(CASE WHEN name='hba1c' THEN first_value END) AS hba1c_first_24h,
          MAX(CASE WHEN name='glucose' THEN first_value END) AS glucose_first_24h,
          MAX(CASE WHEN name='creatinine' THEN first_value END) AS creatinine_first_24h,
          MAX(CASE WHEN name='bun' THEN first_value END) AS bun_first_24h,
          MAX(CASE WHEN name='albumin' THEN first_value END) AS albumin_first_24h,
          MAX(CASE WHEN name='hemoglobin' THEN first_value END) AS hemoglobin_first_24h,
          MAX(CASE WHEN name='wbc' THEN first_value END) AS wbc_first_24h,
          MAX(CASE WHEN name='glucose' THEN mean_value END) AS glucose_mean_24h,
          MAX(CASE WHEN name='glucose' THEN min_value END) AS glucose_min_24h,
          MAX(CASE WHEN name='glucose' THEN max_value END) AS glucose_max_24h,
          MAX(CASE WHEN name='glucose' THEN count_value END) AS glucose_count_24h,
          MAX(CASE WHEN name='creatinine' THEN max_value END) AS creatinine_max_24h,
          MAX(CASE WHEN name='creatinine' THEN count_value END) AS creatinine_count_24h,
          MAX(CASE WHEN name='bun' THEN max_value END) AS bun_max_24h,
          MAX(CASE WHEN name='bun' THEN count_value END) AS bun_count_24h,
          MAX(CASE WHEN name='albumin' THEN min_value END) AS albumin_min_24h,
          MAX(CASE WHEN name='albumin' THEN count_value END) AS albumin_count_24h,
          MAX(CASE WHEN name='hemoglobin' THEN min_value END) AS hemoglobin_min_24h,
          MAX(CASE WHEN name='hemoglobin' THEN count_value END) AS hemoglobin_count_24h,
          MAX(CASE WHEN name='wbc' THEN max_value END) AS wbc_max_24h,
          MAX(CASE WHEN name='wbc' THEN count_value END) AS wbc_count_24h,
          MAX(CASE WHEN name='hba1c' THEN count_value END) AS hba1c_count_24h
        FROM agg
        GROUP BY patientunitstayid
        """
    )
    lab_counts = con.execute(
        """
        SELECT name, COUNT(*) AS row_count, COUNT(DISTINCT patientunitstayid) AS stays
        FROM eicu_lab_events_24h
        GROUP BY name
        ORDER BY stays DESC
        """
    ).fetchdf()
    labs = con.execute("SELECT * FROM eicu_labs_24h_wide").fetchdf()

    if skip_large_tables:
        vitals = pd.DataFrame({"patientunitstayid": cohort_ids["patientunitstayid"]})
        vital_counts = pd.DataFrame()
    else:
        periodic_path = slash(eicu_root / "vitalPeriodic.csv.gz")
        aperiodic_path = slash(eicu_root / "vitalAperiodic.csv.gz")
        vital_range_sql = sql_range_filter(VITAL_PLAUSIBLE_RANGES, "value")
        print("Building eICU vital features from first ICU 24h...")
        con.execute(
            f"""
            CREATE OR REPLACE TABLE eicu_vital_periodic_raw_24h AS
            SELECT
              v.patientunitstayid::BIGINT AS patientunitstayid,
              TRY_CAST(v.observationoffset AS INTEGER) AS offset_min,
              TRY_CAST(v.temperature AS DOUBLE) AS temperature,
              TRY_CAST(v.sao2 AS DOUBLE) AS sao2,
              TRY_CAST(v.heartrate AS DOUBLE) AS heartrate,
              TRY_CAST(v.respiration AS DOUBLE) AS respiration,
              TRY_CAST(v.systemicsystolic AS DOUBLE) AS systemicsystolic,
              TRY_CAST(v.systemicdiastolic AS DOUBLE) AS systemicdiastolic
            FROM read_csv_auto('{periodic_path}', columns={{
              'vitalperiodicid': 'BIGINT',
              'patientunitstayid': 'BIGINT',
              'observationoffset': 'INTEGER',
              'temperature': 'DOUBLE',
              'sao2': 'DOUBLE',
              'heartrate': 'DOUBLE',
              'respiration': 'DOUBLE',
              'cvp': 'DOUBLE',
              'etco2': 'DOUBLE',
              'systemicsystolic': 'DOUBLE',
              'systemicdiastolic': 'DOUBLE',
              'systemicmean': 'DOUBLE',
              'pasystolic': 'DOUBLE',
              'padiastolic': 'DOUBLE',
              'pamean': 'DOUBLE',
              'st1': 'DOUBLE',
              'st2': 'DOUBLE',
              'st3': 'DOUBLE',
              'icp': 'DOUBLE'
            }}, ignore_errors=true, parallel=true) v
            JOIN cohort_ids c ON v.patientunitstayid = c.patientunitstayid
            WHERE TRY_CAST(v.observationoffset AS INTEGER) BETWEEN 0 AND 1440
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE eicu_vital_aperiodic_raw_24h AS
            SELECT
              v.patientunitstayid::BIGINT AS patientunitstayid,
              TRY_CAST(v.observationoffset AS INTEGER) AS offset_min,
              TRY_CAST(v.noninvasivesystolic AS DOUBLE) AS noninvasivesystolic,
              TRY_CAST(v.noninvasivediastolic AS DOUBLE) AS noninvasivediastolic
            FROM read_csv_auto('{aperiodic_path}', columns={{
              'vitalaperiodicid': 'BIGINT',
              'patientunitstayid': 'BIGINT',
              'observationoffset': 'INTEGER',
              'noninvasivesystolic': 'DOUBLE',
              'noninvasivediastolic': 'DOUBLE',
              'noninvasivemean': 'DOUBLE',
              'paop': 'DOUBLE',
              'cardiacoutput': 'DOUBLE',
              'cardiacinput': 'DOUBLE',
              'svr': 'DOUBLE',
              'svri': 'DOUBLE',
              'pvr': 'DOUBLE',
              'pvri': 'DOUBLE'
            }}, ignore_errors=true, parallel=true) v
            JOIN cohort_ids c ON v.patientunitstayid = c.patientunitstayid
            WHERE TRY_CAST(v.observationoffset AS INTEGER) BETWEEN 0 AND 1440
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE eicu_vital_events_24h AS
            WITH events AS (
              SELECT patientunitstayid, offset_min, 'heart_rate' AS name, heartrate AS value, 1 AS source_priority FROM eicu_vital_periodic_raw_24h
              UNION ALL SELECT patientunitstayid, offset_min, 'respiratory_rate', respiration, 1 FROM eicu_vital_periodic_raw_24h
              UNION ALL SELECT patientunitstayid, offset_min, 'temperature', temperature, 1 FROM eicu_vital_periodic_raw_24h
              UNION ALL SELECT patientunitstayid, offset_min, 'spo2', sao2, 1 FROM eicu_vital_periodic_raw_24h
              UNION ALL SELECT patientunitstayid, offset_min, 'systolic_bp', systemicsystolic, 2 FROM eicu_vital_periodic_raw_24h
              UNION ALL SELECT patientunitstayid, offset_min, 'diastolic_bp', systemicdiastolic, 2 FROM eicu_vital_periodic_raw_24h
              UNION ALL SELECT patientunitstayid, offset_min, 'systolic_bp', noninvasivesystolic, 0 FROM eicu_vital_aperiodic_raw_24h
              UNION ALL SELECT patientunitstayid, offset_min, 'diastolic_bp', noninvasivediastolic, 0 FROM eicu_vital_aperiodic_raw_24h
            )
            SELECT patientunitstayid, offset_min, name, value, source_priority
            FROM events
            WHERE value IS NOT NULL
              AND ({vital_range_sql})
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE eicu_vitals_24h_wide AS
            WITH ranked AS (
              SELECT
                patientunitstayid,
                name,
                value,
                ROW_NUMBER() OVER (PARTITION BY patientunitstayid, name ORDER BY offset_min ASC, source_priority ASC) AS rn
              FROM eicu_vital_events_24h
            ),
            agg AS (
              SELECT
                patientunitstayid,
                name,
                MAX(CASE WHEN rn = 1 THEN value END) AS first_value,
                AVG(value) AS mean_value,
                MIN(value) AS min_value,
                MAX(value) AS max_value,
                COUNT(*) AS count_value
              FROM ranked
              GROUP BY patientunitstayid, name
            )
            SELECT
              patientunitstayid,
              MAX(CASE WHEN name='heart_rate' THEN first_value END) AS heart_rate_first_24h,
              MAX(CASE WHEN name='heart_rate' THEN mean_value END) AS heart_rate_mean_24h,
              MAX(CASE WHEN name='heart_rate' THEN max_value END) AS heart_rate_max_24h,
              MAX(CASE WHEN name='heart_rate' THEN count_value END) AS heart_rate_count_24h,
              MAX(CASE WHEN name='respiratory_rate' THEN first_value END) AS respiratory_rate_first_24h,
              MAX(CASE WHEN name='respiratory_rate' THEN mean_value END) AS respiratory_rate_mean_24h,
              MAX(CASE WHEN name='respiratory_rate' THEN max_value END) AS respiratory_rate_max_24h,
              MAX(CASE WHEN name='respiratory_rate' THEN count_value END) AS respiratory_rate_count_24h,
              MAX(CASE WHEN name='temperature' THEN first_value END) AS temperature_first_24h,
              MAX(CASE WHEN name='temperature' THEN mean_value END) AS temperature_mean_24h,
              MAX(CASE WHEN name='temperature' THEN max_value END) AS temperature_max_24h,
              MAX(CASE WHEN name='temperature' THEN count_value END) AS temperature_count_24h,
              MAX(CASE WHEN name='spo2' THEN first_value END) AS spo2_first_24h,
              MAX(CASE WHEN name='spo2' THEN mean_value END) AS spo2_mean_24h,
              MAX(CASE WHEN name='spo2' THEN min_value END) AS spo2_min_24h,
              MAX(CASE WHEN name='spo2' THEN count_value END) AS spo2_count_24h,
              MAX(CASE WHEN name='systolic_bp' THEN first_value END) AS systolic_bp_first_24h,
              MAX(CASE WHEN name='systolic_bp' THEN mean_value END) AS systolic_bp_mean_24h,
              MAX(CASE WHEN name='systolic_bp' THEN min_value END) AS systolic_bp_min_24h,
              MAX(CASE WHEN name='systolic_bp' THEN count_value END) AS systolic_bp_count_24h,
              MAX(CASE WHEN name='diastolic_bp' THEN first_value END) AS diastolic_bp_first_24h,
              MAX(CASE WHEN name='diastolic_bp' THEN mean_value END) AS diastolic_bp_mean_24h,
              MAX(CASE WHEN name='diastolic_bp' THEN min_value END) AS diastolic_bp_min_24h,
              MAX(CASE WHEN name='diastolic_bp' THEN count_value END) AS diastolic_bp_count_24h
            FROM agg
            GROUP BY patientunitstayid
            """
        )
        vital_counts = con.execute(
            """
            SELECT name, COUNT(*) AS row_count, COUNT(DISTINCT patientunitstayid) AS stays
            FROM eicu_vital_events_24h
            GROUP BY name
            ORDER BY stays DESC
            """
        ).fetchdf()
        vitals = con.execute("SELECT * FROM eicu_vitals_24h_wide").fetchdf()

    metadata = {
        "lab_counts": lab_counts.to_dict(orient="records"),
        "vital_counts": vital_counts.to_dict(orient="records") if not vital_counts.empty else [],
        "lab_plausible_ranges": LAB_PLAUSIBLE_RANGES,
        "vital_plausible_ranges": VITAL_PLAUSIBLE_RANGES,
        "time_window_minutes_after_icu_admission": [0, 1440],
    }
    con.close()
    return labs, vitals, metadata


def completeness_table(df: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    rows = []
    for column in columns:
        rows.append(
            {
                "column": column,
                "nonmissing": int(df[column].notna().sum()) if column in df.columns else 0,
                "missing": int(df[column].isna().sum()) if column in df.columns else int(len(df)),
                "nonmissing_rate": float(df[column].notna().mean()) if column in df.columns else 0.0,
            }
        )
    return rows


def write_readme(metadata: dict[str, object], path: Path) -> None:
    minimal = metadata["minimal_cohort"]
    enhanced = metadata["enhanced_cohort"]
    lines = [
        "# eICU Diabetes Hospital Mortality Cohort",
        "",
        "Generated by `scripts/build_eicu_diabetes_hospital_mortality_cohort.py`.",
        "",
        "## Endpoint",
        "",
        "- Primary endpoint: hospital mortality from `patient.hospitaldischargestatus == 'Expired'`.",
        "- Sensitivity endpoint retained: ICU/unit mortality from `patient.unitdischargestatus == 'Expired'`.",
        "- Time window for labs/vitals: offsets 0 to 1440 minutes after ICU admission.",
        "",
        "## Minimal Cohort",
        "",
        f"- Rows: {minimal['rows']:,}",
        f"- Unique patients: {minimal['unique_patients']:,}",
        f"- Hospitals: {minimal['hospitals']:,}",
        f"- Hospital mortality non-missing: {minimal['hospital_mortality_nonmissing']:,}",
        f"- Hospital mortality events: {minimal['hospital_mortality_events']:,}",
        f"- Hospital mortality event rate: {minimal['hospital_mortality_event_rate']:.3%}",
        f"- Unit mortality events: {minimal['unit_mortality_events']:,}",
        f"- Unit mortality event rate: {minimal['unit_mortality_event_rate']:.3%}",
        "",
        "## Enhanced Cohort",
        "",
        f"- Rows: {enhanced['rows']:,}",
        f"- Columns: {enhanced['columns']:,}",
        "",
        "## Files",
        "",
        "- `eicu_crd20_diabetes_minimal_cohort.csv`",
        "- `eicu_crd20_diabetes_labs_24h_wide.csv`",
        "- `eicu_crd20_diabetes_vitals_24h_wide.csv`",
        "- `eicu_crd20_diabetes_lab_vital_enhanced_cohort.csv`",
        "- `eicu_crd20_diabetes_cohort_metadata.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an eICU diabetes ICU-stay cohort for hospital mortality transport studies.")
    parser.add_argument("--eicu-root", type=Path, default=DEFAULT_EICU_ROOT)
    parser.add_argument("--skip-large-tables", action="store_true", help="Skip vitalPeriodic/vitalAperiodic extraction.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.eicu_root.exists():
        raise FileNotFoundError(f"eICU root not found: {args.eicu_root}")

    print("Building minimal eICU diabetes cohort...")
    cohort, minimal_metadata = build_minimal_cohort(args.eicu_root)
    minimal_path = OUT_DIR / "eicu_crd20_diabetes_minimal_cohort.csv"
    cohort.to_csv(minimal_path, index=False)

    labs, vitals, feature_metadata = build_lab_vital_features(args.eicu_root, cohort, skip_large_tables=args.skip_large_tables)
    labs_path = OUT_DIR / "eicu_crd20_diabetes_labs_24h_wide.csv"
    vitals_path = OUT_DIR / "eicu_crd20_diabetes_vitals_24h_wide.csv"
    enhanced_path = OUT_DIR / "eicu_crd20_diabetes_lab_vital_enhanced_cohort.csv"
    labs.to_csv(labs_path, index=False)
    vitals.to_csv(vitals_path, index=False)

    enhanced = cohort.merge(labs, on="patientunitstayid", how="left").merge(vitals, on="patientunitstayid", how="left")
    enhanced.to_csv(enhanced_path, index=False)

    key_columns = [
        "age",
        "female",
        "bmi",
        "hypertension_history",
        "ckd_history",
        "cvd_history",
        "hospital_mortality",
        "unit_mortality",
        "glucose_first_24h",
        "creatinine_first_24h",
        "bun_first_24h",
        "wbc_first_24h",
        "hemoglobin_first_24h",
        "albumin_first_24h",
        "heart_rate_first_24h",
        "systolic_bp_first_24h",
        "diastolic_bp_first_24h",
        "respiratory_rate_first_24h",
        "temperature_first_24h",
        "spo2_first_24h",
    ]
    metadata = {
        "eicu_root": str(args.eicu_root),
        "minimal_output": str(minimal_path),
        "labs_output": str(labs_path),
        "vitals_output": str(vitals_path),
        "enhanced_output": str(enhanced_path),
        "minimal_cohort": minimal_metadata,
        "enhanced_cohort": {
            "rows": int(len(enhanced)),
            "columns": int(len(enhanced.columns)),
        },
        "feature_extraction": feature_metadata,
        "key_column_completeness": completeness_table(enhanced, key_columns),
        "notes": [
            "Primary analysis should use hospital_mortality for endpoint alignment with MIMIC hospital_expire_flag.",
            "unit_mortality is retained for ICU mortality sensitivity analysis.",
            "eICU time variables are offsets in minutes relative to ICU admission; labs and vitals are extracted from offsets 0..1440.",
            "Diabetes candidate definition combines APACHE diabetes flag, diagnosis codes/strings, admission diagnosis, and past history.",
        ],
    }
    metadata_path = OUT_DIR / "eicu_crd20_diabetes_cohort_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(metadata, OUT_DIR / "README_eicu_diabetes_hospital_mortality_cohort.md")
    print(json.dumps(metadata["minimal_cohort"], ensure_ascii=False, indent=2))
    print(f"Wrote {enhanced_path}")


if __name__ == "__main__":
    main()
