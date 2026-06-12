from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_ROOT = Path(r"D:\DATABASE\mimic-iv-3.1")
LIGHT_COHORT = PROJECT_ROOT / "data" / "mimic" / "processed" / "mimic_iv31_diabetes_light_admissions.csv"
OUT_DIR = PROJECT_ROOT / "data" / "mimic" / "processed"
TMP_DIR = PROJECT_ROOT / "outputs" / "duckdb_tmp"
PSQL_EXE = Path(r"D:\MIMIC\PostgreSQL\bin\psql.exe")

LAB_ITEM_MAP = {
    "glucose": [50931, 50809, 52569],
    "creatinine": [50912, 52546],
    "bun": [51006, 52647],
    "hemoglobin": [51222, 50811],
    "wbc": [51301, 51300],
    "albumin": [50862, 53085],
}

LAB_PLAUSIBLE_RANGES = {
    "glucose": (20, 1000),
    "creatinine": (0.1, 30),
    "bun": (1, 300),
    "hemoglobin": (3, 25),
    "wbc": (0.1, 200),
    "albumin": (0.5, 8),
}

VITAL_ITEM_MAP = {
    "heart_rate": [220045],
    "respiratory_rate": [220210],
    "spo2": [220277],
    "systolic_bp": [220050, 220179, 224167, 227243],
    "diastolic_bp": [220051, 220180, 224643, 227242],
    "temperature_c": [223762, 226329, 229236],
    "temperature_f": [223761, 227054],
}

VITAL_PLAUSIBLE_RANGES = {
    "heart_rate": (20, 250),
    "respiratory_rate": (3, 80),
    "spo2": (40, 100),
    "systolic_bp": (50, 300),
    "diastolic_bp": (20, 180),
    "temperature": (25, 45),
}


def slash(path: Path) -> str:
    return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_readable(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as handle:
            handle.read(2)
        return True
    except OSError:
        return False


def sql_list(values: list[int]) -> str:
    return ", ".join(str(v) for v in values)


def case_sql(mapping: dict[str, list[int]], value_name: str = "itemid") -> str:
    return "\n              ".join(
        f"WHEN {value_name} IN ({sql_list(itemids)}) THEN '{name}'" for name, itemids in mapping.items()
    )


def range_sql(mapping: dict[str, tuple[float, float]], name_col: str = "name", value_col: str = "value") -> str:
    return " OR\n          ".join(
        f"({name_col} = '{name}' AND {value_col} BETWEEN {low} AND {high})" for name, (low, high) in mapping.items()
    )


def first_icu_diabetes_cohort() -> tuple[pd.DataFrame, dict[str, object]]:
    light = pd.read_csv(LIGHT_COHORT)
    icu = pd.read_csv(MIMIC_ROOT / "icu" / "icustays.csv")
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    first_icu = (
        icu.sort_values(["subject_id", "hadm_id", "intime", "stay_id"])
        .groupby(["subject_id", "hadm_id"], as_index=False)
        .first()
    )
    cohort = light.merge(
        first_icu[["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime", "los"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    cohort = cohort.rename(
        columns={
            "intime": "icu_intime",
            "outtime": "icu_outtime",
            "los": "icu_los_days",
            "hospital_expire_flag": "hospital_mortality",
        }
    )
    cohort["hospital_mortality"] = pd.to_numeric(cohort["hospital_mortality"], errors="coerce")
    cohort = cohort.sort_values(["subject_id", "hadm_id", "icu_intime"]).reset_index(drop=True)

    metadata = {
        "rows": int(len(cohort)),
        "subjects": int(cohort["subject_id"].nunique()),
        "hospital_mortality_nonmissing": int(cohort["hospital_mortality"].notna().sum()),
        "hospital_mortality_events": int(cohort["hospital_mortality"].fillna(0).sum()),
        "hospital_mortality_event_rate": float(cohort["hospital_mortality"].mean()),
        "notes": [
            "One row per diabetes admission with at least one ICU stay.",
            "When an admission has multiple ICU stays, the first ICU stay by intime is retained.",
            "Primary endpoint is admissions.hospital_expire_flag, renamed hospital_mortality.",
        ],
    }
    return cohort, metadata


def run_psql_copy(query: str, out_path: Path, database: str = "mimiciv31") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    copy_sql = f"\\copy ({query}) TO '{slash(out_path)}' WITH CSV HEADER"
    subprocess.run(
        [
            str(PSQL_EXE),
            "-h",
            "127.0.0.1",
            "-p",
            "5442",
            "-U",
            "postgres",
            "-d",
            database,
            "-c",
            copy_sql,
        ],
        check=True,
    )


def build_features_from_postgres_derived(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read MIMIC-IV derived first-day ICU tables from the local PostgreSQL install."""
    stay_ids = cohort[["stay_id"]].drop_duplicates().copy()
    stay_ids_path = TMP_DIR / "mimic_icu_stay_ids_for_postgres.csv"
    stay_ids.to_csv(stay_ids_path, index=False, quoting=csv.QUOTE_MINIMAL)
    subprocess.run(
        [
            str(PSQL_EXE),
            "-h",
            "127.0.0.1",
            "-p",
            "5442",
            "-U",
            "postgres",
            "-d",
            "mimiciv31",
            "-c",
            "DROP TABLE IF EXISTS public.tmp_mimic_icu_stay_ids; CREATE TABLE public.tmp_mimic_icu_stay_ids (stay_id integer);",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(PSQL_EXE),
            "-h",
            "127.0.0.1",
            "-p",
            "5442",
            "-U",
            "postgres",
            "-d",
            "mimiciv31",
            "-c",
            f"\\copy public.tmp_mimic_icu_stay_ids(stay_id) FROM '{slash(stay_ids_path)}' WITH CSV HEADER",
        ],
        check=True,
    )
    lab_query = """
        SELECT
          l.stay_id,
          l.glucose_min AS glucose_first_24h,
          l.glucose_min AS glucose_min_24h,
          l.glucose_max AS glucose_max_24h,
          l.creatinine_min AS creatinine_first_24h,
          l.creatinine_max AS creatinine_max_24h,
          l.bun_min AS bun_first_24h,
          l.bun_max AS bun_max_24h,
          l.hemoglobin_min AS hemoglobin_first_24h,
          l.hemoglobin_min AS hemoglobin_min_24h,
          l.wbc_max AS wbc_first_24h,
          l.wbc_max AS wbc_max_24h,
          l.albumin_min AS albumin_first_24h,
          l.albumin_min AS albumin_min_24h
        FROM mimiciv_derived.first_day_lab l
        JOIN public.tmp_mimic_icu_stay_ids s ON l.stay_id = s.stay_id
        """
    vital_query = """
        SELECT
          v.stay_id,
          v.heart_rate_mean AS heart_rate_first_24h,
          v.heart_rate_mean AS heart_rate_mean_24h,
          v.heart_rate_max AS heart_rate_max_24h,
          v.resp_rate_mean AS respiratory_rate_first_24h,
          v.resp_rate_mean AS respiratory_rate_mean_24h,
          v.resp_rate_max AS respiratory_rate_max_24h,
          v.spo2_mean AS spo2_first_24h,
          v.spo2_mean AS spo2_mean_24h,
          v.spo2_min AS spo2_min_24h,
          v.sbp_mean AS systolic_bp_first_24h,
          v.sbp_mean AS systolic_bp_mean_24h,
          v.sbp_min AS systolic_bp_min_24h,
          v.dbp_mean AS diastolic_bp_first_24h,
          v.dbp_mean AS diastolic_bp_mean_24h,
          v.dbp_min AS diastolic_bp_min_24h,
          v.temperature_mean AS temperature_first_24h,
          v.temperature_mean AS temperature_mean_24h,
          v.temperature_max AS temperature_max_24h
        FROM mimiciv_derived.first_day_vitalsign v
        JOIN public.tmp_mimic_icu_stay_ids s ON v.stay_id = s.stay_id
        """
    lab_path = TMP_DIR / "mimic_icu_postgres_derived_labs.csv"
    vital_path = TMP_DIR / "mimic_icu_postgres_derived_vitals.csv"
    run_psql_copy(lab_query, lab_path)
    run_psql_copy(vital_query, vital_path)
    subprocess.run(
        [
            str(PSQL_EXE),
            "-h",
            "127.0.0.1",
            "-p",
            "5442",
            "-U",
            "postgres",
            "-d",
            "mimiciv31",
            "-c",
            "DROP TABLE IF EXISTS public.tmp_mimic_icu_stay_ids;",
        ],
        check=True,
    )
    labs = pd.read_csv(lab_path)
    vitals = pd.read_csv(vital_path)

    lab_counts = []
    for name, col in [
        ("glucose", "glucose_first_24h"),
        ("creatinine", "creatinine_first_24h"),
        ("bun", "bun_first_24h"),
        ("hemoglobin", "hemoglobin_first_24h"),
        ("wbc", "wbc_first_24h"),
        ("albumin", "albumin_first_24h"),
    ]:
        lab_counts.append({"name": name, "row_count": int(labs[col].notna().sum()), "stays": int(labs[col].notna().sum())})
    vital_counts = []
    for name, col in [
        ("heart_rate", "heart_rate_first_24h"),
        ("respiratory_rate", "respiratory_rate_first_24h"),
        ("spo2", "spo2_first_24h"),
        ("systolic_bp", "systolic_bp_first_24h"),
        ("diastolic_bp", "diastolic_bp_first_24h"),
        ("temperature", "temperature_first_24h"),
    ]:
        vital_counts.append({"name": name, "row_count": int(vitals[col].notna().sum()), "stays": int(vitals[col].notna().sum())})
    return labs, vitals, pd.DataFrame(lab_counts), pd.DataFrame(vital_counts)


def build_lab_features(con: duckdb.DuckDBPyConnection, cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labevents_path = MIMIC_ROOT / "hosp" / "labevents.csv"
    con.register("mimic_icu_cohort_df", cohort[["hadm_id", "stay_id", "icu_intime"]])
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE mimic_icu_cohort AS
        SELECT
          hadm_id::BIGINT AS hadm_id,
          stay_id::BIGINT AS stay_id,
          icu_intime::TIMESTAMP AS icu_intime
        FROM mimic_icu_cohort_df
        """
    )
    all_itemids = sorted({itemid for itemids in LAB_ITEM_MAP.values() for itemid in itemids})
    print("Building MIMIC ICU lab features from ICU first 24h...")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE mimic_icu_lab_events_24h AS
        WITH raw AS (
          SELECT
            c.stay_id,
            l.hadm_id::BIGINT AS hadm_id,
            l.itemid::INTEGER AS itemid,
            l.charttime::TIMESTAMP AS charttime,
            TRY_CAST(l.valuenum AS DOUBLE) AS value,
            CASE
              {case_sql(LAB_ITEM_MAP, 'l.itemid')}
            END AS name,
            c.icu_intime AS icu_intime
          FROM read_csv_auto('{slash(labevents_path)}', columns={{
            'labevent_id': 'BIGINT',
            'subject_id': 'BIGINT',
            'hadm_id': 'BIGINT',
            'specimen_id': 'BIGINT',
            'itemid': 'INTEGER',
            'order_provider_id': 'VARCHAR',
            'charttime': 'TIMESTAMP',
            'storetime': 'TIMESTAMP',
            'value': 'VARCHAR',
            'valuenum': 'DOUBLE',
            'valueuom': 'VARCHAR',
            'ref_range_lower': 'DOUBLE',
            'ref_range_upper': 'DOUBLE',
            'flag': 'VARCHAR',
            'priority': 'VARCHAR',
            'comments': 'VARCHAR'
          }}, ignore_errors=true, parallel=true) l
          JOIN mimic_icu_cohort c ON l.hadm_id = c.hadm_id
          WHERE l.itemid IN ({sql_list(all_itemids)})
        )
        SELECT stay_id, hadm_id, itemid, charttime, name, value
        FROM raw
        WHERE name IS NOT NULL
          AND value IS NOT NULL
          AND charttime >= icu_intime
          AND charttime <= icu_intime + INTERVAL 24 HOURS
          AND ({range_sql(LAB_PLAUSIBLE_RANGES)})
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE mimic_icu_labs_24h_wide AS
        WITH ranked AS (
          SELECT
            stay_id,
            name,
            value,
            ROW_NUMBER() OVER (PARTITION BY stay_id, name ORDER BY charttime ASC) AS rn
          FROM mimic_icu_lab_events_24h
        ),
        agg AS (
          SELECT
            stay_id,
            name,
            MAX(CASE WHEN rn = 1 THEN value END) AS first_value,
            AVG(value) AS mean_value,
            MIN(value) AS min_value,
            MAX(value) AS max_value,
            COUNT(*) AS count_value
          FROM ranked
          GROUP BY stay_id, name
        )
        SELECT
          stay_id,
          MAX(CASE WHEN name='glucose' THEN first_value END) AS glucose_first_24h,
          MAX(CASE WHEN name='creatinine' THEN first_value END) AS creatinine_first_24h,
          MAX(CASE WHEN name='bun' THEN first_value END) AS bun_first_24h,
          MAX(CASE WHEN name='hemoglobin' THEN first_value END) AS hemoglobin_first_24h,
          MAX(CASE WHEN name='wbc' THEN first_value END) AS wbc_first_24h,
          MAX(CASE WHEN name='albumin' THEN first_value END) AS albumin_first_24h,
          MAX(CASE WHEN name='glucose' THEN mean_value END) AS glucose_mean_24h,
          MAX(CASE WHEN name='glucose' THEN min_value END) AS glucose_min_24h,
          MAX(CASE WHEN name='glucose' THEN max_value END) AS glucose_max_24h,
          MAX(CASE WHEN name='glucose' THEN count_value END) AS glucose_count_24h,
          MAX(CASE WHEN name='creatinine' THEN max_value END) AS creatinine_max_24h,
          MAX(CASE WHEN name='creatinine' THEN count_value END) AS creatinine_count_24h,
          MAX(CASE WHEN name='bun' THEN max_value END) AS bun_max_24h,
          MAX(CASE WHEN name='bun' THEN count_value END) AS bun_count_24h,
          MAX(CASE WHEN name='hemoglobin' THEN min_value END) AS hemoglobin_min_24h,
          MAX(CASE WHEN name='hemoglobin' THEN count_value END) AS hemoglobin_count_24h,
          MAX(CASE WHEN name='wbc' THEN max_value END) AS wbc_max_24h,
          MAX(CASE WHEN name='wbc' THEN count_value END) AS wbc_count_24h,
          MAX(CASE WHEN name='albumin' THEN min_value END) AS albumin_min_24h,
          MAX(CASE WHEN name='albumin' THEN count_value END) AS albumin_count_24h
        FROM agg
        GROUP BY stay_id
        """
    )
    counts = con.execute(
        """
        SELECT name, COUNT(*) AS row_count, COUNT(DISTINCT stay_id) AS stays
        FROM mimic_icu_lab_events_24h
        GROUP BY name
        ORDER BY stays DESC
        """
    ).fetchdf()
    wide = con.execute("SELECT * FROM mimic_icu_labs_24h_wide").fetchdf()
    return wide, counts


def build_vital_features(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    chartevents_path = MIMIC_ROOT / "icu" / "chartevents.csv.gz"
    if not gzip_readable(chartevents_path):
        raise OSError(f"{chartevents_path} is not a readable gzip stream.")
    vital_map_for_sql = {
        "heart_rate": VITAL_ITEM_MAP["heart_rate"],
        "respiratory_rate": VITAL_ITEM_MAP["respiratory_rate"],
        "spo2": VITAL_ITEM_MAP["spo2"],
        "systolic_bp": VITAL_ITEM_MAP["systolic_bp"],
        "diastolic_bp": VITAL_ITEM_MAP["diastolic_bp"],
        "temperature_c": VITAL_ITEM_MAP["temperature_c"],
        "temperature_f": VITAL_ITEM_MAP["temperature_f"],
    }
    all_itemids = sorted({itemid for itemids in vital_map_for_sql.values() for itemid in itemids})
    print("Building MIMIC ICU vital features from ICU first 24h...")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE mimic_icu_vital_events_24h AS
        WITH raw AS (
          SELECT
            c.stay_id,
            ce.itemid::INTEGER AS itemid,
            ce.charttime::TIMESTAMP AS charttime,
            TRY_CAST(ce.valuenum AS DOUBLE) AS raw_value,
            CASE
              {case_sql(vital_map_for_sql, 'ce.itemid')}
            END AS raw_name,
            c.icu_intime AS icu_intime
          FROM read_csv_auto('{slash(chartevents_path)}', columns={{
            'subject_id': 'BIGINT',
            'hadm_id': 'BIGINT',
            'stay_id': 'BIGINT',
            'caregiver_id': 'BIGINT',
            'charttime': 'TIMESTAMP',
            'storetime': 'TIMESTAMP',
            'itemid': 'INTEGER',
            'value': 'VARCHAR',
            'valuenum': 'DOUBLE',
            'valueuom': 'VARCHAR',
            'warning': 'INTEGER'
          }}, ignore_errors=true, parallel=true) ce
          JOIN mimic_icu_cohort c ON ce.stay_id = c.stay_id
          WHERE ce.itemid IN ({sql_list(all_itemids)})
        ),
        mapped AS (
          SELECT
            stay_id,
            itemid,
            charttime,
            CASE
              WHEN raw_name = 'temperature_f' THEN 'temperature'
              WHEN raw_name = 'temperature_c' THEN 'temperature'
              ELSE raw_name
            END AS name,
            CASE
              WHEN raw_name = 'temperature_f' THEN (raw_value - 32) * 5.0 / 9.0
              ELSE raw_value
            END AS value
          FROM raw
          WHERE raw_name IS NOT NULL
            AND raw_value IS NOT NULL
            AND charttime >= icu_intime
            AND charttime <= icu_intime + INTERVAL 24 HOURS
        )
        SELECT stay_id, itemid, charttime, name, value
        FROM mapped
        WHERE {range_sql(VITAL_PLAUSIBLE_RANGES)}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE mimic_icu_vitals_24h_wide AS
        WITH ranked AS (
          SELECT
            stay_id,
            name,
            value,
            ROW_NUMBER() OVER (PARTITION BY stay_id, name ORDER BY charttime ASC) AS rn
          FROM mimic_icu_vital_events_24h
        ),
        agg AS (
          SELECT
            stay_id,
            name,
            MAX(CASE WHEN rn = 1 THEN value END) AS first_value,
            AVG(value) AS mean_value,
            MIN(value) AS min_value,
            MAX(value) AS max_value,
            COUNT(*) AS count_value
          FROM ranked
          GROUP BY stay_id, name
        )
        SELECT
          stay_id,
          MAX(CASE WHEN name='heart_rate' THEN first_value END) AS heart_rate_first_24h,
          MAX(CASE WHEN name='heart_rate' THEN mean_value END) AS heart_rate_mean_24h,
          MAX(CASE WHEN name='heart_rate' THEN max_value END) AS heart_rate_max_24h,
          MAX(CASE WHEN name='heart_rate' THEN count_value END) AS heart_rate_count_24h,
          MAX(CASE WHEN name='respiratory_rate' THEN first_value END) AS respiratory_rate_first_24h,
          MAX(CASE WHEN name='respiratory_rate' THEN mean_value END) AS respiratory_rate_mean_24h,
          MAX(CASE WHEN name='respiratory_rate' THEN max_value END) AS respiratory_rate_max_24h,
          MAX(CASE WHEN name='respiratory_rate' THEN count_value END) AS respiratory_rate_count_24h,
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
          MAX(CASE WHEN name='diastolic_bp' THEN count_value END) AS diastolic_bp_count_24h,
          MAX(CASE WHEN name='temperature' THEN first_value END) AS temperature_first_24h,
          MAX(CASE WHEN name='temperature' THEN mean_value END) AS temperature_mean_24h,
          MAX(CASE WHEN name='temperature' THEN max_value END) AS temperature_max_24h,
          MAX(CASE WHEN name='temperature' THEN count_value END) AS temperature_count_24h
        FROM agg
        GROUP BY stay_id
        """
    )
    counts = con.execute(
        """
        SELECT name, COUNT(*) AS row_count, COUNT(DISTINCT stay_id) AS stays
        FROM mimic_icu_vital_events_24h
        GROUP BY name
        ORDER BY stays DESC
        """
    ).fetchdf()
    wide = con.execute("SELECT * FROM mimic_icu_vitals_24h_wide").fetchdf()
    return wide, counts


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
    cohort = metadata["minimal_cohort"]
    lines = [
        "# MIMIC-IV ICU Diabetes Hospital Mortality Cohort",
        "",
        "Generated by `scripts/build_mimic_icu_diabetes_hospital_mortality_cohort.py`.",
        "",
        "## Endpoint and Unit",
        "",
        "- Unit of analysis: one row per diabetes hospital admission with at least one ICU stay.",
        "- ICU anchor: first ICU stay within the admission.",
        "- Primary endpoint: `admissions.hospital_expire_flag`, renamed `hospital_mortality`.",
        "- Lab/vital window: `icustays.intime <= charttime <= icustays.intime + 24 hours`.",
        "",
        "## Cohort",
        "",
        f"- Rows: {cohort['rows']:,}",
        f"- Subjects: {cohort['subjects']:,}",
        f"- Hospital mortality non-missing: {cohort['hospital_mortality_nonmissing']:,}",
        f"- Hospital mortality events: {cohort['hospital_mortality_events']:,}",
        f"- Hospital mortality event rate: {cohort['hospital_mortality_event_rate']:.3%}",
        "",
        "## Files",
        "",
        "- `mimic_iv31_diabetes_icu_minimal_cohort.csv`",
        "- `mimic_iv31_diabetes_icu_labs_24h_wide.csv`",
        "- `mimic_iv31_diabetes_icu_vitals_24h_wide.csv`",
        "- `mimic_iv31_diabetes_icu_lab_vital_enhanced_cohort.csv`",
        "- `mimic_iv31_diabetes_icu_cohort_metadata.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a MIMIC-IV diabetes ICU cohort aligned to eICU hospital mortality.")
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--source", choices=["postgres-derived", "csv"], default="postgres-derived")
    parser.add_argument(
        "--skip-vitals",
        action="store_true",
        help="Skip chartevents-derived vitals and still write a lab-only aligned cohort.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("Building MIMIC ICU diabetes cohort...")
    cohort, minimal_metadata = first_icu_diabetes_cohort()
    minimal_path = OUT_DIR / "mimic_iv31_diabetes_icu_minimal_cohort.csv"
    cohort.to_csv(minimal_path, index=False)

    vital_error = None
    chartevents_path = MIMIC_ROOT / "icu" / "chartevents.csv.gz"
    if args.source == "postgres-derived":
        print("Building MIMIC ICU labs/vitals from PostgreSQL derived first-day tables...")
        labs, vitals, lab_counts, vital_counts = build_features_from_postgres_derived(cohort)
    else:
        con = duckdb.connect(str(TMP_DIR / "mimic_icu_diabetes_wide.duckdb"))
        con.execute(f"SET temp_directory='{slash(TMP_DIR)}'")
        con.execute(f"SET memory_limit='{args.memory_limit}'")
        con.execute(f"PRAGMA threads={args.threads}")

        labs, lab_counts = build_lab_features(con, cohort)
        if args.skip_vitals:
            vitals = pd.DataFrame({"stay_id": cohort["stay_id"].copy()})
            vital_counts = pd.DataFrame()
            vital_error = "Skipped by --skip-vitals."
        else:
            try:
                vitals, vital_counts = build_vital_features(con)
            except Exception as exc:
                vitals = pd.DataFrame({"stay_id": cohort["stay_id"].copy()})
                vital_counts = pd.DataFrame()
                vital_error = repr(exc)
        con.close()
    labs_path = OUT_DIR / "mimic_iv31_diabetes_icu_labs_24h_wide.csv"
    labs.to_csv(labs_path, index=False)

    vitals_path = OUT_DIR / "mimic_iv31_diabetes_icu_vitals_24h_wide.csv"
    enhanced_path = OUT_DIR / "mimic_iv31_diabetes_icu_lab_vital_enhanced_cohort.csv"
    vitals.to_csv(vitals_path, index=False)
    enhanced = cohort.merge(labs, on="stay_id", how="left").merge(vitals, on="stay_id", how="left")
    enhanced.to_csv(enhanced_path, index=False)

    key_columns = [
        "age",
        "female",
        "bmi",
        "hypertension_history",
        "ckd_history",
        "cvd_history",
        "hospital_mortality",
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
        "mimic_root": str(MIMIC_ROOT),
        "minimal_output": str(minimal_path),
        "labs_output": str(labs_path),
        "vitals_output": str(vitals_path),
        "enhanced_output": str(enhanced_path),
        "minimal_cohort": minimal_metadata,
        "enhanced_cohort": {
            "rows": int(len(enhanced)),
            "columns": int(len(enhanced.columns)),
        },
        "feature_extraction": {
            "lab_counts": lab_counts.to_dict(orient="records"),
            "vital_counts": vital_counts.to_dict(orient="records"),
            "vital_extraction_error": vital_error,
            "chartevents_path": str(chartevents_path),
            "chartevents_sha256": sha256_file(chartevents_path) if chartevents_path.exists() else None,
            "chartevents_gzip_readable": gzip_readable(chartevents_path) if chartevents_path.exists() else False,
            "lab_item_map": LAB_ITEM_MAP,
            "vital_item_map": VITAL_ITEM_MAP,
            "lab_plausible_ranges": LAB_PLAUSIBLE_RANGES,
            "vital_plausible_ranges": VITAL_PLAUSIBLE_RANGES,
            "time_window": "icustays.intime to icustays.intime + 24 hours",
        },
        "key_column_completeness": completeness_table(enhanced, key_columns),
        "notes": [
            "This cohort is aligned to eICU by using ICU-stay anchoring and hospital mortality.",
            "MIMIC uses absolute timestamps; eICU uses offset minutes. Both scripts extract ICU first-24h variables.",
            "Albumin and temperature are retained for sensitivity analyses, not recommended for the primary shared-feature model.",
        ],
    }
    metadata_path = OUT_DIR / "mimic_iv31_diabetes_icu_cohort_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(metadata, OUT_DIR / "README_mimic_icu_diabetes_hospital_mortality_cohort.md")
    print(json.dumps(metadata["minimal_cohort"], ensure_ascii=False, indent=2))
    print(f"Wrote {enhanced_path}")


if __name__ == "__main__":
    main()
