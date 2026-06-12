from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_ROOT = Path(r"D:\DATABASE\mimic-iv-3.1")
LIGHT_COHORT = PROJECT_ROOT / "data" / "mimic" / "processed" / "mimic_iv31_diabetes_light_admissions.csv"
OUT_DIR = PROJECT_ROOT / "data" / "mimic" / "processed"
TMP_DIR = PROJECT_ROOT / "outputs" / "duckdb_tmp"


LAB_ITEM_MAP = {
    "hba1c": [50852],
    "glucose": [50931, 50809, 52569],
    "creatinine": [50912, 52546],
    "albumin": [50862, 53085],
    "hemoglobin": [51222, 50811],
    "wbc": [51301, 51300],
    "total_cholesterol": [50907],
    "hdl_cholesterol": [50904],
    "ldl_cholesterol": [50905, 50906],
    "triglycerides": [51000],
    "uacr": [51070],
}

PLAUSIBLE_RANGES = {
    "hba1c": (3, 20),
    "glucose": (20, 1000),
    "creatinine": (0.1, 30),
    "albumin": (0.5, 8),
    "hemoglobin": (3, 25),
    "wbc": (0.1, 200),
    "total_cholesterol": (30, 1000),
    "hdl_cholesterol": (5, 200),
    "ldl_cholesterol": (5, 500),
    "triglycerides": (10, 5000),
    "uacr": (0, 100000),
}


def sql_list(values: list[int]) -> str:
    return ", ".join(str(v) for v in values)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    labevents_path = MIMIC_ROOT / "hosp" / "labevents.csv"

    con = duckdb.connect(str(TMP_DIR / "mimic_labs_24h.duckdb"))
    con.execute(f"SET temp_directory='{str(TMP_DIR).replace('\\', '/')}'")
    con.execute("SET memory_limit='8GB'")
    con.execute("PRAGMA threads=4")

    cohort = pd.read_csv(LIGHT_COHORT, usecols=["hadm_id", "admittime"])
    cohort["hadm_id"] = cohort["hadm_id"].astype("int64")
    cohort["admittime"] = pd.to_datetime(cohort["admittime"], errors="coerce")
    con.register("cohort_df", cohort)
    con.execute("CREATE OR REPLACE TEMP TABLE cohort AS SELECT hadm_id, admittime::TIMESTAMP AS admittime FROM cohort_df")

    case_parts = []
    for lab, itemids in LAB_ITEM_MAP.items():
        case_parts.append(f"WHEN itemid IN ({sql_list(itemids)}) THEN '{lab}'")
    case_sql = "\n        ".join(case_parts)
    itemids_all = sorted({itemid for itemids in LAB_ITEM_MAP.values() for itemid in itemids})

    range_parts = []
    for lab, (low, high) in PLAUSIBLE_RANGES.items():
        range_parts.append(f"(lab_name = '{lab}' AND valuenum BETWEEN {low} AND {high})")
    range_sql = " OR\n          ".join(range_parts)

    # DuckDB reads and filters the CSV lazily. HADM and itemid predicates are applied before materialization.
    query_events = f"""
    CREATE OR REPLACE TABLE target_labs_24h AS
    WITH raw AS (
      SELECT
        l.hadm_id::BIGINT AS hadm_id,
        l.itemid::INTEGER AS itemid,
        l.charttime::TIMESTAMP AS charttime,
        TRY_CAST(l.valuenum AS DOUBLE) AS valuenum,
        l.valueuom AS valueuom,
        CASE
        {case_sql}
        END AS lab_name,
        c.admittime AS admittime
      FROM read_csv_auto('{str(labevents_path).replace('\\', '/')}', columns={{
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
      JOIN cohort c ON l.hadm_id = c.hadm_id
      WHERE l.itemid IN ({sql_list(itemids_all)})
    )
    SELECT hadm_id, itemid, lab_name, charttime, valuenum, valueuom
    FROM raw
    WHERE lab_name IS NOT NULL
      AND valuenum IS NOT NULL
      AND charttime >= admittime
      AND charttime <= admittime + INTERVAL 24 HOURS
      AND ({range_sql})
    """
    print("Creating target_labs_24h with DuckDB...")
    con.execute(query_events)

    events_long_out = OUT_DIR / "mimic_iv31_diabetes_labevents_target_24h_long.csv"
    labs_wide_out = OUT_DIR / "mimic_iv31_diabetes_labs_24h_wide.csv"
    enhanced_out = OUT_DIR / "mimic_iv31_diabetes_lab_enhanced_admissions.csv"

    con.execute(
        f"""
        COPY (
          SELECT hadm_id, itemid, lab_name, charttime, valuenum, valueuom
          FROM target_labs_24h
          ORDER BY hadm_id, lab_name, charttime
        ) TO '{str(events_long_out).replace('\\', '/')}' (HEADER, DELIMITER ',')
        """
    )

    # first() is deterministic after ordering via row_number.
    query_wide = """
    CREATE OR REPLACE TABLE labs_24h_wide AS
    WITH ranked AS (
      SELECT
        hadm_id,
        lab_name,
        valuenum,
        ROW_NUMBER() OVER (PARTITION BY hadm_id, lab_name ORDER BY charttime ASC) AS rn
      FROM target_labs_24h
    ),
    agg AS (
      SELECT
        hadm_id,
        lab_name,
        MAX(CASE WHEN rn = 1 THEN valuenum END) AS first_value,
        AVG(valuenum) AS mean_value,
        MIN(valuenum) AS min_value,
        MAX(valuenum) AS max_value,
        COUNT(*) AS count_value
      FROM ranked
      GROUP BY hadm_id, lab_name
    )
    SELECT
      hadm_id,
      MAX(CASE WHEN lab_name='hba1c' THEN first_value END) AS hba1c_first_24h,
      MAX(CASE WHEN lab_name='glucose' THEN first_value END) AS glucose_first_24h,
      MAX(CASE WHEN lab_name='creatinine' THEN first_value END) AS creatinine_first_24h,
      MAX(CASE WHEN lab_name='albumin' THEN first_value END) AS albumin_first_24h,
      MAX(CASE WHEN lab_name='hemoglobin' THEN first_value END) AS hemoglobin_first_24h,
      MAX(CASE WHEN lab_name='wbc' THEN first_value END) AS wbc_first_24h,
      MAX(CASE WHEN lab_name='total_cholesterol' THEN first_value END) AS total_cholesterol_first_24h,
      MAX(CASE WHEN lab_name='hdl_cholesterol' THEN first_value END) AS hdl_cholesterol_first_24h,
      MAX(CASE WHEN lab_name='ldl_cholesterol' THEN first_value END) AS ldl_cholesterol_first_24h,
      MAX(CASE WHEN lab_name='triglycerides' THEN first_value END) AS triglycerides_first_24h,
      MAX(CASE WHEN lab_name='uacr' THEN first_value END) AS uacr_first_24h,
      MAX(CASE WHEN lab_name='glucose' THEN mean_value END) AS glucose_mean_24h,
      MAX(CASE WHEN lab_name='glucose' THEN min_value END) AS glucose_min_24h,
      MAX(CASE WHEN lab_name='glucose' THEN max_value END) AS glucose_max_24h,
      MAX(CASE WHEN lab_name='glucose' THEN count_value END) AS glucose_count_24h,
      MAX(CASE WHEN lab_name='creatinine' THEN mean_value END) AS creatinine_mean_24h,
      MAX(CASE WHEN lab_name='creatinine' THEN count_value END) AS creatinine_count_24h,
      MAX(CASE WHEN lab_name='hba1c' THEN count_value END) AS hba1c_count_24h
    FROM agg
    GROUP BY hadm_id
    """
    con.execute(query_wide)
    con.execute(f"COPY labs_24h_wide TO '{str(labs_wide_out).replace('\\', '/')}' (HEADER, DELIMITER ',')")

    light = pd.read_csv(LIGHT_COHORT)
    labs = pd.read_csv(labs_wide_out)
    enhanced = light.merge(labs, on="hadm_id", how="left")
    enhanced.to_csv(enhanced_out, index=False)

    counts = con.execute(
        """
        SELECT lab_name, COUNT(*) AS row_count, COUNT(DISTINCT hadm_id) AS admissions
        FROM target_labs_24h
        GROUP BY lab_name
        ORDER BY admissions DESC
        """
    ).fetchdf()
    metadata = {
        "labevents_path": str(labevents_path),
        "light_cohort": str(LIGHT_COHORT),
        "events_long_output": str(events_long_out),
        "labs_wide_output": str(labs_wide_out),
        "enhanced_cohort_output": str(enhanced_out),
        "admissions_with_any_target_lab_24h": int(labs["hadm_id"].nunique()),
        "lab_counts": counts.to_dict(orient="records"),
        "lab_item_map": LAB_ITEM_MAP,
        "plausible_ranges": PLAUSIBLE_RANGES,
    }
    metadata_path = OUT_DIR / "mimic_iv31_diabetes_labs_24h_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    con.close()


if __name__ == "__main__":
    main()
