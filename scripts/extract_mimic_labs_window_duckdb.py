from __future__ import annotations

import argparse
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


def window_label(start_hours: int, end_hours: int) -> str:
    def fmt(value: int) -> str:
        return f"m{abs(value)}" if value < 0 else f"p{value}"

    return f"{fmt(start_hours)}_{fmt(end_hours)}h"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MIMIC-IV diabetes admission labs within a configurable admission-time window.")
    parser.add_argument("--start-hours", type=int, default=-24, help="Window start relative to admittime, in hours.")
    parser.add_argument("--end-hours", type=int, default=24, help="Window end relative to admittime, in hours.")
    parser.add_argument("--memory-limit", default="8GB", help="DuckDB memory limit.")
    parser.add_argument("--threads", type=int, default=4, help="DuckDB worker threads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_hours > args.end_hours:
        raise ValueError("--start-hours must be <= --end-hours")
    label = window_label(args.start_hours, args.end_hours)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    labevents_path = MIMIC_ROOT / "hosp" / "labevents.csv"

    con = duckdb.connect(str(TMP_DIR / f"mimic_labs_{label}.duckdb"))
    con.execute(f"SET temp_directory='{str(TMP_DIR).replace('\\', '/')}'")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA threads={args.threads}")

    cohort = pd.read_csv(LIGHT_COHORT, usecols=["hadm_id", "admittime"])
    cohort["hadm_id"] = cohort["hadm_id"].astype("int64")
    cohort["admittime"] = pd.to_datetime(cohort["admittime"], errors="coerce")
    con.register("cohort_df", cohort)
    con.execute("CREATE OR REPLACE TEMP TABLE cohort AS SELECT hadm_id, admittime::TIMESTAMP AS admittime FROM cohort_df")

    case_sql = "\n        ".join(
        f"WHEN itemid IN ({sql_list(itemids)}) THEN '{lab}'"
        for lab, itemids in LAB_ITEM_MAP.items()
    )
    range_sql = " OR\n          ".join(
        f"(lab_name = '{lab}' AND valuenum BETWEEN {low} AND {high})"
        for lab, (low, high) in PLAUSIBLE_RANGES.items()
    )
    itemids_all = sorted({itemid for itemids in LAB_ITEM_MAP.values() for itemid in itemids})
    lower_expr = f"admittime {'+' if args.start_hours >= 0 else '-'} INTERVAL {abs(args.start_hours)} HOURS"
    upper_expr = f"admittime {'+' if args.end_hours >= 0 else '-'} INTERVAL {abs(args.end_hours)} HOURS"

    print(f"Creating target_labs_{label} with DuckDB...")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE target_labs_{label} AS
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
          AND charttime >= {lower_expr}
          AND charttime <= {upper_expr}
          AND ({range_sql})
        """
    )

    events_long_out = OUT_DIR / f"mimic_iv31_diabetes_labevents_target_{label}_long.csv"
    labs_wide_out = OUT_DIR / f"mimic_iv31_diabetes_labs_{label}_wide.csv"
    enhanced_out = OUT_DIR / f"mimic_iv31_diabetes_lab_enhanced_{label}_admissions.csv"
    table = f"target_labs_{label}"

    con.execute(
        f"""
        COPY (
          SELECT hadm_id, itemid, lab_name, charttime, valuenum, valueuom
          FROM {table}
          ORDER BY hadm_id, lab_name, charttime
        ) TO '{str(events_long_out).replace('\\', '/')}' (HEADER, DELIMITER ',')
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE labs_{label}_wide AS
        WITH ranked AS (
          SELECT
            hadm_id,
            lab_name,
            valuenum,
            ROW_NUMBER() OVER (PARTITION BY hadm_id, lab_name ORDER BY charttime ASC) AS rn
          FROM {table}
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
          MAX(CASE WHEN lab_name='hba1c' THEN first_value END) AS hba1c_first_{label},
          MAX(CASE WHEN lab_name='glucose' THEN first_value END) AS glucose_first_{label},
          MAX(CASE WHEN lab_name='creatinine' THEN first_value END) AS creatinine_first_{label},
          MAX(CASE WHEN lab_name='albumin' THEN first_value END) AS albumin_first_{label},
          MAX(CASE WHEN lab_name='hemoglobin' THEN first_value END) AS hemoglobin_first_{label},
          MAX(CASE WHEN lab_name='wbc' THEN first_value END) AS wbc_first_{label},
          MAX(CASE WHEN lab_name='total_cholesterol' THEN first_value END) AS total_cholesterol_first_{label},
          MAX(CASE WHEN lab_name='hdl_cholesterol' THEN first_value END) AS hdl_cholesterol_first_{label},
          MAX(CASE WHEN lab_name='ldl_cholesterol' THEN first_value END) AS ldl_cholesterol_first_{label},
          MAX(CASE WHEN lab_name='triglycerides' THEN first_value END) AS triglycerides_first_{label},
          MAX(CASE WHEN lab_name='uacr' THEN first_value END) AS uacr_first_{label},
          MAX(CASE WHEN lab_name='glucose' THEN mean_value END) AS glucose_mean_{label},
          MAX(CASE WHEN lab_name='glucose' THEN min_value END) AS glucose_min_{label},
          MAX(CASE WHEN lab_name='glucose' THEN max_value END) AS glucose_max_{label},
          MAX(CASE WHEN lab_name='glucose' THEN count_value END) AS glucose_count_{label},
          MAX(CASE WHEN lab_name='creatinine' THEN mean_value END) AS creatinine_mean_{label},
          MAX(CASE WHEN lab_name='creatinine' THEN count_value END) AS creatinine_count_{label},
          MAX(CASE WHEN lab_name='hba1c' THEN count_value END) AS hba1c_count_{label}
        FROM agg
        GROUP BY hadm_id
        """
    )
    con.execute(f"COPY labs_{label}_wide TO '{str(labs_wide_out).replace('\\', '/')}' (HEADER, DELIMITER ',')")

    light = pd.read_csv(LIGHT_COHORT)
    labs = pd.read_csv(labs_wide_out)
    enhanced = light.merge(labs, on="hadm_id", how="left")
    enhanced.to_csv(enhanced_out, index=False)

    counts = con.execute(
        f"""
        SELECT lab_name, COUNT(*) AS row_count, COUNT(DISTINCT hadm_id) AS admissions
        FROM {table}
        GROUP BY lab_name
        ORDER BY admissions DESC
        """
    ).fetchdf()
    metadata = {
        "window_label": label,
        "start_hours": args.start_hours,
        "end_hours": args.end_hours,
        "labevents_path": str(labevents_path),
        "light_cohort": str(LIGHT_COHORT),
        "events_long_output": str(events_long_out),
        "labs_wide_output": str(labs_wide_out),
        "enhanced_cohort_output": str(enhanced_out),
        "admissions_with_any_target_lab_window": int(labs["hadm_id"].nunique()),
        "lab_counts": counts.to_dict(orient="records"),
        "lab_item_map": LAB_ITEM_MAP,
        "plausible_ranges": PLAUSIBLE_RANGES,
    }
    metadata_path = OUT_DIR / f"mimic_iv31_diabetes_labs_{label}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    con.close()


if __name__ == "__main__":
    main()
