from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_ROOT = Path(r"D:\DATABASE\mimic-iv-3.1")
LIGHT_COHORT = PROJECT_ROOT / "data" / "mimic" / "processed" / "mimic_iv31_diabetes_light_admissions.csv"
OUT_DIR = PROJECT_ROOT / "data" / "mimic" / "processed"


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

ITEM_TO_LAB = {itemid: lab for lab, itemids in LAB_ITEM_MAP.items() for itemid in itemids}


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


def clean_lab_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lab_name"] = df["itemid"].map(ITEM_TO_LAB)
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    df = df.dropna(subset=["lab_name", "valuenum"])
    keep = np.ones(len(df), dtype=bool)
    for lab, (low, high) in PLAUSIBLE_RANGES.items():
        mask = df["lab_name"].eq(lab)
        keep[mask.to_numpy()] = df.loc[mask, "valuenum"].between(low, high).to_numpy()
    return df.loc[keep].copy()


def aggregate_labs(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["hadm_id"])
    events = events.sort_values(["hadm_id", "lab_name", "charttime"])
    grouped = events.groupby(["hadm_id", "lab_name"])["valuenum"]
    agg = grouped.agg(["first", "mean", "min", "max", "count"]).reset_index()
    wide_parts = []
    for stat in ["first", "mean", "min", "max", "count"]:
        part = agg.pivot(index="hadm_id", columns="lab_name", values=stat)
        part = part.rename(columns={col: f"{col}_{stat}_24h" for col in part.columns})
        wide_parts.append(part)
    wide = pd.concat(wide_parts, axis=1).reset_index()
    return wide


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(LIGHT_COHORT, usecols=["hadm_id", "admittime"])
    cohort["admittime"] = pd.to_datetime(cohort["admittime"], errors="coerce")
    cohort = cohort.dropna(subset=["hadm_id", "admittime"]).copy()
    cohort["hadm_id"] = cohort["hadm_id"].astype("int64")
    hadm_to_admit = cohort.set_index("hadm_id")["admittime"]
    target_hadm = set(hadm_to_admit.index)
    target_items = set(ITEM_TO_LAB)

    labevents_path = MIMIC_ROOT / "hosp" / "labevents.csv"
    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
    chunksize = 2_000_000
    kept_chunks: list[pd.DataFrame] = []
    scanned_rows = 0
    kept_target_rows = 0
    kept_window_rows = 0

    for i, chunk in enumerate(pd.read_csv(labevents_path, usecols=usecols, chunksize=chunksize), start=1):
        scanned_rows += len(chunk)
        chunk = chunk[chunk["hadm_id"].notna()]
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(target_hadm) & chunk["itemid"].isin(target_items)]
        kept_target_rows += len(chunk)
        if chunk.empty:
            if i % 10 == 0:
                print(f"chunk {i}: scanned={scanned_rows:,}, target_rows={kept_target_rows:,}, window_rows={kept_window_rows:,}")
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["admittime"] = chunk["hadm_id"].map(hadm_to_admit)
        delta_hours = (chunk["charttime"] - chunk["admittime"]).dt.total_seconds() / 3600
        chunk = chunk[delta_hours.between(0, 24, inclusive="both")]
        chunk = clean_lab_values(chunk)
        kept_window_rows += len(chunk)
        if not chunk.empty:
            kept_chunks.append(chunk[["hadm_id", "itemid", "lab_name", "charttime", "valuenum", "valueuom"]])
        if i % 10 == 0:
            print(f"chunk {i}: scanned={scanned_rows:,}, target_rows={kept_target_rows:,}, window_rows={kept_window_rows:,}")

    events = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame()
    events_out = OUT_DIR / "mimic_iv31_diabetes_labevents_target_24h_long.csv"
    labs_wide_out = OUT_DIR / "mimic_iv31_diabetes_labs_24h_wide.csv"
    enhanced_out = OUT_DIR / "mimic_iv31_diabetes_lab_enhanced_admissions.csv"
    events.to_csv(events_out, index=False)
    labs_wide = aggregate_labs(events)
    labs_wide.to_csv(labs_wide_out, index=False)

    full = pd.read_csv(LIGHT_COHORT)
    full = full.merge(labs_wide, on="hadm_id", how="left")
    full.to_csv(enhanced_out, index=False)

    metadata = {
        "labevents_path": str(labevents_path),
        "light_cohort": str(LIGHT_COHORT),
        "events_long_output": str(events_out),
        "labs_wide_output": str(labs_wide_out),
        "enhanced_cohort_output": str(enhanced_out),
        "scanned_rows": int(scanned_rows),
        "target_item_and_hadm_rows": int(kept_target_rows),
        "valid_target_rows_in_0_24h": int(kept_window_rows),
        "admissions_with_any_target_lab_24h": int(labs_wide["hadm_id"].nunique()) if not labs_wide.empty else 0,
        "lab_item_map": LAB_ITEM_MAP,
        "plausible_ranges": PLAUSIBLE_RANGES,
    }
    metadata_path = OUT_DIR / "mimic_iv31_diabetes_labs_24h_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
