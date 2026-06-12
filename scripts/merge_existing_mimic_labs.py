from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "mimic" / "processed"


def merge_labs(labs_file: str, output_file: str) -> dict:
    light_path = PROCESSED / "mimic_iv31_diabetes_light_admissions.csv"
    labs_path = PROCESSED / labs_file
    output_path = PROCESSED / output_file
    light = pd.read_csv(light_path)
    labs = pd.read_csv(labs_path)
    merged = light.merge(labs, on="hadm_id", how="left")
    merged.to_csv(output_path, index=False)
    return {
        "labs_file": str(labs_path),
        "output_file": str(output_path),
        "rows": int(len(merged)),
        "columns": int(len(merged.columns)),
        "hospital_deaths": int(merged["hospital_expire_flag"].sum()),
        "death_within_30d": int(merged["death_within_30d_discharge"].sum()),
        "death_within_1y": int(merged["death_within_1y_discharge"].sum()),
    }


def main() -> None:
    results = [
        merge_labs("mimic_iv31_diabetes_labs_24h_wide.csv", "mimic_iv31_diabetes_lab_enhanced_admissions.csv"),
        merge_labs("mimic_iv31_diabetes_labs_m24_p24h_wide.csv", "mimic_iv31_diabetes_lab_enhanced_m24_p24h_admissions.csv"),
    ]
    metadata_path = PROCESSED / "mimic_iv31_diabetes_lab_merge_metadata.json"
    metadata_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
