from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NHANES_PROJECT = Path(r"D:\糖尿病医工结合")
MIMIC_ROOT = Path(r"D:\DATABASE\mimic-iv-3.1")
OUT_DIR = PROJECT_ROOT / "outputs" / "inventory"


def file_info(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3) if path.exists() and path.is_file() else None,
    }


def read_mimic_version() -> str | None:
    changelog = MIMIC_ROOT / "CHANGELOG.txt"
    if not changelog.exists():
        return None
    text = changelog.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"MIMIC-IV v\d+\.\d+", text)
    return match.group(0) if match else None


def summarize_nhanes() -> dict:
    processed = PROJECT_ROOT / "data" / "nhanes" / "processed"
    files = {
        "nhanes_2005_2018": processed / "nhanes_2005_2018_diabetes_ckd_mortality_scan.csv",
        "nhanes3_external_validation": processed / "nhanes3_diabetes_ckd_mortality_external_validation.csv",
        "nhanes_2017_2018_demo": processed / "nhanes_2017_2018_diabetes_demo.csv",
    }
    summary: dict[str, object] = {
        "source_project": str(NHANES_PROJECT),
        "files": {name: file_info(path) for name, path in files.items()},
        "datasets": {},
    }
    for name, path in files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        dataset = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": list(df.columns),
        }
        for col in [
            "diabetes",
            "ckd_egfr_or_uacr",
            "all_cause_death",
            "death_within_5y",
            "death_5y",
            "known_5y_outcome",
        ]:
            if col in df.columns:
                dataset[f"{col}_sum"] = int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
                dataset[f"{col}_nonmissing"] = int(df[col].notna().sum())
        summary["datasets"][name] = dataset
    return summary


def summarize_mimic() -> dict:
    files = {
        "patients": MIMIC_ROOT / "hosp" / "patients.csv",
        "admissions": MIMIC_ROOT / "hosp" / "admissions.csv",
        "diagnoses_icd": MIMIC_ROOT / "hosp" / "diagnoses_icd.csv",
        "icustays": MIMIC_ROOT / "icu" / "icustays.csv",
        "d_labitems": MIMIC_ROOT / "hosp" / "d_labitems.csv",
        "labevents": MIMIC_ROOT / "hosp" / "labevents.csv",
        "omr": MIMIC_ROOT / "hosp" / "omr.csv",
        "prescriptions_gz": MIMIC_ROOT / "hosp" / "prescriptions.csv.gz",
        "chartevents_gz": MIMIC_ROOT / "icu" / "chartevents.csv.gz",
    }
    summary: dict[str, object] = {
        "root": str(MIMIC_ROOT),
        "version": read_mimic_version(),
        "files": {name: file_info(path) for name, path in files.items()},
    }
    required = ["patients", "admissions", "diagnoses_icd", "icustays"]
    if not all(files[name].exists() for name in required):
        summary["cohort_counts"] = "Required MIMIC core files not all present."
        return summary

    patients = pd.read_csv(files["patients"], usecols=["subject_id", "gender", "anchor_age", "dod"])
    admissions = pd.read_csv(
        files["admissions"],
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "race", "hospital_expire_flag"],
    )
    icustays = pd.read_csv(files["icustays"], usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    diagnoses = pd.read_csv(
        files["diagnoses_icd"],
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": "string"},
    )
    code = diagnoses["icd_code"].str.upper().str.replace(".", "", regex=False)
    diabetes = (
        (diagnoses["icd_version"].eq(9) & code.str.startswith("250"))
        | (diagnoses["icd_version"].eq(10) & code.str.match(r"^E0[8-9]|^E1[0-3]", na=False))
    )
    diabetes_admissions = diagnoses.loc[diabetes, ["subject_id", "hadm_id"]].drop_duplicates()
    diabetes_icu = icustays.merge(diabetes_admissions, on=["subject_id", "hadm_id"], how="inner")
    diabetes_hosp = admissions.merge(diabetes_admissions, on=["subject_id", "hadm_id"], how="inner")
    diabetes_icu_hadm = diabetes_icu[["subject_id", "hadm_id"]].drop_duplicates()
    diabetes_icu_hosp = diabetes_icu_hadm.merge(
        admissions[["subject_id", "hadm_id", "hospital_expire_flag"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    diabetes_subject_ids = set(diabetes_admissions["subject_id"].unique())
    diabetes_patients = patients[patients["subject_id"].isin(diabetes_subject_ids)]
    summary["cohort_counts"] = {
        "patients_n": int(len(patients)),
        "admissions_n": int(len(admissions)),
        "icu_stays_n": int(len(icustays)),
        "diagnoses_rows_n": int(len(diagnoses)),
        "diabetes_subjects_n": int(diabetes_admissions["subject_id"].nunique()),
        "diabetes_admissions_n": int(diabetes_admissions["hadm_id"].nunique()),
        "diabetes_icu_stays_n": int(diabetes_icu["stay_id"].nunique()),
        "diabetes_icu_subjects_n": int(diabetes_icu["subject_id"].nunique()),
        "diabetes_icu_admissions_n": int(diabetes_icu["hadm_id"].nunique()),
        "diabetes_hospital_deaths_n": int(diabetes_hosp["hospital_expire_flag"].sum()),
        "diabetes_admissions_with_deathtime_n": int(diabetes_hosp["deathtime"].notna().sum()),
        "diabetes_subjects_with_dod_n": int(diabetes_patients["dod"].notna().sum()),
        "diabetes_icu_hospital_deaths_n": int(diabetes_icu_hosp["hospital_expire_flag"].sum()),
    }
    return summary


def write_markdown(report: dict, path: Path) -> None:
    nh = report["nhanes"]
    mi = report["mimic"]
    lines = [
        "# Local Data Inventory",
        "",
        "This file is generated by `scripts/local_data_inventory.py`.",
        "",
        "## NHANES",
        "",
        f"- Source project: `{nh['source_project']}`",
    ]
    for name, data in nh["datasets"].items():
        lines.append(f"- `{name}`: {data['rows']} rows, {data['columns']} columns")
    lines.extend(
        [
            "",
            "## MIMIC-IV",
            "",
            f"- Root: `{mi['root']}`",
            f"- Version: `{mi['version']}`",
        ]
    )
    counts = mi.get("cohort_counts")
    if isinstance(counts, dict):
        for key, value in counts.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append(f"- Cohort counts: {counts}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "nhanes": summarize_nhanes(),
        "mimic": summarize_mimic(),
    }
    json_path = OUT_DIR / "local_data_inventory.json"
    md_path = OUT_DIR / "local_data_inventory.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
