from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EICU_ROOT = Path(r"D:\eicu-crd-2.0\physionet.org\files\eicu-crd\2.0")
OUT_DIR = PROJECT_ROOT / "outputs" / "eicu_feasibility_audit"

DIABETES_ICD_RE = re.compile(r"(^|[^0-9A-Za-z])(250(\.|\b)|E0[89](\.|\b)|E1[0-3](\.|\b))", re.I)
DIABETES_TEXT_RE = re.compile(
    r"\b(diabetes mellitus|diabetic ketoacidosis|insulin dependent diabetes|"
    r"non-insulin dependent diabetes|noninsulin dependent diabetes)\b",
    re.I,
)


def read_gzip_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


def count_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(root: Path) -> list[dict[str, str]]:
    sums_path = root / "SHA256SUMS.txt"
    results: list[dict[str, str]] = []
    if not sums_path.exists():
        return [{"file": "SHA256SUMS.txt", "status": "missing", "expected": "", "actual": ""}]

    for line in sums_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        expected, filename = re.split(r"\s+", line.strip(), maxsplit=1)
        path = root / filename
        if not path.exists():
            results.append({"file": filename, "status": "missing", "expected": expected, "actual": ""})
            continue
        actual = sha256_file(path)
        results.append(
            {
                "file": filename,
                "status": "ok" if actual.lower() == expected.lower() else "bad",
                "expected": expected.lower(),
                "actual": actual.lower(),
            }
        )
    return results


def table_dictionary(root: Path) -> list[dict[str, object]]:
    tables: list[dict[str, object]] = []
    for path in sorted(root.glob("*.csv.gz")):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader, [])
            sample = next(reader, [])
        tables.append(
            {
                "file": path.name,
                "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                "n_columns": len(columns),
                "columns": columns,
                "sample_first_fields": sample[: min(len(sample), 8)],
            }
        )
    return tables


def summarize_patient(root: Path) -> tuple[dict[str, object], set[str]]:
    patient_path = root / "patient.csv.gz"
    patientunitstayids: set[str] = set()
    uniquepids: set[str] = set()
    health_system_stays: set[str] = set()
    hospitals: set[str] = set()
    hospital_status = Counter()
    unit_status = Counter()
    gender = Counter()
    unit_type = Counter()
    adult_stays = 0
    age_ge_90 = 0
    age_missing_or_unparsed = 0

    for row in read_gzip_rows(patient_path):
        patientunitstayids.add(row["patientunitstayid"])
        if row.get("uniquepid"):
            uniquepids.add(row["uniquepid"])
        if row.get("patienthealthsystemstayid"):
            health_system_stays.add(row["patienthealthsystemstayid"])
        if row.get("hospitalid"):
            hospitals.add(row["hospitalid"])
        hospital_status[row.get("hospitaldischargestatus", "")] += 1
        unit_status[row.get("unitdischargestatus", "")] += 1
        gender[row.get("gender", "")] += 1
        unit_type[row.get("unittype", "")] += 1

        age = row.get("age", "")
        if age == "> 89":
            age_ge_90 += 1
            adult_stays += 1
        else:
            try:
                adult_stays += int(float(age) >= 18)
            except ValueError:
                age_missing_or_unparsed += 1

    return (
        {
            "patientunitstay_rows": len(patientunitstayids),
            "unique_patients_uniquepid": len(uniquepids),
            "health_system_stays": len(health_system_stays),
            "hospitals": len(hospitals),
            "adult_stays": adult_stays,
            "age_ge_90_coded_stays": age_ge_90,
            "age_missing_or_unparsed_stays": age_missing_or_unparsed,
            "hospitaldischargestatus": dict(hospital_status),
            "unitdischargestatus": dict(unit_status),
            "gender": dict(gender),
            "unittype": dict(unit_type),
        },
        patientunitstayids,
    )


def text_has_diabetes(text: str) -> bool:
    return bool(DIABETES_ICD_RE.search(text) or DIABETES_TEXT_RE.search(text))


def diabetes_candidates(root: Path) -> tuple[dict[str, object], set[str]]:
    apache_ids: set[str] = set()
    diagnosis_ids: set[str] = set()
    admission_dx_ids: set[str] = set()
    past_history_ids: set[str] = set()
    examples: dict[str, list[str]] = {
        "diagnosis": [],
        "admissionDx": [],
        "pastHistory": [],
    }
    row_counts: dict[str, int] = {}
    match_rows: dict[str, int] = {}
    apache_distribution = Counter()

    n = 0
    for row in read_gzip_rows(root / "apachePredVar.csv.gz"):
        n += 1
        value = (row.get("diabetes") or "").strip()
        apache_distribution[value] += 1
        if value == "1":
            apache_ids.add(row["patientunitstayid"])
    row_counts["apachePredVar"] = n
    match_rows["apachePredVar"] = len(apache_ids)

    n = 0
    matched = 0
    for row in read_gzip_rows(root / "diagnosis.csv.gz"):
        n += 1
        text = " ".join([row.get("icd9code", ""), row.get("diagnosisstring", "")])
        if text_has_diabetes(text):
            matched += 1
            diagnosis_ids.add(row["patientunitstayid"])
            if len(examples["diagnosis"]) < 5:
                examples["diagnosis"].append(text)
    row_counts["diagnosis"] = n
    match_rows["diagnosis"] = matched

    n = 0
    matched = 0
    for row in read_gzip_rows(root / "admissionDx.csv.gz"):
        n += 1
        text = " ".join([row.get("admitdxpath", ""), row.get("admitdxname", ""), row.get("admitdxtext", "")])
        if text_has_diabetes(text):
            matched += 1
            admission_dx_ids.add(row["patientunitstayid"])
            if len(examples["admissionDx"]) < 5:
                examples["admissionDx"].append(text)
    row_counts["admissionDx"] = n
    match_rows["admissionDx"] = matched

    n = 0
    matched = 0
    for row in read_gzip_rows(root / "pastHistory.csv.gz"):
        n += 1
        text = " ".join(
            [row.get("pasthistorypath", ""), row.get("pasthistoryvalue", ""), row.get("pasthistoryvaluetext", "")]
        )
        if text_has_diabetes(text):
            matched += 1
            past_history_ids.add(row["patientunitstayid"])
            if len(examples["pastHistory"]) < 5:
                examples["pastHistory"].append(text)
    row_counts["pastHistory"] = n
    match_rows["pastHistory"] = matched

    union_ids = apache_ids | diagnosis_ids | admission_dx_ids | past_history_ids
    summary = {
        "row_counts": row_counts,
        "matched_rows": match_rows,
        "apache_diabetes_distribution": dict(apache_distribution),
        "apache_diabetes_stays": len(apache_ids),
        "diagnosis_diabetes_stays": len(diagnosis_ids),
        "admissionDx_diabetes_stays": len(admission_dx_ids),
        "pastHistory_diabetes_stays": len(past_history_ids),
        "union_diabetes_candidate_stays": len(union_ids),
        "apache_only_stays": len(apache_ids - diagnosis_ids - admission_dx_ids - past_history_ids),
        "diagnosis_only_stays": len(diagnosis_ids - apache_ids - admission_dx_ids - past_history_ids),
        "admissionDx_only_stays": len(admission_dx_ids - apache_ids - diagnosis_ids - past_history_ids),
        "pastHistory_only_stays": len(past_history_ids - apache_ids - diagnosis_ids - admission_dx_ids),
        "examples": examples,
    }
    return summary, union_ids


def diabetes_endpoint_counts(root: Path, diabetes_ids: set[str]) -> dict[str, object]:
    hospital_status = Counter()
    unit_status = Counter()
    adult_stays = 0
    age_ge_90 = 0
    age_missing_or_unparsed = 0

    for row in read_gzip_rows(root / "patient.csv.gz"):
        if row["patientunitstayid"] not in diabetes_ids:
            continue
        hospital_status[row.get("hospitaldischargestatus", "")] += 1
        unit_status[row.get("unitdischargestatus", "")] += 1
        age = row.get("age", "")
        if age == "> 89":
            age_ge_90 += 1
            adult_stays += 1
        else:
            try:
                adult_stays += int(float(age) >= 18)
            except ValueError:
                age_missing_or_unparsed += 1

    return {
        "adult_diabetes_candidate_stays": adult_stays,
        "age_ge_90_coded_diabetes_stays": age_ge_90,
        "age_missing_or_unparsed_diabetes_stays": age_missing_or_unparsed,
        "hospitaldischargestatus": dict(hospital_status),
        "unitdischargestatus": dict(unit_status),
    }


def summarize_apache_results(root: Path) -> dict[str, object]:
    rows = 0
    versions = Counter()
    actual_icu = Counter()
    actual_hospital = Counter()
    stays = set()
    for row in read_gzip_rows(root / "apachePatientResult.csv.gz"):
        rows += 1
        stays.add(row["patientunitstayid"])
        versions[row.get("apacheversion", "")] += 1
        actual_icu[row.get("actualicumortality", "")] += 1
        actual_hospital[row.get("actualhospitalmortality", "")] += 1
    return {
        "rows": rows,
        "unique_patientunitstayids": len(stays),
        "apacheversion": dict(versions),
        "actualicumortality": dict(actual_icu),
        "actualhospitalmortality": dict(actual_hospital),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(report: dict[str, object], path: Path) -> None:
    integrity = report["sha256_integrity"]
    integrity_counts = Counter(item["status"] for item in integrity)
    patient = report["patient_summary"]
    diabetes = report["diabetes_candidate_summary"]
    diabetes_endpoints = report["diabetes_endpoint_summary"]
    file_inventory = report["file_inventory"]

    lines = [
        "# eICU Feasibility Audit",
        "",
        "Generated by `scripts/eicu_feasibility_audit.py`.",
        "",
        "## Download Integrity",
        "",
        f"- eICU root: `{report['eicu_root']}`",
        f"- SHA256 statuses: {dict(integrity_counts)}",
        f"- CSV/GZ tables: {len(file_inventory)}",
        f"- Total compressed CSV/GZ size: {sum(item['size_mb'] for item in file_inventory):.1f} MB",
        "",
        "## Core Endpoints",
        "",
        f"- ICU stays (`patientunitstayid`): {patient['patientunitstay_rows']:,}",
        f"- Unique patients (`uniquepid`): {patient['unique_patients_uniquepid']:,}",
        f"- Hospitals: {patient['hospitals']:,}",
        f"- Hospital discharge status: {patient['hospitaldischargestatus']}",
        f"- Unit discharge status: {patient['unitdischargestatus']}",
        "",
        "## Diabetes Candidate Cohort",
        "",
        f"- APACHE diabetes flag stays: {diabetes['apache_diabetes_stays']:,}",
        f"- Diagnosis table diabetes stays: {diabetes['diagnosis_diabetes_stays']:,}",
        f"- Admission diagnosis diabetes stays: {diabetes['admissionDx_diabetes_stays']:,}",
        f"- Past-history diabetes stays: {diabetes['pastHistory_diabetes_stays']:,}",
        f"- Union diabetes candidate stays: {diabetes['union_diabetes_candidate_stays']:,}",
        f"- Adult diabetes candidate stays: {diabetes_endpoints['adult_diabetes_candidate_stays']:,}",
        f"- Diabetes hospital discharge status: {diabetes_endpoints['hospitaldischargestatus']}",
        f"- Diabetes unit discharge status: {diabetes_endpoints['unitdischargestatus']}",
        "",
        "## Initial Interpretation",
        "",
        "- eICU has usable ICU and hospital mortality endpoints through `patient.csv.gz` and supporting APACHE result fields.",
        "- Diabetes can be defined from multiple sources; the first robust definition should probably combine APACHE diabetes, diagnosis strings/codes, and past-history diabetes.",
        "- The primary MIMIC-eICU endpoint should be ICU or hospital mortality, not one-year mortality.",
        "- The sample and event counts are large enough for transportability, subgroup, recalibration, and decision-curve analyses.",
        "",
        "## Output Files",
        "",
        "- `eicu_feasibility_summary.json`",
        "- `eicu_sha256_integrity.csv`",
        "- `eicu_table_dictionary.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit eICU-CRD download integrity and analysis feasibility.")
    parser.add_argument("--eicu-root", type=Path, default=DEFAULT_EICU_ROOT)
    args = parser.parse_args()

    root = args.eicu_root
    if not root.exists():
        raise FileNotFoundError(f"eICU root not found: {root}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    integrity = verify_sha256(root)
    tables = table_dictionary(root)
    patient_summary, patient_ids = summarize_patient(root)
    diabetes_summary, diabetes_ids = diabetes_candidates(root)
    diabetes_endpoint_summary = diabetes_endpoint_counts(root, diabetes_ids)
    apache_result_summary = summarize_apache_results(root)

    report = {
        "eicu_root": str(root),
        "sha256_integrity": integrity,
        "file_inventory": tables,
        "patient_summary": patient_summary,
        "diabetes_candidate_summary": diabetes_summary,
        "diabetes_endpoint_summary": diabetes_endpoint_summary,
        "apache_result_summary": apache_result_summary,
        "derived_notes": {
            "patient_stay_coverage_diabetes_union": round(len(diabetes_ids) / len(patient_ids), 6),
            "recommended_primary_eicu_endpoint": "hospitaldischargestatus or unitdischargestatus",
            "not_available_for_primary_endpoint": "one-year mortality is not present in core eICU-CRD tables",
        },
    }

    json_path = OUT_DIR / "eicu_feasibility_summary.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUT_DIR / "eicu_sha256_integrity.csv", integrity)
    write_csv(
        OUT_DIR / "eicu_table_dictionary.csv",
        [
            {
                "file": item["file"],
                "size_mb": item["size_mb"],
                "n_columns": item["n_columns"],
                "columns": "|".join(item["columns"]),
                "sample_first_fields": "|".join(item["sample_first_fields"]),
            }
            for item in tables
        ],
    )
    write_markdown(report, OUT_DIR / "README_eicu_feasibility_audit.md")

    print(f"Wrote {json_path}")
    print(f"Wrote {OUT_DIR / 'eicu_sha256_integrity.csv'}")
    print(f"Wrote {OUT_DIR / 'eicu_table_dictionary.csv'}")
    print(f"Wrote {OUT_DIR / 'README_eicu_feasibility_audit.md'}")


if __name__ == "__main__":
    main()
