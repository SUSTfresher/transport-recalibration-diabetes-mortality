from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "domain_shift"


COMMON_FEATURES = [
    "age",
    "female",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "hypertension_history",
    "ckd_history",
    "cvd_history",
]


OPTIONAL_LAB_FEATURES = [
    "hba1c",
    "glucose",
    "creatinine",
    "egfr",
    "uacr",
    "total_cholesterol",
    "hdl_cholesterol",
    "albumin",
    "hemoglobin",
    "wbc",
]

LAB_ENHANCED_FEATURES = [
    "hba1c",
    "glucose",
    "creatinine",
    "albumin",
    "hemoglobin",
    "wbc",
    "total_cholesterol",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
]


def load_nhanes() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "nhanes" / "processed" / "nhanes_2005_2018_diabetes_ckd_mortality_scan.csv"
    df = pd.read_csv(path)
    df = df[df["diabetes"].eq(1)].copy()
    out = pd.DataFrame(
        {
            "source": "NHANES",
            "source_id": df["SEQN"].astype(str),
            "age": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
            "female": pd.to_numeric(df["RIAGENDR"], errors="coerce").eq(2).astype(int),
            "bmi": pd.to_numeric(df["BMXBMI"], errors="coerce"),
            "systolic_bp": pd.to_numeric(df["systolic_bp"], errors="coerce"),
            "diastolic_bp": pd.to_numeric(df["diastolic_bp"], errors="coerce"),
            "hypertension_history": pd.to_numeric(df["hypertension_history"], errors="coerce"),
            "ckd_history": pd.to_numeric(df["ckd_egfr_or_uacr"], errors="coerce"),
            "cvd_history": pd.to_numeric(df["cvd_history"], errors="coerce"),
            "all_cause_death": pd.to_numeric(df["all_cause_death"], errors="coerce"),
            "death_within_5y": pd.to_numeric(df["death_within_5y"], errors="coerce"),
            "hba1c": pd.to_numeric(df["LBXGH"], errors="coerce"),
            "glucose": pd.to_numeric(df["LBXGLU"], errors="coerce"),
            "creatinine": pd.to_numeric(df["LBXSCR"], errors="coerce"),
            "egfr": pd.to_numeric(df["egfr"], errors="coerce"),
            "uacr": pd.to_numeric(df["uacr"], errors="coerce"),
            "total_cholesterol": pd.to_numeric(df["LBXTC"], errors="coerce"),
            "hdl_cholesterol": pd.to_numeric(df["LBDHDD"], errors="coerce"),
            "triglycerides": pd.to_numeric(df["LBXTR"], errors="coerce"),
        }
    )
    for feature in ["albumin", "hemoglobin", "wbc", "ldl_cholesterol"]:
        out[feature] = np.nan
    return out


def load_mimic() -> pd.DataFrame:
    enhanced = PROJECT_ROOT / "data" / "mimic" / "processed" / "mimic_iv31_diabetes_lab_enhanced_admissions.csv"
    light = PROJECT_ROOT / "data" / "mimic" / "processed" / "mimic_iv31_diabetes_light_admissions.csv"
    path = enhanced if enhanced.exists() else light
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "source": "MIMIC-IV",
            "source_id": df["hadm_id"].astype(str),
            "age": pd.to_numeric(df["age"], errors="coerce"),
            "female": pd.to_numeric(df["female"], errors="coerce"),
            "bmi": pd.to_numeric(df["bmi"], errors="coerce"),
            "systolic_bp": pd.to_numeric(df["systolic_bp"], errors="coerce"),
            "diastolic_bp": pd.to_numeric(df["diastolic_bp"], errors="coerce"),
            "hypertension_history": pd.to_numeric(df["hypertension_history"], errors="coerce"),
            "ckd_history": pd.to_numeric(df["ckd_history"], errors="coerce"),
            "cvd_history": pd.to_numeric(df["cvd_history"], errors="coerce"),
            "hospital_expire_flag": pd.to_numeric(df["hospital_expire_flag"], errors="coerce"),
            "death_within_30d_discharge": pd.to_numeric(df["death_within_30d_discharge"], errors="coerce"),
            "death_within_1y_discharge": pd.to_numeric(df["death_within_1y_discharge"], errors="coerce"),
            "has_icu_stay": pd.to_numeric(df["has_icu_stay"], errors="coerce"),
        }
    )
    lab_source = {
        "hba1c": "hba1c_first_24h",
        "glucose": "glucose_first_24h",
        "creatinine": "creatinine_first_24h",
        "albumin": "albumin_first_24h",
        "hemoglobin": "hemoglobin_first_24h",
        "wbc": "wbc_first_24h",
        "total_cholesterol": "total_cholesterol_first_24h",
        "hdl_cholesterol": "hdl_cholesterol_first_24h",
        "ldl_cholesterol": "ldl_cholesterol_first_24h",
        "triglycerides": "triglycerides_first_24h",
        "uacr": "uacr_first_24h",
    }
    for feature, source_col in lab_source.items():
        out[feature] = pd.to_numeric(df[source_col], errors="coerce") if source_col in df.columns else np.nan
    out.attrs["mimic_input_path"] = str(path)
    return out


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, group in df.groupby("source"):
        for col in COMMON_FEATURES + OPTIONAL_LAB_FEATURES:
            rows.append(
                {
                    "source": source,
                    "feature": col,
                    "n": int(len(group)),
                    "missing_n": int(group[col].isna().sum()) if col in group else int(len(group)),
                    "missing_pct": float(group[col].isna().mean()) if col in group else 1.0,
                    "nonmissing_n": int(group[col].notna().sum()) if col in group else 0,
                }
            )
    return pd.DataFrame(rows)


def smd_cont(x1: pd.Series, x2: pd.Series) -> float:
    x1 = pd.to_numeric(x1, errors="coerce").dropna()
    x2 = pd.to_numeric(x2, errors="coerce").dropna()
    if len(x1) < 2 or len(x2) < 2:
        return np.nan
    pooled = np.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2)
    if pooled == 0:
        return 0.0
    return float((x2.mean() - x1.mean()) / pooled)


def smd_binary(x1: pd.Series, x2: pd.Series) -> float:
    x1 = pd.to_numeric(x1, errors="coerce").dropna()
    x2 = pd.to_numeric(x2, errors="coerce").dropna()
    if len(x1) == 0 or len(x2) == 0:
        return np.nan
    p1 = x1.mean()
    p2 = x2.mean()
    pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    if pooled == 0:
        return 0.0
    return float((p2 - p1) / pooled)


def summarize_shift(nhanes: pd.DataFrame, mimic: pd.DataFrame) -> pd.DataFrame:
    binary = {"female", "hypertension_history", "ckd_history", "cvd_history"}
    rows = []
    for feature in COMMON_FEATURES:
        is_binary = feature in binary
        rows.append(
            {
                "feature": feature,
                "type": "binary" if is_binary else "continuous",
                "nhanes_nonmissing": int(nhanes[feature].notna().sum()),
                "mimic_nonmissing": int(mimic[feature].notna().sum()),
                "nhanes_mean_or_prop": float(pd.to_numeric(nhanes[feature], errors="coerce").mean()),
                "mimic_mean_or_prop": float(pd.to_numeric(mimic[feature], errors="coerce").mean()),
                "smd_mimic_minus_nhanes": smd_binary(nhanes[feature], mimic[feature]) if is_binary else smd_cont(nhanes[feature], mimic[feature]),
            }
        )
    return pd.DataFrame(rows).sort_values("smd_mimic_minus_nhanes", key=lambda s: s.abs(), ascending=False)


def source_classifier(df: pd.DataFrame, features: list[str]) -> dict:
    data = df[["source"] + features].copy()
    data = data.dropna(how="all", subset=features)
    # Downsample the very large MIMIC table to keep source classification interpretable and fast.
    source_counts = data["source"].value_counts()
    if len(source_counts) < 2 or source_counts.min() < 50:
        return {
            "skipped": True,
            "reason": "Fewer than two sources or fewer than 50 rows in the smallest source after dropping all-missing features.",
            "source_counts": source_counts.to_dict(),
        }
    min_n = source_counts.min()
    balanced = pd.concat(
        [
            group.sample(n=min_n, random_state=20260604)
            for _, group in data.groupby("source", sort=False)
        ],
        ignore_index=True,
    )
    y = balanced["source"].eq("MIMIC-IV").astype(int)
    x = balanced[features]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
    )
    models = {
        "logistic_regression": Pipeline(
            [
                ("preprocess", pre),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", pre),
                ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=20260604, n_jobs=-1)),
            ]
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260604)
    results = {}
    for name, model in models.items():
        pred = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
        results[name] = {
            "auc": float(roc_auc_score(y, pred)),
            "n_balanced": int(len(balanced)),
            "n_per_source": int(min_n),
        }
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nhanes = load_nhanes()
    mimic = load_mimic()
    combined = pd.concat([nhanes, mimic], ignore_index=True, sort=False)
    combined.to_csv(OUT_DIR / "nhanes_mimic_common_feature_table.csv", index=False)

    missing = summarize_missing(combined)
    missing.to_csv(OUT_DIR / "common_feature_missingness.csv", index=False)

    shift = summarize_shift(nhanes, mimic)
    shift.to_csv(OUT_DIR / "common_feature_smd.csv", index=False)

    common_available = [f for f in COMMON_FEATURES if combined[f].notna().any()]
    lab_enhanced_available = [
        f
        for f in COMMON_FEATURES + LAB_ENHANCED_FEATURES
        if combined.groupby("source")[f].apply(lambda s: s.notna().sum()).min() >= 50
    ]
    classifier_results = {
        "basic_features": source_classifier(combined, common_available),
        "lab_enhanced_features_min50_per_source": source_classifier(combined, lab_enhanced_available),
    }
    metadata = {
        "nhanes_rows_diabetes": int(len(nhanes)),
        "mimic_rows_diabetes_admissions": int(len(mimic)),
        "common_features_used": common_available,
        "lab_enhanced_features_used": lab_enhanced_available,
        "mimic_input_path": mimic.attrs.get("mimic_input_path"),
        "optional_lab_features_status": "Uses MIMIC 0-24h labevents if lab-enhanced cohort exists; otherwise labs are missing in MIMIC.",
        "source_classifier": classifier_results,
        "interpretation": "Higher source-classifier AUC indicates stronger covariate/domain shift between NHANES and MIMIC-IV.",
    }
    (OUT_DIR / "domain_shift_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("\nTop SMD features:")
    print(shift.to_string(index=False))


if __name__ == "__main__":
    main()
