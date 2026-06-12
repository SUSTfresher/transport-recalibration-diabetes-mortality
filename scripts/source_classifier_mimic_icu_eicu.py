from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_PATH = PROJECT_ROOT / "data" / "mimic" / "processed" / "mimic_iv31_diabetes_icu_lab_vital_enhanced_cohort.csv"
EICU_PATH = PROJECT_ROOT / "data" / "eicu" / "processed" / "eicu_crd20_diabetes_lab_vital_enhanced_cohort.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "mimic_icu_eicu_source_classifier"

BASIC_NUMERIC = ["age", "bmi"]
BASIC_BINARY = ["female", "hypertension_history", "ckd_history", "cvd_history"]
LABS = [
    "glucose_first_24h",
    "creatinine_first_24h",
    "bun_first_24h",
    "wbc_first_24h",
    "hemoglobin_first_24h",
]
SENSITIVITY_LABS = ["albumin_first_24h"]
VITALS = [
    "heart_rate_first_24h",
    "systolic_bp_first_24h",
    "diastolic_bp_first_24h",
    "respiratory_rate_first_24h",
    "spo2_first_24h",
]


def load_dataset() -> pd.DataFrame:
    mimic = pd.read_csv(MIMIC_PATH)
    eicu = pd.read_csv(EICU_PATH)
    mimic = mimic.loc[mimic["hospital_mortality"].notna()].copy()
    eicu = eicu.loc[eicu["hospital_mortality"].notna()].copy()
    mimic["source"] = 0
    eicu["source"] = 1
    mimic["source_name"] = "MIMIC-IV ICU"
    eicu["source_name"] = "eICU"

    # Harmonize MIMIC OMR blood-pressure columns to the first-24h names only for the basic model.
    # ICU-vital models should wait for a repaired MIMIC chartevents file.
    for target, source in [
        ("systolic_bp_first_24h", "systolic_bp"),
        ("diastolic_bp_first_24h", "diastolic_bp"),
    ]:
        if target not in mimic.columns and source in mimic.columns:
            mimic[target] = mimic[source]

    all_cols = sorted(set(mimic.columns).union(eicu.columns).union({"source", "source_name"}))
    mimic = mimic.reindex(columns=all_cols)
    eicu = eicu.reindex(columns=all_cols)
    df = pd.concat([mimic, eicu], ignore_index=True)
    for col in BASIC_NUMERIC + BASIC_BINARY + LABS + SENSITIVITY_LABS + VITALS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_pipeline(numeric: list[str], binary: list[str], model_name: str) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "binary",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                binary,
            ),
        ]
    )
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=2000)
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=25,
            n_jobs=-1,
            random_state=20260611,
        )
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate_feature_set(df: pd.DataFrame, feature_set: str, numeric: list[str], binary: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    x = df[numeric + binary]
    y = df["source"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260611)
    for model_name in ["logistic_regression", "random_forest"]:
        model = make_pipeline(numeric, binary, model_name)
        pred = cross_val_predict(model, x, y, cv=cv, method="predict_proba", n_jobs=None)[:, 1]
        auc = roc_auc_score(y, pred)
        fpr, tpr, thresholds = roc_curve(y, pred)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
            OUT_DIR / f"roc_{feature_set}_{model_name}.csv", index=False
        )
        pd.DataFrame({"source": y, "prediction_eicu": pred}).to_csv(
            OUT_DIR / f"predictions_{feature_set}_{model_name}.csv", index=False
        )
        rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "roc_auc": float(auc),
                "n": int(len(y)),
                "mimic_n": int((df["source"] == 0).sum()),
                "eicu_n": int((df["source"] == 1).sum()),
                "features": numeric + binary,
            }
        )
    return rows


def summarize_features(df: pd.DataFrame, feature_sets: dict[str, tuple[list[str], list[str]]]) -> pd.DataFrame:
    rows = []
    for name, (numeric, binary) in feature_sets.items():
        for col in numeric + binary:
            for source_name, sub in df.groupby("source_name"):
                rows.append(
                    {
                        "feature_set": name,
                        "feature": col,
                        "source": source_name,
                        "nonmissing_rate": float(sub[col].notna().mean()),
                        "mean": float(sub[col].mean()) if pd.api.types.is_numeric_dtype(sub[col]) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    # Balance the source-classifier sample to avoid AUC being influenced by class prevalence.
    n_mimic = int((df["source"] == 0).sum())
    n_eicu = int((df["source"] == 1).sum())
    n = min(n_mimic, n_eicu)
    balanced = pd.concat(
        [
            df.loc[df["source"].eq(0)].sample(n=n, random_state=20260611),
            df.loc[df["source"].eq(1)].sample(n=n, random_state=20260611),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=20260611).reset_index(drop=True)

    feature_sets = {
        "basic": (BASIC_NUMERIC, BASIC_BINARY),
        "basic_labs": (BASIC_NUMERIC + LABS, BASIC_BINARY),
        "basic_labs_albumin_sensitivity": (BASIC_NUMERIC + LABS + SENSITIVITY_LABS, BASIC_BINARY),
        "basic_labs_eicu_vitals_diagnostic": (BASIC_NUMERIC + LABS + VITALS, BASIC_BINARY),
    }
    metrics: list[dict[str, object]] = []
    for feature_set, (numeric, binary) in feature_sets.items():
        metrics.extend(evaluate_feature_set(balanced, feature_set, numeric, binary))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT_DIR / "mimic_icu_eicu_source_classifier_summary.csv", index=False)
    feature_summary = summarize_features(balanced, feature_sets)
    feature_summary.to_csv(OUT_DIR / "mimic_icu_eicu_feature_summary.csv", index=False)
    metadata = {
        "mimic_path": str(MIMIC_PATH),
        "eicu_path": str(EICU_PATH),
        "balanced_n_per_source": n,
        "source_counts_full": {
            "mimic": n_mimic,
            "eicu": n_eicu,
        },
        "notes": [
            "The diagnostic vital feature set is not a primary result until MIMIC chartevents is repaired.",
            "Albumin is retained only as a sensitivity feature because coverage is limited in both databases.",
        ],
        "metrics": metrics,
    }
    (OUT_DIR / "mimic_icu_eicu_source_classifier_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
