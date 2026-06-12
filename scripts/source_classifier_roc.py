from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from domain_shift_nhanes_mimic import COMMON_FEATURES, LAB_ENHANCED_FEATURES, load_mimic, load_nhanes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "domain_shift"


def available_features(df: pd.DataFrame, candidate_features: list[str], min_per_source: int = 50) -> list[str]:
    out = []
    for feature in candidate_features:
        if feature not in df.columns:
            continue
        per_source = df.groupby("source")[feature].apply(lambda s: int(s.notna().sum()))
        if per_source.min() >= min_per_source:
            out.append(feature)
    return out


def make_models(features: list[str]) -> dict[str, Pipeline]:
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
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", pre),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", pre),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=10,
                        random_state=20260608,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def run_feature_set(df: pd.DataFrame, feature_set: str, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    data = df[["source", "source_id"] + features].dropna(how="all", subset=features).copy()
    source_counts = data["source"].value_counts()
    if len(source_counts) != 2:
        raise ValueError(f"Expected two sources, got {source_counts.to_dict()}")
    min_n = int(source_counts.min())
    balanced = pd.concat(
        [group.sample(n=min_n, random_state=20260608) for _, group in data.groupby("source", sort=False)],
        ignore_index=True,
    )
    y = balanced["source"].eq("MIMIC-IV").astype(int)
    x = balanced[features]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260608)

    prediction_frames = []
    roc_frames = []
    summary_rows = []
    for model_name, model in make_models(features).items():
        pred = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
        fpr, tpr, thresholds = roc_curve(y, pred)
        auc = float(roc_auc_score(y, pred))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "source": balanced["source"].to_numpy(),
                    "source_id": balanced["source_id"].astype(str).to_numpy(),
                    "source_binary_mimic": y.to_numpy(),
                    "prediction_mimic_probability": pred,
                }
            )
        )
        roc_frames.append(
            pd.DataFrame(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "fpr": fpr,
                    "tpr": tpr,
                    "threshold": thresholds,
                }
            )
        )
        summary_rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "roc_auc": auc,
                "n_balanced": int(len(balanced)),
                "n_per_source": min_n,
                "features": ";".join(features),
            }
        )
    return pd.concat(prediction_frames, ignore_index=True), pd.concat(roc_frames, ignore_index=True), summary_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nhanes = load_nhanes()
    mimic = load_mimic()
    combined = pd.concat([nhanes, mimic], ignore_index=True, sort=False)

    feature_sets = {
        "basic_features": available_features(combined, COMMON_FEATURES, min_per_source=50),
        "lab_enhanced_features": available_features(combined, COMMON_FEATURES + LAB_ENHANCED_FEATURES, min_per_source=50),
    }

    all_predictions = []
    all_roc = []
    all_summary = []
    for feature_set, features in feature_sets.items():
        pred, roc, summary = run_feature_set(combined, feature_set, features)
        all_predictions.append(pred)
        all_roc.append(roc)
        all_summary.extend(summary)

    pd.concat(all_predictions, ignore_index=True).to_csv(OUT_DIR / "source_classifier_predictions.csv", index=False)
    pd.concat(all_roc, ignore_index=True).to_csv(OUT_DIR / "source_classifier_roc_curve.csv", index=False)
    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(OUT_DIR / "source_classifier_roc_summary.csv", index=False)
    metadata = {
        "purpose": "Cross-validated source classifier distinguishing NHANES from MIMIC-IV diabetes records.",
        "feature_sets": feature_sets,
        "cv": "5-fold stratified cross-validation on balanced NHANES/MIMIC sample.",
        "interpretation": "Higher AUC means the two data sources are more separable and transportability risk is higher.",
    }
    (OUT_DIR / "source_classifier_roc_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
