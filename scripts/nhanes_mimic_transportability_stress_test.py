from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NHANES_PATH = PROJECT_ROOT / "data" / "nhanes" / "processed" / "nhanes_2005_2018_diabetes_ckd_mortality_scan.csv"
MIMIC_PATH = PROJECT_ROOT / "data" / "mimic" / "processed" / "mimic_iv31_diabetes_lab_enhanced_admissions.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "nhanes_mimic_transportability"


COMMON_BASE = [
    "age",
    "female",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "hypertension_history",
    "ckd_history",
    "cvd_history",
]
COMMON_LABS = [
    "hba1c",
    "glucose",
    "creatinine",
]
ALL_FEATURES = COMMON_BASE + COMMON_LABS


def load_nhanes() -> pd.DataFrame:
    df = pd.read_csv(NHANES_PATH)
    df = df[df["diabetes"].eq(1)].copy()
    out = pd.DataFrame(
        {
            "source": "NHANES",
            "id": df["SEQN"].astype(str),
            "cycle": df["cycle"],
            "age": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
            "female": pd.to_numeric(df["RIAGENDR"], errors="coerce").eq(2).astype(int),
            "bmi": pd.to_numeric(df["BMXBMI"], errors="coerce"),
            "systolic_bp": pd.to_numeric(df["systolic_bp"], errors="coerce"),
            "diastolic_bp": pd.to_numeric(df["diastolic_bp"], errors="coerce"),
            "hypertension_history": pd.to_numeric(df["hypertension_history"], errors="coerce"),
            "ckd_history": pd.to_numeric(df["ckd_egfr_or_uacr"], errors="coerce"),
            "cvd_history": pd.to_numeric(df["cvd_history"], errors="coerce"),
            "hba1c": pd.to_numeric(df["LBXGH"], errors="coerce"),
            "glucose": pd.to_numeric(df["LBXGLU"], errors="coerce"),
            "creatinine": pd.to_numeric(df["LBXSCR"], errors="coerce"),
            "outcome": pd.to_numeric(df["death_within_5y"], errors="coerce"),
            "outcome_name": "death_within_5y",
        }
    )
    out["known_outcome"] = out["outcome"].notna()
    for feature in COMMON_LABS:
        out[f"{feature}_measured"] = out[feature].notna().astype(int)
    return out


def load_mimic() -> pd.DataFrame:
    df = pd.read_csv(MIMIC_PATH)
    out = pd.DataFrame(
        {
            "source": "MIMIC-IV",
            "id": df["hadm_id"].astype(str),
            "cycle": "MIMIC-IV v3.1",
            "age": pd.to_numeric(df["age"], errors="coerce"),
            "female": pd.to_numeric(df["female"], errors="coerce"),
            "bmi": pd.to_numeric(df["bmi"], errors="coerce"),
            "systolic_bp": pd.to_numeric(df["systolic_bp"], errors="coerce"),
            "diastolic_bp": pd.to_numeric(df["diastolic_bp"], errors="coerce"),
            "hypertension_history": pd.to_numeric(df["hypertension_history"], errors="coerce"),
            "ckd_history": pd.to_numeric(df["ckd_history"], errors="coerce"),
            "cvd_history": pd.to_numeric(df["cvd_history"], errors="coerce"),
            "hba1c": pd.to_numeric(df["hba1c_first_24h"], errors="coerce"),
            "glucose": pd.to_numeric(df["glucose_first_24h"], errors="coerce"),
            "creatinine": pd.to_numeric(df["creatinine_first_24h"], errors="coerce"),
            "outcome": pd.to_numeric(df["hospital_expire_flag"], errors="coerce"),
            "outcome_name": "hospital_expire_flag",
        }
    )
    out["known_outcome"] = out["outcome"].notna()
    for feature in COMMON_LABS:
        out[f"{feature}_measured"] = out[feature].notna().astype(int)
    return out


def make_model(model_name: str, features: list[str]) -> Pipeline:
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
        ]
    )
    if model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_name == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=20260604,
        )
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocess", pre), ("model", clf)])


def temporal_nhanes_split(nhanes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = nhanes[nhanes["known_outcome"]].copy()
    train_cycles = {"2005-2006", "2007-2008", "2009-2010", "2011-2012"}
    test_cycles = {"2013-2014"}
    train = eligible[eligible["cycle"].isin(train_cycles)].copy()
    test = eligible[eligible["cycle"].isin(test_cycles)].copy()
    return train, test


def mimic_split(mimic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Deterministic admission-level split for stress test. MIMIC internal model already uses grouped split.
    rng = np.random.default_rng(20260604)
    mask = rng.random(len(mimic)) < 0.75
    return mimic[mask].copy(), mimic[~mask].copy()


def evaluate(y_true: pd.Series, pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, pred)) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, pred)) if len(np.unique(y_true)) == 2 else np.nan,
        "brier_score": float(brier_score_loss(y_true, pred)),
        "mean_prediction": float(np.mean(pred)),
        "prediction_p05": float(np.quantile(pred, 0.05)),
        "prediction_p50": float(np.quantile(pred, 0.50)),
        "prediction_p95": float(np.quantile(pred, 0.95)),
    }


def calibration_table(y_true: pd.Series, pred: np.ndarray, label: str) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(y_true, pred, n_bins=10, strategy="quantile")
    return pd.DataFrame(
        {
            "analysis": label,
            "mean_predicted_probability": prob_pred,
            "observed_probability": prob_true,
        }
    )


def fit_and_score(
    train: pd.DataFrame,
    internal_test: pd.DataFrame,
    transport_test: pd.DataFrame,
    source_name: str,
    transport_name: str,
    feature_set_name: str,
    features: list[str],
    model_name: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    model = make_model(model_name, features)
    train = train.dropna(subset=["outcome"]).copy()
    internal_test = internal_test.dropna(subset=["outcome"]).copy()
    transport_test = transport_test.dropna(subset=["outcome"]).copy()
    model.fit(train[features], train["outcome"].astype(int))
    rows = []
    cal_rows = []
    pred_rows = []
    for target_name, target in [("internal_test", internal_test), ("transport_test", transport_test)]:
        pred = model.predict_proba(target[features])[:, 1]
        metrics = evaluate(target["outcome"].astype(int), pred)
        metrics.update(
            {
                "train_source": source_name,
                "test_target": source_name if target_name == "internal_test" else transport_name,
                "test_type": target_name,
                "feature_set": feature_set_name,
                "model": model_name,
                "outcome_train": train["outcome_name"].iloc[0],
                "outcome_test": target["outcome_name"].iloc[0],
            }
        )
        rows.append(metrics)
        label = f"{source_name}_to_{metrics['test_target']}_{feature_set_name}_{model_name}"
        cal_rows.append(calibration_table(target["outcome"].astype(int), pred, label))
        pred_rows.append(
            pd.DataFrame(
                {
                    "id": target["id"].to_numpy(),
                    "train_source": source_name,
                    "test_target": metrics["test_target"],
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "outcome": target["outcome"].to_numpy(),
                    "prediction": pred,
                }
            )
        )
    models_dir = OUT_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / f"{source_name}_{feature_set_name}_{model_name}.joblib")
    return rows, pd.concat(cal_rows, ignore_index=True), pd.concat(pred_rows, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nhanes = load_nhanes()
    mimic = load_mimic()
    combined = pd.concat([nhanes, mimic], ignore_index=True, sort=False)
    combined.to_csv(OUT_DIR / "nhanes_mimic_common_transport_table.csv", index=False)

    missing = []
    for source, group in combined.groupby("source"):
        for feature in ALL_FEATURES:
            missing.append(
                {
                    "source": source,
                    "feature": feature,
                    "n": int(len(group)),
                    "missing_pct": float(group[feature].isna().mean()),
                    "nonmissing_n": int(group[feature].notna().sum()),
                }
            )
    pd.DataFrame(missing).to_csv(OUT_DIR / "common_feature_missingness.csv", index=False)

    nh_train, nh_test = temporal_nhanes_split(nhanes)
    mimic_train, mimic_test = mimic_split(mimic)
    feature_sets = {
        "base": COMMON_BASE,
        "base_labs": COMMON_BASE + COMMON_LABS + [f"{f}_measured" for f in COMMON_LABS],
    }
    all_metrics = []
    all_calibration = []
    all_predictions = []
    for feature_set_name, features in feature_sets.items():
        for model_name in ["logistic_regression", "hist_gradient_boosting"]:
            rows, cal, pred = fit_and_score(
                nh_train,
                nh_test,
                mimic_test,
                "NHANES",
                "MIMIC-IV",
                feature_set_name,
                features,
                model_name,
            )
            all_metrics.extend(rows)
            all_calibration.append(cal)
            all_predictions.append(pred)
            rows, cal, pred = fit_and_score(
                mimic_train,
                mimic_test,
                nh_test,
                "MIMIC-IV",
                "NHANES",
                feature_set_name,
                features,
                model_name,
            )
            all_metrics.extend(rows)
            all_calibration.append(cal)
            all_predictions.append(pred)

    pd.DataFrame(all_metrics).to_csv(OUT_DIR / "transportability_metrics.csv", index=False)
    pd.concat(all_calibration, ignore_index=True).to_csv(OUT_DIR / "transportability_calibration.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT_DIR / "transportability_predictions.csv", index=False)
    metadata = {
        "purpose": "Stress test of cross-database risk score transport between NHANES 5-year mortality and MIMIC-IV in-hospital mortality. Outcome mismatch means this is not formal external validation.",
        "nhanes_train_cycles": ["2005-2006", "2007-2008", "2009-2010", "2011-2012"],
        "nhanes_test_cycles": ["2013-2014"],
        "mimic_split": "Deterministic random 75/25 admission-level split for stress test.",
        "feature_sets": feature_sets,
        "nhanes_rows": int(len(nhanes)),
        "mimic_rows": int(len(mimic)),
    }
    (OUT_DIR / "transportability_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(all_metrics).to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
