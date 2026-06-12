from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "mimic" / "processed" / "mimic_iv31_diabetes_lab_enhanced_admissions.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "mimic_inhospital_mortality"


CORE_NUMERIC = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
]
CORE_BINARY = [
    "female",
    "hypertension_history",
    "ckd_history",
    "cvd_history",
]
CORE_CATEGORICAL = [
    "admission_type",
    "race_group",
]
LAB_FIRST = [
    "hba1c_first_24h",
    "glucose_first_24h",
    "creatinine_first_24h",
    "albumin_first_24h",
    "hemoglobin_first_24h",
    "wbc_first_24h",
    "total_cholesterol_first_24h",
    "hdl_cholesterol_first_24h",
    "ldl_cholesterol_first_24h",
    "triglycerides_first_24h",
]
LAB_COUNTS = [
    "glucose_count_24h",
    "creatinine_count_24h",
    "hba1c_count_24h",
]


def race_group(value: object) -> str:
    text = str(value).upper()
    if "WHITE" in text:
        return "White"
    if "BLACK" in text or "AFRICAN" in text:
        return "Black"
    if "ASIAN" in text:
        return "Asian"
    if "HISPANIC" in text or "LATINO" in text:
        return "Hispanic"
    if "UNKNOWN" in text or "UNABLE" in text or "DECLINED" in text or text in {"NAN", ""}:
        return "Unknown"
    return "Other"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["race_group"] = df["race"].map(race_group)
    for col in CORE_NUMERIC + CORE_BINARY + LAB_FIRST + LAB_COUNTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Missingness indicators capture measurement-process information but are not future outcomes.
    for col in LAB_FIRST:
        df[f"{col}_measured"] = df[col].notna().astype(int)
    for col in LAB_COUNTS:
        df[col] = df[col].fillna(0)
    return df


def make_preprocessor(numeric: list[str], binary: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "binary",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                binary,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def make_models(preprocessor: ColumnTransformer, pos_weight: float) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=20260604,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=20260604,
                    ),
                ),
            ]
        ),
    }


def threshold_metrics(y_true: pd.Series, pred: np.ndarray, threshold: float) -> dict:
    y_hat = pred >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    return {
        "threshold": threshold,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def decision_curve(y_true: pd.Series, pred: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    n = len(y_true)
    prevalence = float(np.mean(y_true))
    rows = []
    y = np.asarray(y_true)
    for threshold in thresholds:
        pred_pos = pred >= threshold
        tp = np.sum(pred_pos & (y == 1))
        fp = np.sum(pred_pos & (y == 0))
        net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        treat_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        rows.append(
            {
                "threshold": threshold,
                "net_benefit_model": net_benefit,
                "net_benefit_treat_all": treat_all,
                "net_benefit_treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names: list[str] = []
    for transformer_name, transformer, cols in preprocessor.transformers_:
        if transformer_name == "remainder":
            continue
        if transformer_name == "categorical":
            ohe = transformer.named_steps["onehot"]
            names.extend(ohe.get_feature_names_out(cols).tolist())
        else:
            names.extend(cols)
    return names


def model_importance(model: Pipeline, out_path: Path) -> None:
    names = feature_names(model.named_steps["preprocess"])
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = estimator.coef_.ravel()
        imp = pd.DataFrame({"feature": names, "coefficient": values, "abs_coefficient": np.abs(values)})
        imp = imp.sort_values("abs_coefficient", ascending=False)
    elif hasattr(estimator, "feature_importances_"):
        imp = pd.DataFrame({"feature": names, "importance": estimator.feature_importances_}).sort_values("importance", ascending=False)
    else:
        return
    imp.to_csv(out_path, index=False)


def evaluate_model(model_name: str, feature_set: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, out_dir: Path) -> dict:
    pred = model.predict_proba(x_test)[:, 1]
    fpr, tpr, roc_thresholds = roc_curve(y_test, pred)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, pred)
    prob_true, prob_pred = calibration_curve(y_test, pred, n_bins=10, strategy="quantile")
    thresholds = np.array([0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20])

    prefix = f"{feature_set}_{model_name}"
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(out_dir / f"roc_{prefix}.csv", index=False)
    pd.DataFrame({"precision": precision, "recall": recall}).to_csv(out_dir / f"precision_recall_{prefix}.csv", index=False)
    pd.DataFrame({"mean_predicted_probability": prob_pred, "observed_probability": prob_true}).to_csv(out_dir / f"calibration_{prefix}.csv", index=False)
    decision_curve(y_test, pred, np.linspace(0.005, 0.30, 60)).to_csv(out_dir / f"decision_curve_{prefix}.csv", index=False)
    pd.DataFrame({"y_true": y_test.to_numpy(), "prediction": pred}).to_csv(out_dir / f"predictions_{prefix}.csv", index=False)

    metrics = {
        "feature_set": feature_set,
        "model": model_name,
        "roc_auc": float(roc_auc_score(y_test, pred)),
        "pr_auc": float(average_precision_score(y_test, pred)),
        "brier_score": float(brier_score_loss(y_test, pred)),
        "test_n": int(len(y_test)),
        "test_events": int(y_test.sum()),
        "test_event_rate": float(y_test.mean()),
    }
    for threshold in thresholds:
        metrics.update({f"{k}_at_{threshold:.2f}": v for k, v in threshold_metrics(y_test, pred, float(threshold)).items() if k != "threshold"})
    return metrics


def run_feature_set(df: pd.DataFrame, feature_set: str, numeric: list[str], binary: list[str], categorical: list[str], out_dir: Path) -> list[dict]:
    all_features = numeric + binary + categorical
    modeling = df.dropna(subset=["hospital_expire_flag", "subject_id"]).copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=20260604)
    train_idx, test_idx = next(splitter.split(modeling, modeling["hospital_expire_flag"], groups=modeling["subject_id"]))
    train = modeling.iloc[train_idx].copy()
    test = modeling.iloc[test_idx].copy()
    x_train = train[all_features]
    y_train = train["hospital_expire_flag"].astype(int)
    x_test = test[all_features]
    y_test = test["hospital_expire_flag"].astype(int)
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    preprocessor = make_preprocessor(numeric, binary, categorical)
    models = make_models(preprocessor, pos_weight)
    metrics = []
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for model_name, model in models.items():
        print(f"Training {feature_set}/{model_name}...")
        model.fit(x_train, y_train)
        metrics.append(evaluate_model(model_name, feature_set, model, x_test, y_test, out_dir))
        model_importance(model, out_dir / f"feature_importance_{feature_set}_{model_name}.csv")
        joblib.dump(model, models_dir / f"{feature_set}_{model_name}.joblib")

    split_info = {
        "feature_set": feature_set,
        "features": all_features,
        "train_n": int(len(train)),
        "train_events": int(y_train.sum()),
        "test_n": int(len(test)),
        "test_events": int(y_test.sum()),
        "group_split": "GroupShuffleSplit by subject_id, test_size=0.25",
    }
    (out_dir / f"split_{feature_set}.json").write_text(json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    measured_indicators = [f"{col}_measured" for col in LAB_FIRST]
    lab_numeric = LAB_FIRST + LAB_COUNTS

    feature_sets = {
        "core": {
            "numeric": CORE_NUMERIC,
            "binary": CORE_BINARY,
            "categorical": CORE_CATEGORICAL,
        },
        "core_labs": {
            "numeric": CORE_NUMERIC + lab_numeric,
            "binary": CORE_BINARY + measured_indicators,
            "categorical": CORE_CATEGORICAL,
        },
    }
    all_metrics: list[dict] = []
    missing_rows = []
    for feature_set, spec in feature_sets.items():
        features = spec["numeric"] + spec["binary"] + spec["categorical"]
        missing = df[features].isna().mean().reset_index()
        missing.columns = ["feature", "missing_pct"]
        missing["feature_set"] = feature_set
        missing_rows.append(missing)
        all_metrics.extend(run_feature_set(df, feature_set, spec["numeric"], spec["binary"], spec["categorical"], OUT_DIR))

    pd.DataFrame(all_metrics).sort_values(["feature_set", "roc_auc"], ascending=[True, False]).to_csv(OUT_DIR / "model_performance.csv", index=False)
    pd.concat(missing_rows, ignore_index=True).to_csv(OUT_DIR / "feature_missingness.csv", index=False)
    metadata = {
        "data_path": str(DATA_PATH),
        "outcome": "hospital_expire_flag",
        "n": int(len(df)),
        "events": int(df["hospital_expire_flag"].sum()),
        "event_rate": float(df["hospital_expire_flag"].mean()),
        "leakage_excluded": [
            "length_of_stay_days",
            "dischtime",
            "death_within_30d_discharge",
            "death_within_1y_discharge",
            "has_icu_stay",
            "icu_stay_count",
        ],
        "feature_sets": feature_sets,
    }
    (OUT_DIR / "modeling_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(pd.DataFrame(all_metrics).sort_values(["feature_set", "roc_auc"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
