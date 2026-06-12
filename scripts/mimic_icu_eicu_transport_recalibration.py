from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_PATH = PROJECT_ROOT / "data" / "mimic" / "processed" / "mimic_iv31_diabetes_icu_lab_vital_enhanced_cohort.csv"
EICU_PATH = PROJECT_ROOT / "data" / "eicu" / "processed" / "eicu_crd20_diabetes_lab_vital_enhanced_cohort.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "mimic_icu_eicu_transport_recalibration"

RANDOM_SEED = 20260611
TRAIN_FRACTION = 0.75
N_BOOTSTRAP = 200
N_RECALIBRATION_REPEATS = 200
EVENT_TARGETS = [25, 50, 100, 200]
DCA_THRESHOLDS = [0.20, 0.25, 0.30]
EPS = 1e-6

SOURCE_CONFIG = {
    "MIMIC-IV ICU": {
        "path": MIMIC_PATH,
        "id_col": "hadm_id",
        "patient_col": "subject_id",
    },
    "eICU": {
        "path": EICU_PATH,
        "id_col": "patientunitstayid",
        "patient_col": "uniquepid",
    },
}

PRIMARY_NUMERIC = [
    "age",
    "glucose_first_24h",
    "creatinine_first_24h",
    "bun_first_24h",
    "wbc_first_24h",
    "hemoglobin_first_24h",
    "heart_rate_first_24h",
    "systolic_bp_first_24h",
    "diastolic_bp_first_24h",
    "spo2_first_24h",
]
PRIMARY_BINARY = [
    "female",
    "hypertension_history",
    "ckd_history",
    "cvd_history",
]

RESTRICTED_NUMERIC = [
    "age",
    "bmi",
    "systolic_bp_first_24h",
    "diastolic_bp_first_24h",
    "glucose_first_24h",
    "creatinine_first_24h",
]
RESTRICTED_BINARY = [
    "female",
    "hypertension_history",
    "cvd_history",
]

FEATURE_SETS = {
    "icu_native_primary": {
        "numeric": PRIMARY_NUMERIC,
        "binary": PRIMARY_BINARY,
        "description": "ICU-native primary set excluding BMI, albumin, temperature, and respiratory rate.",
    },
    "nhanes_compatible_restricted": {
        "numeric": RESTRICTED_NUMERIC,
        "binary": RESTRICTED_BINARY,
        "description": "Restricted bridge set compatible with the existing NHANES stress-test feature family.",
    },
}

METRIC_COLS = [
    "roc_auc",
    "pr_auc",
    "brier_score",
    "ece_10bin",
    "mean_prediction",
    "event_rate",
    "calibration_slope",
    "calibration_intercept",
]


def source_slug(source: str) -> str:
    return source.lower().replace("-iv", "").replace(" ", "_").replace("/", "_")


def direction_label(train_source: str, test_target: str) -> str:
    return f"{source_slug(train_source)}_to_{source_slug(test_target)}"


def normalize_id(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    return out.str.replace(r"\.0$", "", regex=True)


def inv_logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expected_calibration_error(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> float:
    if len(y) == 0:
        return np.nan
    order = np.argsort(pred)
    y_sorted = y[order]
    pred_sorted = pred[order]
    bins = np.array_split(np.arange(len(pred_sorted)), min(n_bins, len(pred_sorted)))
    ece = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        ece += (len(idx) / len(pred_sorted)) * abs(float(y_sorted[idx].mean()) - float(pred_sorted[idx].mean()))
    return float(ece)


def calibration_regression(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return {
            "calibration_slope": np.nan,
            "calibration_intercept": np.nan,
            "calibration_converged": False,
        }
    x = logit(pred)
    design = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0], dtype=float)
    converged = False
    for _ in range(50):
        eta = design @ beta
        prob = inv_logit(eta)
        weight = np.clip(prob * (1 - prob), 1e-9, None)
        grad = design.T @ (prob - y)
        hessian = design.T @ (design * weight[:, None])
        hessian += np.eye(2) * 1e-7
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            break
        beta -= step
        if not np.all(np.isfinite(beta)):
            break
        if float(np.max(np.abs(step))) < 1e-7:
            converged = True
            break
    return {
        "calibration_intercept": float(beta[0]) if np.all(np.isfinite(beta)) else np.nan,
        "calibration_slope": float(beta[1]) if np.all(np.isfinite(beta)) else np.nan,
        "calibration_converged": bool(converged),
    }


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float | int | bool]:
    y = np.asarray(y_true, dtype=int)
    pred = np.clip(np.asarray(pred, dtype=float), EPS, 1 - EPS)
    has_two_classes = len(np.unique(y)) == 2
    cal = calibration_regression(y, pred)
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()) if len(y) else np.nan,
        "roc_auc": float(roc_auc_score(y, pred)) if has_two_classes else np.nan,
        "pr_auc": float(average_precision_score(y, pred)) if has_two_classes else np.nan,
        "brier_score": float(brier_score_loss(y, pred)) if len(y) else np.nan,
        "ece_10bin": expected_calibration_error(y, pred),
        "mean_prediction": float(pred.mean()) if len(pred) else np.nan,
        "prediction_p05": float(np.quantile(pred, 0.05)) if len(pred) else np.nan,
        "prediction_p50": float(np.quantile(pred, 0.50)) if len(pred) else np.nan,
        "prediction_p95": float(np.quantile(pred, 0.95)) if len(pred) else np.nan,
        "calibration_slope": cal["calibration_slope"],
        "calibration_intercept": cal["calibration_intercept"],
        "calibration_converged": cal["calibration_converged"],
    }


def load_source(source: str) -> pd.DataFrame:
    cfg = SOURCE_CONFIG[source]
    raw = pd.read_csv(cfg["path"])
    raw = raw.loc[raw["hospital_mortality"].notna()].copy()
    all_features = sorted({col for spec in FEATURE_SETS.values() for col in spec["numeric"] + spec["binary"]})
    cols = [cfg["id_col"], cfg["patient_col"], "hospital_mortality"] + all_features
    missing = [col for col in cols if col not in raw.columns]
    if missing:
        raise KeyError(f"{source} is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "source": source,
            "id": raw[cfg["id_col"]].astype(str),
            "patient_id": raw[cfg["patient_col"]].astype(str),
            "outcome": pd.to_numeric(raw["hospital_mortality"], errors="coerce").astype(int),
        }
    )
    for col in all_features:
        out[col] = pd.to_numeric(raw[col], errors="coerce")
    return out


def split_by_patient(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRACTION, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, y=df["outcome"], groups=df["patient_id"]))
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    train["split"] = "development"
    test["split"] = "holdout"
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_model(model_name: str, numeric: list[str], binary: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
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
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                binary,
            ),
        ]
    )
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def prediction_frame(
    df: pd.DataFrame,
    pred: np.ndarray,
    train_source: str,
    test_target: str,
    test_type: str,
    feature_set: str,
    model_name: str,
) -> pd.DataFrame:
    pred = np.clip(np.asarray(pred, dtype=float), EPS, 1 - EPS)
    return pd.DataFrame(
        {
            "id": df["id"].to_numpy(),
            "patient_id": df["patient_id"].to_numpy(),
            "train_source": train_source,
            "test_target": test_target,
            "direction": direction_label(train_source, test_target),
            "test_type": test_type,
            "feature_set": feature_set,
            "model": model_name,
            "outcome": df["outcome"].astype(int).to_numpy(),
            "prediction": pred,
            "age": df["age"].to_numpy(),
            "female": df["female"].to_numpy(),
            "hypertension_history": df["hypertension_history"].to_numpy(),
            "ckd_history": df["ckd_history"].to_numpy(),
            "cvd_history": df["cvd_history"].to_numpy(),
        }
    )


def fit_direction_models(
    splits: dict[str, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models_dir = OUT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    prediction_parts = []
    calibration_pool_parts = []
    sources = list(splits.keys())

    for train_source in sources:
        target_source = [s for s in sources if s != train_source][0]
        source_train = splits[train_source]["development"]
        source_test = splits[train_source]["holdout"]
        target_train = splits[target_source]["development"]
        target_test = splits[target_source]["holdout"]

        for feature_set, spec in FEATURE_SETS.items():
            numeric = spec["numeric"]
            binary = spec["binary"]
            features = numeric + binary
            for model_name in ["logistic_regression", "hist_gradient_boosting"]:
                model = make_model(model_name, numeric, binary)
                model.fit(source_train[features], source_train["outcome"].astype(int))
                model_path = models_dir / f"{source_slug(train_source)}_{feature_set}_{model_name}.joblib"
                joblib.dump(model, model_path)

                internal_pred = model.predict_proba(source_test[features])[:, 1]
                transport_pred = model.predict_proba(target_test[features])[:, 1]
                calibration_pool_pred = model.predict_proba(target_train[features])[:, 1]

                prediction_parts.append(
                    prediction_frame(
                        source_test,
                        internal_pred,
                        train_source,
                        train_source,
                        "internal_test",
                        feature_set,
                        model_name,
                    )
                )
                prediction_parts.append(
                    prediction_frame(
                        target_test,
                        transport_pred,
                        train_source,
                        target_source,
                        "transport_test",
                        feature_set,
                        model_name,
                    )
                )
                calibration_pool_parts.append(
                    prediction_frame(
                        target_train,
                        calibration_pool_pred,
                        train_source,
                        target_source,
                        "target_development_calibration_pool",
                        feature_set,
                        model_name,
                    )
                )
                print(f"Fit {train_source} -> {target_source}: {feature_set}, {model_name}")

    return pd.concat(prediction_parts, ignore_index=True), pd.concat(calibration_pool_parts, ignore_index=True)


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["train_source", "test_target", "direction", "test_type", "feature_set", "model"]
    rows = []
    for key, group in predictions.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, key))
        row.update(metrics(group["outcome"].astype(int).to_numpy(), group["prediction"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_curve_rows(predictions: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    group_cols = ["train_source", "test_target", "direction", "test_type", "feature_set", "model"]
    rows = []
    for key, group in predictions.groupby(group_cols, sort=False):
        y = group["outcome"].astype(int).to_numpy()
        pred = group["prediction"].to_numpy()
        order = np.argsort(pred)
        y_sorted = y[order]
        pred_sorted = pred[order]
        bins = np.array_split(np.arange(len(group)), min(n_bins, len(group)))
        for i, idx in enumerate(bins, start=1):
            if len(idx) == 0:
                continue
            row = dict(zip(group_cols, key))
            row.update(
                {
                    "bin": i,
                    "bin_n": int(len(idx)),
                    "mean_predicted_probability": float(pred_sorted[idx].mean()),
                    "observed_probability": float(y_sorted[idx].mean()),
                    "events": int(y_sorted[idx].sum()),
                    "prediction_min": float(pred_sorted[idx].min()),
                    "prediction_max": float(pred_sorted[idx].max()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    group_cols = ["train_source", "test_target", "direction", "test_type", "feature_set", "model"]
    long_rows = []
    summary_rows = []
    for key, group in predictions.groupby(group_cols, sort=False):
        y = group["outcome"].astype(int).to_numpy()
        pred = group["prediction"].to_numpy()
        n = len(group)
        id_cols = dict(zip(group_cols, key))
        id_cols.update({"n": int(n), "events": int(y.sum())})
        point = metrics(y, pred)
        group_long = []
        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            m = metrics(y[idx], pred[idx])
            m.update(id_cols)
            m["bootstrap"] = b
            group_long.append(m)
        long_rows.extend(group_long)
        boot = pd.DataFrame(group_long)
        for metric in METRIC_COLS:
            vals = boot[metric].dropna().to_numpy()
            summary_rows.append(
                {
                    **id_cols,
                    "metric": metric,
                    "point": point[metric],
                    "ci_lower": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                    "ci_upper": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                    "bootstrap_n_valid": int(len(vals)),
                }
            )
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def intercept_only_offset(y_cal: np.ndarray, pred_cal: np.ndarray) -> float:
    target = float(np.clip(np.mean(y_cal), EPS, 1 - EPS))
    base = logit(pred_cal)
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = (lo + hi) / 2
        current = float(np.mean(inv_logit(base + mid)))
        if current < target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def fit_platt(y_cal: np.ndarray, pred_cal: np.ndarray) -> tuple[float, float]:
    fit = calibration_regression(y_cal, pred_cal)
    return float(fit["calibration_intercept"]), float(fit["calibration_slope"])


def apply_recalibration(
    method: str,
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_eval: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    if method == "raw":
        return np.clip(pred_eval, EPS, 1 - EPS), {}
    if method == "intercept_only":
        offset = intercept_only_offset(y_cal, pred_cal)
        pred = inv_logit(logit(pred_eval) + offset)
        return np.clip(pred, EPS, 1 - EPS), {"recalibration_intercept_offset": offset}
    if method == "platt":
        intercept, slope = fit_platt(y_cal, pred_cal)
        pred = inv_logit(intercept + slope * logit(pred_eval))
        return np.clip(pred, EPS, 1 - EPS), {
            "recalibration_intercept_parameter": intercept,
            "recalibration_slope_parameter": slope,
        }
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        model.fit(pred_cal, y_cal)
        pred = model.predict(pred_eval)
        return np.clip(pred, EPS, 1 - EPS), {}
    raise ValueError(method)


def sample_by_event_count(y: np.ndarray, event_target: int, rng: np.random.Generator) -> np.ndarray:
    event_idx = np.flatnonzero(y == 1)
    nonevent_idx = np.flatnonzero(y == 0)
    if event_target >= len(event_idx):
        raise ValueError(f"event_target={event_target} exceeds available events={len(event_idx)}")
    event_rate = len(event_idx) / len(y)
    total_n = int(round(event_target / event_rate))
    nonevent_target = max(1, total_n - event_target)
    if nonevent_target >= len(nonevent_idx):
        nonevent_target = len(nonevent_idx)
    chosen_events = rng.choice(event_idx, size=event_target, replace=False)
    chosen_nonevents = rng.choice(nonevent_idx, size=nonevent_target, replace=False)
    return np.concatenate([chosen_events, chosen_nonevents])


def run_event_count_recalibration(
    transport_predictions: pd.DataFrame,
    calibration_pool_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    group_cols = ["train_source", "test_target", "direction", "feature_set", "model"]
    rows = []
    for key, pool_group in calibration_pool_predictions.groupby(group_cols, sort=False):
        selector = np.ones(len(transport_predictions), dtype=bool)
        for col, value in zip(group_cols, key):
            selector &= transport_predictions[col].eq(value).to_numpy()
        selector &= transport_predictions["test_type"].eq("transport_test").to_numpy()
        test_group = transport_predictions.loc[selector].copy()
        if test_group.empty:
            continue

        y_pool = pool_group["outcome"].astype(int).to_numpy()
        pred_pool = pool_group["prediction"].to_numpy()
        y_test = test_group["outcome"].astype(int).to_numpy()
        pred_test = test_group["prediction"].to_numpy()

        base = dict(zip(group_cols, key))
        raw_metrics = metrics(y_test, pred_test)
        raw_metrics.update(
            {
                **base,
                "method": "raw",
                "event_target": 0,
                "repeat": 0,
                "calibration_n": 0,
                "calibration_events": 0,
                "calibration_event_rate": np.nan,
                "evaluation_n": int(len(y_test)),
                "evaluation_events": int(y_test.sum()),
            }
        )
        rows.append(raw_metrics)

        for event_target in EVENT_TARGETS:
            if event_target >= int(y_pool.sum()):
                continue
            for repeat in range(N_RECALIBRATION_REPEATS):
                cal_idx = sample_by_event_count(y_pool, event_target, rng)
                y_cal = y_pool[cal_idx]
                pred_cal = pred_pool[cal_idx]
                for method in ["intercept_only", "platt", "isotonic"]:
                    pred_eval, params = apply_recalibration(method, y_cal, pred_cal, pred_test)
                    m = metrics(y_test, pred_eval)
                    m.update(
                        {
                            **base,
                            "method": method,
                            "event_target": event_target,
                            "repeat": repeat,
                            "calibration_n": int(len(cal_idx)),
                            "calibration_events": int(y_cal.sum()),
                            "calibration_event_rate": float(y_cal.mean()),
                            "evaluation_n": int(len(y_test)),
                            "evaluation_events": int(y_test.sum()),
                        }
                    )
                    m.update(params)
                    rows.append(m)
    long = pd.DataFrame(rows)
    summary = summarize_recalibration(long)
    return long, summary


def summarize_recalibration(long: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["train_source", "test_target", "direction", "feature_set", "model", "method", "event_target"]
    metric_cols = METRIC_COLS + ["calibration_n", "calibration_events", "calibration_event_rate", "evaluation_n", "evaluation_events"]
    rows = []
    for key, group in long.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, key))
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors="coerce").dropna().to_numpy()
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            row[f"{col}_ci_lower"] = float(np.quantile(vals, 0.025)) if len(vals) else np.nan
            row[f"{col}_ci_upper"] = float(np.quantile(vals, 0.975)) if len(vals) else np.nan
        row["n_repeats"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def net_benefit(y: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    treated = pred >= threshold
    n = len(y)
    tp = int(((y == 1) & treated).sum())
    fp = int(((y == 0) & treated).sum())
    return float(tp / n - fp / n * (threshold / (1 - threshold)))


def build_dca(
    predictions: pd.DataFrame,
    calibration_pool_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows = []
    eval_parts = []
    for train_source, target_source in [("MIMIC-IV ICU", "eICU"), ("eICU", "MIMIC-IV ICU")]:
        direction = direction_label(train_source, target_source)
        raw = predictions[
            predictions["train_source"].eq(train_source)
            & predictions["test_target"].eq(target_source)
            & predictions["test_type"].eq("transport_test")
            & predictions["feature_set"].eq("icu_native_primary")
            & predictions["model"].eq("logistic_regression")
        ].copy()
        cal_pool = calibration_pool_predictions[
            calibration_pool_predictions["train_source"].eq(train_source)
            & calibration_pool_predictions["test_target"].eq(target_source)
            & calibration_pool_predictions["feature_set"].eq("icu_native_primary")
            & calibration_pool_predictions["model"].eq("logistic_regression")
        ].copy()
        internal_logit = predictions[
            predictions["train_source"].eq(target_source)
            & predictions["test_target"].eq(target_source)
            & predictions["test_type"].eq("internal_test")
            & predictions["feature_set"].eq("icu_native_primary")
            & predictions["model"].eq("logistic_regression")
        ][["id", "prediction"]].rename(columns={"prediction": "internal_logistic"})
        internal_hgb = predictions[
            predictions["train_source"].eq(target_source)
            & predictions["test_target"].eq(target_source)
            & predictions["test_type"].eq("internal_test")
            & predictions["feature_set"].eq("icu_native_primary")
            & predictions["model"].eq("hist_gradient_boosting")
        ][["id", "prediction"]].rename(columns={"prediction": "internal_hgb"})

        raw["id_norm"] = normalize_id(raw["id"])
        internal_logit["id_norm"] = normalize_id(internal_logit["id"])
        internal_hgb["id_norm"] = normalize_id(internal_hgb["id"])
        eval_df = raw[["id_norm", "id", "outcome", "prediction"]].rename(columns={"prediction": "raw_transport"})
        eval_df = eval_df.merge(internal_logit[["id_norm", "internal_logistic"]], on="id_norm", how="inner")
        eval_df = eval_df.merge(internal_hgb[["id_norm", "internal_hgb"]], on="id_norm", how="inner")

        y_pool = cal_pool["outcome"].astype(int).to_numpy()
        pred_pool = cal_pool["prediction"].to_numpy()
        y_eval = eval_df["outcome"].astype(int).to_numpy()
        pred_eval_raw = eval_df["raw_transport"].to_numpy()
        rng = np.random.default_rng(RANDOM_SEED + len(curve_rows) + 100)
        cal_idx = sample_by_event_count(y_pool, 100, rng)
        platt_pred, _ = apply_recalibration("platt", y_pool[cal_idx], pred_pool[cal_idx], pred_eval_raw)
        eval_df["platt_100_events"] = platt_pred
        eval_df["direction"] = direction
        eval_df["train_source"] = train_source
        eval_df["test_target"] = target_source
        eval_df["calibration_n"] = int(len(cal_idx))
        eval_df["calibration_events"] = int(y_pool[cal_idx].sum())
        eval_parts.append(eval_df)

        event_rate = float(y_eval.mean())
        model_predictions = {
            "Raw transport logistic": pred_eval_raw,
            "Platt 100 events": platt_pred,
            "Internal logistic benchmark": eval_df["internal_logistic"].to_numpy(),
            "Internal HGB benchmark": eval_df["internal_hgb"].to_numpy(),
        }
        for threshold in DCA_THRESHOLDS:
            curve_rows.append(
                {
                    "direction": direction,
                    "train_source": train_source,
                    "test_target": target_source,
                    "strategy": "Treat none",
                    "threshold": threshold,
                    "net_benefit": 0.0,
                    "calibration_n": 0,
                    "calibration_events": 0,
                }
            )
            curve_rows.append(
                {
                    "direction": direction,
                    "train_source": train_source,
                    "test_target": target_source,
                    "strategy": "Treat all",
                    "threshold": threshold,
                    "net_benefit": event_rate - (1 - event_rate) * threshold / (1 - threshold),
                    "calibration_n": 0,
                    "calibration_events": 0,
                }
            )
            for strategy, pred in model_predictions.items():
                curve_rows.append(
                    {
                        "direction": direction,
                        "train_source": train_source,
                        "test_target": target_source,
                        "strategy": strategy,
                        "threshold": threshold,
                        "net_benefit": net_benefit(y_eval, pred, threshold),
                        "calibration_n": int(len(cal_idx)) if strategy == "Platt 100 events" else 0,
                        "calibration_events": int(y_pool[cal_idx].sum()) if strategy == "Platt 100 events" else 0,
                    }
                )
    return pd.DataFrame(curve_rows), pd.concat(eval_parts, ignore_index=True)


def subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["train_source", "test_target", "direction", "test_type", "feature_set", "model"]
    subgroup_defs = [
        ("Overall", "Overall", np.ones(len(predictions), dtype=bool)),
        ("Age", "Age <65", predictions["age"].lt(65).fillna(False).to_numpy()),
        ("Age", "Age >=65", predictions["age"].ge(65).fillna(False).to_numpy()),
        ("CKD history", "No CKD", predictions["ckd_history"].eq(0).fillna(False).to_numpy()),
        ("CKD history", "CKD", predictions["ckd_history"].eq(1).fillna(False).to_numpy()),
        ("CVD history", "No CVD", predictions["cvd_history"].eq(0).fillna(False).to_numpy()),
        ("CVD history", "CVD", predictions["cvd_history"].eq(1).fillna(False).to_numpy()),
    ]
    rows = []
    for key, group in predictions.groupby(group_cols, sort=False):
        base = dict(zip(group_cols, key))
        group_index = group.index.to_numpy()
        for subgroup_type, subgroup, mask_all in subgroup_defs:
            sub = group.loc[np.intersect1d(group_index, np.flatnonzero(mask_all), assume_unique=False)]
            if len(sub) < 100:
                continue
            m = metrics(sub["outcome"].astype(int).to_numpy(), sub["prediction"].to_numpy())
            m.update(base)
            m.update({"subgroup_type": subgroup_type, "subgroup": subgroup})
            rows.append(m)
    return pd.DataFrame(rows)


def common_feature_missingness(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_set, spec in FEATURE_SETS.items():
        for feature in spec["numeric"] + spec["binary"]:
            for source, group in combined.groupby("source", sort=False):
                rows.append(
                    {
                        "feature_set": feature_set,
                        "feature": feature,
                        "source": source,
                        "n": int(len(group)),
                        "nonmissing_n": int(group[feature].notna().sum()),
                        "nonmissing_rate": float(group[feature].notna().mean()),
                        "mean": float(group[feature].mean()) if pd.api.types.is_numeric_dtype(group[feature]) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def split_summary(splits: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for source, split_map in splits.items():
        for split_name, df in split_map.items():
            rows.append(
                {
                    "source": source,
                    "split": split_name,
                    "n": int(len(df)),
                    "patients": int(df["patient_id"].nunique()),
                    "events": int(df["outcome"].sum()),
                    "event_rate": float(df["outcome"].mean()),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_readme(
    metrics_df: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    recalibration_summary: pd.DataFrame,
    dca: pd.DataFrame,
    split_df: pd.DataFrame,
) -> None:
    focus_metrics = metrics_df[
        metrics_df["test_type"].eq("transport_test")
        & metrics_df["feature_set"].eq("icu_native_primary")
    ].copy()
    focus_metrics = focus_metrics.sort_values(["direction", "model"])
    metric_cols = [
        "direction",
        "model",
        "n",
        "events",
        "event_rate",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "ece_10bin",
        "calibration_slope",
        "calibration_intercept",
        "mean_prediction",
    ]

    boot_focus = bootstrap_summary[
        bootstrap_summary["test_type"].eq("transport_test")
        & bootstrap_summary["feature_set"].eq("icu_native_primary")
        & bootstrap_summary["model"].eq("logistic_regression")
        & bootstrap_summary["metric"].isin(["roc_auc", "brier_score", "ece_10bin", "calibration_slope", "calibration_intercept"])
    ].copy()
    boot_focus = boot_focus.sort_values(["direction", "metric"])
    boot_cols = ["direction", "metric", "point", "ci_lower", "ci_upper"]

    recal_focus = recalibration_summary[
        recalibration_summary["feature_set"].eq("icu_native_primary")
        & recalibration_summary["model"].eq("logistic_regression")
        & (
            recalibration_summary["method"].eq("raw")
            | recalibration_summary["method"].isin(["intercept_only", "platt", "isotonic"])
        )
    ].copy()
    recal_focus = recal_focus[
        recal_focus["event_target"].isin([0, 25, 50, 100, 200])
    ].sort_values(["direction", "event_target", "method"])
    recal_cols = [
        "direction",
        "method",
        "event_target",
        "brier_score_mean",
        "ece_10bin_mean",
        "calibration_slope_mean",
        "calibration_intercept_mean",
        "roc_auc_mean",
    ]

    dca_focus = dca.sort_values(["direction", "threshold", "strategy"])
    dca_cols = ["direction", "strategy", "threshold", "net_benefit", "calibration_events"]

    text = f"""# MIMIC-IV ICU <-> eICU Transport and Recalibration

Generated on 2026-06-11.

## Design

- Endpoint: hospital mortality in both databases.
- Unit: one diabetes ICU admission/stay row; MIMIC keeps the first ICU stay per hospital admission.
- Split: patient-level {TRAIN_FRACTION:.0%}/{1 - TRAIN_FRACTION:.0%} development/holdout split within each database.
- Main transport directions: MIMIC-IV ICU -> eICU and eICU -> MIMIC-IV ICU.
- Primary model: logistic regression with the same class-balanced setup used in the existing NHANES -> MIMIC scripts.
- Benchmark model: HistGradientBoostingClassifier.
- Local recalibration samples are drawn from the target-site development split and evaluated on the fixed target-site holdout split.

## Feature Sets

- `icu_native_primary`: age, sex, hypertension, CKD, CVD, glucose, creatinine, BUN, WBC, hemoglobin, HR, SBP, DBP, SpO2.
- `nhanes_compatible_restricted`: age, sex, BMI, SBP, DBP, hypertension, CVD, glucose, creatinine.

Albumin and temperature are excluded from the primary model. Albumin can be added later as a focused sensitivity analysis if needed.

## Split Summary

{markdown_table(split_df, ["source", "split", "n", "patients", "events", "event_rate"])}

## Primary Transport Performance

{markdown_table(focus_metrics, metric_cols)}

## Bootstrap CIs, Primary Logistic Transport

{markdown_table(boot_focus, boot_cols)}

## Event-Count Recalibration, Primary Logistic Transport

{markdown_table(recal_focus, recal_cols)}

## Decision Curve, Selected Thresholds

{markdown_table(dca_focus, dca_cols)}

## Outputs

```text
outputs\\mimic_icu_eicu_transport_recalibration\\mimic_icu_eicu_common_transport_table.csv
outputs\\mimic_icu_eicu_transport_recalibration\\common_feature_missingness.csv
outputs\\mimic_icu_eicu_transport_recalibration\\split_summary.csv
outputs\\mimic_icu_eicu_transport_recalibration\\transport_predictions.csv
outputs\\mimic_icu_eicu_transport_recalibration\\target_development_calibration_pool_predictions.csv
outputs\\mimic_icu_eicu_transport_recalibration\\transport_metrics.csv
outputs\\mimic_icu_eicu_transport_recalibration\\transport_calibration_curve.csv
outputs\\mimic_icu_eicu_transport_recalibration\\transport_bootstrap_long.csv
outputs\\mimic_icu_eicu_transport_recalibration\\transport_bootstrap_ci.csv
outputs\\mimic_icu_eicu_transport_recalibration\\event_count_recalibration_long.csv
outputs\\mimic_icu_eicu_transport_recalibration\\event_count_recalibration_summary.csv
outputs\\mimic_icu_eicu_transport_recalibration\\decision_curve_selected_thresholds.csv
outputs\\mimic_icu_eicu_transport_recalibration\\decision_curve_eval_predictions.csv
outputs\\mimic_icu_eicu_transport_recalibration\\subgroup_transport_metrics.csv
outputs\\mimic_icu_eicu_transport_recalibration\\analysis_metadata.json
outputs\\mimic_icu_eicu_transport_recalibration\\README_mimic_icu_eicu_transport_recalibration.md
```
"""
    (OUT_DIR / "README_mimic_icu_eicu_transport_recalibration.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mimic = load_source("MIMIC-IV ICU")
    eicu = load_source("eICU")
    combined = pd.concat([mimic, eicu], ignore_index=True, sort=False)
    combined.to_csv(OUT_DIR / "mimic_icu_eicu_common_transport_table.csv", index=False)
    common_feature_missingness(combined).to_csv(OUT_DIR / "common_feature_missingness.csv", index=False)

    splits: dict[str, dict[str, pd.DataFrame]] = {}
    for i, (source, df) in enumerate([("MIMIC-IV ICU", mimic), ("eICU", eicu)]):
        development, holdout = split_by_patient(df, RANDOM_SEED + i)
        splits[source] = {"development": development, "holdout": holdout}
    split_df = split_summary(splits)
    split_df.to_csv(OUT_DIR / "split_summary.csv", index=False)

    predictions, calibration_pool_predictions = fit_direction_models(splits)
    predictions.to_csv(OUT_DIR / "transport_predictions.csv", index=False)
    calibration_pool_predictions.to_csv(OUT_DIR / "target_development_calibration_pool_predictions.csv", index=False)

    metrics_df = metric_rows(predictions)
    metrics_df.to_csv(OUT_DIR / "transport_metrics.csv", index=False)
    calibration_curve_rows(predictions).to_csv(OUT_DIR / "transport_calibration_curve.csv", index=False)

    bootstrap_long, bootstrap_summary = bootstrap_ci(predictions)
    bootstrap_long.to_csv(OUT_DIR / "transport_bootstrap_long.csv", index=False)
    bootstrap_summary.to_csv(OUT_DIR / "transport_bootstrap_ci.csv", index=False)

    recalibration_long, recalibration_summary = run_event_count_recalibration(predictions, calibration_pool_predictions)
    recalibration_long.to_csv(OUT_DIR / "event_count_recalibration_long.csv", index=False)
    recalibration_summary.to_csv(OUT_DIR / "event_count_recalibration_summary.csv", index=False)

    dca, dca_eval = build_dca(predictions, calibration_pool_predictions)
    dca.to_csv(OUT_DIR / "decision_curve_selected_thresholds.csv", index=False)
    dca_eval.to_csv(OUT_DIR / "decision_curve_eval_predictions.csv", index=False)

    subgroup = subgroup_metrics(predictions)
    subgroup.to_csv(OUT_DIR / "subgroup_transport_metrics.csv", index=False)

    metadata = {
        "purpose": "MIMIC-IV ICU <-> eICU hospital mortality transportability, recalibration, DCA, and subgroup analysis.",
        "endpoint": {
            "mimic": "admissions.hospital_expire_flag == 1, exported as hospital_mortality",
            "eicu": "hospitaldischargestatus == 'Expired', exported as hospital_mortality",
        },
        "source_paths": {source: str(cfg["path"]) for source, cfg in SOURCE_CONFIG.items()},
        "feature_sets": FEATURE_SETS,
        "split": {
            "type": "patient-level GroupShuffleSplit",
            "train_fraction": TRAIN_FRACTION,
            "random_seed": RANDOM_SEED,
        },
        "models": {
            "primary": "logistic_regression, class_weight='balanced'",
            "benchmark": "hist_gradient_boosting",
        },
        "bootstrap": {
            "n": N_BOOTSTRAP,
            "ci": "Percentile 2.5% and 97.5% intervals.",
        },
        "event_count_recalibration": {
            "event_targets": EVENT_TARGETS,
            "n_repeats": N_RECALIBRATION_REPEATS,
            "methods": ["intercept_only", "platt", "isotonic"],
            "calibration_pool": "Target-site development split.",
            "evaluation": "Fixed target-site holdout split.",
        },
        "decision_curve": {
            "thresholds": DCA_THRESHOLDS,
            "strategies": [
                "Treat none",
                "Treat all",
                "Raw transport logistic",
                "Platt 100 events",
                "Internal logistic benchmark",
                "Internal HGB benchmark",
            ],
        },
        "outputs": {
            "common_table": str(OUT_DIR / "mimic_icu_eicu_common_transport_table.csv"),
            "transport_metrics": str(OUT_DIR / "transport_metrics.csv"),
            "bootstrap_ci": str(OUT_DIR / "transport_bootstrap_ci.csv"),
            "recalibration_summary": str(OUT_DIR / "event_count_recalibration_summary.csv"),
            "decision_curve": str(OUT_DIR / "decision_curve_selected_thresholds.csv"),
            "subgroup": str(OUT_DIR / "subgroup_transport_metrics.csv"),
        },
    }
    (OUT_DIR / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(metrics_df, bootstrap_summary, recalibration_summary, dca, split_df)

    focus = metrics_df[
        metrics_df["test_type"].eq("transport_test")
        & metrics_df["feature_set"].eq("icu_native_primary")
    ].copy()
    print(focus.to_string(index=False))
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
