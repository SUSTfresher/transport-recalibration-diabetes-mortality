from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import mimic_icu_eicu_transport_recalibration as icu
import nhanes_mimic_oneyear_mortality_transport as nm
import sensitivity_unweighted_logistic as sens


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "sensitivity_unweighted_full_recalibration_dca"

RANDOM_SEED = 20260611
EVENT_TARGETS = [25, 50, 100, 200]
N_REPEATS = 200
DCA_THRESHOLDS = [0.20, 0.25, 0.30]
EPS = 1e-6


def sample_by_event_count(y: np.ndarray, event_target: int, rng: np.random.Generator) -> np.ndarray:
    event_idx = np.flatnonzero(y == 1)
    nonevent_idx = np.flatnonzero(y == 0)
    if event_target >= len(event_idx):
        raise ValueError(f"event_target={event_target} exceeds available events={len(event_idx)}")
    event_rate = len(event_idx) / len(y)
    total_n = int(round(event_target / event_rate))
    nonevent_target = max(1, total_n - event_target)
    nonevent_target = min(nonevent_target, len(nonevent_idx))
    chosen_events = rng.choice(event_idx, size=event_target, replace=False)
    chosen_nonevents = rng.choice(nonevent_idx, size=nonevent_target, replace=False)
    return np.concatenate([chosen_events, chosen_nonevents])


def intercept_only_offset(y_cal: np.ndarray, pred_cal: np.ndarray) -> float:
    target = float(np.clip(np.mean(y_cal), EPS, 1 - EPS))
    base = sens.logit(pred_cal)
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = (lo + hi) / 2
        current = float(np.mean(sens.inv_logit(base + mid)))
        if current < target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def apply_recalibration(method: str, y_cal: np.ndarray, pred_cal: np.ndarray, pred_eval: np.ndarray) -> np.ndarray:
    pred_eval = np.clip(np.asarray(pred_eval, dtype=float), EPS, 1 - EPS)
    if method == "raw":
        return pred_eval
    if method == "intercept_only":
        offset = intercept_only_offset(y_cal, pred_cal)
        return np.clip(sens.inv_logit(sens.logit(pred_eval) + offset), EPS, 1 - EPS)
    if method == "platt":
        model = LogisticRegression(max_iter=5000, solver="lbfgs", C=1e12)
        model.fit(sens.logit(pred_cal).reshape(-1, 1), y_cal.astype(int))
        pred = model.predict_proba(sens.logit(pred_eval).reshape(-1, 1))[:, 1]
        return np.clip(pred, EPS, 1 - EPS)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        model.fit(pred_cal, y_cal)
        pred = model.predict(pred_eval)
        return np.clip(pred, EPS, 1 - EPS)
    raise ValueError(method)


def net_benefit(y: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    treated = pred >= threshold
    n = len(y)
    tp = int(((y == 1) & treated).sum())
    fp = int(((y == 0) & treated).sum())
    return float(tp / n - fp / n * threshold / (1 - threshold))


def nhanes_prediction_set() -> dict[str, object]:
    nhanes = nm.load_nhanes()
    mimic = nm.load_mimic()
    nh_train, _ = nm.temporal_nhanes_split(nhanes)
    mimic_train, mimic_test = nm.mimic_split(mimic)
    features = nm.COMMON_BASE

    source_train = nh_train.dropna(subset=["outcome"]).copy()
    target_train = mimic_train.dropna(subset=["outcome"]).copy()
    target_eval = mimic_test.dropna(subset=["outcome"]).copy()

    source_model = sens.make_unweighted_nhanes_model(features)
    source_model.fit(source_train[features], source_train["outcome"].astype(int))

    internal_logistic = sens.make_unweighted_nhanes_model(features)
    internal_logistic.fit(target_train[features], target_train["outcome"].astype(int))

    internal_hgb = nm.make_model("hist_gradient_boosting", features)
    internal_hgb.fit(target_train[features], target_train["outcome"].astype(int))

    eval_df = pd.DataFrame(
        {
            "id": target_eval["id"].to_numpy(),
            "outcome": target_eval["outcome"].astype(int).to_numpy(),
            "raw_transport": source_model.predict_proba(target_eval[features])[:, 1],
            "internal_logistic": internal_logistic.predict_proba(target_eval[features])[:, 1],
            "internal_hgb": internal_hgb.predict_proba(target_eval[features])[:, 1],
        }
    )
    eval_df["row_id"] = np.arange(len(eval_df))
    cal_df = eval_df[["row_id", "outcome", "raw_transport"]].copy()
    return {
        "scenario": "Extreme cross-setting stress test",
        "direction": "NHANES -> MIMIC-IV",
        "endpoint": "1-year mortality",
        "feature_set": "NHANES-compatible base",
        "calibration_design": "target_evaluation_resampling_excluding_calibration_rows",
        "eval_df": eval_df,
        "cal_df": cal_df,
        "exclude_calibration_from_evaluation": True,
    }


def icu_prediction_sets() -> list[dict[str, object]]:
    sources = {
        "MIMIC-IV ICU": icu.load_source("MIMIC-IV ICU"),
        "eICU": icu.load_source("eICU"),
    }
    splits: dict[str, dict[str, pd.DataFrame]] = {}
    for i, source in enumerate(["MIMIC-IV ICU", "eICU"]):
        development, holdout = icu.split_by_patient(sources[source], icu.RANDOM_SEED + i)
        splits[source] = {"development": development, "holdout": holdout}

    spec = icu.FEATURE_SETS["icu_native_primary"]
    numeric = spec["numeric"]
    binary = spec["binary"]
    features = numeric + binary
    sets = []
    for train_source, target_source in [("MIMIC-IV ICU", "eICU"), ("eICU", "MIMIC-IV ICU")]:
        source_train = splits[train_source]["development"]
        target_train = splits[target_source]["development"]
        target_eval = splits[target_source]["holdout"]

        source_model = sens.make_unweighted_icu_model(numeric, binary)
        source_model.fit(source_train[features], source_train["outcome"].astype(int))

        internal_logistic = sens.make_unweighted_icu_model(numeric, binary)
        internal_logistic.fit(target_train[features], target_train["outcome"].astype(int))

        internal_hgb = icu.make_model("hist_gradient_boosting", numeric, binary)
        internal_hgb.fit(target_train[features], target_train["outcome"].astype(int))

        eval_df = pd.DataFrame(
            {
                "id": target_eval["id"].to_numpy(),
                "outcome": target_eval["outcome"].astype(int).to_numpy(),
                "raw_transport": source_model.predict_proba(target_eval[features])[:, 1],
                "internal_logistic": internal_logistic.predict_proba(target_eval[features])[:, 1],
                "internal_hgb": internal_hgb.predict_proba(target_eval[features])[:, 1],
            }
        )
        eval_df["row_id"] = np.arange(len(eval_df))
        cal_df = pd.DataFrame(
            {
                "row_id": np.arange(len(target_train)),
                "outcome": target_train["outcome"].astype(int).to_numpy(),
                "raw_transport": source_model.predict_proba(target_train[features])[:, 1],
            }
        )
        sets.append(
            {
                "scenario": "Realistic ICU deployment",
                "direction": f"{train_source} -> {target_source}",
                "endpoint": "Hospital mortality",
                "feature_set": "ICU-native primary",
                "calibration_design": "target_development_calibration_pool_fixed_holdout_evaluation",
                "eval_df": eval_df,
                "cal_df": cal_df,
                "exclude_calibration_from_evaluation": False,
            }
        )
    return sets


def prediction_sets_to_frames(prediction_sets: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_rows = []
    cal_rows = []
    for item in prediction_sets:
        meta = {
            "scenario": item["scenario"],
            "direction": item["direction"],
            "endpoint": item["endpoint"],
            "feature_set": item["feature_set"],
            "model": "logistic_regression_unweighted",
            "calibration_design": item["calibration_design"],
        }
        eval_df = item["eval_df"].copy()
        cal_df = item["cal_df"].copy()
        for key, value in meta.items():
            eval_df[key] = value
            cal_df[key] = value
        eval_rows.append(eval_df)
        cal_rows.append(cal_df)
    return pd.concat(eval_rows, ignore_index=True), pd.concat(cal_rows, ignore_index=True)


def run_event_count_recalibration(prediction_sets: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for item in prediction_sets:
        eval_df = item["eval_df"].reset_index(drop=True)
        cal_df = item["cal_df"].reset_index(drop=True)
        y_eval_full = eval_df["outcome"].astype(int).to_numpy()
        pred_eval_full = np.clip(eval_df["raw_transport"].to_numpy(), EPS, 1 - EPS)
        y_cal_pool = cal_df["outcome"].astype(int).to_numpy()
        pred_cal_pool = np.clip(cal_df["raw_transport"].to_numpy(), EPS, 1 - EPS)
        base = {
            "scenario": item["scenario"],
            "direction": item["direction"],
            "endpoint": item["endpoint"],
            "feature_set": item["feature_set"],
            "model": "logistic_regression_unweighted",
            "calibration_design": item["calibration_design"],
        }

        raw_metrics = sens.point_metrics(y_eval_full, pred_eval_full)
        raw_metrics.update(
            {
                **base,
                "method": "raw",
                "event_target": 0,
                "repeat": 0,
                "calibration_n": 0,
                "calibration_events": 0,
                "calibration_event_rate": np.nan,
                "evaluation_n": int(len(y_eval_full)),
                "evaluation_events": int(y_eval_full.sum()),
            }
        )
        rows.append(raw_metrics)

        for event_target in EVENT_TARGETS:
            if event_target >= int(y_cal_pool.sum()):
                continue
            for repeat in range(N_REPEATS):
                cal_idx = sample_by_event_count(y_cal_pool, event_target, rng)
                y_cal = y_cal_pool[cal_idx]
                pred_cal = pred_cal_pool[cal_idx]
                if item["exclude_calibration_from_evaluation"]:
                    eval_mask = np.ones(len(eval_df), dtype=bool)
                    eval_mask[cal_idx] = False
                    y_eval = y_eval_full[eval_mask]
                    pred_eval_raw = pred_eval_full[eval_mask]
                else:
                    y_eval = y_eval_full
                    pred_eval_raw = pred_eval_full

                for method in ["intercept_only", "platt", "isotonic"]:
                    pred_eval = apply_recalibration(method, y_cal, pred_cal, pred_eval_raw)
                    m = sens.point_metrics(y_eval, pred_eval)
                    m.update(
                        {
                            **base,
                            "method": method,
                            "event_target": event_target,
                            "repeat": repeat,
                            "calibration_n": int(len(cal_idx)),
                            "calibration_events": int(y_cal.sum()),
                            "calibration_event_rate": float(y_cal.mean()),
                            "evaluation_n": int(len(y_eval)),
                            "evaluation_events": int(y_eval.sum()),
                        }
                    )
                    rows.append(m)
    long = pd.DataFrame(rows)
    return long, summarize_recalibration(long)


def summarize_recalibration(long: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scenario", "direction", "endpoint", "feature_set", "model", "calibration_design", "method", "event_target"]
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "ece_10bin",
        "ece_10bin_equal_width",
        "mean_prediction",
        "event_rate",
        "calibration_slope",
        "calibration_intercept",
        "calibration_n",
        "calibration_events",
        "calibration_event_rate",
        "evaluation_n",
        "evaluation_events",
    ]
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


def build_dca(prediction_sets: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    eval_parts = []
    for i, item in enumerate(prediction_sets):
        rng = np.random.default_rng(RANDOM_SEED + 1000 + i)
        eval_df = item["eval_df"].reset_index(drop=True).copy()
        cal_df = item["cal_df"].reset_index(drop=True)
        y_cal_pool = cal_df["outcome"].astype(int).to_numpy()
        pred_cal_pool = np.clip(cal_df["raw_transport"].to_numpy(), EPS, 1 - EPS)
        cal_idx = sample_by_event_count(y_cal_pool, 100, rng)
        y_cal = y_cal_pool[cal_idx]
        pred_cal = pred_cal_pool[cal_idx]

        if item["exclude_calibration_from_evaluation"]:
            eval_mask = np.ones(len(eval_df), dtype=bool)
            eval_mask[cal_idx] = False
            eval_df = eval_df.loc[eval_mask].reset_index(drop=True)

        y_eval = eval_df["outcome"].astype(int).to_numpy()
        raw = np.clip(eval_df["raw_transport"].to_numpy(), EPS, 1 - EPS)
        intercept = apply_recalibration("intercept_only", y_cal, pred_cal, raw)
        platt = apply_recalibration("platt", y_cal, pred_cal, raw)
        eval_df["intercept_only_100_events"] = intercept
        eval_df["platt_100_events"] = platt
        eval_df["scenario"] = item["scenario"]
        eval_df["direction"] = item["direction"]
        eval_df["endpoint"] = item["endpoint"]
        eval_df["feature_set"] = item["feature_set"]
        eval_df["calibration_n"] = int(len(cal_idx))
        eval_df["calibration_events"] = int(y_cal.sum())
        eval_parts.append(eval_df)

        strategies = {
            "Treat none": None,
            "Treat all": None,
            "Raw transport logistic": raw,
            "Intercept-only 100 events": intercept,
            "Platt 100 events": platt,
            "Internal logistic benchmark": eval_df["internal_logistic"].to_numpy(),
            "Internal HGB benchmark": eval_df["internal_hgb"].to_numpy(),
        }
        event_rate = float(y_eval.mean())
        for threshold in DCA_THRESHOLDS:
            for strategy, pred in strategies.items():
                if strategy == "Treat none":
                    nb = 0.0
                elif strategy == "Treat all":
                    nb = event_rate - (1 - event_rate) * threshold / (1 - threshold)
                else:
                    nb = net_benefit(y_eval, pred, threshold)
                rows.append(
                    {
                        "scenario": item["scenario"],
                        "direction": item["direction"],
                        "endpoint": item["endpoint"],
                        "feature_set": item["feature_set"],
                        "model": "logistic_regression_unweighted",
                        "strategy": strategy,
                        "threshold": threshold,
                        "net_benefit": nb,
                        "calibration_n": int(len(cal_idx)) if "100 events" in strategy else 0,
                        "calibration_events": int(y_cal.sum()) if "100 events" in strategy else 0,
                        "evaluation_n": int(len(y_eval)),
                        "evaluation_events": int(y_eval.sum()),
                    }
                )
    return pd.DataFrame(rows), pd.concat(eval_parts, ignore_index=True)


def fmt_interval(row: pd.Series, metric: str) -> str:
    mean = row[f"{metric}_mean"]
    lo = row[f"{metric}_ci_lower"]
    hi = row[f"{metric}_ci_upper"]
    if pd.isna(mean):
        return ""
    if pd.isna(lo) or pd.isna(hi) or row["n_repeats"] == 1:
        return f"{mean:.3f}"
    return f"{mean:.3f} ({lo:.3f}-{hi:.3f})"


def write_readme(summary: pd.DataFrame, dca: pd.DataFrame) -> None:
    selected = summary[
        summary["method"].eq("raw")
        | ((summary["event_target"].eq(100)) & summary["method"].isin(["intercept_only", "platt", "isotonic"]))
    ].copy()
    lines = [
        "# Unweighted Logistic Full Sensitivity",
        "",
        "This analysis repeats the three primary transport directions with unweighted logistic regression and then reruns event-count recalibration and selected-threshold decision-curve analysis.",
        "",
        "## Selected recalibration results",
        "",
        "| Direction | Method | Events | ECE | Equal-width ECE | Slope | Mean predicted risk |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in selected.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["direction"]),
                    str(row["method"]),
                    str(int(row["event_target"])),
                    fmt_interval(row, "ece_10bin"),
                    fmt_interval(row, "ece_10bin_equal_width"),
                    fmt_interval(row, "calibration_slope"),
                    fmt_interval(row, "mean_prediction"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Selected DCA results",
            "",
            "| Direction | Strategy | Threshold | Net benefit |",
            "| --- | --- | --- | --- |",
        ]
    )
    focus = dca[dca["threshold"].isin(DCA_THRESHOLDS)].copy()
    for _, row in focus.iterrows():
        lines.append(f"| {row['direction']} | {row['strategy']} | {row['threshold']:.2f} | {row['net_benefit']:.4f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The unweighted analysis tests whether class weighting drove the apparent need for recalibration. It should be read as a major sensitivity analysis rather than a replacement for the locked primary class-weighted results unless the manuscript is later re-centered on unweighted logistic regression.",
        ]
    )
    (OUT_DIR / "README_unweighted_full_recalibration_dca.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_sets = [nhanes_prediction_set(), *icu_prediction_sets()]
    eval_predictions, calibration_pools = prediction_sets_to_frames(prediction_sets)
    eval_predictions.to_csv(OUT_DIR / "unweighted_transport_eval_predictions.csv", index=False)
    calibration_pools.to_csv(OUT_DIR / "unweighted_calibration_pool_predictions.csv", index=False)

    metrics_rows = []
    for item in prediction_sets:
        y = item["eval_df"]["outcome"].astype(int).to_numpy()
        pred = item["eval_df"]["raw_transport"].to_numpy()
        row = sens.point_metrics(y, pred)
        row.update(
            {
                "scenario": item["scenario"],
                "direction": item["direction"],
                "endpoint": item["endpoint"],
                "feature_set": item["feature_set"],
                "model": "logistic_regression_unweighted",
                "calibration_design": item["calibration_design"],
            }
        )
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "unweighted_transport_metrics.csv", index=False)

    recalibration_long, recalibration_summary = run_event_count_recalibration(prediction_sets)
    recalibration_long.to_csv(OUT_DIR / "unweighted_event_count_recalibration_long.csv", index=False)
    recalibration_summary.to_csv(OUT_DIR / "unweighted_event_count_recalibration_summary.csv", index=False)

    dca, dca_eval = build_dca(prediction_sets)
    dca.to_csv(OUT_DIR / "unweighted_decision_curve_selected_thresholds.csv", index=False)
    dca_eval.to_csv(OUT_DIR / "unweighted_decision_curve_eval_predictions.csv", index=False)

    metadata = {
        "purpose": "Full sensitivity analysis rerunning transport, event-count recalibration, and selected-threshold DCA with unweighted logistic regression.",
        "event_targets": EVENT_TARGETS,
        "n_repeats": N_REPEATS,
        "dca_thresholds": DCA_THRESHOLDS,
        "nhanes_calibration_design": "Calibration samples drawn from the MIMIC target evaluation set and excluded from replicate evaluation, matching the original NHANES-to-MIMIC event-count simulation.",
        "icu_calibration_design": "Calibration samples drawn from target-site development split and evaluated on fixed target-site holdout, matching the original ICU-to-ICU simulation.",
        "outputs": {
            "metrics": str(OUT_DIR / "unweighted_transport_metrics.csv"),
            "recalibration_summary": str(OUT_DIR / "unweighted_event_count_recalibration_summary.csv"),
            "dca": str(OUT_DIR / "unweighted_decision_curve_selected_thresholds.csv"),
        },
    }
    (OUT_DIR / "unweighted_full_sensitivity_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme(recalibration_summary, dca)

    print(metrics_df.to_string(index=False))
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
