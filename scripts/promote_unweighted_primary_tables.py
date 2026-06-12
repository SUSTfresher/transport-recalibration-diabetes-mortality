from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import mimic_icu_eicu_transport_recalibration as icu
import nhanes_mimic_oneyear_mortality_transport as nm
import sensitivity_unweighted_logistic as sens


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "manuscript_tables"
CURRENT_DIR = ROOT / "outputs" / "current_submission" / "tables"

PRED_PATH = ROOT / "outputs" / "sensitivity_unweighted_full_recalibration_dca" / "unweighted_transport_eval_predictions.csv"
RECAL_PATH = ROOT / "outputs" / "sensitivity_unweighted_full_recalibration_dca" / "unweighted_event_count_recalibration_summary.csv"
DCA_PATH = ROOT / "outputs" / "sensitivity_unweighted_full_recalibration_dca" / "unweighted_decision_curve_selected_thresholds.csv"

N_BOOTSTRAP = 1000
RANDOM_SEED = 20260611

DIRECTION_ORDER = [
    "NHANES -> MIMIC-IV",
    "MIMIC-IV ICU -> eICU",
    "eICU -> MIMIC-IV ICU",
]

ENDPOINT_NOTES = {
    "NHANES -> MIMIC-IV": "NHANES death within 12 months; MIMIC-IV death within 1 year after discharge.",
    "MIMIC-IV ICU -> eICU": "MIMIC-IV hospital_expire_flag; eICU hospitaldischargestatus == Expired.",
    "eICU -> MIMIC-IV ICU": "eICU hospitaldischargestatus == Expired; MIMIC-IV hospital_expire_flag.",
}

METRICS = [
    "event_rate",
    "mean_prediction",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "ece_10bin",
    "ece_10bin_equal_width",
    "calibration_slope",
    "calibration_intercept",
]


def fmt_decimal(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def fmt_ci(mean: float, lo: float | None, hi: float | None, digits: int = 3) -> str:
    if pd.isna(mean):
        return "NA"
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return fmt_decimal(mean, digits)
    if np.isclose(float(mean), float(lo)) and np.isclose(float(mean), float(hi)):
        return fmt_decimal(mean, digits)
    return f"{float(mean):.{digits}f} ({float(lo):.{digits}f}-{float(hi):.{digits}f})"


def markdown_table(df: pd.DataFrame) -> str:
    display = df.fillna("NA").astype(str)
    cols = list(display.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(row[col] for col in cols) + " |")
    return "\n".join(lines)


def write_table(name: str, df: pd.DataFrame, notes: list[str]) -> None:
    csv_path = OUT_DIR / f"{name}.csv"
    md_path = OUT_DIR / f"{name}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    note_text = "\n".join(f"- {note}" for note in notes)
    md_path.write_text(f"# {name}\n\n{markdown_table(df)}\n\n## Notes\n\n{note_text}\n", encoding="utf-8")


def table3_bootstrap() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preds = pd.read_csv(PRED_PATH)
    rng = np.random.default_rng(RANDOM_SEED)
    summaries: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []

    for direction in DIRECTION_ORDER:
        group = preds[preds["direction"].eq(direction)].copy()
        if group.empty:
            raise ValueError(f"Missing unweighted predictions for {direction}")
        y = group["outcome"].astype(int).to_numpy()
        pred = group["raw_transport"].to_numpy()
        point = sens.point_metrics(y, pred)
        id_cols = {
            "scenario": group["scenario"].iloc[0],
            "direction": direction,
            "endpoint": group["endpoint"].iloc[0],
            "endpoint_note": ENDPOINT_NOTES[direction],
            "feature_set": group["feature_set"].iloc[0],
            "model": "Logistic regression (unweighted)",
            "n": int(len(group)),
            "events": int(y.sum()),
        }

        boot_rows = []
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, len(group), size=len(group))
            row = dict(id_cols)
            row["bootstrap"] = i
            row.update(sens.point_metrics(y[idx], pred[idx]))
            boot_rows.append(row)
        long_rows.extend(boot_rows)
        boot = pd.DataFrame(boot_rows)

        summary = dict(id_cols)
        for metric in METRICS:
            vals = pd.to_numeric(boot[metric], errors="coerce").dropna().to_numpy()
            summary[f"{metric}_point"] = point[metric]
            summary[f"{metric}_ci_lower"] = float(np.quantile(vals, 0.025)) if len(vals) else np.nan
            summary[f"{metric}_ci_upper"] = float(np.quantile(vals, 0.975)) if len(vals) else np.nan
            summary[f"{metric}_bootstrap_n_valid"] = int(len(vals))
        summaries.append(summary)

    numeric = pd.DataFrame(summaries)
    formatted_rows = []
    for _, row in numeric.iterrows():
        formatted_rows.append(
            {
                "Scenario": row["scenario"],
                "Direction": row["direction"],
                "Endpoint": row["endpoint"],
                "Feature set": row["feature_set"],
                "Model": row["model"],
                "Target evaluation N": int(row["n"]),
                "Target evaluation events": int(row["events"]),
                "Event rate": fmt_ci(row["event_rate_point"], row["event_rate_ci_lower"], row["event_rate_ci_upper"]),
                "Mean predicted risk": fmt_ci(row["mean_prediction_point"], row["mean_prediction_ci_lower"], row["mean_prediction_ci_upper"]),
                "AUC": fmt_ci(row["roc_auc_point"], row["roc_auc_ci_lower"], row["roc_auc_ci_upper"]),
                "PR AUC": fmt_ci(row["pr_auc_point"], row["pr_auc_ci_lower"], row["pr_auc_ci_upper"]),
                "Brier score": fmt_ci(row["brier_score_point"], row["brier_score_ci_lower"], row["brier_score_ci_upper"]),
                "ECE": fmt_ci(row["ece_10bin_point"], row["ece_10bin_ci_lower"], row["ece_10bin_ci_upper"]),
                "Equal-width ECE": fmt_ci(
                    row["ece_10bin_equal_width_point"],
                    row["ece_10bin_equal_width_ci_lower"],
                    row["ece_10bin_equal_width_ci_upper"],
                ),
                "Calibration slope": fmt_ci(
                    row["calibration_slope_point"],
                    row["calibration_slope_ci_lower"],
                    row["calibration_slope_ci_upper"],
                ),
                "Calibration intercept": fmt_ci(
                    row["calibration_intercept_point"],
                    row["calibration_intercept_ci_lower"],
                    row["calibration_intercept_ci_upper"],
                ),
            }
        )

    formatted = pd.DataFrame(formatted_rows)
    long = pd.DataFrame(long_rows)
    return formatted, numeric, long


def write_table3(formatted: pd.DataFrame, numeric: pd.DataFrame, long: pd.DataFrame) -> None:
    formatted.to_csv(OUT_DIR / "Table_3_cross_scenario_transport_performance.csv", index=False, encoding="utf-8-sig")
    numeric.to_csv(OUT_DIR / "Table_3_cross_scenario_transport_performance_numeric.csv", index=False)
    long.to_csv(OUT_DIR / "Table_3_cross_scenario_transport_bootstrap_long.csv", index=False)
    metadata = {
        "generated_on": "2026-06-11",
        "primary_model": "Unweighted logistic regression",
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "metrics": METRICS,
        "ci": "Percentile 2.5% and 97.5% bootstrap intervals.",
        "prediction_source": str(PRED_PATH),
        "endpoint_note": "NHANES -> MIMIC-IV uses one-year mortality; MIMIC-IV ICU <-> eICU uses hospital mortality.",
        "class_weighted_results": "Retained as Supplementary_Table_class_weighted_transport_performance.* and Supplementary_Table_class_weight_sensitivity.*",
    }
    (OUT_DIR / "Table_3_cross_scenario_transport_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    readme = f"""# Table 3 Cross-Scenario Transport Performance

Generated on 2026-06-11.

This table fixes the manuscript's core transport-performance comparison after promoting unweighted logistic regression to the primary probability model.

Important endpoint note: the NHANES -> MIMIC-IV stress-test uses a one-year mortality endpoint, whereas the MIMIC-IV ICU <-> eICU deployment analyses use harmonized hospital mortality. The NHANES and MIMIC-IV one-year endpoints also differ in ascertainment mechanism: NHANES uses linked community mortality follow-up, whereas MIMIC-IV uses hospital-episode data plus post-discharge mortality. These rows should be compared as endpoint-specific deployment scenarios; the cross-scenario contrast is intended to compare transport failure modes and recalibration needs, not endpoint-equivalent absolute accuracy. Do not interpret row-to-row differences in AUC, ECE, or Brier score as absolute performance differences under a shared endpoint.

Model note: Table 3 reports unweighted logistic regression. The former class-weighted primary analysis is retained in the supplementary class-weighting sensitivity tables.

All displayed intervals are percentile 95% confidence intervals from {N_BOOTSTRAP} bootstrap resamples of the target/evaluation cohort.

{markdown_table(formatted)}

## Outputs

```text
outputs\\manuscript_tables\\Table_3_cross_scenario_transport_performance.csv
outputs\\manuscript_tables\\Table_3_cross_scenario_transport_performance_numeric.csv
outputs\\manuscript_tables\\Table_3_cross_scenario_transport_bootstrap_long.csv
outputs\\manuscript_tables\\Table_3_cross_scenario_transport_metadata.json
outputs\\manuscript_tables\\README_Table_3_cross_scenario_transport_performance.md
```
"""
    (OUT_DIR / "README_Table_3_cross_scenario_transport_performance.md").write_text(readme, encoding="utf-8")


def build_table4() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rec = pd.read_csv(RECAL_PATH)
    rec = rec[
        rec["direction"].isin(DIRECTION_ORDER)
        & rec["method"].isin(["raw", "intercept_only", "platt", "isotonic"])
        & rec["event_target"].isin([0, 25, 50, 100, 200])
    ].copy()
    method_labels = {
        "raw": "Raw transport",
        "intercept_only": "Intercept-only",
        "platt": "Platt",
        "isotonic": "Isotonic",
    }
    method_order = {"raw": 0, "intercept_only": 1, "platt": 2, "isotonic": 3}
    rec["Direction"] = rec["direction"]
    rec["Method"] = rec["method"].map(method_labels)
    rec["Local outcome events"] = rec["event_target"].astype(int).astype(str)
    rec["ece_mean"] = rec["ece_10bin_mean"]
    rec["ece_ci_lower"] = rec["ece_10bin_ci_lower"]
    rec["ece_ci_upper"] = rec["ece_10bin_ci_upper"]
    rec["calibration_n_mean"] = rec["calibration_n_mean"]
    rec["calibration_events_mean"] = rec["calibration_events_mean"]
    rec["ECE (95% empirical interval)"] = rec.apply(
        lambda r: fmt_decimal(r["ece_mean"], 3)
        if r["method"] == "raw"
        else fmt_ci(r["ece_mean"], r["ece_ci_lower"], r["ece_ci_upper"], 3),
        axis=1,
    )
    rec["_direction_order"] = rec["Direction"].map({d: i for i, d in enumerate(DIRECTION_ORDER)})
    rec["_method_order"] = rec["method"].map(method_order)
    rec["_event_order"] = rec["event_target"].astype(int)
    rec = rec.sort_values(["_direction_order", "_method_order", "_event_order"])

    numeric_long = rec[
        [
            "Direction",
            "method",
            "event_target",
            "ece_mean",
            "ece_ci_lower",
            "ece_ci_upper",
            "ece_10bin_equal_width_mean",
            "ece_10bin_equal_width_ci_lower",
            "ece_10bin_equal_width_ci_upper",
            "calibration_slope_mean",
            "calibration_slope_ci_lower",
            "calibration_slope_ci_upper",
            "mean_prediction_mean",
            "mean_prediction_ci_lower",
            "mean_prediction_ci_upper",
            "calibration_n_mean",
            "calibration_events_mean",
            "n_repeats",
            "Method",
            "Local outcome events",
            "ECE (95% empirical interval)",
        ]
    ].copy()

    wide = rec.pivot_table(
        index=["_method_order", "_event_order", "Method", "Local outcome events"],
        columns="Direction",
        values="ECE (95% empirical interval)",
        aggfunc="first",
    ).reset_index()
    for direction in DIRECTION_ORDER:
        if direction not in wide.columns:
            wide[direction] = "NA"
    wide = wide.sort_values(["_method_order", "_event_order"])
    wide = wide[["Method", "Local outcome events"] + DIRECTION_ORDER]

    panel = numeric_long[
        [
            "Direction",
            "Method",
            "Local outcome events",
            "ECE (95% empirical interval)",
            "calibration_n_mean",
            "calibration_events_mean",
            "n_repeats",
        ]
    ].rename(
        columns={
            "calibration_n_mean": "Mean calibration N",
            "calibration_events_mean": "Mean local events",
            "n_repeats": "Repeats",
        }
    )
    return wide, numeric_long, panel


def write_table4(wide: pd.DataFrame, numeric_long: pd.DataFrame, panel: pd.DataFrame) -> None:
    write_table(
        "Table_4_recalibration_by_event_count",
        wide,
        [
            "Primary model is unweighted logistic regression.",
            "Cells show ECE with empirical 95% intervals across 200 local recalibration samples when intervals are available.",
            "Raw transport is a point estimate before local recalibration.",
            "Event counts refer to local target-site outcome events used for recalibration.",
        ],
    )
    numeric_long.to_csv(OUT_DIR / "Table_4_recalibration_by_event_count_numeric_long.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(OUT_DIR / "Table_4_recalibration_by_event_count_panel.csv", index=False, encoding="utf-8-sig")
    sections = ["# Table_4_recalibration_by_event_count_panel\n"]
    for direction in DIRECTION_ORDER:
        part = panel[panel["Direction"].eq(direction)].drop(columns=["Direction"]).copy()
        sections.append(f"## {direction}\n")
        sections.append(markdown_table(part))
        sections.append("")
    sections.append("## Notes\n")
    sections.append("- Primary model is unweighted logistic regression.")
    sections.append("- Raw transport is a point estimate before local recalibration; interval columns apply to repeated recalibration samples.")
    (OUT_DIR / "Table_4_recalibration_by_event_count_panel.md").write_text("\n".join(sections), encoding="utf-8")


def build_table6() -> pd.DataFrame:
    dca = pd.read_csv(DCA_PATH)
    dca = dca[dca["direction"].isin(DIRECTION_ORDER) & dca["threshold"].round(2).isin([0.20, 0.25, 0.30])].copy()
    keep = [
        "Treat none",
        "Treat all",
        "Raw transport logistic",
        "Intercept-only 100 events",
        "Platt 100 events",
        "Internal HGB benchmark",
    ]
    dca = dca[dca["strategy"].isin(keep)]
    wide = dca.pivot_table(
        index=["direction", "threshold"],
        columns="strategy",
        values="net_benefit",
        aggfunc="first",
    ).reset_index()
    for col in keep:
        if col not in wide:
            wide[col] = np.nan
        wide[col] = wide[col].map(lambda x: fmt_decimal(x, 4))
    wide["_direction_order"] = wide["direction"].map({d: i for i, d in enumerate(DIRECTION_ORDER)})
    wide = wide.sort_values(["_direction_order", "threshold"]).drop(columns="_direction_order")
    wide["Threshold"] = wide["threshold"].map(lambda x: f"{float(x):.2f}")
    wide = wide.rename(columns={"direction": "Direction"})
    return wide[
        [
            "Direction",
            "Threshold",
            "Treat none",
            "Treat all",
            "Raw transport logistic",
            "Intercept-only 100 events",
            "Platt 100 events",
            "Internal HGB benchmark",
        ]
    ]


def subgroup_target_features() -> pd.DataFrame:
    frames = []

    mimic = nm.load_mimic()
    _, mimic_test = nm.mimic_split(mimic)
    frames.append(
        pd.DataFrame(
            {
                "direction": "NHANES -> MIMIC-IV",
                "id": mimic_test["id"].astype(str),
                "age": pd.to_numeric(mimic_test["age"], errors="coerce"),
                "ckd_history": pd.to_numeric(mimic_test["ckd_history"], errors="coerce"),
                "cvd_history": pd.to_numeric(mimic_test["cvd_history"], errors="coerce"),
            }
        )
    )

    sources = {
        "MIMIC-IV ICU": icu.load_source("MIMIC-IV ICU"),
        "eICU": icu.load_source("eICU"),
    }
    splits: dict[str, dict[str, pd.DataFrame]] = {}
    for i, source in enumerate(["MIMIC-IV ICU", "eICU"]):
        development, holdout = icu.split_by_patient(sources[source], icu.RANDOM_SEED + i)
        splits[source] = {"development": development, "holdout": holdout}

    for direction, target_source in [
        ("MIMIC-IV ICU -> eICU", "eICU"),
        ("eICU -> MIMIC-IV ICU", "MIMIC-IV ICU"),
    ]:
        target = splits[target_source]["holdout"]
        frames.append(
            pd.DataFrame(
                {
                    "direction": direction,
                    "id": target["id"].astype(str),
                    "age": pd.to_numeric(target["age"], errors="coerce"),
                    "ckd_history": pd.to_numeric(target["ckd_history"], errors="coerce"),
                    "cvd_history": pd.to_numeric(target["cvd_history"], errors="coerce"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_table5() -> pd.DataFrame:
    preds = pd.read_csv(PRED_PATH, dtype={"id": str})
    features = subgroup_target_features()
    merged = preds.merge(features, on=["direction", "id"], how="left")
    subgroup_defs = [
        ("Age", "Age <65", lambda df: pd.to_numeric(df["age"], errors="coerce").lt(65)),
        ("Age", "Age >=65", lambda df: pd.to_numeric(df["age"], errors="coerce").ge(65)),
        ("CKD history", "No CKD", lambda df: pd.to_numeric(df["ckd_history"], errors="coerce").eq(0)),
        ("CKD history", "CKD", lambda df: pd.to_numeric(df["ckd_history"], errors="coerce").eq(1)),
        ("CVD history", "No CVD", lambda df: pd.to_numeric(df["cvd_history"], errors="coerce").eq(0)),
        ("CVD history", "CVD", lambda df: pd.to_numeric(df["cvd_history"], errors="coerce").eq(1)),
    ]
    rows = []
    for direction in DIRECTION_ORDER:
        group = merged[merged["direction"].eq(direction)].copy()
        for subgroup_type, subgroup, mask_fn in subgroup_defs:
            mask = mask_fn(group).fillna(False).to_numpy()
            sub = group.loc[mask].copy()
            if len(sub) < 100:
                continue
            y = sub["outcome"].astype(int).to_numpy()
            pred = sub["raw_transport"].to_numpy()
            m = sens.point_metrics(y, pred)
            rows.append(
                {
                    "Direction": direction,
                    "Subgroup type": subgroup_type,
                    "Subgroup": subgroup,
                    "N": int(len(sub)),
                    "Events": int(y.sum()),
                    "Event rate": fmt_decimal(m["event_rate"], 3),
                    "AUC": fmt_decimal(m["roc_auc"], 3),
                    "ECE": fmt_decimal(m["ece_10bin"], 3),
                    "Equal-width ECE": fmt_decimal(m["ece_10bin_equal_width"], 3),
                    "Calibration slope": fmt_decimal(m["calibration_slope"], 3),
                }
            )
    out = pd.DataFrame(rows)
    subgroup_order = {
        ("Age", "Age <65"): 1,
        ("Age", "Age >=65"): 2,
        ("CKD history", "No CKD"): 3,
        ("CKD history", "CKD"): 4,
        ("CVD history", "No CVD"): 5,
        ("CVD history", "CVD"): 6,
    }
    out["_direction_order"] = out["Direction"].map({d: i for i, d in enumerate(DIRECTION_ORDER)})
    out["_subgroup_order"] = out.apply(lambda r: subgroup_order.get((r["Subgroup type"], r["Subgroup"]), 99), axis=1)
    return out.sort_values(["_direction_order", "_subgroup_order"]).drop(columns=["_direction_order", "_subgroup_order"])


def write_table5(table5: pd.DataFrame) -> None:
    write_table(
        "Table_5_subgroup_transportability",
        table5,
        [
            "Primary model is unweighted logistic regression.",
            "Rows report raw transported logistic-regression performance within target-site subgroups.",
            "AUC is ROC AUC; ECE is 10-bin equal-frequency expected calibration error.",
            "Subgroup ECE is descriptive and may partly reflect subgroup event-rate differences.",
        ],
    )


def write_table6(table6: pd.DataFrame) -> None:
    write_table(
        "Table_6_decision_curve_selected_thresholds",
        table6,
        [
            "Primary model is unweighted logistic regression.",
            "Net benefit is reported at thresholds 0.20, 0.25, and 0.30.",
            "Internal HGB benchmark is the target-site histogram-gradient-boosting model.",
            "Thresholds are illustrative cutoffs for high-risk clinical review rather than validated treatment thresholds.",
        ],
    )


def copy_to_current_submission() -> None:
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    names = [
        "Table_3_cross_scenario_transport_performance.csv",
        "Table_3_cross_scenario_transport_performance_numeric.csv",
        "Table_3_cross_scenario_transport_bootstrap_long.csv",
        "Table_3_cross_scenario_transport_metadata.json",
        "README_Table_3_cross_scenario_transport_performance.md",
        "Table_4_recalibration_by_event_count.csv",
        "Table_4_recalibration_by_event_count.md",
        "Table_4_recalibration_by_event_count_numeric_long.csv",
        "Table_4_recalibration_by_event_count_panel.csv",
        "Table_4_recalibration_by_event_count_panel.md",
        "Table_5_subgroup_transportability.csv",
        "Table_5_subgroup_transportability.md",
        "Table_6_decision_curve_selected_thresholds.csv",
        "Table_6_decision_curve_selected_thresholds.md",
    ]
    for name in names:
        shutil.copy2(OUT_DIR / name, CURRENT_DIR / name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    formatted, numeric, long = table3_bootstrap()
    write_table3(formatted, numeric, long)

    table4, table4_numeric, table4_panel = build_table4()
    write_table4(table4, table4_numeric, table4_panel)

    table5 = build_table5()
    write_table5(table5)

    table6 = build_table6()
    write_table6(table6)

    metadata = {
        "primary_model": "unweighted logistic regression",
        "table3_rows": int(len(formatted)),
        "table4_rows": int(len(table4)),
        "table5_rows": int(len(table5)),
        "table6_rows": int(len(table6)),
        "directions": DIRECTION_ORDER,
        "source_files": {
            "predictions": str(PRED_PATH),
            "recalibration": str(RECAL_PATH),
            "decision_curve": str(DCA_PATH),
        },
        "class_weighted_backup": "Supplementary_Table_class_weighted_* files were copied before overwriting primary tables.",
    }
    (OUT_DIR / "primary_unweighted_tables_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.copy2(OUT_DIR / "primary_unweighted_tables_metadata.json", CURRENT_DIR / "primary_unweighted_tables_metadata.json")
    copy_to_current_submission()

    print(json.dumps(metadata, indent=2))
    print(formatted.to_string(index=False))
    print(table4.to_string(index=False))
    print(table5.to_string(index=False))
    print(table6.to_string(index=False))


if __name__ == "__main__":
    main()
