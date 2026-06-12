from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "figures_feasibility"

SMD_PATH = PROJECT_ROOT / "outputs" / "domain_shift" / "common_feature_smd.csv"
MISS_PATH = PROJECT_ROOT / "outputs" / "domain_shift" / "common_feature_missingness.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "transportability_metrics.csv"
CAL_PATH = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "transportability_calibration.csv"
RECAL_PATH = PROJECT_ROOT / "outputs" / "local_recalibration_simulation" / "local_recalibration_summary.csv"
SOURCE_ROC_PATH = PROJECT_ROOT / "outputs" / "domain_shift" / "source_classifier_roc_curve.csv"
SOURCE_ROC_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "domain_shift" / "source_classifier_roc_summary.csv"
BOOT_CI_PATH = PROJECT_ROOT / "outputs" / "nhanes_mimic_oneyear_mortality_transport" / "transportability_bootstrap_ci.csv"
EVENT_RECAL_PATH = PROJECT_ROOT / "outputs" / "local_recalibration_by_event_count" / "local_recalibration_by_event_summary.csv"
EVENT_RECAL_LONG_PATH = PROJECT_ROOT / "outputs" / "local_recalibration_by_event_count" / "local_recalibration_by_event_results_long.csv"
SUBGROUP_PATH = PROJECT_ROOT / "outputs" / "subgroup_transportability" / "subgroup_transportability_metrics.csv"
DCA_PATH = PROJECT_ROOT / "outputs" / "decision_curve" / "decision_curve_transportability.csv"


FEATURE_LABELS = {
    "age": "Age",
    "female": "Female",
    "bmi": "BMI",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "hypertension_history": "Hypertension",
    "ckd_history": "CKD",
    "cvd_history": "CVD",
    "hba1c": "HbA1c",
    "glucose": "Glucose",
    "creatinine": "Creatinine",
    "egfr": "eGFR",
    "uacr": "UACR",
    "total_cholesterol": "Total cholesterol",
    "hdl_cholesterol": "HDL cholesterol",
    "albumin": "Albumin",
    "hemoglobin": "Hemoglobin",
    "wbc": "WBC",
}


SCENARIO_LABELS = {
    "NHANES_to_MIMIC_1y_base_logistic_regression": "Base logistic",
    "NHANES_to_MIMIC_1y_base_labs_logistic_regression": "Base + labs logistic",
    "NHANES_to_MIMIC_1y_base_hist_gradient_boosting": "Base HGB",
    "NHANES_to_MIMIC_1y_base_labs_hist_gradient_boosting": "Base + labs HGB",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_smd() -> None:
    smd = pd.read_csv(SMD_PATH)
    smd["abs_smd"] = smd["smd_mimic_minus_nhanes"].abs()
    smd = smd.sort_values("abs_smd", ascending=True)
    smd["label"] = smd["feature"].map(FEATURE_LABELS).fillna(smd["feature"])
    colors = np.where(smd["smd_mimic_minus_nhanes"] >= 0, "#2f6f9f", "#b85c38")

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.barh(smd["label"], smd["smd_mimic_minus_nhanes"], color=colors)
    ax.axvline(0, color="#2b2b2b", linewidth=0.8)
    ax.axvline(0.1, color="#8a8a8a", linewidth=0.8, linestyle="--")
    ax.axvline(-0.1, color="#8a8a8a", linewidth=0.8, linestyle="--")
    ax.axvline(0.2, color="#b0b0b0", linewidth=0.8, linestyle=":")
    ax.axvline(-0.2, color="#b0b0b0", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Standardized mean difference, MIMIC-IV minus NHANES")
    ax.set_ylabel("")
    ax.set_title("Common-feature distribution shift")
    save_fig(fig, "figure_1_common_feature_smd")


def plot_missingness() -> None:
    miss = pd.read_csv(MISS_PATH)
    keep = list(FEATURE_LABELS.keys())
    miss = miss[miss["feature"].isin(keep)].copy()
    miss["feature_label"] = miss["feature"].map(FEATURE_LABELS)
    pivot = miss.pivot_table(index="feature_label", columns="source", values="missing_pct", aggfunc="first")
    pivot = pivot.loc[[FEATURE_LABELS[f] for f in keep if FEATURE_LABELS[f] in pivot.index]]

    fig, ax = plt.subplots(figsize=(5.4, 6.6))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Missing proportion"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Common-variable missingness")
    save_fig(fig, "figure_2_common_feature_missingness")


def transport_label(row: pd.Series) -> str:
    train = row["train_source"]
    target = row["test_target"]
    feature = "Base + labs" if row["feature_set"] == "base_labs" else "Base"
    model = "HGB" if row["model"] == "hist_gradient_boosting" else "Logistic"
    return f"{train} to {target}\n{feature}, {model}"


def plot_transport_metrics() -> None:
    metrics = pd.read_csv(METRICS_PATH)
    metrics = metrics[metrics["test_type"].isin(["internal_test", "transport_test"])].copy()
    metrics["label"] = metrics.apply(transport_label, axis=1)
    metrics["direction"] = metrics["train_source"] + " to " + metrics["test_target"]
    order = [
        "NHANES to NHANES",
        "NHANES to MIMIC-IV",
        "MIMIC-IV to MIMIC-IV",
        "MIMIC-IV to NHANES",
    ]
    metrics["direction"] = pd.Categorical(metrics["direction"], categories=order, ordered=True)
    metrics = metrics.sort_values(["direction", "feature_set", "model"])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    color_map = {
        "internal_test": "#3a7d44",
        "transport_test": "#c05a2b",
    }
    y = np.arange(len(metrics))
    for ax, metric, label in [
        (axes[0], "roc_auc", "ROC AUC"),
        (axes[1], "brier_score", "Brier score"),
    ]:
        ax.barh(y, metrics[metric], color=metrics["test_type"].map(color_map))
        ax.set_xlabel(label)
        ax.set_yticks(y)
        ax.set_yticklabels(metrics["label"] if ax is axes[0] else [])
        ax.invert_yaxis()
        if metric == "roc_auc":
            ax.set_xlim(0.45, 0.82)
            ax.axvline(0.5, color="#777777", linewidth=0.8, linestyle="--")
        else:
            ax.set_xlim(0, max(0.45, metrics[metric].max() * 1.08))
        ax.set_title(label)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map["internal_test"], label="Internal test"),
        plt.Rectangle((0, 0), 1, 1, color=color_map["transport_test"], label="Transport test"),
    ]
    axes[1].legend(handles=handles, loc="lower right", frameon=False)
    fig.suptitle("One-year mortality model performance across settings", y=1.02, fontsize=12)
    save_fig(fig, "figure_3_oneyear_transport_metrics")


def plot_calibration() -> None:
    cal = pd.read_csv(CAL_PATH)
    selected = [
        "NHANES_to_MIMIC-IV_base_logistic_regression",
        "MIMIC-IV_to_MIMIC-IV_base_hist_gradient_boosting",
        "MIMIC-IV_to_NHANES_base_hist_gradient_boosting",
        "NHANES_to_NHANES_base_logistic_regression",
    ]
    labels = {
        "NHANES_to_MIMIC-IV_base_logistic_regression": "NHANES to MIMIC, base logistic",
        "MIMIC-IV_to_MIMIC-IV_base_hist_gradient_boosting": "MIMIC internal, base HGB",
        "MIMIC-IV_to_NHANES_base_hist_gradient_boosting": "MIMIC to NHANES, base HGB",
        "NHANES_to_NHANES_base_logistic_regression": "NHANES internal, base logistic",
    }
    cal = cal[cal["analysis"].isin(selected)].copy()
    cal["label"] = cal["analysis"].map(labels)

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for label, group in cal.groupby("label", sort=False):
        ax.plot(
            group["mean_predicted_probability"],
            group["observed_probability"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=label,
        )
    ax.plot([0, 1], [0, 1], color="#555555", linewidth=0.9, linestyle="--", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration drift after transport")
    ax.legend(frameon=False, loc="upper left")
    save_fig(fig, "figure_4_oneyear_calibration_drift")


def plot_recalibration_curve() -> None:
    recal = pd.read_csv(RECAL_PATH)
    focus = "NHANES_to_MIMIC_1y_base_logistic_regression"
    recal = recal[recal["prediction_set"].eq(focus)].copy()
    recal["scenario_label"] = recal["prediction_set"].map(SCENARIO_LABELS)
    recal["fraction_pct"] = recal["fraction"] * 100
    recal["method_label"] = recal["method"].map(
        {
            "raw": "Raw",
            "intercept_only": "Intercept only",
            "platt": "Platt",
            "isotonic": "Isotonic",
        }
    )
    keep_methods = ["Raw", "Intercept only", "Platt", "Isotonic"]
    recal = recal[recal["method_label"].isin(keep_methods)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    palette = {
        "Raw": "#5b5b5b",
        "Intercept only": "#7a5195",
        "Platt": "#1f78b4",
        "Isotonic": "#b15928",
    }
    for ax, metric, ylabel in [
        (axes[0], "brier_score_mean", "Brier score"),
        (axes[1], "ece_10bin_mean", "Expected calibration error"),
    ]:
        for method, sub in recal.groupby("method_label", sort=False):
            if method == "Raw":
                raw_val = sub.loc[sub["fraction_pct"].eq(0), metric]
                if raw_val.empty:
                    continue
                ax.hlines(
                    raw_val.iloc[0],
                    xmin=0,
                    xmax=20,
                    color=palette[method],
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.8,
                    label=method,
                )
                continue
            ax.plot(
                sub["fraction_pct"],
                sub[metric],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=palette[method],
                alpha=0.95,
                label=method,
            )
        ax.set_xlabel("Local calibration sample (%)")
        ax.set_title(ylabel)
    axes[0].set_ylabel("Brier score")
    axes[1].set_ylabel("Expected calibration error")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Local recalibration repairs NHANES-to-MIMIC calibration", y=1.02, fontsize=12)
    save_fig(fig, "figure_5_local_recalibration_curves")


def plot_source_classifier_roc() -> None:
    if not SOURCE_ROC_PATH.exists() or not SOURCE_ROC_SUMMARY_PATH.exists():
        return
    roc = pd.read_csv(SOURCE_ROC_PATH)
    summary = pd.read_csv(SOURCE_ROC_SUMMARY_PATH)
    feature_labels = {
        "basic_features": "Basic features",
        "lab_enhanced_features": "Lab-enhanced features",
    }
    model_labels = {
        "logistic_regression": "Logistic",
        "random_forest": "Random forest",
    }
    colors = {
        ("basic_features", "logistic_regression"): "#4c78a8",
        ("basic_features", "random_forest"): "#f58518",
        ("lab_enhanced_features", "logistic_regression"): "#54a24b",
        ("lab_enhanced_features", "random_forest"): "#b279a2",
    }

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    for (feature_set, model), group in roc.groupby(["feature_set", "model"], sort=False):
        auc = summary.loc[
            summary["feature_set"].eq(feature_set) & summary["model"].eq(model),
            "roc_auc",
        ].iloc[0]
        label = f"{feature_labels.get(feature_set, feature_set)}, {model_labels.get(model, model)} (AUC {auc:.3f})"
        ax.plot(group["fpr"], group["tpr"], linewidth=1.8, color=colors.get((feature_set, model)), label=label)
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Source classifier ROC: NHANES vs MIMIC-IV")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "figure_6_source_classifier_roc")


def plot_transport_bootstrap_ci() -> None:
    if not BOOT_CI_PATH.exists():
        return
    ci = pd.read_csv(BOOT_CI_PATH)
    ci = ci[
        ci["metric"].isin(["roc_auc", "brier_score"])
        & ci["feature_set"].eq("base")
        & ci["model"].isin(["logistic_regression", "hist_gradient_boosting"])
    ].copy()
    ci["analysis"] = ci["train_source"] + " to " + ci["test_target"] + ", " + ci["model"].map(
        {"logistic_regression": "Logistic", "hist_gradient_boosting": "HGB"}
    )
    order = [
        "NHANES to NHANES, Logistic",
        "NHANES to MIMIC-IV, Logistic",
        "MIMIC-IV to MIMIC-IV, Logistic",
        "MIMIC-IV to NHANES, Logistic",
        "NHANES to NHANES, HGB",
        "NHANES to MIMIC-IV, HGB",
        "MIMIC-IV to MIMIC-IV, HGB",
        "MIMIC-IV to NHANES, HGB",
    ]
    ci["analysis"] = pd.Categorical(ci["analysis"], categories=order, ordered=True)
    ci = ci.sort_values("analysis")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    metric_labels = {"roc_auc": "ROC AUC", "brier_score": "Brier score"}
    for ax, metric in zip(axes, ["roc_auc", "brier_score"]):
        sub = ci[ci["metric"].eq(metric)].copy()
        y = np.arange(len(sub))
        xerr = np.vstack([sub["point"] - sub["ci_lower"], sub["ci_upper"] - sub["point"]])
        ax.errorbar(sub["point"], y, xerr=xerr, fmt="o", color="#2f6f9f", ecolor="#6f9fc1", capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["analysis"] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(metric_labels[metric])
        ax.set_title(metric_labels[metric])
        if metric == "roc_auc":
            ax.axvline(0.5, color="#777777", linestyle="--", linewidth=0.8)
            ax.set_xlim(0.35, 0.9)
        else:
            ax.set_xlim(0, 0.32)
    fig.suptitle("Bootstrap uncertainty for one-year transport metrics", y=1.02, fontsize=12)
    save_fig(fig, "figure_7_transport_bootstrap_ci")


def plot_recalibration_by_event_count() -> None:
    if not EVENT_RECAL_LONG_PATH.exists():
        return
    recal_long = pd.read_csv(EVENT_RECAL_LONG_PATH)
    focus = "NHANES_to_MIMIC_1y_base_logistic_regression"
    recal_long = recal_long[recal_long["prediction_set"].eq(focus)].copy()
    metric_cols = ["brier_score", "ece_10bin"]
    rows = []
    for key, group in recal_long.groupby(["prediction_set", "method", "event_target"], sort=False):
        row = dict(zip(["prediction_set", "method", "event_target"], key))
        for metric in metric_cols:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_ci_lower"] = values.quantile(0.025)
            row[f"{metric}_ci_upper"] = values.quantile(0.975)
        rows.append(row)
    recal = pd.DataFrame(rows)
    recal["method_label"] = recal["method"].map(
        {
            "raw": "Raw",
            "intercept_only": "Intercept only",
            "platt": "Platt",
            "isotonic": "Isotonic",
        }
    )
    recal = recal[recal["method_label"].isin(["Raw", "Intercept only", "Platt", "Isotonic"])].copy()
    palette = {
        "Raw": "#5b5b5b",
        "Intercept only": "#7a5195",
        "Platt": "#1f78b4",
        "Isotonic": "#b15928",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "brier_score_mean", "Brier score"),
        (axes[1], "ece_10bin_mean", "Expected calibration error"),
    ]:
        lower = metric.replace("_mean", "_ci_lower")
        upper = metric.replace("_mean", "_ci_upper")
        for method, group in recal.groupby("method_label", sort=False):
            if method == "Raw":
                raw_val = group.loc[group["event_target"].eq(0), metric]
                if not raw_val.empty:
                    ax.hlines(raw_val.iloc[0], xmin=25, xmax=1000, color=palette[method], linestyle="--", linewidth=1.2, label=method)
                continue
            sub = group[group["event_target"] > 0].sort_values("event_target").copy()
            x = sub["event_target"].to_numpy(dtype=float)
            y = sub[metric].to_numpy(dtype=float)
            y_lower = sub[lower].to_numpy(dtype=float)
            y_upper = sub[upper].to_numpy(dtype=float)
            ax.fill_between(x, y_lower, y_upper, color=palette[method], alpha=0.13, linewidth=0)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=palette[method],
                label=method,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Local calibration events")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks([25, 50, 100, 200, 500, 1000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Local recalibration by number of target-site events", y=1.02, fontsize=12)
    save_fig(fig, "figure_8_recalibration_by_event_count")


def plot_subgroup_transportability() -> None:
    if not SUBGROUP_PATH.exists():
        return
    sub = pd.read_csv(SUBGROUP_PATH)
    focus = sub[
        sub["train_source"].eq("NHANES")
        & sub["test_target"].eq("MIMIC-IV")
        & sub["feature_set"].eq("base")
        & sub["model"].eq("logistic_regression")
        & ~sub["subgroup_type"].eq("Overall")
    ].copy()
    focus["label"] = focus["subgroup_type"] + ": " + focus["subgroup"]
    focus = focus.sort_values(["subgroup_type", "subgroup"])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.6), sharey=True)
    y = np.arange(len(focus))
    for ax, metric, title, color in [
        (axes[0], "ece_10bin", "Expected calibration error", "#2f6f9f"),
        (axes[1], "brier_score", "Brier score", "#b85c38"),
    ]:
        ax.barh(y, focus[metric], color=color, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(focus["label"] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.set_title(title)
    fig.suptitle("Subgroup calibration drift: NHANES to MIMIC-IV base logistic", y=1.02, fontsize=12)
    save_fig(fig, "figure_9_subgroup_transportability")


def plot_decision_curve() -> None:
    if not DCA_PATH.exists():
        return
    dca = pd.read_csv(DCA_PATH)
    keep = [
        "Treat none",
        "Treat all",
        "NHANES raw logistic",
        "NHANES Platt 100 events",
        "MIMIC internal HGB",
    ]
    dca = dca[dca["model"].isin(keep)].copy()
    colors = {
        "Treat none": "#666666",
        "Treat all": "#999999",
        "NHANES raw logistic": "#c05a2b",
        "NHANES Platt 100 events": "#1f78b4",
        "MIMIC internal HGB": "#3a7d44",
    }
    styles = {
        "Treat none": ":",
        "Treat all": "--",
        "NHANES raw logistic": "-",
        "NHANES Platt 100 events": "-",
        "MIMIC internal HGB": "-",
    }

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for model, group in dca.groupby("model", sort=False):
        ax.plot(group["threshold"], group["net_benefit"], label=model, color=colors[model], linestyle=styles[model], linewidth=1.8)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xlim(0.01, 0.50)
    ax.set_ylim(-0.08, 0.20)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve in MIMIC-IV target setting")
    ax.legend(frameon=False, loc="upper right")
    save_fig(fig, "figure_10_decision_curve_transportability")


def write_readme() -> None:
    readme = """# Feasibility Figures

Generated by:

```powershell
cd D:\\多库联合_医工结合
D:\\糖尿病医工结合\\.venv\\Scripts\\python.exe scripts\\make_feasibility_figures.py
```

These are first-pass feasibility figures for presentation and manuscript planning. They are not final journal-polished figures.

## Figure Files

- `figure_1_common_feature_smd`: NHANES vs MIMIC-IV standardized mean differences for common baseline features.
- `figure_2_common_feature_missingness`: missingness of common clinical and laboratory variables across NHANES and MIMIC-IV.
- `figure_3_oneyear_transport_metrics`: one-year mortality internal vs transport performance by ROC AUC and Brier score.
- `figure_4_oneyear_calibration_drift`: calibration curves showing cross-setting calibration drift.
- `figure_5_local_recalibration_curves`: local calibration sample-size curves for the NHANES-to-MIMIC base logistic model.
- `figure_6_source_classifier_roc`: cross-validated ROC curves for source classifiers distinguishing NHANES from MIMIC-IV.
- `figure_7_transport_bootstrap_ci`: bootstrap 95% confidence intervals for selected one-year transport metrics.
- `figure_8_recalibration_by_event_count`: recalibration curves by local target-site event count with empirical 2.5%-97.5% uncertainty bands.
- `figure_9_subgroup_transportability`: subgroup ECE and Brier score for NHANES-to-MIMIC base logistic transport.
- `figure_10_decision_curve_transportability`: decision curves in the MIMIC-IV target setting.

## Interpretation

The figures support the current study frame:

- NHANES and MIMIC-IV diabetes cohorts differ substantially in covariates and measurement process.
- One-year mortality endpoint harmonization does not remove transportability problems.
- Transported models often retain moderate discrimination but have severe calibration drift.
- Small local calibration samples can substantially repair calibration, giving a direct template for the future Xi'an hospital cohort.
- Event-count recalibration curves are more deployable than fraction-only curves because they answer how many local outcomes a hospital may need.
- Subgroup plots show whether transport failures concentrate in clinically important groups.
- Decision curves show that recalibration can change clinical net benefit at moderate and high risk thresholds.
"""
    (OUT_DIR / "README_figures_feasibility.md").write_text(readme, encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_smd()
    plot_missingness()
    plot_transport_metrics()
    plot_calibration()
    plot_recalibration_curve()
    plot_source_classifier_roc()
    plot_transport_bootstrap_ci()
    plot_recalibration_by_event_count()
    plot_subgroup_transportability()
    plot_decision_curve()
    write_readme()
    print(f"Wrote feasibility figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
