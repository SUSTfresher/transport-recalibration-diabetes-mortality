from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "current_submission" / "tables"
FIG_DIR = ROOT / "outputs" / "current_submission" / "figures" / "figure2"
SOURCE_DIR = ROOT / "outputs" / "current_submission" / "source_data"


DIRECTION_ORDER = [
    "NHANES -> MIMIC-IV",
    "MIMIC-IV ICU -> eICU",
    "eICU -> MIMIC-IV ICU",
]

DIRECTION_LABELS = {
    "NHANES -> MIMIC-IV": "NHANES\n-> MIMIC-IV",
    "MIMIC-IV ICU -> eICU": "MIMIC-IV ICU\n-> eICU",
    "eICU -> MIMIC-IV ICU": "eICU\n-> MIMIC-IV ICU",
}

FAILURE_LABELS = {
    "NHANES -> MIMIC-IV": "Severe slope\ndistortion",
    "MIMIC-IV ICU -> eICU": "Moderate slope\ndistortion",
    "eICU -> MIMIC-IV ICU": "Near-calibrated\ntransport",
}

COLORS = {
    "NHANES -> MIMIC-IV": "#4E79A7",
    "MIMIC-IV ICU -> eICU": "#59A14F",
    "eICU -> MIMIC-IV ICU": "#B07AA1",
}

METRIC_COLORS = {
    "ECE": "#6B6F76",
    "Brier score": "#A7BBC7",
}

SELECTED_ACTIONS = {
    "NHANES -> MIMIC-IV": {
        "method": "platt",
        "method_label": "Platt",
        "event_target": 100,
    },
    "MIMIC-IV ICU -> eICU": {
        "method": "platt",
        "method_label": "Platt",
        "event_target": 100,
    },
    "eICU -> MIMIC-IV ICU": {
        "method": "raw",
        "method_label": "Retain raw",
        "event_target": 0,
    },
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_raw_metrics() -> pd.DataFrame:
    path = TABLE_DIR / "Table_3_cross_scenario_transport_performance_numeric.csv"
    df = pd.read_csv(path)
    df = df[df["direction"].isin(DIRECTION_ORDER)].copy()
    df["direction"] = pd.Categorical(df["direction"], DIRECTION_ORDER, ordered=True)
    df = df.sort_values("direction").reset_index(drop=True)
    return df


def load_selected_updates(raw: pd.DataFrame) -> dict[str, dict[str, object]]:
    path = TABLE_DIR / "Table_4_recalibration_by_event_count_numeric_long.csv"
    recal = pd.read_csv(path)
    updates: dict[str, dict[str, object]] = {}

    for _, raw_row in raw.iterrows():
        direction = str(raw_row["direction"])
        action = SELECTED_ACTIONS[direction]
        event_target = int(action["event_target"])
        method = str(action["method"])

        if method == "raw":
            updates[direction] = {
                "method": str(action["method_label"]),
                "event_target": event_target,
                "slope": float(raw_row["calibration_slope_point"]),
                "ece": float(raw_row["ece_10bin_point"]),
                "ece_ci_lower": float(raw_row["ece_10bin_ci_lower"]),
                "ece_ci_upper": float(raw_row["ece_10bin_ci_upper"]),
            }
            continue

        row = recal[
            (recal["Direction"] == direction)
            & (recal["method"] == method)
            & (recal["event_target"] == event_target)
        ].iloc[0]
        updates[direction] = {
            "method": f"{action['method_label']} {event_target} events",
            "event_target": event_target,
            "slope": float(row["calibration_slope_mean"]),
            "ece": float(row["ece_mean"]),
            "ece_ci_lower": float(row["ece_ci_lower"]),
            "ece_ci_upper": float(row["ece_ci_upper"]),
        }

    return updates


def build_source_data(raw: pd.DataFrame, selected_updates: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in raw.iterrows():
        direction = str(row["direction"])
        base = {
            "direction": direction,
            "scenario": row["scenario"],
            "endpoint": row["endpoint"],
            "feature_set": row["feature_set"],
            "model": row["model"],
            "n": int(row["n"]),
            "events": int(row["events"]),
        }

        rows.append(
            {
                **base,
                "panel": "2a",
                "metric": "Transport AUC",
                "method": "Raw transport",
                "event_count": 0,
                "value": row["roc_auc_point"],
                "ci_lower": row["roc_auc_ci_lower"],
                "ci_upper": row["roc_auc_ci_upper"],
                "notes": "Point and 95% bootstrap CI from Table 3.",
            }
        )

        for metric, value_col, lo_col, hi_col in [
            ("ECE", "ece_10bin_point", "ece_10bin_ci_lower", "ece_10bin_ci_upper"),
            (
                "Brier score",
                "brier_score_point",
                "brier_score_ci_lower",
                "brier_score_ci_upper",
            ),
        ]:
            rows.append(
                {
                    **base,
                    "panel": "2b",
                    "metric": metric,
                    "method": "Raw transport",
                    "event_count": 0,
                    "value": row[value_col],
                    "ci_lower": row[lo_col],
                    "ci_upper": row[hi_col],
                    "notes": "Point and 95% bootstrap CI from Table 3.",
                }
            )

        for metric, value_col, lo_col, hi_col in [
            (
                "Calibration slope",
                "calibration_slope_point",
                "calibration_slope_ci_lower",
                "calibration_slope_ci_upper",
            ),
            (
                "Calibration intercept",
                "calibration_intercept_point",
                "calibration_intercept_ci_lower",
                "calibration_intercept_ci_upper",
            ),
        ]:
            rows.append(
                {
                    **base,
                    "panel": "2c",
                    "metric": metric,
                    "method": "Raw transport",
                    "event_count": 0,
                    "value": row[value_col],
                    "ci_lower": row[lo_col],
                    "ci_upper": row[hi_col],
                    "notes": "Calibration-state map coordinate from raw transported model.",
                }
            )

        update = selected_updates[direction]
        rows.append(
            {
                **base,
                "panel": "2d",
                "metric": "Calibration slope",
                "method": "Raw transport",
                "event_count": 0,
                "value": row["calibration_slope_point"],
                "ci_lower": row["calibration_slope_ci_lower"],
                "ci_upper": row["calibration_slope_ci_upper"],
                "notes": "Raw transported slope before local recalibration.",
            }
        )
        rows.append(
            {
                **base,
                "panel": "2d",
                "metric": "Calibration slope",
                "method": update["method"],
                "event_count": update["event_target"],
                "value": update["slope"],
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "notes": "Selected diagnostic action from current Table 4; near-calibrated transport retains raw slope.",
            }
        )
        rows.append(
            {
                **base,
                "panel": "2d",
                "metric": "ECE",
                "method": update["method"],
                "event_count": update["event_target"],
                "value": update["ece"],
                "ci_lower": update["ece_ci_lower"],
                "ci_upper": update["ece_ci_upper"],
                "notes": "Empirical interval across 200 local calibration samples when updated; raw bootstrap CI when no update selected.",
            }
        )

    source = pd.DataFrame(rows)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    source.to_csv(SOURCE_DIR / "Figure_2_source_data.csv", index=False)
    return source


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def draw_panel_a(ax: plt.Axes, raw: pd.DataFrame) -> None:
    y = np.arange(len(raw))[::-1]
    for i, (_, row) in enumerate(raw.iterrows()):
        yi = y[i]
        direction = str(row["direction"])
        auc = row["roc_auc_point"]
        lo = row["roc_auc_ci_lower"]
        hi = row["roc_auc_ci_upper"]
        ax.plot([lo, hi], [yi, yi], color=COLORS[direction], lw=1.8, solid_capstyle="round")
        ax.scatter(auc, yi, s=34, color=COLORS[direction], edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(hi + 0.006, yi, f"{auc:.3f}", va="center", ha="left", fontsize=6.4)

    ax.set_yticks(y, [DIRECTION_LABELS[str(d)] for d in raw["direction"]])
    ax.set_xlim(0.64, 0.78)
    ax.set_xlabel("Transport AUC")
    ax.set_title("Discrimination was partially retained", loc="left", pad=5)
    ax.grid(axis="x", color="#E7E7E7", lw=0.5)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "a")


def draw_panel_b(ax: plt.Axes, raw: pd.DataFrame) -> None:
    x = np.arange(len(raw))
    width = 0.32
    metrics = [
        ("ECE", "ece_10bin_point", -width / 2),
        ("Brier score", "brier_score_point", width / 2),
    ]
    for metric, col, offset in metrics:
        values = raw[col].to_numpy(dtype=float)
        ax.bar(
            x + offset,
            values,
            width=width,
            color=METRIC_COLORS[metric],
            edgecolor="white",
            linewidth=0.5,
            label=metric,
        )
        for xi, val in zip(x + offset, values):
            ax.text(xi, val + 0.008, f"{val:.3f}", ha="center", va="bottom", fontsize=5.8)

    ax.set_xticks(x, [DIRECTION_LABELS[str(d)] for d in raw["direction"]])
    ax.set_ylim(0, 0.20)
    ax.set_ylabel("Metric value")
    ax.set_title("Probability-scale error varied by direction", loc="left", pad=5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, handlelength=1.0, columnspacing=0.8)
    ax.grid(axis="y", color="#E7E7E7", lw=0.5)
    panel_label(ax, "b")


def draw_panel_c(ax: plt.Axes, raw: pd.DataFrame) -> None:
    ax.axvline(1.0, color="#9AA0A6", ls=(0, (3, 2)), lw=0.9)
    ax.axhline(0.0, color="#9AA0A6", ls=(0, (3, 2)), lw=0.9)
    ax.scatter([1.0], [0.0], s=28, marker="*", color="#3C4043", zorder=2)
    ax.text(1.012, 0.02, "Ideal", ha="left", va="bottom", fontsize=6.2, color="#3C4043")

    label_offsets = {
        "NHANES -> MIMIC-IV": (0.05, 0.05, "left"),
        "MIMIC-IV ICU -> eICU": (0.05, 0.04, "left"),
        "eICU -> MIMIC-IV ICU": (-0.05, -0.11, "right"),
    }
    for _, row in raw.iterrows():
        direction = str(row["direction"])
        slope = row["calibration_slope_point"]
        intercept = row["calibration_intercept_point"]
        ax.scatter(
            slope,
            intercept,
            s=54,
            color=COLORS[direction],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        dx, dy, ha = label_offsets[direction]
        ax.text(
            slope + dx,
            intercept + dy,
            f"{FAILURE_LABELS[direction]}\n{slope:.2f}, {intercept:.2f}",
            ha=ha,
            va="center",
            fontsize=6.2,
            color="#202124",
        )

    ax.set_xlim(0.42, 1.32)
    ax.set_ylim(-1.48, 0.72)
    ax.set_xlabel("Calibration slope")
    ax.set_ylabel("Calibration intercept")
    ax.set_title("Calibration state separated failure modes", loc="left", pad=5)
    ax.grid(color="#ECECEC", lw=0.5)
    panel_label(ax, "c")


def draw_panel_d(ax: plt.Axes, raw: pd.DataFrame, selected_updates: dict[str, dict[str, object]]) -> None:
    x = np.arange(len(raw))
    raw_slope = raw["calibration_slope_point"].to_numpy(dtype=float)
    updated_slope = np.array([selected_updates[str(d)]["slope"] for d in raw["direction"]], dtype=float)
    updated_ece = np.array([selected_updates[str(d)]["ece"] for d in raw["direction"]], dtype=float)
    raw_ece = raw["ece_10bin_point"].to_numpy(dtype=float)

    ax.axhline(1.0, color="#9AA0A6", ls=(0, (3, 2)), lw=0.9)
    for i, direction in enumerate(raw["direction"]):
        direction = str(direction)
        raw_x = i - 0.045
        update_x = i + 0.045
        arrow = FancyArrowPatch(
            (raw_x, raw_slope[i]),
            (update_x, updated_slope[i]),
            arrowstyle="-|>",
            mutation_scale=7.5,
            lw=1.5,
            color="#C4C7C5",
            zorder=1,
            shrinkA=2.0,
            shrinkB=2.0,
        )
        ax.add_patch(arrow)
        ax.scatter(raw_x, raw_slope[i], s=34, color="#C4C7C5", edgecolor="white", linewidth=0.7, zorder=3)
        ax.scatter(update_x, updated_slope[i], s=46, color=COLORS[direction], edgecolor="white", linewidth=0.8, zorder=4)
        update = selected_updates[direction]
        if int(update["event_target"]) == 0:
            action_label = "Retain\nraw"
            ece_label = f"ECE {raw_ece[i]:.3f}\nretained"
            action_x = i - 0.08
        else:
            action_label = f"{str(update['method']).replace(' 100 events', '')}\n100 events"
            ece_label = f"ECE {raw_ece[i]:.3f}\n-> {updated_ece[i]:.3f}"
            action_x = i
        ax.text(action_x, 1.315, action_label, ha="center", va="top", fontsize=5.8)
        ax.text(i, 0.45, ece_label, ha="center", va="bottom", fontsize=5.7, color="#3C4043")

    ax.set_xticks(x, [DIRECTION_LABELS[str(d)] for d in raw["direction"]])
    ax.set_xlim(-0.20, len(raw) - 0.80)
    ax.set_ylim(0.42, 1.35)
    ax.set_ylabel("Calibration slope")
    ax.set_title("Diagnostic action matched the calibration state", loc="left", pad=5)
    ax.grid(axis="y", color="#E7E7E7", lw=0.5)
    panel_label(ax, "d")


def make_figure(raw: pd.DataFrame, selected_updates: dict[str, dict[str, object]]) -> plt.Figure:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.2, 5.35), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.985,
        top=0.94,
        bottom=0.12,
        wspace=0.36,
        hspace=0.55,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a, raw)
    draw_panel_b(ax_b, raw)
    draw_panel_c(ax_c, raw)
    draw_panel_d(ax_d, raw, selected_updates)
    return fig


def save_outputs(fig: plt.Figure) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / "Figure_2_calibration_state_map"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> None:
    raw = load_raw_metrics()
    selected_updates = load_selected_updates(raw)
    build_source_data(raw, selected_updates)
    fig = make_figure(raw, selected_updates)
    save_outputs(fig)
    plt.close(fig)
    print(f"Wrote figure outputs to {FIG_DIR}")
    print(f"Wrote source data to {SOURCE_DIR / 'Figure_2_source_data.csv'}")


if __name__ == "__main__":
    main()
