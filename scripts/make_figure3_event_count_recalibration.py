from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "current_submission" / "tables"
FIG_DIR = ROOT / "outputs" / "current_submission" / "figures" / "figure3"
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

SHORT_TITLES = {
    "NHANES -> MIMIC-IV": "NHANES -> MIMIC-IV",
    "MIMIC-IV ICU -> eICU": "MIMIC-IV ICU -> eICU",
    "eICU -> MIMIC-IV ICU": "eICU -> MIMIC-IV ICU",
}

X_POS = {0: 0, 25: 1, 50: 2, 100: 3, 200: 4}
X_TICKS = [0, 1, 2, 3, 4]
X_LABELS = ["Raw", "25", "50", "100", "200"]

METHOD_ORDER = ["intercept_only", "platt", "isotonic"]
METHOD_LABELS = {
    "intercept_only": "Intercept-only",
    "platt": "Platt",
    "isotonic": "Isotonic",
}
METHOD_COLORS = {
    "intercept_only": "#4E79A7",
    "platt": "#59A14F",
    "isotonic": "#B07AA1",
}

DIRECTION_COLORS = {
    "NHANES -> MIMIC-IV": "#4E79A7",
    "MIMIC-IV ICU -> eICU": "#59A14F",
    "eICU -> MIMIC-IV ICU": "#B07AA1",
}

DCA_COLORS = {
    "Raw transport": "#6B6F76",
    "Intercept-only 100 events": "#4E79A7",
    "Platt 100 events": "#59A14F",
    "Internal HGB": "#E15759",
}

SELECTED_SLOPE_METHODS = {
    "NHANES -> MIMIC-IV": {
        "method": "platt",
        "selected_method": "Platt",
        "line_label": "NHANES\nPlatt",
    },
    "MIMIC-IV ICU -> eICU": {
        "method": "platt",
        "selected_method": "Platt",
        "line_label": "MIMIC/eICU\nPlatt",
    },
    "eICU -> MIMIC-IV ICU": {
        "method": "intercept_only",
        "selected_method": "Intercept-only",
        "line_label": "eICU/MIMIC\nminimal update",
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
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.12) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def load_ece_data() -> pd.DataFrame:
    path = TABLE_DIR / "Table_4_recalibration_by_event_count_numeric_long.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"Direction": "direction"})
    df = df[
        [
            "direction",
            "method",
            "event_target",
            "ece_mean",
            "ece_ci_lower",
            "ece_ci_upper",
            "calibration_slope_mean",
            "calibration_slope_ci_lower",
            "calibration_slope_ci_upper",
        ]
    ].copy()
    df = df[df["direction"].isin(DIRECTION_ORDER)]
    df["direction"] = pd.Categorical(df["direction"], DIRECTION_ORDER, ordered=True)
    df["event_target"] = df["event_target"].astype(int)
    return df.sort_values(["direction", "method", "event_target"]).reset_index(drop=True)


def build_ece_plot_data(ece: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for direction in DIRECTION_ORDER:
        raw = ece[(ece["direction"] == direction) & (ece["method"] == "raw")].iloc[0]
        for method in METHOD_ORDER:
            rows.append(
                {
                    "direction": direction,
                    "method": method,
                    "event_target": 0,
                    "ece_mean": raw["ece_mean"],
                    "ece_ci_lower": raw["ece_ci_lower"],
                    "ece_ci_upper": raw["ece_ci_upper"],
                    "calibration_slope_mean": raw["calibration_slope_mean"],
                    "calibration_slope_ci_lower": raw["calibration_slope_ci_lower"],
                    "calibration_slope_ci_upper": raw["calibration_slope_ci_upper"],
                    "source_method": "raw",
                }
            )
            subset = ece[(ece["direction"] == direction) & (ece["method"] == method)]
            for _, row in subset.iterrows():
                rows.append(
                    {
                        "direction": direction,
                        "method": method,
                        "event_target": int(row["event_target"]),
                        "ece_mean": row["ece_mean"],
                        "ece_ci_lower": row["ece_ci_lower"],
                        "ece_ci_upper": row["ece_ci_upper"],
                        "calibration_slope_mean": row["calibration_slope_mean"],
                        "calibration_slope_ci_lower": row["calibration_slope_ci_lower"],
                        "calibration_slope_ci_upper": row["calibration_slope_ci_upper"],
                        "source_method": method,
                    }
                )

    plot = pd.DataFrame(rows)
    plot["x_pos"] = plot["event_target"].map(X_POS)
    return plot


def load_slope_data() -> pd.DataFrame:
    recal = load_ece_data()
    rows: list[pd.DataFrame] = []
    for direction, action in SELECTED_SLOPE_METHODS.items():
        raw = recal[(recal["direction"] == direction) & (recal["method"] == "raw")].copy()
        selected = recal[(recal["direction"] == direction) & (recal["method"] == action["method"])].copy()
        d = pd.concat([raw, selected], ignore_index=True)
        d["selected_method"] = action["selected_method"]
        d["line_label"] = action["line_label"]
        rows.append(d)

    slope = pd.concat(rows, ignore_index=True)
    slope["direction"] = pd.Categorical(slope["direction"], DIRECTION_ORDER, ordered=True)
    slope["event_target"] = slope["event_target"].astype(int)
    slope["x_pos"] = slope["event_target"].map(X_POS)
    return slope.sort_values(["direction", "event_target", "method"]).reset_index(drop=True)


def load_dca_data() -> pd.DataFrame:
    dca = pd.read_csv(TABLE_DIR / "Table_6_decision_curve_selected_thresholds.csv")
    strategy_map = {
        "Raw transport logistic": "Raw transport",
        "Intercept-only 100 events": "Intercept-only 100 events",
        "Platt 100 events": "Platt 100 events",
        "Internal HGB benchmark": "Internal HGB",
    }
    dca = dca.rename(columns={"Direction": "direction", "Threshold": "threshold"})
    long = dca.melt(
        id_vars=["direction", "threshold"],
        value_vars=list(strategy_map.keys()),
        var_name="strategy_raw",
        value_name="net_benefit",
    )
    long["strategy"] = long["strategy_raw"].map(strategy_map)
    long["direction"] = pd.Categorical(long["direction"], DIRECTION_ORDER, ordered=True)
    long["threshold"] = long["threshold"].astype(float)
    long["net_benefit"] = long["net_benefit"].astype(float)
    return long.sort_values(["direction", "strategy", "threshold"]).reset_index(drop=True)


def write_source_data(ece_plot: pd.DataFrame, slope: pd.DataFrame, dca: pd.DataFrame) -> pd.DataFrame:
    ece_source = ece_plot.assign(
        panel=np.where(ece_plot["direction"] == "NHANES -> MIMIC-IV", "3a", "3b"),
        metric="ECE",
        value=ece_plot["ece_mean"],
        ci_lower=ece_plot["ece_ci_lower"],
        ci_upper=ece_plot["ece_ci_upper"],
        threshold=np.nan,
        method_or_strategy=ece_plot["method"].map(METHOD_LABELS),
        notes=np.where(
            ece_plot["event_target"] == 0,
            "Raw ECE duplicated to anchor each recalibration-method trajectory.",
            "ECE mean and empirical interval across 200 local calibration samples.",
        ),
    )[
        [
            "panel",
            "direction",
            "metric",
            "method_or_strategy",
            "event_target",
            "threshold",
            "value",
            "ci_lower",
            "ci_upper",
            "notes",
        ]
    ]

    slope_source = slope.assign(
        panel="3c",
        metric="Calibration slope",
        value=slope["calibration_slope_mean"],
        ci_lower=slope["calibration_slope_ci_lower"],
        ci_upper=slope["calibration_slope_ci_upper"],
        threshold=np.nan,
        method_or_strategy=np.where(
            slope["event_target"] == 0,
            "Raw transport",
            slope["selected_method"] + " selected update",
        ),
        notes="Selected diagnostic-action slope trajectory from current Table 4; intervals retained as source data.",
    )[
        [
            "panel",
            "direction",
            "metric",
            "method_or_strategy",
            "event_target",
            "threshold",
            "value",
            "ci_lower",
            "ci_upper",
            "notes",
        ]
    ]

    dca_source = dca.assign(
        panel="3d",
        metric="Net benefit",
        method_or_strategy=dca["strategy"],
        event_target=np.nan,
        value=dca["net_benefit"],
        ci_lower=np.nan,
        ci_upper=np.nan,
        notes="Selected threshold decision-curve summary from Table 6.",
    )[
        [
            "panel",
            "direction",
            "metric",
            "method_or_strategy",
            "event_target",
            "threshold",
            "value",
            "ci_lower",
            "ci_upper",
            "notes",
        ]
    ]

    out = pd.concat([ece_source, slope_source, dca_source], ignore_index=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SOURCE_DIR / "Figure_3_source_data.csv", index=False)
    return out


def draw_ece_axis(
    ax: plt.Axes,
    data: pd.DataFrame,
    direction: str,
    title: str,
    show_ylabel: bool = True,
    show_legend: bool = False,
    y_max: float = 0.38,
) -> None:
    subset = data[data["direction"] == direction]
    for method in METHOD_ORDER:
        m = subset[subset["method"] == method].sort_values("event_target")
        color = METHOD_COLORS[method]
        ax.plot(m["x_pos"], m["ece_mean"], color=color, lw=1.5, marker="o", ms=3.5)
        m_ci = m[m["event_target"] > 0]
        ax.errorbar(
            m_ci["x_pos"],
            m_ci["ece_mean"],
            yerr=[
                m_ci["ece_mean"] - m_ci["ece_ci_lower"],
                m_ci["ece_ci_upper"] - m_ci["ece_mean"],
            ],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=1.8,
            alpha=0.65,
        )

    raw_y = subset[subset["event_target"] == 0]["ece_mean"].iloc[0]
    ax.scatter([0], [raw_y], s=28, color="#6B6F76", edgecolor="white", linewidth=0.6, zorder=4)
    ax.text(0.08, raw_y + 0.014, f"Raw {raw_y:.3f}", ha="left", va="bottom", fontsize=5.8, color="#3C4043")

    ax.set_title(title, loc="left", pad=5)
    ax.set_xticks(X_TICKS, X_LABELS)
    ax.set_xlabel("Local outcome events")
    if show_ylabel:
        ax.set_ylabel("ECE")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", color="#E7E7E7", lw=0.5)
    ax.grid(axis="x", color="#F0F0F0", lw=0.4)
    if show_legend:
        handles = [
            Line2D([0], [0], color=METHOD_COLORS[m], lw=1.6, marker="o", ms=3.5, label=METHOD_LABELS[m])
            for m in METHOD_ORDER
        ]
        ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=1, handlelength=1.4)


def draw_slope_axis(ax: plt.Axes, slope: pd.DataFrame) -> None:
    label_offsets = {
        "NHANES -> MIMIC-IV": -0.040,
        "MIMIC-IV ICU -> eICU": -0.005,
        "eICU -> MIMIC-IV ICU": 0.052,
    }
    x_offsets = {
        "NHANES -> MIMIC-IV": 0.10,
        "MIMIC-IV ICU -> eICU": 0.10,
        "eICU -> MIMIC-IV ICU": 0.10,
    }
    for direction in DIRECTION_ORDER:
        d = slope[slope["direction"] == direction].sort_values("event_target")
        ax.plot(
            d["x_pos"],
            d["calibration_slope_mean"],
            color=DIRECTION_COLORS[direction],
            lw=1.6,
            marker="o",
            ms=3.8,
            label=DIRECTION_LABELS[direction].replace("\n", " "),
        )
        last = d[d["event_target"] == 200].iloc[0]
        ax.text(
            last["x_pos"] + x_offsets[direction],
            last["calibration_slope_mean"] + label_offsets[direction],
            str(last["line_label"]),
            ha="left",
            va="center",
            fontsize=5.8,
            color=DIRECTION_COLORS[direction],
            linespacing=0.9,
        )

    ax.axhline(1.0, color="#9AA0A6", lw=0.9, ls=(0, (3, 2)))
    ax.text(0.06, 1.015, "Ideal slope", ha="left", va="bottom", fontsize=5.8, color="#3C4043")
    ax.set_title("Selected actions moved or preserved the slope", loc="left", pad=5)
    ax.set_xticks(X_TICKS, X_LABELS)
    ax.set_xlabel("Local outcome events")
    ax.set_ylabel("Calibration slope")
    ax.set_xlim(-0.12, 5.05)
    ax.set_ylim(0.35, 1.42)
    ax.grid(axis="y", color="#E7E7E7", lw=0.5)
    ax.grid(axis="x", color="#F0F0F0", lw=0.4)


def draw_dca_axis(ax: plt.Axes, dca: pd.DataFrame, direction: str, title: str, show_ylabel: bool) -> None:
    subset = dca[dca["direction"] == direction]
    for strategy in ["Raw transport", "Intercept-only 100 events", "Platt 100 events", "Internal HGB"]:
        s = subset[subset["strategy"] == strategy].sort_values("threshold")
        ax.plot(
            s["threshold"] * 100,
            s["net_benefit"],
            color=DCA_COLORS[strategy],
            marker="o",
            ms=3.0,
            lw=1.25,
            label=strategy,
        )

    ax.axhline(0, color="#9AA0A6", lw=0.8, ls=(0, (3, 2)))
    ax.set_title(title, fontsize=6.4, pad=4)
    ax.set_xlim(19, 31)
    ax.set_ylim(-0.014, 0.075)
    ax.set_xticks([20, 25, 30])
    ax.set_xlabel("Threshold (%)", fontsize=6.2)
    if show_ylabel:
        ax.set_ylabel("Net benefit", fontsize=6.4)
    else:
        ax.set_yticklabels([])
    ax.grid(axis="y", color="#E7E7E7", lw=0.5)


def make_figure(ece_plot: pd.DataFrame, slope: pd.DataFrame, dca: pd.DataFrame) -> plt.Figure:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.2, 5.65), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        top=0.94,
        bottom=0.105,
        wspace=0.30,
        hspace=0.56,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    draw_ece_axis(
        ax_a,
        ece_plot,
        "NHANES -> MIMIC-IV",
        "Extreme shift: recalibration repaired ECE",
        show_ylabel=True,
        show_legend=True,
        y_max=0.18,
    )
    panel_label(ax_a, "a")

    sub_b = gs[0, 1].subgridspec(1, 2, wspace=0.18)
    ax_b1 = fig.add_subplot(sub_b[0, 0])
    ax_b2 = fig.add_subplot(sub_b[0, 1])
    draw_ece_axis(ax_b1, ece_plot, "MIMIC-IV ICU -> eICU", "MIMIC-IV ICU -> eICU", show_ylabel=True, y_max=0.075)
    draw_ece_axis(ax_b2, ece_plot, "eICU -> MIMIC-IV ICU", "eICU -> MIMIC-IV ICU", show_ylabel=False, y_max=0.075)
    panel_label(ax_b1, "b", x=-0.25)

    ax_c = fig.add_subplot(gs[1, 0])
    draw_slope_axis(ax_c, slope)
    panel_label(ax_c, "c")

    sub_d = gs[1, 1].subgridspec(1, 3, wspace=0.16)
    axes_d = [
        fig.add_subplot(sub_d[0, 0]),
        fig.add_subplot(sub_d[0, 1]),
        fig.add_subplot(sub_d[0, 2]),
    ]
    for i, (ax, direction) in enumerate(zip(axes_d, DIRECTION_ORDER)):
        draw_dca_axis(ax, dca, direction, SHORT_TITLES[direction], show_ylabel=(i == 0))
    panel_label(axes_d[0], "d", x=-0.37)
    handles = [
        Line2D([0], [0], color=DCA_COLORS[s], lw=1.4, marker="o", ms=3.0, label=label)
        for s, label in [
            ("Raw transport", "Raw"),
            ("Intercept-only 100 events", "Intercept 100"),
            ("Platt 100 events", "Platt 100"),
            ("Internal HGB", "Internal HGB"),
        ]
    ]
    axes_d[0].legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.10),
        ncol=4,
        handlelength=1.0,
        columnspacing=0.45,
    )

    return fig


def save_outputs(fig: plt.Figure) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / "Figure_3_event_count_recalibration"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> None:
    ece = load_ece_data()
    ece_plot = build_ece_plot_data(ece)
    slope = load_slope_data()
    dca = load_dca_data()
    write_source_data(ece_plot, slope, dca)
    fig = make_figure(ece_plot, slope, dca)
    save_outputs(fig)
    plt.close(fig)
    print(f"Wrote figure outputs to {FIG_DIR}")
    print(f"Wrote source data to {SOURCE_DIR / 'Figure_3_source_data.csv'}")


if __name__ == "__main__":
    main()
