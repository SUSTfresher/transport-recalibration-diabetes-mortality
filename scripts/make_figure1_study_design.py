from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "current_submission" / "tables"
FIG_DIR = ROOT / "outputs" / "current_submission" / "figures" / "figure1"
SOURCE_DIR = ROOT / "outputs" / "current_submission" / "source_data"


DATA_COLORS = {
    "NHANES": "#4E79A7",
    "MIMIC-IV ICU": "#59A14F",
    "eICU": "#B07AA1",
    "MIMIC-IV": "#59A14F",
}

METHOD_COLORS = {
    "Logistic AUC": "#6B6F76",
    "Random forest AUC": "#59A14F",
}

DIRECTION_ROWS = [
    {
        "direction": "NHANES -> MIMIC-IV",
        "scenario": "Extreme stress test",
        "endpoint": "1-year mortality",
        "source": "NHANES",
        "source_n": 6564,
        "source_events": 166,
        "target": "MIMIC-IV",
        "target_n": 33201,
        "target_events": 6297,
    },
    {
        "direction": "MIMIC-IV ICU -> eICU",
        "scenario": "ICU deployment",
        "endpoint": "Hospital mortality",
        "source": "MIMIC-IV ICU",
        "source_n": 25906,
        "source_events": 2988,
        "target": "eICU",
        "target_n": 13971,
        "target_events": 1263,
    },
    {
        "direction": "eICU -> MIMIC-IV ICU",
        "scenario": "ICU deployment",
        "endpoint": "Hospital mortality",
        "source": "eICU",
        "source_n": 55486,
        "source_events": 5242,
        "target": "MIMIC-IV ICU",
        "target_n": 6395,
        "target_events": 712,
    },
]


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
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.07) -> None:
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


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    color: str,
    text_color: str = "#202124",
    lw: float = 0.9,
    alpha: float = 0.12,
    fontsize: float = 6.5,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=mpl.colors.to_rgba(color, alpha),
        edgecolor=color,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        linespacing=1.0,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#6B6F76",
    rad: float = 0.0,
    lw: float = 1.1,
    mutation_scale: float = 9,
) -> None:
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)


def load_source_classifier() -> pd.DataFrame:
    path = TABLE_DIR / "Table_2_source_classifier_auc.csv"
    return pd.read_csv(path)


def build_source_data(source_auc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    rows.extend(
        [
            {
                "panel": "1a",
                "record_type": "transport_direction",
                **row,
                "metric": np.nan,
                "value": np.nan,
                "notes": "Design schematic transport direction.",
            }
            for row in DIRECTION_ROWS
        ]
    )

    for row in DIRECTION_ROWS:
        for role in ["source", "target"]:
            rows.append(
                {
                    "panel": "1b",
                    "record_type": f"{role}_cohort",
                    "direction": row["direction"],
                    "scenario": row["scenario"],
                    "endpoint": row["endpoint"],
                    "source": row["source"],
                    "source_n": row["source_n"],
                    "source_events": row["source_events"],
                    "target": row["target"],
                    "target_n": row["target_n"],
                    "target_events": row["target_events"],
                    "metric": f"{role}_events",
                    "value": row[f"{role}_events"],
                    "notes": "Source cohort size and target evaluation size for primary transport direction.",
                }
            )

    for _, row in source_auc.iterrows():
        contrast = str(row["Source contrast"])
        panel = "1c" if contrast == "NHANES vs MIMIC-IV" else "1d"
        for metric_col in ["Logistic AUC", "Random forest AUC"]:
            rows.append(
                {
                    "panel": panel,
                    "record_type": "source_classifier_auc",
                    "direction": contrast,
                    "scenario": row["Scenario"],
                    "endpoint": np.nan,
                    "source": np.nan,
                    "source_n": np.nan,
                    "source_events": np.nan,
                    "target": np.nan,
                    "target_n": np.nan,
                    "target_events": np.nan,
                    "feature_block": row["Feature block"],
                    "metric": metric_col,
                    "value": row[metric_col],
                    "n_balanced": row["N balanced"],
                    "n_per_source": row["N per source"],
                    "notes": "Feature-block source-classifier AUC from Table 2.",
                }
            )

    source = pd.DataFrame(rows)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    source.to_csv(SOURCE_DIR / "Figure_1_source_data.csv", index=False)
    return source


def draw_panel_a(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("Three data sources and transport directions", loc="left", pad=5)

    rounded_box(
        ax,
        (0.03, 0.62),
        0.22,
        0.20,
        "NHANES\npopulation\nsurvey",
        DATA_COLORS["NHANES"],
        fontsize=6.8,
    )
    rounded_box(
        ax,
        (0.38, 0.62),
        0.24,
        0.20,
        "MIMIC-IV ICU\nsingle-system\nICU",
        DATA_COLORS["MIMIC-IV ICU"],
        fontsize=6.8,
    )
    rounded_box(
        ax,
        (0.75, 0.62),
        0.22,
        0.20,
        "eICU\nmulticenter\nICU",
        DATA_COLORS["eICU"],
        fontsize=6.8,
    )

    arrow(ax, (0.25, 0.72), (0.38, 0.72), DATA_COLORS["NHANES"], rad=0.0, lw=1.25)
    ax.text(
        0.315,
        0.86,
        "Extreme\nstress test",
        ha="center",
        va="bottom",
        fontsize=5.9,
        color="#3C4043",
        linespacing=0.95,
    )
    arrow(ax, (0.62, 0.76), (0.75, 0.76), DATA_COLORS["MIMIC-IV ICU"], rad=0.0, lw=1.25)
    arrow(ax, (0.75, 0.68), (0.62, 0.68), DATA_COLORS["eICU"], rad=0.0, lw=1.25)
    ax.text(
        0.685,
        0.86,
        "Bidirectional\nICU deployment",
        ha="center",
        va="bottom",
        fontsize=5.9,
        color="#3C4043",
        linespacing=0.95,
    )

    step_y = 0.18
    steps = [
        ("1. Quantify\nsource shift", 0.05),
        ("2. Evaluate\ndiscrimination", 0.30),
        ("3. Diagnose\nslope/intercept", 0.55),
        ("4. Select\nupdating strategy", 0.80),
    ]
    for label, x in steps:
        rounded_box(
            ax,
            (x, step_y),
            0.17,
            0.16,
            label,
            "#9AA0A6",
            alpha=0.10,
            fontsize=6.0,
        )
    for (_, x0), (_, x1) in zip(steps[:-1], steps[1:]):
        arrow(ax, (x0 + 0.17, step_y + 0.075), (x1, step_y + 0.075), "#9AA0A6", lw=1.0)

    ax.text(
        0.05,
        0.42,
        "Scenario-specific endpoints: 1-year mortality for NHANES -> MIMIC-IV;\n"
        "hospital mortality for MIMIC-IV ICU <-> eICU.",
        ha="left",
        va="center",
        fontsize=6.0,
        color="#3C4043",
        linespacing=1.05,
    )
    panel_label(ax, "a")


def fmt_count(n: int, events: int) -> str:
    pct = events / n * 100
    return f"{n:,}\n{events:,} events ({pct:.1f}%)"


def fmt_count_short(n: int, events: int) -> str:
    pct = events / n * 100
    return f"N={n:,}\nDeaths={events:,} ({pct:.1f}%)"


def draw_panel_b(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("Cohort sizes and endpoints by direction", loc="left", pad=11)

    x_cols = [0.02, 0.30, 0.55, 0.78]
    headers = ["Direction", "Endpoint", "Source", "Target evaluation"]
    for x, header in zip(x_cols, headers):
        ax.text(x, 0.83, header, ha="left", va="center", fontsize=5.9, fontweight="bold")
    ax.plot([0.02, 0.98], [0.78, 0.78], color="#DADCE0", lw=0.8)

    y_vals = [0.64, 0.43, 0.22]
    for y, row in zip(y_vals, DIRECTION_ROWS):
        ax.plot([0.02, 0.98], [y - 0.095, y - 0.095], color="#ECECEC", lw=0.6)
        ax.text(
            x_cols[0],
            y,
            row["direction"].replace(" -> ", "\n-> "),
            ha="left",
            va="center",
            fontsize=5.5,
            linespacing=0.95,
        )
        ax.text(
            x_cols[1],
            y,
            row["endpoint"].replace(" ", "\n", 1),
            ha="left",
            va="center",
            fontsize=5.5,
            linespacing=0.95,
        )
        ax.text(
            x_cols[2],
            y,
            f"{row['source']}\n{fmt_count_short(row['source_n'], row['source_events'])}",
            ha="left",
            va="center",
            fontsize=4.85,
            linespacing=0.92,
            color=DATA_COLORS.get(row["source"], "#202124"),
        )
        ax.text(
            x_cols[3],
            y,
            f"{row['target']}\n{fmt_count_short(row['target_n'], row['target_events'])}",
            ha="left",
            va="center",
            fontsize=4.85,
            linespacing=0.92,
            color=DATA_COLORS.get(row["target"], "#202124"),
        )

    ax.text(
        0.02,
        0.045,
        "Target evaluation sizes correspond to primary transport-performance analyses.",
        ha="left",
        va="center",
        fontsize=5.4,
        color="#5F6368",
    )
    panel_label(ax, "b")


def draw_source_auc_axis(
    ax: plt.Axes,
    source_auc: pd.DataFrame,
    contrast: str,
    title: str,
    show_legend: bool = False,
) -> None:
    subset = source_auc[source_auc["Source contrast"] == contrast].copy()
    x = np.arange(len(subset))
    labels = [str(v).replace(" + ", " +\n") for v in subset["Feature block"]]
    for metric_col in ["Logistic AUC", "Random forest AUC"]:
        values = subset[metric_col].to_numpy(dtype=float)
        ax.plot(
            x,
            values,
            color=METHOD_COLORS[metric_col],
            marker="o",
            lw=1.6,
            ms=4.0,
            label=metric_col.replace(" AUC", ""),
        )
        for xi, val in zip(x, values):
            ax.text(xi, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=5.5)

    ax.set_xticks(x, labels)
    ax.set_ylim(0.58, 1.02)
    ax.set_ylabel("Source-classifier AUC")
    ax.set_title(title, loc="left", pad=5)
    ax.axhline(0.5, color="#9AA0A6", lw=0.8, ls=(0, (3, 2)))
    ax.grid(axis="y", color="#E7E7E7", lw=0.5)
    ax.grid(axis="x", color="#F0F0F0", lw=0.4)
    if show_legend:
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), handlelength=1.2)


def make_figure(source_auc: pd.DataFrame) -> plt.Figure:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.6, 5.25), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.065,
        right=0.985,
        top=0.94,
        bottom=0.11,
        wspace=0.32,
        hspace=0.55,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_source_auc_axis(
        ax_c,
        source_auc,
        "NHANES vs MIMIC-IV",
        "NHANES-MIMIC separability increased with labs",
        show_legend=True,
    )
    panel_label(ax_c, "c")
    draw_source_auc_axis(
        ax_d,
        source_auc,
        "MIMIC-IV ICU vs eICU",
        "ICU separability increased with labs and vitals",
        show_legend=False,
    )
    panel_label(ax_d, "d")
    return fig


def save_outputs(fig: plt.Figure) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / "Figure_1_study_design_source_shift"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> None:
    source_auc = load_source_classifier()
    build_source_data(source_auc)
    fig = make_figure(source_auc)
    save_outputs(fig)
    plt.close(fig)
    print(f"Wrote figure outputs to {FIG_DIR}")
    print(f"Wrote source data to {SOURCE_DIR / 'Figure_1_source_data.csv'}")


if __name__ == "__main__":
    main()
