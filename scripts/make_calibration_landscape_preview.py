from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "nature_style_main_figure_preview"
TABLE4B = PROJECT_ROOT / "outputs" / "manuscript_tables" / "Table_4b_recalibration_uncertainty_intervals.csv"


INK = "#242424"
MUTED = "#6E6E6E"
RAW = "#C87944"
PLATT = "#7657A8"
ISO = "#6F9B7B"
INTERCEPT = "#8B8B8B"
FLOOR = "#F4F4F4"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def load_points() -> pd.DataFrame:
    table = pd.read_csv(TABLE4B)
    df = table[
        table["prediction_set"].eq("NHANES_to_MIMIC_1y_base_logistic_regression")
        & table["method"].isin(["raw", "intercept_only", "isotonic", "platt"])
        & table["event_target"].isin([0, 25, 50, 100, 200])
    ].copy()
    df["label"] = df["method"].map(
        {
            "raw": "Raw transport",
            "intercept_only": "Intercept-only",
            "isotonic": "Isotonic",
            "platt": "Platt",
        }
    )
    return df


def label3d(ax, x: float, y: float, z: float, text: str, color: str, dx=0.0, dy=0.0, dz=0.0) -> None:
    ax.text(
        x + dx,
        y + dy,
        z + dz,
        text,
        color=color,
        fontsize=6.6,
        ha="left",
        va="center",
        zorder=20,
    )


def make_landscape() -> None:
    df = load_points()
    fig = plt.figure(figsize=(7.15, 4.50))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    sx = [0.20, 1.40]
    sy = [-1.62, 0.52]
    xx, yy = np.meshgrid(sx, sy)
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color=FLOOR, alpha=0.88, shade=False, linewidth=0, zorder=0)

    for level, alpha in [(0.02, 0.060), (0.05, 0.040), (0.10, 0.025)]:
        ax.plot_surface(xx, yy, np.full_like(xx, level), color="#BCA7D8", alpha=alpha, shade=False, linewidth=0, zorder=0)

    colors = {"raw": RAW, "intercept_only": INTERCEPT, "isotonic": ISO, "platt": PLATT}
    for method in ["intercept_only", "isotonic", "platt"]:
        s = df[df["method"].eq(method)].sort_values("event_target")
        ax.plot(
            s["calibration_slope_mean"],
            s["calibration_intercept_mean"],
            np.zeros(len(s)),
            color=colors[method],
            lw=1.3,
            ls=":",
            alpha=0.45,
            zorder=2,
        )
        for _, row in s.iterrows():
            ax.plot(
                [row["calibration_slope_mean"], row["calibration_slope_mean"]],
                [row["calibration_intercept_mean"], row["calibration_intercept_mean"]],
                [0, row["ece_10bin_mean"]],
                color=colors[method],
                lw=0.8,
                alpha=0.28,
                zorder=3,
            )
        ax.plot(
            s["calibration_slope_mean"],
            s["calibration_intercept_mean"],
            s["ece_10bin_mean"],
            color=colors[method],
            lw=2.4,
            marker="o",
            ms=4.4,
            mec="white",
            mew=0.6,
            zorder=5,
        )
    raw = df[df["method"].eq("raw")].iloc[0]
    ax.scatter(
        [raw["calibration_slope_mean"]],
        [raw["calibration_intercept_mean"]],
        [raw["ece_10bin_mean"]],
        s=82,
        color=RAW,
        edgecolor="white",
        linewidth=0.7,
        depthshade=False,
        zorder=10,
    )
    ax.plot(
        [raw["calibration_slope_mean"], raw["calibration_slope_mean"]],
        [raw["calibration_intercept_mean"], raw["calibration_intercept_mean"]],
        [0, raw["ece_10bin_mean"]],
        color=RAW,
        lw=1.1,
        alpha=0.40,
        zorder=3,
    )
    ax.scatter(
        [raw["calibration_slope_mean"]],
        [raw["calibration_intercept_mean"]],
        [0],
        s=28,
        color=RAW,
        alpha=0.32,
        depthshade=False,
        zorder=2,
    )
    ax.scatter([1], [0], [0], s=92, marker="*", color=INK, depthshade=False, zorder=11)

    platt100 = df[df["method"].eq("platt") & df["event_target"].eq(100)].iloc[0]
    platt200 = df[df["method"].eq("platt") & df["event_target"].eq(200)].iloc[0]
    iso200 = df[df["method"].eq("isotonic") & df["event_target"].eq(200)].iloc[0]
    int200 = df[df["method"].eq("intercept_only") & df["event_target"].eq(200)].iloc[0]

    label3d(ax, raw["calibration_slope_mean"], raw["calibration_intercept_mean"], raw["ece_10bin_mean"], "Raw transport\nslope 0.45\nintercept -1.51", RAW, 0.035, -0.050, 0.010)
    label3d(ax, platt100["calibration_slope_mean"], platt100["calibration_intercept_mean"], platt100["ece_10bin_mean"], "Platt 100", PLATT, 0.105, 0.060, 0.030)
    label3d(ax, platt200["calibration_slope_mean"], platt200["calibration_intercept_mean"], platt200["ece_10bin_mean"], "Platt 200", PLATT, 0.055, -0.125, 0.020)
    label3d(ax, iso200["calibration_slope_mean"], iso200["calibration_intercept_mean"], iso200["ece_10bin_mean"], "Isotonic 200", ISO, 0.045, -0.040, 0.020)
    label3d(ax, int200["calibration_slope_mean"], int200["calibration_intercept_mean"], int200["ece_10bin_mean"], "Intercept-only", INTERCEPT, 0.040, 0.020, 0.022)
    label3d(ax, 1, 0, 0, "Ideal", INK, -0.135, 0.045, 0.018)

    ax.set_xlim(0.20, 1.40)
    ax.set_ylim(-1.62, 0.52)
    ax.set_zlim(0.00, 0.31)
    ax.set_xlabel("Calibration slope", labelpad=4, fontsize=7.0)
    ax.set_ylabel("Calibration intercept", labelpad=5, fontsize=7.0)
    ax.set_zlabel("ECE", labelpad=2, fontsize=7.0)
    ax.set_xticks([0.25, 0.50, 0.75, 1.00, 1.25])
    ax.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5])
    ax.set_zticks([0.00, 0.10, 0.20, 0.30])
    ax.view_init(elev=20, azim=-49)
    ax.set_box_aspect((1.70, 1.05, 0.66))
    ax.tick_params(labelsize=6.2, pad=-1)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = (0.89, 0.89, 0.89, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.55
        axis._axinfo["axisline"]["color"] = (0.25, 0.25, 0.25, 1.0)
    ax.xaxis.pane.set_facecolor((0.98, 0.98, 0.98, 0.38))
    ax.yaxis.pane.set_facecolor((0.98, 0.98, 0.98, 0.28))
    ax.zaxis.pane.set_facecolor((0.98, 0.98, 0.98, 0.18))

    ax.text2D(
        0.010,
        0.985,
        "Calibration-state trajectory after transport and local recalibration",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=INK,
    )
    ax.text2D(
        0.010,
        0.935,
        "Observed states are plotted in slope-intercept-ECE space; floor projections show the path back toward ideal calibration.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=MUTED,
    )

    handles = [
        Line2D([0], [0], color=INTERCEPT, marker="o", lw=1.7, label="Intercept-only"),
        Line2D([0], [0], color=ISO, marker="o", lw=1.7, label="Isotonic"),
        Line2D([0], [0], color=PLATT, marker="o", lw=1.7, label="Platt"),
        Line2D([0], [0], color=RAW, marker="o", lw=0, label="Raw transport"),
        Line2D([0], [0], color=INK, marker="*", lw=0, markersize=8, label="Ideal"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.865), fontsize=5.3, handlelength=1.3, labelspacing=0.26, borderpad=0.1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    for ext, kwargs in {
        "png": {"dpi": 450},
        "svg": {},
        "pdf": {},
    }.items():
        fig.savefig(OUT_DIR / f"calibration_landscape_3d_preview.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    apply_style()
    make_landscape()
    print(f"Wrote calibration landscape preview to {OUT_DIR}")


if __name__ == "__main__":
    main()
