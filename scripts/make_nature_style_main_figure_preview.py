from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "nature_style_main_figure_preview"

TABLE3 = PROJECT_ROOT / "outputs" / "manuscript_tables" / "Table_3_transportability_metrics_ci.csv"
TABLE4B = PROJECT_ROOT / "outputs" / "manuscript_tables" / "Table_4b_recalibration_uncertainty_intervals.csv"
DCA_PATH = PROJECT_ROOT / "outputs" / "decision_curve" / "decision_curve_transportability.csv"


INK = "#222222"
MUTED = "#6F6F6F"
GRID = "#E8E8E8"
NHANES = "#8A76B5"
MIMIC = "#5B987A"
RAW = "#C87944"
PLATT = "#7657A8"
ISO = "#7A9F84"
INTERCEPT = "#878787"
PALE_PURPLE = "#F1EDF8"
PALE_GREEN = "#EEF6F1"
PALE_WARM = "#F8EEE8"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        "png": {"dpi": 450},
        "svg": {},
        "pdf": {},
    }.items():
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def first_number(value: str) -> float:
    return float(str(value).split(" ", 1)[0])


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=INK,
    )


def add_database(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    title: str,
    subtitle: str,
) -> None:
    body = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=color,
        edgecolor="none",
        transform=ax.transAxes,
        alpha=0.95,
        zorder=2,
    )
    ax.add_patch(body)
    for i in range(4):
        ax.add_patch(
            patches.Ellipse(
                (x + w / 2, y + h - i * h / 4),
                w,
                h * 0.20,
                facecolor="#FFFFFF" if i == 0 else color,
                edgecolor="white",
                lw=0.8,
                alpha=0.62 if i == 0 else 0.25,
                transform=ax.transAxes,
                zorder=3,
            )
        )
    ax.text(x + w / 2, y + h * 0.60, title, transform=ax.transAxes, ha="center", va="center", color="white", fontsize=8.2, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.34, subtitle, transform=ax.transAxes, ha="center", va="center", color="white", fontsize=6.2)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    label: str | None = None,
    rad: float = 0.0,
    label_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    ax.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.5,
            color=color,
            transform=ax.transAxes,
            zorder=4,
        )
    )
    if label:
        ax.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.7,
            color=color,
            fontweight="bold",
        )


def calibration_mini(ax: plt.Axes, x: float, y: float, w: float, h: float, mode: str) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor="white",
            edgecolor="#D5D5D5",
            lw=0.7,
            transform=ax.transAxes,
            zorder=3,
        )
    )
    xs = np.linspace(0.12, 0.88, 36)
    ax.plot(x + xs * w, y + xs * h, color="#AAAAAA", lw=0.75, ls="--", transform=ax.transAxes, zorder=4)
    if mode == "raw":
        ys = 0.10 + 0.30 * xs + 0.025 * np.sin(xs * 7)
        color = RAW
    else:
        ys = 0.08 + 0.84 * xs + 0.015 * np.sin(xs * 7)
        color = PLATT
    ax.plot(x + xs * w, y + ys * h, color=color, lw=1.7, transform=ax.transAxes, zorder=5)


def draw_schematic(ax: plt.Axes) -> None:
    ax.set_axis_off()
    panel_label(ax, "a", x=-0.012, y=0.99)
    ax.text(
        0.02,
        0.94,
        "Local recalibration converts transport failure into deployable risk prediction",
        transform=ax.transAxes,
        fontsize=9.6,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.02,
        0.875,
        "Raw transport preserves moderate discrimination but severely underestimates mortality risk.",
        transform=ax.transAxes,
        fontsize=6.8,
        color=MUTED,
    )

    add_database(ax, 0.055, 0.48, 0.145, 0.28, NHANES, "NHANES", "source")
    add_database(ax, 0.805, 0.48, 0.145, 0.28, MIMIC, "MIMIC-IV", "target")

    raw_box = patches.FancyBboxPatch(
        (0.285, 0.50),
        0.25,
        0.255,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        facecolor=PALE_WARM,
        edgecolor="#E5C5AF",
        lw=0.8,
        transform=ax.transAxes,
        zorder=1,
    )
    platt_box = patches.FancyBboxPatch(
        (0.565, 0.16),
        0.255,
        0.255,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        facecolor=PALE_PURPLE,
        edgecolor="#CEC2E2",
        lw=0.8,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(raw_box)
    ax.add_patch(platt_box)

    calibration_mini(ax, 0.306, 0.535, 0.105, 0.155, "raw")
    calibration_mini(ax, 0.585, 0.195, 0.100, 0.155, "platt")
    ax.text(0.427, 0.680, "Raw transport", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=RAW)
    ax.text(0.427, 0.622, "slope 0.45\nintercept -1.51\nECE 0.286", transform=ax.transAxes, fontsize=6.6, color=INK, va="top")
    ax.text(0.700, 0.340, "Platt recalibration", transform=ax.transAxes, fontsize=7.1, fontweight="bold", color=PLATT)
    ax.text(0.700, 0.282, "100-200 events\nslope ~= 1\nintercept ~= 0", transform=ax.transAxes, fontsize=6.3, color=INK, va="top")

    arrow(ax, (0.205, 0.635), (0.285, 0.635), RAW)
    arrow(ax, (0.535, 0.635), (0.805, 0.635), RAW, "direct transport", rad=0.03, label_offset=(0.0, 0.055))
    arrow(ax, (0.805, 0.46), (0.690, 0.405), PLATT, rad=-0.15)
    arrow(ax, (0.690, 0.405), (0.565, 0.315), PLATT, rad=-0.08)
    ax.text(0.705, 0.455, "target outcomes", transform=ax.transAxes, fontsize=6.4, color=PLATT, fontweight="bold")

    ax.add_patch(patches.Rectangle((0.055, 0.075), 0.895, 0.010, transform=ax.transAxes, facecolor="#EFEFEF", edgecolor="none"))
    ax.text(0.060, 0.120, "model development", transform=ax.transAxes, fontsize=6.4, color=MUTED)
    ax.text(0.815, 0.120, "target-site use", transform=ax.transAxes, fontsize=6.4, color=MUTED)


def plot_state_map(ax: plt.Axes) -> None:
    t4b = pd.read_csv(TABLE4B)
    sub = t4b[
        t4b["prediction_set"].eq("NHANES_to_MIMIC_1y_base_logistic_regression")
        & t4b["method"].isin(["raw", "intercept_only", "isotonic", "platt"])
        & t4b["event_target"].isin([0, 25, 50, 100, 200])
    ].copy()
    colors = {"raw": RAW, "intercept_only": INTERCEPT, "isotonic": ISO, "platt": PLATT}
    labels = {"raw": "Raw", "intercept_only": "Intercept-only", "isotonic": "Isotonic", "platt": "Platt"}

    for method in ["intercept_only", "isotonic", "platt"]:
        s = sub[sub["method"].eq(method)].sort_values("event_target")
        ax.plot(
            s["calibration_slope_mean"],
            s["calibration_intercept_mean"],
            color=colors[method],
            lw=1.5,
            marker="o",
            ms=4.0,
            mec="white",
            mew=0.5,
            label=labels[method],
        )
        if method in ["platt", "isotonic"]:
            end = s[s["event_target"].eq(200)].iloc[0]
            if method == "platt":
                label_x = end["calibration_slope_mean"] + 0.085
                label_y = end["calibration_intercept_mean"] - 0.075
            else:
                label_x = end["calibration_slope_mean"] + 0.025
                label_y = end["calibration_intercept_mean"]
            ax.text(
                label_x,
                label_y,
                f"{labels[method]} 200",
                fontsize=5.9,
                color=colors[method],
                va="center",
            )
    raw = sub[sub["method"].eq("raw")].iloc[0]
    ax.scatter(
        [raw["calibration_slope_mean"]],
        [raw["calibration_intercept_mean"]],
        s=130,
        color=RAW,
        edgecolor="white",
        linewidth=0.7,
        label="Raw",
        zorder=5,
    )
    ax.scatter([1], [0], s=150, marker="*", color=INK, label="Ideal", zorder=6)
    ax.text(raw["calibration_slope_mean"] + 0.035, raw["calibration_intercept_mean"] + 0.18, "Raw transport\nslope 0.45\nintercept -1.51", color=RAW, fontsize=6.0, va="bottom")
    ax.text(0.86, 0.105, "Ideal", color=INK, fontsize=6.0, va="bottom")
    int_end = sub[sub["method"].eq("intercept_only") & sub["event_target"].eq(200)].iloc[0]
    ax.text(int_end["calibration_slope_mean"] + 0.030, int_end["calibration_intercept_mean"], "Intercept-only", fontsize=5.9, color=INTERCEPT, va="center")

    ax.axvline(1, color=INK, ls="--", lw=0.75, alpha=0.65)
    ax.axhline(0, color=INK, ls="--", lw=0.75, alpha=0.65)
    ax.set_xlim(0.20, 1.42)
    ax.set_ylim(-1.62, 0.55)
    ax.set_xlabel("Calibration slope")
    ax.set_ylabel("Calibration intercept")
    ax.set_title("State map", loc="left", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(color=GRID, lw=0.5)
    panel_label(ax, "b")


def plot_slope_intervals(ax: plt.Axes) -> None:
    t4b = pd.read_csv(TABLE4B)
    sub = t4b[
        t4b["prediction_set"].eq("NHANES_to_MIMIC_1y_base_logistic_regression")
        & t4b["method"].isin(["intercept_only", "platt"])
        & t4b["event_target"].isin([25, 50, 100, 200])
    ].copy()
    y_lookup = {
        ("intercept_only", 25): 8,
        ("intercept_only", 50): 7,
        ("intercept_only", 100): 6,
        ("intercept_only", 200): 5,
        ("platt", 25): 3,
        ("platt", 50): 2,
        ("platt", 100): 1,
        ("platt", 200): 0,
    }
    colors = {"intercept_only": INTERCEPT, "platt": PLATT}
    rows = []
    for method in ["intercept_only", "platt"]:
        for event in [25, 50, 100, 200]:
            r = sub[sub["method"].eq(method) & sub["event_target"].eq(event)].iloc[0]
            rows.append(
                {
                    "y": y_lookup[(method, event)],
                    "label": str(event),
                    "mean": r["calibration_slope_mean"],
                    "lo": r["calibration_slope_ci_lower"],
                    "hi": r["calibration_slope_ci_upper"],
                    "color": colors[method],
                }
            )
    for row in rows:
        ax.plot([row["lo"], row["hi"]], [row["y"], row["y"]], color=row["color"], lw=1.45, alpha=0.88)
        ax.plot(row["mean"], row["y"], "o", color=row["color"], ms=3.4)
    ax.axvline(1, color=INK, ls="--", lw=0.8)
    ax.set_yticks([8, 7, 6, 5, 3, 2, 1, 0])
    ax.set_yticklabels(["25", "50", "100", "200", "25", "50", "100", "200"])
    ax.text(0.32, 8.42, "Intercept-only", ha="left", va="center", fontsize=6.4, color=INTERCEPT, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    ax.text(0.32, 3.42, "Platt", ha="left", va="center", fontsize=6.4, color=PLATT, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    ax.set_xlim(0.30, 1.72)
    ax.set_ylim(-0.7, 8.7)
    ax.set_xlabel("Calibration slope")
    ax.set_ylabel("Local events")
    ax.set_title("Slope recovery", loc="left", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(axis="x", color=GRID, lw=0.5)
    panel_label(ax, "c")


def plot_decision_curve(ax: plt.Axes) -> None:
    dca = pd.read_csv(DCA_PATH)
    keep = ["Treat none", "Treat all", "NHANES raw logistic", "NHANES Platt 100 events", "MIMIC internal HGB"]
    dca = dca[dca["model"].isin(keep)].copy()
    colors = {
        "Treat none": "#777777",
        "Treat all": "#BEBEBE",
        "NHANES raw logistic": RAW,
        "NHANES Platt 100 events": PLATT,
        "MIMIC internal HGB": MIMIC,
    }
    styles = {"Treat none": ":", "Treat all": "--"}
    labels = {
        "Treat none": "Treat none",
        "Treat all": "Treat all",
        "NHANES raw logistic": "Raw transport",
        "NHANES Platt 100 events": "Platt 100 events",
        "MIMIC internal HGB": "MIMIC internal",
    }
    for model in keep:
        g = dca[dca["model"].eq(model)].sort_values("threshold")
        lw = 1.65 if model in ["NHANES Platt 100 events", "MIMIC internal HGB"] else 1.15
        ax.plot(g["threshold"], g["net_benefit"], color=colors[model], lw=lw, ls=styles.get(model, "-"), label=labels[model])
        if model in ["NHANES raw logistic", "NHANES Platt 100 events", "MIMIC internal HGB"]:
            point = g[g["threshold"].between(0.28, 0.31)].iloc[-1]
            ax.text(point["threshold"] + 0.006, point["net_benefit"], labels[model], color=colors[model], fontsize=5.9, va="center")
        elif model == "Treat all":
            point = g[g["threshold"].between(0.12, 0.14)].iloc[0]
            ax.text(point["threshold"] - 0.012, point["net_benefit"] - 0.012, "Treat all", color=colors[model], fontsize=5.7, ha="right", va="center")
        elif model == "Treat none":
            ax.text(0.245, 0.005, "Treat none", color=colors[model], fontsize=5.7, ha="center", va="bottom")
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xlim(0.05, 0.35)
    ax.set_ylim(-0.07, 0.14)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Clinical utility", loc="left", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(axis="y", color=GRID, lw=0.5)
    panel_label(ax, "d")


def make_figure() -> None:
    fig = plt.figure(figsize=(7.4, 5.65))
    gs = fig.add_gridspec(2, 12, height_ratios=[0.98, 1.42], hspace=0.28, wspace=1.08)

    ax_a = fig.add_subplot(gs[0, :])
    draw_schematic(ax_a)

    ax_b = fig.add_subplot(gs[1, :4])
    plot_state_map(ax_b)

    ax_c = fig.add_subplot(gs[1, 4:8])
    plot_slope_intervals(ax_c)

    ax_d = fig.add_subplot(gs[1, 8:])
    plot_decision_curve(ax_d)

    save(fig, "nature_style_main_figure_preview")


def main() -> None:
    apply_style()
    make_figure()
    print(f"Wrote preview figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
