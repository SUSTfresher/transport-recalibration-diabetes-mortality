from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_PROCESSED = PROJECT_ROOT / "data" / "mimic" / "processed"
OUT_DIR = PROJECT_ROOT / "outputs" / "domain_shift"


WINDOWS = {
    "p0_p24h": MIMIC_PROCESSED / "mimic_iv31_diabetes_labs_24h_metadata.json",
    "m24_p24h": MIMIC_PROCESSED / "mimic_iv31_diabetes_labs_m24_p24h_metadata.json",
}


def load_counts(label: str, path: Path) -> tuple[int, pd.DataFrame]:
    data = json.loads(path.read_text(encoding="utf-8"))
    any_lab = data.get("admissions_with_any_target_lab_24h", data.get("admissions_with_any_target_lab_window"))
    counts = pd.DataFrame(data["lab_counts"])
    counts["window"] = label
    return int(any_lab), counts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    any_rows = []
    count_frames = []
    for label, path in WINDOWS.items():
        any_lab, counts = load_counts(label, path)
        any_rows.append({"lab_name": "any_target_lab", "window": label, "admissions": any_lab})
        count_frames.append(counts[["lab_name", "window", "row_count", "admissions"]])
    counts_long = pd.concat([pd.DataFrame(any_rows)] + count_frames, ignore_index=True)
    admissions = counts_long.pivot_table(index="lab_name", columns="window", values="admissions", aggfunc="first").reset_index()
    admissions["absolute_gain"] = admissions["m24_p24h"] - admissions["p0_p24h"]
    admissions["relative_gain_pct"] = admissions["absolute_gain"] / admissions["p0_p24h"] * 100
    admissions = admissions.sort_values("absolute_gain", ascending=False)

    out_csv = OUT_DIR / "mimic_lab_window_coverage_comparison.csv"
    admissions.to_csv(out_csv, index=False)

    lines = [
        "# MIMIC Lab Window Coverage Comparison",
        "",
        "Comparison of target laboratory coverage in MIMIC-IV diabetes admissions.",
        "",
        "| Lab | 0 to +24h admissions | -24 to +24h admissions | Absolute gain | Relative gain (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in admissions.iterrows():
        lines.append(
            f"| {row['lab_name']} | {int(row['p0_p24h'])} | {int(row['m24_p24h'])} | {int(row['absolute_gain'])} | {row['relative_gain_pct']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: expanding the window from 0/+24h to -24/+24h only modestly increases coverage. The main issue is measurement-process sparsity rather than a narrow baseline window.",
        ]
    )
    out_md = OUT_DIR / "README_mimic_lab_window_comparison.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(admissions.to_string(index=False))


if __name__ == "__main__":
    main()
