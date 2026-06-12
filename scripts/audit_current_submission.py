from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "outputs" / "current_submission"
DOC = SUB / "docs" / "transport_recalibration_three_scenario_manuscript_draft.md"
REPORT = SUB / "submission_audit_report.md"


REQUIRED_FILES = [
    "docs/transport_recalibration_three_scenario_manuscript_draft.md",
    "docs/citation_insertion_map_three_scenario.md",
    "docs/TRIPOD_AI_checklist_draft.md",
    "docs/submission_statements.md",
    "README_current_submission.md",
    "tables/Table_1_baseline_characteristics.csv",
    "tables/Table_2_source_classifier_auc.csv",
    "tables/Table_3_cross_scenario_transport_performance.csv",
    "tables/Table_3_cross_scenario_transport_performance_numeric.csv",
    "tables/Table_4_recalibration_by_event_count.csv",
    "tables/Table_4_recalibration_by_event_count_numeric_long.csv",
    "tables/Table_5_subgroup_transportability.csv",
    "tables/Table_6_decision_curve_selected_thresholds.csv",
    "source_data/Figure_1_source_data.csv",
    "source_data/Figure_2_source_data.csv",
    "source_data/Figure_3_source_data.csv",
]

FIGURE_BASES = {
    "Figure 1": "figures/figure1/Figure_1_study_design_source_shift",
    "Figure 2": "figures/figure2/Figure_2_calibration_state_map",
    "Figure 3": "figures/figure3/Figure_3_event_count_recalibration",
}

EXPECTED_NUMBERS = {
    "AUC 0.674": "0.674",
    "AUC 0.707": "0.707",
    "AUC 0.737": "0.737",
    "ECE 0.146": "0.146",
    "ECE 0.059": "0.059",
    "ECE 0.016": "0.016",
    "slope 0.497": "0.497",
    "slope 0.563": "0.563",
    "slope 1.198": "1.198",
}

STALE_PATTERNS = [
    "predicted recalibration difficulty",
    "degree of distribution shift",
    "primary class-balanced",
    "class-balanced primary",
    "failure-matched 100-event",
    "selected recalibration strategy for each direction",
    "0.349",
    "0.348",
    "0.668",
    "1.055",
    "0.449",
]


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def warn(ok: bool) -> str:
    return "PASS" if ok else "WARN"


def check_required_files(lines: list[str]) -> bool:
    lines.append("## Required Files\n")
    all_ok = True
    for rel in REQUIRED_FILES:
        path = SUB / rel
        ok = path.exists() and path.stat().st_size > 0
        all_ok &= ok
        size = path.stat().st_size if path.exists() else 0
        lines.append(f"- {status(ok)} `{rel}` ({size:,} bytes)")

    for fig, base in FIGURE_BASES.items():
        for ext in [".svg", ".pdf", ".png", ".tiff"]:
            rel = base + ext
            path = SUB / rel
            ok = path.exists() and path.stat().st_size > 0
            all_ok &= ok
            size = path.stat().st_size if path.exists() else 0
            lines.append(f"- {status(ok)} `{rel}` ({size:,} bytes)")
    lines.append("")
    return all_ok


def check_manuscript(lines: list[str]) -> bool:
    lines.append("## Manuscript Text Checks\n")
    text = DOC.read_text(encoding="utf-8")
    all_ok = True

    abstract = text.split("## Abstract", 1)[1].split("Keywords:", 1)[0]
    abstract_words = len([w for w in abstract.replace("\n", " ").split() if w])
    ok = abstract_words <= 150
    all_ok &= ok
    lines.append(f"- {status(ok)} abstract word count: {abstract_words}")

    for label, value in EXPECTED_NUMBERS.items():
        ok = value in text
        all_ok &= ok
        lines.append(f"- {status(ok)} manuscript contains {label}")

    for i in range(1, 7):
        ok = bool(re.search(rf"\bTable {i}\b", text))
        all_ok &= ok
        lines.append(f"- {status(ok)} Table {i} is referenced")

    for i in range(1, 4):
        ok = bool(re.search(rf"\bFigure {i}\b", text))
        all_ok &= ok
        lines.append(f"- {status(ok)} Figure {i} legend/reference is present")

    for pattern in STALE_PATTERNS:
        ok = pattern not in text
        all_ok &= ok
        lines.append(f"- {status(ok)} stale text absent: `{pattern}`")

    lines.append("")
    return all_ok


def check_tables(lines: list[str]) -> bool:
    lines.append("## Table Consistency Checks\n")
    all_ok = True

    table3 = pd.read_csv(SUB / "tables" / "Table_3_cross_scenario_transport_performance_numeric.csv")
    ok = len(table3) == 3 and set(table3["model"]) == {"Logistic regression (unweighted)"}
    all_ok &= ok
    lines.append(f"- {status(ok)} Table 3 has three unweighted primary rows")

    expected_dirs = {
        "NHANES -> MIMIC-IV",
        "MIMIC-IV ICU -> eICU",
        "eICU -> MIMIC-IV ICU",
    }
    ok = set(table3["direction"]) == expected_dirs
    all_ok &= ok
    lines.append(f"- {status(ok)} Table 3 directions match manuscript design")

    table4 = pd.read_csv(SUB / "tables" / "Table_4_recalibration_by_event_count_numeric_long.csv")
    expected_methods = {"raw", "intercept_only", "platt", "isotonic"}
    ok = set(table4["method"]) == expected_methods
    all_ok &= ok
    lines.append(f"- {status(ok)} Table 4 methods include raw/intercept-only/Platt/isotonic")

    raw_rows = table4[table4["method"] == "raw"]
    ok = len(raw_rows) == 3 and set(raw_rows["event_target"]) == {0}
    all_ok &= ok
    lines.append(f"- {status(ok)} Table 4 raw rows are event_target=0 for three directions")

    table6 = pd.read_csv(SUB / "tables" / "Table_6_decision_curve_selected_thresholds.csv")
    needed_cols = {
        "Raw transport logistic",
        "Intercept-only 100 events",
        "Platt 100 events",
        "Internal HGB benchmark",
    }
    ok = needed_cols.issubset(table6.columns) and len(table6) == 9
    all_ok &= ok
    lines.append(f"- {status(ok)} Table 6 has 3 directions x 3 thresholds and required strategies")

    supp = list((SUB / "tables").glob("Supplementary_Table_class_weighted_*.csv"))
    ok = len(supp) >= 4
    all_ok &= ok
    lines.append(f"- {status(ok)} class-weighted supplementary CSVs present ({len(supp)})")

    source_internal = SUB / "tables" / "Supplementary_Table_source_internal_calibration.csv"
    ok = source_internal.exists() and len(pd.read_csv(source_internal)) == 3
    all_ok &= ok
    lines.append(f"- {status(ok)} source-internal calibration supplementary table present")

    lines.append("")
    return all_ok


def check_figures(lines: list[str]) -> bool:
    lines.append("## Figure Source-Data Checks\n")
    all_ok = True

    expected_rows = {
        "Figure_1_source_data.csv": 19,
        "Figure_2_source_data.csv": 24,
        "Figure_3_source_data.csv": 96,
    }
    for name, n_rows in expected_rows.items():
        df = pd.read_csv(SUB / "source_data" / name)
        ok = len(df) == n_rows
        all_ok &= ok
        lines.append(f"- {status(ok)} {name} row count = {len(df)}")

    fig2 = pd.read_csv(SUB / "source_data" / "Figure_2_source_data.csv")
    ok = "Retain raw" in set(fig2["method"])
    all_ok &= ok
    lines.append(f"- {status(ok)} Figure 2 source data records retain-raw action")

    fig3 = pd.read_csv(SUB / "source_data" / "Figure_3_source_data.csv")
    dca_strategies = set(fig3.loc[fig3["panel"] == "3d", "method_or_strategy"])
    ok = {
        "Raw transport",
        "Intercept-only 100 events",
        "Platt 100 events",
        "Internal HGB",
    }.issubset(dca_strategies)
    all_ok &= ok
    lines.append(f"- {status(ok)} Figure 3 DCA source data contains all plotted strategies")

    lines.append("")
    return all_ok


def check_remaining_risks(lines: list[str]) -> None:
    lines.append("## Remaining Manual Items\n")
    items = [
        "Confirm journal-specific figure dimension and TIFF requirements before upload.",
        "Transfer TRIPOD_AI_checklist_draft.md into the journal's official checklist format.",
        "Confirm local ethics exemption/approval wording for secondary de-identified data analysis.",
        "Add public code repository URL and archive DOI, or decide whether these will be supplied at acceptance.",
        "Fill author initials, funding, competing interests, and acknowledgements in submission_statements.md.",
        "Confirm whether Supplementary Tables should be split into separate Excel sheets or uploaded as CSV/MD.",
        "Run a final human read for wording around endpoint differences and DCA threshold interpretation.",
    ]
    for item in items:
        lines.append(f"- TODO {item}")
    lines.append("")


def main() -> None:
    lines: list[str] = [
        "# Current Submission Audit Report",
        "",
        "Scope: manuscript-facing files in `outputs/current_submission`.",
        "",
    ]
    results = [
        check_required_files(lines),
        check_manuscript(lines),
        check_tables(lines),
        check_figures(lines),
    ]
    check_remaining_risks(lines)

    overall = "PASS" if all(results) else "FAIL"
    lines.insert(2, f"Overall status: **{overall}**")
    lines.insert(3, "")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote audit report to {REPORT}")
    print(f"Overall status: {overall}")


if __name__ == "__main__":
    main()
