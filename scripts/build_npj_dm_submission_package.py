from __future__ import annotations

import csv
import re
import shutil
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "outputs" / "current_submission"
TARGET = ROOT / "outputs" / "npj_dm_submission_ready"

MANUSCRIPT = CURRENT / "docs" / "transport_recalibration_three_scenario_manuscript_draft.md"
TABLE_DIR = CURRENT / "tables"
FIG_DIR = CURRENT / "figures"
SOURCE_DATA_DIR = CURRENT / "source_data"

PANDOC = shutil.which("pandoc")
SOFFICE = shutil.which("soffice") or shutil.which("libreoffice")

ARTICLE_TITLE = "Diagnosing and repairing transport failure in diabetes mortality prediction across clinical databases"
RUNNING_TITLE = "Transport failure in mortality prediction"
ARTICLE_TYPE = "Article"

AUTHORS = "Mingwei Wang^1, Haozhen Liu^1, Peng Dong^2, Jingxin Tang^1, Chenyue Guo^1, Yifan Yan^1, Yixuan Zhang^1"
AFFILIATIONS = (
    "1. School of Electronic Information and Artificial Intelligence, Shaanxi University of Science and Technology, "
    "Xi'an, Shaanxi 710021, China\n"
    "2. Department of Endocrinology, the Second Affiliated Hospital, Xi'an Jiaotong University, "
    "Xi'an, Shaanxi 710049, China"
)
CORRESPONDING = (
    "Peng Dong, Department of Endocrinology, the Second Affiliated Hospital, Xi'an Jiaotong University, "
    "Xi'an, Shaanxi 710049, China. Email: dongpeng1807@xjtu.edu.cn"
)
AUTHOR_EMAILS = (
    "Mingwei Wang: wangmingwei@sust.edu.cn; Haozhen Liu: 251612059@sust.edu.cn; "
    "Peng Dong: dongpeng1807@xjtu.edu.cn; Jingxin Tang: 3346652875@qq.com; "
    "Chenyue Guo: 3307849257@qq.com; Yifan Yan: 627229346@qq.com; "
    "Yixuan Zhang: 3113482059@qq.com"
)
KEYWORDS = "clinical AI; transportability; calibration; recalibration; dataset shift; diabetes; mortality prediction; MIMIC-IV; eICU; NHANES"

FUNDING_TEXT = (
    "This work was supported by the Qinchuangyuan Scientists and Engineers Team Construction Project of the "
    "Shaanxi Provincial Department of Science and Technology (2024QCY-KXJ-181), the International Science and "
    "Technology Cooperation Program of Shaanxi Province (China-Iran) (2024GH-YBXM-06), the University and "
    "Research Institute Service Enterprise Project of Xi'an Municipal Science and Technology Bureau "
    "(25GXKJRC00055), and the Technology Innovation Guidance Program (Fund) of the Shaanxi Provincial "
    "Department of Science and Technology (2025YFBT-22-01/02, 2025ZC-SXFF3-11/12/13, "
    "2025ZC-SXFF3-35/36/37). The funders had no role in study design, data collection, analysis, "
    "interpretation, manuscript preparation, or the decision to submit the article for publication."
)

AUTHOR_CONTRIBUTIONS = (
    "M.W. and P.D. conceived and supervised the study. H.L. performed data curation, cohort construction, "
    "statistical analysis, model development, recalibration simulations, visualization, and drafted the manuscript. "
    "P.D. provided clinical interpretation and critically revised the manuscript for important intellectual content. "
    "J.T., C.G., Y.Y., and Y.Z. contributed to data checking, literature review, result verification, and manuscript "
    "revision. M.W. provided methodological supervision and critical revision. All authors reviewed and approved the "
    "final manuscript and agree to be accountable for the integrity of the work."
)

COMPETING_INTERESTS = "All authors declare no financial or non-financial competing interests."

ACKNOWLEDGEMENTS = (
    "The authors thank the developers and maintainers of NHANES, MIMIC-IV, eICU, and PhysioNet for making these "
    "resources available to the research community. " + FUNDING_TEXT
)

AI_DISCLOSURE = (
    "During manuscript preparation, the authors used ChatGPT and Codex to support language clarity, structural "
    "compression, code review, figure-generation workflow organization, consistency checks, and submission-package "
    "formatting. The authors reviewed and edited all AI-assisted output and take full responsibility for the content "
    "of the publication."
)

ETHICS_TEXT = (
    "This study used de-identified public or credentialed-access secondary data. NHANES protocols were approved by "
    "the National Center for Health Statistics Research Ethics Review Board, and participant consent was obtained by "
    "the original study. MIMIC-IV and eICU are de-identified databases available through PhysioNet under required "
    "credentialing, data-use training, and data-use agreements. No new human participant recruitment, intervention, "
    "or patient-care decision was conducted for this retrospective secondary analysis."
)


def reset_target() -> None:
    managed = [
        "01_manuscript",
        "02_submission_text",
        "03_main_figures",
        "04_supplementary_information",
        "05_source_data",
        "06_reporting_checklists",
        "07_admin_checklist",
    ]
    TARGET.mkdir(parents=True, exist_ok=True)
    for sub in managed:
        path = TARGET / sub
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def section(text: str, start: str, end: str | None = None) -> str:
    pattern = rf"^## {re.escape(start)}\s*$"
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"Missing section {start}")
    start_idx = m.end()
    if end is None:
        end_idx = len(text)
    else:
        n = re.search(rf"^## {re.escape(end)}\s*$", text[start_idx:], flags=re.MULTILINE)
        if not n:
            raise ValueError(f"Missing section {end}")
        end_idx = start_idx + n.start()
    return text[start_idx:end_idx].strip()


def abstract_text(text: str) -> str:
    abstract = section(text, "Abstract", "Introduction")
    abstract = re.split(r"^Keywords:", abstract, flags=re.MULTILINE)[0]
    return abstract.strip()


def inject_ethics(methods: str) -> str:
    ethics_section = f"### Ethics and data governance\n\n{ETHICS_TEXT}\n\n"
    if "### Ethics and data governance" in methods:
        return methods
    return methods.replace("### Endpoint alignment", ethics_section + "### Endpoint alignment")


def csv_to_markdown(path: Path) -> str:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows[1:]:
        row = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(cell.replace("\n", " ") for cell in row[: len(header)]) + " |")
    return "\n".join(lines)


def clean_existing_table_md(path: Path, title: str) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return f"## {title}\n" + "\n".join(lines).strip() + "\n"


def rows_to_markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        row = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row[: len(header)]) + " |")
    return "\n".join(lines)


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fmt_intish(value: str) -> str:
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return value


def table3_markdown() -> str:
    rows = []
    for row in read_dict_rows(TABLE_DIR / "Table_3_cross_scenario_transport_performance.csv"):
        rows.append(
            [
                row["Direction"],
                row["Endpoint"],
                f'{row["Target evaluation N"]} ({row["Target evaluation events"]})',
                row["Event rate"],
                row["AUC"],
                row["ECE"],
                row["Calibration slope"],
                row["Calibration intercept"],
            ]
        )

    notes = (
        "Important endpoint note: the NHANES -> MIMIC-IV stress test uses a one-year mortality endpoint, whereas "
        "the MIMIC-IV ICU <-> eICU deployment analyses use harmonized hospital mortality. Rows should be compared "
        "as endpoint-specific deployment scenarios; the cross-scenario contrast is intended to compare transport "
        "failure modes and recalibration needs, not endpoint-equivalent absolute accuracy. All displayed intervals "
        "are percentile 95% confidence intervals from 1,000 bootstrap resamples of the target evaluation cohort. "
        "Full performance metrics, including mean predicted risk, PR AUC, Brier score, and equal-width ECE, are "
        "provided in Supplementary Data 1."
    )
    return (
        "## Table 3 | Cross-scenario transport performance\n\n"
        + rows_to_markdown(
            [
                "Direction",
                "Endpoint",
                "Target N (events)",
                "Event rate",
                "AUC",
                "ECE",
                "Calibration slope",
                "Calibration intercept",
            ],
            rows,
        )
        + "\n\n"
        + notes
        + "\n"
    )


def table4_markdown() -> str:
    all_rows = read_dict_rows(TABLE_DIR / "Table_4_recalibration_by_event_count_numeric_long.csv")
    directions = []
    for row in all_rows:
        if row["Direction"] not in directions:
            directions.append(row["Direction"])

    parts = [
        "## Table 4 | Event-count recalibration by direction and method",
        "",
    ]
    panel_labels = ["a", "b", "c"]
    for idx, direction in enumerate(directions):
        panel_rows = []
        for row in all_rows:
            if row["Direction"] != direction:
                continue
            panel_rows.append(
                [
                    row["Method"],
                    row["Local outcome events"],
                    row["ECE (95% empirical interval)"],
                    fmt_intish(row["calibration_n_mean"]),
                    row["n_repeats"],
                ]
            )
        label = panel_labels[idx] if idx < len(panel_labels) else str(idx + 1)
        parts.extend(
            [
                f"**Panel {label}. {direction}**",
                "",
                rows_to_markdown(
                    ["Method", "Local events", "ECE (95% empirical interval)", "Mean calibration N", "Repeats"],
                    panel_rows,
                ),
                "",
            ]
        )

    parts.append(
        "Primary model is unweighted logistic regression. Raw transport is a point estimate before local "
        "recalibration; interval columns apply to repeated recalibration samples. Full recalibration summaries, "
        "including equal-width ECE and calibration-slope intervals, are provided in Supplementary Data 1."
    )
    return "\n".join(parts) + "\n"


def table5_markdown() -> str:
    all_rows = read_dict_rows(TABLE_DIR / "Table_5_subgroup_transportability.csv")
    directions = []
    for row in all_rows:
        if row["Direction"] not in directions:
            directions.append(row["Direction"])

    parts = [
        "## Table 5 | Subgroup transportability by age CKD and CVD",
        "",
    ]
    panel_labels = ["a", "b", "c"]
    for idx, direction in enumerate(directions):
        panel_rows = []
        for row in all_rows:
            if row["Direction"] != direction:
                continue
            panel_rows.append(
                [
                    f'{row["Subgroup type"]}: {row["Subgroup"]}',
                    f'{row["N"]} ({row["Events"]})',
                    row["Event rate"],
                    row["AUC"],
                    row["ECE"],
                    row["Calibration slope"],
                ]
            )
        label = panel_labels[idx] if idx < len(panel_labels) else str(idx + 1)
        parts.extend(
            [
                f"**Panel {label}. {direction}**",
                "",
                rows_to_markdown(
                    ["Subgroup", "N (events)", "Event rate", "AUC", "ECE", "Calibration slope"],
                    panel_rows,
                ),
                "",
            ]
        )

    parts.append(
        "Primary model is unweighted logistic regression. Rows report raw transported performance within target-site "
        "subgroups. ECE is 10-bin equal-frequency expected calibration error. Subgroup ECE is descriptive and may "
        "partly reflect subgroup event-rate differences. Full subgroup metrics, including equal-width ECE, are "
        "provided in Supplementary Data 1."
    )
    return "\n".join(parts) + "\n"


def table6_markdown() -> str:
    all_rows = read_dict_rows(TABLE_DIR / "Table_6_decision_curve_selected_thresholds.csv")
    directions = []
    for row in all_rows:
        if row["Direction"] not in directions:
            directions.append(row["Direction"])

    parts = [
        "## Table 6 | Decision-curve net benefit at selected thresholds",
        "",
    ]
    panel_labels = ["a", "b", "c"]
    for idx, direction in enumerate(directions):
        panel_rows = []
        for row in all_rows:
            if row["Direction"] != direction:
                continue
            panel_rows.append(
                [
                    row["Threshold"],
                    row["Treat none"],
                    row["Treat all"],
                    row["Raw transport logistic"],
                    row["Intercept-only 100 events"],
                    row["Platt 100 events"],
                    row["Internal HGB benchmark"],
                ]
            )
        label = panel_labels[idx] if idx < len(panel_labels) else str(idx + 1)
        parts.extend(
            [
                f"**Panel {label}. {direction}**",
                "",
                rows_to_markdown(
                    [
                        "Threshold",
                        "Treat none",
                        "Treat all",
                        "Raw transport",
                        "Intercept-only",
                        "Platt",
                        "Internal HGB",
                    ],
                    panel_rows,
                ),
                "",
            ]
        )

    parts.append(
        "Net benefit is reported at thresholds 0.20, 0.25, and 0.30. Intercept-only and Platt columns use "
        "100 local outcome events. Internal HGB benchmark is the target-site histogram-gradient-boosting model. "
        "Thresholds are illustrative cutoffs for high-risk clinical review rather than validated treatment "
        "thresholds."
    )
    return "\n".join(parts) + "\n"


def build_tables_markdown() -> str:
    parts = [
        clean_existing_table_md(TABLE_DIR / "Table_1_baseline_characteristics.md", "Table 1 | Baseline characteristics of the three diabetes cohorts"),
        clean_existing_table_md(TABLE_DIR / "Table_2_source_classifier_auc.md", "Table 2 | Source-classifier AUCs by scenario and feature block"),
        table3_markdown(),
        table4_markdown(),
        table5_markdown(),
        table6_markdown(),
    ]
    return "\n\n".join(parts)


def build_main_manuscript() -> str:
    raw = MANUSCRIPT.read_text(encoding="utf-8")
    abstract = abstract_text(raw)
    intro = section(raw, "Introduction", "Results")
    results = section(raw, "Results", "Discussion")
    discussion = section(raw, "Discussion", "Methods")
    methods = inject_ethics(section(raw, "Methods", "Data availability"))
    data_avail = section(raw, "Data availability", "Code availability")
    code_avail = section(raw, "Code availability", "Tables")
    fig_legends = section(raw, "Figure legends", "References")
    references = section(raw, "References", None)

    title_page = f"""# {ARTICLE_TITLE}

Article type: {ARTICLE_TYPE}

Running title: {RUNNING_TITLE}

Authors: {AUTHORS}

Affiliations:

{AFFILIATIONS}

Corresponding author: {CORRESPONDING}

Author emails: {AUTHOR_EMAILS}

"""

    front = f"""{title_page}
## Abstract

{abstract}

## Introduction

{intro}
"""
    body = f"""
## Results

{results}

## Discussion

{discussion}

## Methods

{methods}

## Data availability

{data_avail}

## Code availability

{code_avail}

## Acknowledgements

{ACKNOWLEDGEMENTS}

## Author contributions

{AUTHOR_CONTRIBUTIONS}

## Competing interests

{COMPETING_INTERESTS}

## AI assistance disclosure

{AI_DISCLOSURE}

## References

{references}

## Figure legends and embedded figures

{fig_legends}

![Figure 1](../03_main_figures/Figure_1_study_design_source_shift.png)

![Figure 2](../03_main_figures/Figure_2_calibration_state_map.png)

![Figure 3](../03_main_figures/Figure_3_event_count_recalibration.png)

## Tables

{build_tables_markdown()}
"""
    return front + body


def write_text_files(main_md: str) -> None:
    manuscript_dir = TARGET / "01_manuscript"
    text_dir = TARGET / "02_submission_text"

    (manuscript_dir / "NPJDM_main_manuscript_with_tables_and_figures.md").write_text(main_md, encoding="utf-8")

    title_page = f"""Title: {ARTICLE_TITLE}
Article type: {ARTICLE_TYPE}
Running title: {RUNNING_TITLE}
Authors: {AUTHORS}
Affiliations: {AFFILIATIONS.replace(chr(10), '; ')}
Corresponding author: {CORRESPONDING}
Author emails: {AUTHOR_EMAILS}
Keywords: {KEYWORDS}
"""
    (text_dir / "NPJDM_title_page_information.txt").write_text(title_page, encoding="utf-8")

    abstract = abstract_text(MANUSCRIPT.read_text(encoding="utf-8"))
    data_avail = section(MANUSCRIPT.read_text(encoding="utf-8"), "Data availability", "Code availability")
    code_avail = section(MANUSCRIPT.read_text(encoding="utf-8"), "Code availability", "Tables")
    submission_text = f"""# Submission System Text

## Article type
{ARTICLE_TYPE}

## Title
{ARTICLE_TITLE}

## Abstract
{abstract}

Abstract word count: {len(abstract.split())}

## Keywords
{KEYWORDS}

## Data availability
{data_avail}

## Code availability
{code_avail}

## Ethics
{ETHICS_TEXT}

## Acknowledgements and funding
{ACKNOWLEDGEMENTS}

## Author contributions
{AUTHOR_CONTRIBUTIONS}

## Competing interests
{COMPETING_INTERESTS}

## AI assistance disclosure
{AI_DISCLOSURE}
"""
    (text_dir / "NPJDM_submission_system_text_to_paste.md").write_text(submission_text, encoding="utf-8")

    cover = f"""Dear Editors,

Please consider our Article, "{ARTICLE_TITLE}", for publication in npj Digital Medicine.

This study addresses a practical barrier in clinical AI deployment: transported models may retain discrimination while failing on the probability scale. Using NHANES, MIMIC-IV, and eICU, we evaluate diabetes mortality model transport across an extreme population-to-hospital stress test and a bidirectional ICU-to-ICU deployment scenario. The central contribution is an operational diagnostic sequence that separates source-shift detection from probability-scale diagnosis and uses calibration slope, intercept, and local event-count experiments to guide recalibration strategy.

The manuscript is, to our knowledge, not under consideration elsewhere. All authors have approved the submission. The analysis uses de-identified public or credentialed-access secondary data, and no new human participant recruitment or intervention was conducted. Code is publicly available at https://github.com/SUSTfresher/transport-recalibration-diabetes-mortality and archived at https://doi.org/10.5281/zenodo.20657894.

Sincerely,

Peng Dong
Corresponding author
Department of Endocrinology, the Second Affiliated Hospital, Xi'an Jiaotong University
Email: dongpeng1807@xjtu.edu.cn
"""
    (text_dir / "NPJDM_cover_letter.md").write_text(cover, encoding="utf-8")

    competing = f"# Competing interests\n\n{COMPETING_INTERESTS}\n"
    (text_dir / "NPJDM_competing_interests_statement.md").write_text(competing, encoding="utf-8")


def copy_figures() -> None:
    target = TARGET / "03_main_figures"
    figure_map = {
        "figure1": "Figure_1_study_design_source_shift",
        "figure2": "Figure_2_calibration_state_map",
        "figure3": "Figure_3_event_count_recalibration",
    }
    for subdir, base in figure_map.items():
        for ext in [".png", ".pdf", ".svg"]:
            src = FIG_DIR / subdir / f"{base}{ext}"
            if src.exists():
                shutil.copy2(src, target / src.name)


def build_supplementary_information() -> str:
    supp_parts = [
        "# Supplementary Information",
        "",
        f"Supplementary information for {ARTICLE_TITLE}.",
        "",
        "No Supplementary Methods are provided; all Methods are reported in the main manuscript.",
        "",
        clean_existing_table_md(TABLE_DIR / "Supplementary_Table_source_internal_calibration.md", "Supplementary Table 1 | Source-internal calibration checks"),
        clean_existing_table_md(TABLE_DIR / "Supplementary_Table_class_weight_sensitivity.md", "Supplementary Table 2 | Class-weighted versus unweighted logistic transport sensitivity"),
        clean_existing_table_md(TABLE_DIR / "Supplementary_Table_class_weighted_transport_performance.md", "Supplementary Table 3 | Class-weighted transport performance"),
        clean_existing_table_md(TABLE_DIR / "Supplementary_Table_class_weighted_recalibration_by_event_count_panel.md", "Supplementary Table 4 | Class-weighted event-count recalibration"),
        clean_existing_table_md(TABLE_DIR / "Supplementary_Table_class_weighted_subgroup_transportability.md", "Supplementary Table 5 | Class-weighted subgroup transportability"),
        clean_existing_table_md(TABLE_DIR / "Supplementary_Table_class_weighted_decision_curve_selected_thresholds.md", "Supplementary Table 6 | Class-weighted decision-curve sensitivity"),
    ]
    return "\n\n".join(supp_parts)


def copy_source_data_and_tables() -> None:
    target = TARGET / "05_source_data"
    for path in SOURCE_DATA_DIR.glob("*.csv"):
        shutil.copy2(path, target / path.name)
    for path in TABLE_DIR.glob("*.csv"):
        if "bootstrap_long" in path.name:
            continue
        shutil.copy2(path, target / path.name)

    zip_path = target / "Supplementary_Data_1_source_data_and_tables.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(target.glob("*.csv")):
            zf.write(path, arcname=path.name)


def write_admin_files(main_md: str, supp_md: str) -> None:
    check_dir = TARGET / "07_admin_checklist"
    abstract = abstract_text(MANUSCRIPT.read_text(encoding="utf-8"))
    title_words = ARTICLE_TITLE.split()
    figure_legend_words = []
    for line in section(MANUSCRIPT.read_text(encoding="utf-8"), "Figure legends", "References").splitlines():
        if line.startswith("**Figure"):
            figure_legend_words.append(len(re.sub(r"[*|]", "", line).split()))

    checklist = f"""# npj Digital Medicine Submission Checklist

Generated for Article submission.

Format basis: npj Digital Medicine content-type guidance and submission guidelines.

Official sources checked:

- https://www.nature.com/npjdigitalmed/content-types
- https://www.nature.com/npjdigitalmed/for-authors-and-referees/submission-guidelines

## Official format checks

- PASS Article title has {len(title_words)} words and no punctuation: {ARTICLE_TITLE}
- PASS Abstract has {len(abstract.split())} words and no subheadings.
- PASS Introduction has no subheadings.
- PASS Results uses subheadings.
- PASS Discussion has no subheadings and includes limitations within the text, with no separate Limitations or Conclusions section.
- PASS Methods uses subheadings and includes ethics/data-governance wording.
- PASS Data availability section is present.
- PASS Code availability section is present with GitHub and Zenodo DOI.
- PASS Acknowledgements include funding and funder-role statement; no separate Funding section is used.
- PASS Author contributions use initials and refer to all authors.
- PASS Competing interests statement is present.
- PASS References count is 38, below the npj Article guide limit of 60.
- PASS Figure legends word counts are all below 350: {figure_legend_words}.
- PASS Main tables are placed at the end of the text document.
- PASS Main display tables use Word-safe compact layouts; full numeric columns are retained in Supplementary Data 1.
- PASS Supplementary Information is combined into one file, with no Supplementary Methods.

## Files to upload

- `01_manuscript/NPJDM_main_manuscript_with_tables_and_figures.docx`
- `02_submission_text/NPJDM_cover_letter.md`
- `03_main_figures/Figure_1_study_design_source_shift.pdf`
- `03_main_figures/Figure_2_calibration_state_map.pdf`
- `03_main_figures/Figure_3_event_count_recalibration.pdf`
- `04_supplementary_information/NPJDM_supplementary_information.pdf`
- `05_source_data/Supplementary_Data_1_source_data_and_tables.zip`
- `06_reporting_checklists/TRIPOD_AI_checklist_draft.md`

## Manual confirmation before upload

- Confirm all authors and affiliations are final.
- Confirm local institutional policy does not require an additional exemption letter for this secondary analysis.
- Confirm the journal accepts the AI assistance disclosure wording in the manuscript or cover-letter metadata.
- If the submission system requests separate source-data files instead of a ZIP, upload the CSV files in `05_source_data` individually.
- If the submission system requires Nature reference style at first submission, convert the numbered reference list using Zotero/EndNote before upload.
"""
    (check_dir / "NPJDM_submission_checklist.md").write_text(checklist, encoding="utf-8")

    readme = f"""# npj Digital Medicine Submission Package

This folder contains a submission-ready package formatted for an npj Digital Medicine Article.

The package was generated from:

```text
outputs/current_submission
```

Primary files:

- `01_manuscript`: main manuscript in Markdown and Word format.
- `02_submission_text`: title page information, cover letter, and text blocks for the submission system.
- `03_main_figures`: separate figure files in PDF, PNG, and SVG.
- `04_supplementary_information`: single Supplementary Information file.
- `05_source_data`: source-data and table CSVs plus a zipped Supplementary Data package.
- `06_reporting_checklists`: TRIPOD+AI checklist draft.
- `07_admin_checklist`: npj DM format checklist and remaining upload notes.
"""
    (TARGET / "README_NPJDM_SUBMISSION_PACKAGE.md").write_text(readme, encoding="utf-8")


def convert_with_pandoc(md_path: Path, docx_path: Path, resource_paths: list[Path] | None = None) -> bool:
    if not PANDOC:
        return False
    cmd = [PANDOC, str(md_path), "-f", "gfm", "-t", "docx", "-o", str(docx_path)]
    if resource_paths:
        cmd.extend(["--resource-path", ";".join(str(p) for p in resource_paths)])
    subprocess.run(cmd, check=True, cwd=TARGET)
    return True


def convert_docx_to_pdf(docx_path: Path, out_dir: Path) -> bool:
    if not SOFFICE:
        return False
    subprocess.run(
        [SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(docx_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return True


def convert_outputs() -> None:
    manuscript_md = TARGET / "01_manuscript" / "NPJDM_main_manuscript_with_tables_and_figures.md"
    manuscript_docx = TARGET / "01_manuscript" / "NPJDM_main_manuscript_with_tables_and_figures.docx"
    convert_with_pandoc(
        manuscript_md,
        manuscript_docx,
        [TARGET, TARGET / "01_manuscript", TARGET / "03_main_figures"],
    )

    supp_md = TARGET / "04_supplementary_information" / "NPJDM_supplementary_information.md"
    supp_docx = TARGET / "04_supplementary_information" / "NPJDM_supplementary_information.docx"
    if convert_with_pandoc(supp_md, supp_docx, [TARGET, TARGET / "04_supplementary_information"]):
        convert_docx_to_pdf(supp_docx, TARGET / "04_supplementary_information")


def copy_reporting_checklists() -> None:
    target = TARGET / "06_reporting_checklists"
    shutil.copy2(CURRENT / "docs" / "TRIPOD_AI_checklist_draft.md", target / "TRIPOD_AI_checklist_draft.md")


def main() -> None:
    reset_target()
    copy_figures()
    main_md = build_main_manuscript()
    write_text_files(main_md)

    supp_md = build_supplementary_information()
    (TARGET / "04_supplementary_information" / "NPJDM_supplementary_information.md").write_text(supp_md, encoding="utf-8")

    copy_source_data_and_tables()
    copy_reporting_checklists()
    write_admin_files(main_md, supp_md)
    convert_outputs()

    print(f"Wrote npj Digital Medicine submission package to {TARGET}")
    print(f"Pandoc: {'found' if PANDOC else 'missing'}")
    print(f"LibreOffice: {'found' if SOFFICE else 'missing'}")


if __name__ == "__main__":
    main()
