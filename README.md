# Transport recalibration in diabetes mortality prediction

This repository contains the analysis code and manuscript-facing reproducibility materials for a three-database study of clinical AI transportability and local recalibration in diabetes mortality prediction.

The study evaluates model transport across:

- NHANES to MIMIC-IV as an extreme population-survey-to-hospital stress test.
- MIMIC-IV ICU to eICU as a realistic ICU-to-ICU deployment scenario.
- eICU to MIMIC-IV ICU as the reverse ICU deployment scenario.

The central analysis asks whether transported models retain discrimination, how calibration fails, and how many local target-site outcome events are needed to choose an appropriate recalibration strategy.

## Repository Contents

- `scripts/`: cohort construction, source-shift assessment, transport evaluation, recalibration simulations, decision-curve analysis, subgroup analysis, manuscript-table generation, figure generation, and audit scripts.
- `outputs/current_submission/docs/`: current manuscript draft and supporting submission documents.
- `outputs/current_submission/`: manuscript-facing materials only, including current tables, lightweight figure exports, and figure source data.
- `requirements.txt`: Python package requirements used by the analysis scripts.

## Data Access

This repository does not redistribute restricted clinical data or patient-level derived datasets.

- NHANES data are publicly available from the US National Center for Health Statistics.
- MIMIC-IV and eICU are available through PhysioNet to credentialed users who complete the required training and data-use agreements.
- Local patient-level processed files under `data/` are intentionally excluded from version control.
- Large intermediate prediction files and temporary analysis outputs are intentionally excluded from version control.

The included figure source-data files and manuscript tables are aggregate manuscript-facing outputs, not restricted source data.

## Environment

Create a Python environment and install the required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Core package requirements are listed in `requirements.txt`.

## Reproducibility Notes

The analysis scripts assume that credentialed MIMIC-IV/eICU files and processed local cohorts are available in the local paths configured inside the scripts or adapted by the user. Restricted source files are not included here.

Common entry points:

```powershell
# Generate or promote current manuscript tables
.\.venv\Scripts\python.exe scripts\promote_unweighted_primary_tables.py

# Regenerate the three main figures
.\.venv\Scripts\python.exe scripts\make_figure1_study_design.py
.\.venv\Scripts\python.exe scripts\make_figure2_calibration_state_map.py
.\.venv\Scripts\python.exe scripts\make_figure3_event_count_recalibration.py

# Audit the manuscript-facing submission folder
.\.venv\Scripts\python.exe scripts\audit_current_submission.py
```

## Current Submission Materials

The current manuscript-facing package is located at:

```text
outputs/current_submission/
```

The latest audit report is:

```text
outputs/current_submission/submission_audit_report.md
```

## Code Availability

Public GitHub repository:

```text
https://github.com/SUSTfresher/transport-recalibration-diabetes-mortality
```

Archived release:

```text
https://doi.org/10.5281/zenodo.20657894
```
