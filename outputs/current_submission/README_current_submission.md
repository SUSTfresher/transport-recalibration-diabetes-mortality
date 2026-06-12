# Current Submission Folder

This folder contains the current manuscript-facing materials only.

- docs: current three-scenario manuscript draft, citation insertion map, TRIPOD+AI checklist draft, and submission statements draft.
- tables: locked Table 1-6 files, Table 3 cross-scenario performance, and table README/metadata.
- figures: current Figure 1-3 exports in SVG, PDF, PNG, and TIFF.
- source_data: source-data CSV files for Figure 1-3.
- submission_audit_report.md: automated consistency audit for manuscript-facing files.

Large intermediate prediction files, temporary DuckDB files, old feasibility figures, old two-database tables, and model joblib files were removed from `outputs` to keep the workspace readable. Regenerate primary tables with `scripts/promote_unweighted_primary_tables.py` after rerunning the upstream analyses. Regenerate figures with `scripts/make_figure1_study_design.py`, `scripts/make_figure2_calibration_state_map.py`, and `scripts/make_figure3_event_count_recalibration.py`. Re-run the consistency audit with `scripts/audit_current_submission.py`.
