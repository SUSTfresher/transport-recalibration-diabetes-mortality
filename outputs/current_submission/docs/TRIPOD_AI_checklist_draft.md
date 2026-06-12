# TRIPOD+AI Checklist Draft

This working checklist maps the current manuscript to TRIPOD+AI reporting domains. It is intended as a submission-preparation aid; transfer the final answers to the journal's required checklist format before upload.

| Item | Domain | Current status | Manuscript location | Notes / actions |
|---|---|---|---|---|
| 1 | Title | Complete | Title | Identifies transport failure, recalibration, diabetes mortality prediction, and three database settings. |
| 2 | Abstract | Complete | Abstract | Structured concepts are present in unstructured abstract: source/target settings, transport directions, primary findings, endpoint caveat, and recalibration implication. |
| 3a | Background and rationale | Complete | Introduction paragraphs 1-3 | Explains model transport, calibration, dataset shift, and clinical threshold relevance. |
| 3b | Objectives | Complete | Introduction final paragraph | States discrimination, calibration failure mode, and local recalibration questions. |
| 4a | Data sources and study setting | Complete | Methods: Data sources | Describes NHANES, MIMIC-IV, and eICU sources with versions/citations. |
| 4b | Eligibility criteria | Complete | Methods: Data sources; Participants/features | Diabetes cohorts, ICU inclusion, first ICU stay rule, and known endpoint status described. |
| 4c | Study design and transport setting | Complete | Methods: Study design; Figure 1 | Three transport directions and endpoint-specific scenario design described. |
| 5a | Outcome definition | Complete | Methods: Data sources; Endpoint alignment | One-year mortality and hospital mortality definitions are explicit. |
| 5b | Outcome ascertainment timing | Complete | Methods: Data sources; Endpoint alignment | NHANES 12-month mortality and MIMIC/eICU hospital mortality windows described. |
| 6a | Candidate predictors/features | Complete | Methods: Participants and feature sets | NHANES-compatible and ICU-native feature sets described. |
| 6b | Predictor measurement timing | Complete | Methods: Participants and feature sets | First 24-hour ICU window and eICU offset-time extraction described. |
| 6c | Predictor handling and harmonization | Complete | Methods: Participants and feature sets; Missing data | Notes comorbidity harmonization, albumin/temperature exclusion, imputation. |
| 7 | Sample size | Complete | Methods: Study design; Results first paragraph | Existing data inclusion, event counts, holdout splits, and recalibration event targets reported. |
| 8 | Missing data | Complete | Methods: Missing data and preprocessing; Table 1 | Median/mode imputation and coverage-driven exclusions described. |
| 9a | Model type and training | Complete | Methods: Model development and transport evaluation | Logistic regression primary, HGB benchmark, scikit-learn pipelines, fixed seeds described. |
| 9b | Hyperparameters and tuning | Complete | Methods: Model development and transport evaluation | HGB hyperparameters stated; logistic model class stated. |
| 9c | Class imbalance handling | Complete | Methods: Model development and transport evaluation; Results sensitivity paragraph | Unweighted logistic primary; class-weighted logistic retained as sensitivity. |
| 10a | Validation/transport design | Complete | Methods: Study design; Model development and transport evaluation | Source training and target transport evaluation described for each scenario. |
| 10b | Internal benchmark | Complete | Methods: Model development and transport evaluation; Table 6 | Internal target-site HGB benchmark included for DCA comparison. |
| 11a | Performance measures | Complete | Methods: Performance and calibration metrics | AUC, average precision, Brier, mean risk, ECE, slope/intercept described. |
| 11b | Calibration assessment | Complete | Methods: Performance and calibration metrics; Figure 2 | Calibration equation, ideal slope/intercept, ECE binning, and equal-width robustness check described. |
| 11c | Clinical utility | Complete | Methods: Decision-curve analysis; Table 6; Figure 3 | DCA thresholds and interpretation as high-risk review cutoffs described. |
| 12a | Recalibration/model updating | Complete | Methods: Local recalibration experiments | Intercept-only, Platt, isotonic, event targets, repeated sampling described. |
| 12b | Recalibration sample separation | Complete | Methods: Local recalibration experiments | NHANES-to-MIMIC and ICU calibration/evaluation separation described. |
| 13 | Statistical uncertainty | Complete | Methods: Statistical uncertainty | Bootstrap CIs and empirical recalibration intervals described. |
| 14 | Participant flow / cohort description | Complete | Results first paragraph; Table 1; Figure 1 | Cohort sizes and event counts reported for all data sources and transport directions. |
| 15 | Model performance results | Complete | Results; Tables 3-6; Figures 2-3 | Transport, recalibration, subgroup, and DCA results reported. |
| 16 | Subgroup or heterogeneity analyses | Complete | Results subgroup paragraph; Table 5 | Age, CKD, and CVD subgroup analyses reported with prevalence caveat. |
| 17 | Sensitivity analyses | Complete | Results class-weighting paragraph; Supplementary Tables | Class-weighted sensitivity and equal-width ECE robustness are reported. |
| 18 | Interpretation | Complete | Discussion paragraphs 1-3 | Diagnostic-repair framework and distinction from source classifier AUC described. |
| 19 | Limitations | Complete | Discussion limitations paragraph | Endpoint differences, ECE binning, class weighting, subgroup prevalence, and secondary-data limits described. |
| 20 | Generalizability and transportability | Complete | Discussion; endpoint-difference paragraph | Cross-scenario interpretation limits and ICU directional asymmetry discussed. |
| 21 | Clinical implications | Complete | Discussion; Decision-curve methods/results | Emphasizes diagnostic workflow and local event-count scale; not a deployed clinical system. |
| 22 | Patient/public involvement | Not applicable / confirm | Not currently included | Secondary database study. Confirm whether journal requires a statement. |
| 23 | Ethics and data governance | Needs author confirmation | Submission statements draft | Draft text prepared; local institutional exemption/approval wording must be confirmed. |
| 24 | Data availability | Partial | Data availability; submission statements draft | Public/restricted data routes described. Add repository links for source data package if journal requires separate upload metadata. |
| 25 | Code availability | Complete | Code availability; submission statements draft | Public GitHub repository and Zenodo archive DOI provided. |
| 26 | Conflicts, funding, author contributions | Needs author confirmation | Submission statements draft | Placeholders prepared; author initials, funding, and competing interests must be finalized. |
| 27 | Deployment/human-AI interaction | Complete / not applicable | Methods: Statistical uncertainty and reproducibility paragraph | Retrospective research models only; no UI, alerting logic, or clinical workflow integration implemented. |

## Open Items Before Upload

- Transfer this mapping into the journal's official TRIPOD+AI checklist template.
- Confirm local ethics/exemption wording for secondary de-identified data analysis.
- Fill author contributions, funding, competing interests, and acknowledgements.
- Confirm whether patient/public involvement statement is required by the target journal.
