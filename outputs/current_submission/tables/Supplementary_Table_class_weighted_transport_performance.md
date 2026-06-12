# Table 3 Cross-Scenario Transport Performance

Generated on 2026-06-11.

This table fixes the manuscript's core transport-performance comparison across the two planned transport scenarios.

Important endpoint note: the NHANES -> MIMIC-IV stress-test uses a one-year mortality endpoint, whereas the MIMIC-IV ICU <-> eICU deployment analyses use harmonized hospital mortality. These rows should be compared as endpoint-specific deployment scenarios; the cross-scenario contrast is intended to compare transport failure modes and recalibration needs, not endpoint-equivalent absolute accuracy.

Model note: Table 3 reports the prespecified class-weighted logistic regression primary analysis. The unweighted logistic sensitivity analysis is reported separately in `Supplementary_Table_class_weight_sensitivity.csv`.

All displayed intervals are percentile 95% confidence intervals from 1000 bootstrap resamples of the target/evaluation cohort.

| Scenario | Direction | Endpoint | Feature set | Model | N | Events | Event rate | Mean predicted risk | AUC | PR AUC | Brier score | ECE | Calibration slope | Calibration intercept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Extreme cross-setting stress test | NHANES -> MIMIC-IV | 1-year mortality | NHANES-compatible base | Logistic regression (class-weighted) | 33201 | 6297 | 0.190 (0.186-0.194) | 0.476 (0.473-0.479) | 0.674 (0.667-0.681) | 0.321 (0.311-0.332) | 0.261 (0.258-0.263) | 0.286 (0.282-0.291) | 0.449 (0.427-0.471) | -1.510 (-1.541--1.481) |
| Realistic ICU deployment | MIMIC-IV ICU -> eICU | Hospital mortality | ICU-native primary | Logistic regression (class-weighted) | 13971 | 1263 | 0.090 (0.086-0.095) | 0.440 (0.436-0.444) | 0.718 (0.703-0.732) | 0.234 (0.214-0.256) | 0.226 (0.222-0.229) | 0.349 (0.344-0.355) | 0.668 (0.619-0.717) | -2.372 (-2.436--2.308) |
| Realistic ICU deployment | eICU -> MIMIC-IV ICU | Hospital mortality | ICU-native primary | Logistic regression (class-weighted) | 6395 | 712 | 0.111 (0.104-0.119) | 0.459 (0.455-0.463) | 0.738 (0.718-0.757) | 0.289 (0.259-0.327) | 0.220 (0.217-0.224) | 0.348 (0.340-0.355) | 1.055 (0.916-1.184) | -2.163 (-2.249--2.075) |

## Outputs

```text
outputs\manuscript_tables\Table_3_cross_scenario_transport_performance.csv
outputs\manuscript_tables\Table_3_cross_scenario_transport_performance_numeric.csv
outputs\manuscript_tables\Table_3_cross_scenario_transport_bootstrap_long.csv
outputs\manuscript_tables\Table_3_cross_scenario_transport_metadata.json
outputs\manuscript_tables\README_Table_3_cross_scenario_transport_performance.md
```
