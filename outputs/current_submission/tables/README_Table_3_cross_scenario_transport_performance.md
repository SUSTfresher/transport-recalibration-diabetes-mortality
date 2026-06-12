# Table 3 Cross-Scenario Transport Performance

Generated on 2026-06-11.

This table fixes the manuscript's core transport-performance comparison after promoting unweighted logistic regression to the primary probability model.

Important endpoint note: the NHANES -> MIMIC-IV stress-test uses a one-year mortality endpoint, whereas the MIMIC-IV ICU <-> eICU deployment analyses use harmonized hospital mortality. The NHANES and MIMIC-IV one-year endpoints also differ in ascertainment mechanism: NHANES uses linked community mortality follow-up, whereas MIMIC-IV uses hospital-episode data plus post-discharge mortality. These rows should be compared as endpoint-specific deployment scenarios; the cross-scenario contrast is intended to compare transport failure modes and recalibration needs, not endpoint-equivalent absolute accuracy. Do not interpret row-to-row differences in AUC, ECE, or Brier score as absolute performance differences under a shared endpoint.

Model note: Table 3 reports unweighted logistic regression. The former class-weighted primary analysis is retained in the supplementary class-weighting sensitivity tables.

All displayed intervals are percentile 95% confidence intervals from 1000 bootstrap resamples of the target/evaluation cohort.

| Scenario | Direction | Endpoint | Feature set | Model | Target evaluation N | Target evaluation events | Event rate | Mean predicted risk | AUC | PR AUC | Brier score | ECE | Equal-width ECE | Calibration slope | Calibration intercept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Extreme cross-setting stress test | NHANES -> MIMIC-IV | 1-year mortality | NHANES-compatible base | Logistic regression (unweighted) | 33201 | 6297 | 0.190 (0.186-0.194) | 0.043 (0.043-0.044) | 0.674 (0.667-0.682) | 0.319 (0.309-0.331) | 0.168 (0.165-0.172) | 0.146 (0.142-0.150) | 0.146 (0.142-0.150) | 0.497 (0.473-0.521) | 0.307 (0.226-0.391) |
| Realistic ICU deployment | MIMIC-IV ICU -> eICU | Hospital mortality | ICU-native primary | Logistic regression (unweighted) | 13971 | 1263 | 0.090 (0.086-0.095) | 0.142 (0.139-0.145) | 0.707 (0.691-0.721) | 0.234 (0.214-0.255) | 0.087 (0.084-0.090) | 0.059 (0.055-0.064) | 0.052 (0.048-0.057) | 0.563 (0.519-0.606) | -1.232 (-1.317--1.143) |
| Realistic ICU deployment | eICU -> MIMIC-IV ICU | Hospital mortality | ICU-native primary | Logistic regression (unweighted) | 6395 | 712 | 0.111 (0.104-0.119) | 0.100 (0.098-0.102) | 0.737 (0.717-0.757) | 0.293 (0.263-0.332) | 0.091 (0.085-0.096) | 0.016 (0.013-0.026) | 0.015 (0.009-0.023) | 1.198 (1.050-1.338) | 0.524 (0.193-0.821) |

## Outputs

```text
outputs\manuscript_tables\Table_3_cross_scenario_transport_performance.csv
outputs\manuscript_tables\Table_3_cross_scenario_transport_performance_numeric.csv
outputs\manuscript_tables\Table_3_cross_scenario_transport_bootstrap_long.csv
outputs\manuscript_tables\Table_3_cross_scenario_transport_metadata.json
outputs\manuscript_tables\README_Table_3_cross_scenario_transport_performance.md
```
