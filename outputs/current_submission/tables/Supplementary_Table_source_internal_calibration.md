# Supplementary_Table_source_internal_calibration

| Source | Internal evaluation | Endpoint | Feature set | N | Events | Event rate | Mean predicted risk | AUC | PR AUC | Brier score | ECE | Equal-width ECE | Calibration slope | Calibration intercept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NHANES | NHANES 2013-2014 temporal holdout | 1-year mortality | NHANES-compatible base | 923 | 17 | 0.018 | 0.024 | 0.702 | 0.097 | 0.018 | 0.015 | 0.005 | 0.716 | -1.182 |
| MIMIC-IV ICU | MIMIC-IV ICU patient-level holdout | Hospital mortality | ICU-native primary | 6395 | 712 | 0.111 | 0.114 | 0.739 | 0.355 | 0.087 | 0.014 | 0.010 | 1.047 | 0.058 |
| eICU | eICU patient-level holdout | Hospital mortality | ICU-native primary | 13971 | 1263 | 0.090 | 0.095 | 0.740 | 0.240 | 0.076 | 0.007 | 0.009 | 1.043 | 0.023 |

## Notes

- This supplementary table reports unweighted logistic model performance on the source dataset's internal holdout or temporal holdout.
- The ICU source models were close to calibrated on their source holdouts, supporting interpretation of ICU-to-ICU slope distortion as a transport phenomenon.
- The NHANES temporal holdout contained few one-year mortality events, so its internal calibration slope is imprecise and the NHANES-to-MIMIC analysis remains a cross-setting stress test.
