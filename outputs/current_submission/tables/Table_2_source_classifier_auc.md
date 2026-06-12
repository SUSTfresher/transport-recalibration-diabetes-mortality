# Table_2_source_classifier_auc

| Scenario | Source contrast | Feature block | Logistic AUC | Random forest AUC | N balanced | N per source |
| --- | --- | --- | --- | --- | --- | --- |
| Extreme cross-setting stress test | NHANES vs MIMIC-IV | Basic | 0.707 | 0.812 | 13532 | 6766 |
| Extreme cross-setting stress test | NHANES vs MIMIC-IV | Basic + labs | 0.743 | 0.984 | 13532 | 6766 |
| Realistic ICU deployment | MIMIC-IV ICU vs eICU | Basic | 0.628 | 0.784 | 51812 | 25906 / 25906 |
| Realistic ICU deployment | MIMIC-IV ICU vs eICU | Basic + labs | 0.738 | 0.878 | 51812 | 25906 / 25906 |
| Realistic ICU deployment | MIMIC-IV ICU vs eICU | Basic + labs + vitals | 0.748 | 0.959 | 51812 | 25906 / 25906 |

## Notes

- Source-classifier AUC quantifies feature-distribution separability between source databases.
- The MIMIC-IV ICU vs eICU source contrast is contrast-level and applies to both ICU transport directions.
- Albumin sensitivity is omitted from the main table because albumin was excluded from the primary ICU-native model.
