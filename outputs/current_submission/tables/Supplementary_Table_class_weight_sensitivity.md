# Supplementary Table. Class-weighted versus unweighted logistic transport sensitivity

This sensitivity analysis compares the primary unweighted transported logistic regressions with otherwise identical class-weighted logistic regressions, using the same cohorts, deterministic splits, feature sets, and preprocessing. ECE is reported using both 10 equal-frequency bins and 10 equal-width bins.

| Direction | Endpoint | Event rate | Mean predicted risk, class-weighted | Mean predicted risk, unweighted | AUC, class-weighted | AUC, unweighted | ECE equal-frequency, class-weighted | ECE equal-frequency, unweighted | ECE equal-width, class-weighted | ECE equal-width, unweighted | Calibration slope, class-weighted | Calibration slope, unweighted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NHANES -> MIMIC-IV | 1-year mortality | 0.190 | 0.476 | 0.043 | 0.674 | 0.674 | 0.286 | 0.146 | 0.286 | 0.146 | 0.449 | 0.497 |
| MIMIC-IV ICU -> eICU | Hospital mortality | 0.090 | 0.440 | 0.142 | 0.718 | 0.707 | 0.349 | 0.059 | 0.349 | 0.052 | 0.667 | 0.563 |
| eICU -> MIMIC-IV ICU | Hospital mortality | 0.111 | 0.459 | 0.100 | 0.738 | 0.737 | 0.348 | 0.016 | 0.348 | 0.015 | 1.055 | 1.198 |

Interpretation: class weighting had little effect on discrimination but materially changed absolute probability levels and ECE. This supports the decision to use unweighted logistic regression as the primary probability model and to interpret class-weighted results as a sensitivity analysis when the model is intended for risk estimation.
