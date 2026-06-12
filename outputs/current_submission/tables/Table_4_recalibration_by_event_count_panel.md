# Table_4_recalibration_by_event_count_panel

## NHANES -> MIMIC-IV

| Method | Local outcome events | ECE (95% empirical interval) | Mean calibration N | Mean local events | Repeats |
| --- | --- | --- | --- | --- | --- |
| Raw transport | 0 | 0.146 | 0.0 | 0.0 | 1 |
| Intercept-only | 25 | 0.064 (0.058-0.074) | 132.0 | 25.0 | 200 |
| Intercept-only | 50 | 0.064 (0.059-0.071) | 264.0 | 50.0 | 200 |
| Intercept-only | 100 | 0.064 (0.059-0.068) | 527.0 | 100.0 | 200 |
| Intercept-only | 200 | 0.064 (0.060-0.067) | 1055.0 | 200.0 | 200 |
| Platt | 25 | 0.028 (0.012-0.068) | 132.0 | 25.0 | 200 |
| Platt | 50 | 0.021 (0.012-0.046) | 264.0 | 50.0 | 200 |
| Platt | 100 | 0.018 (0.011-0.032) | 527.0 | 100.0 | 200 |
| Platt | 200 | 0.015 (0.011-0.025) | 1055.0 | 200.0 | 200 |
| Isotonic | 25 | 0.044 (0.016-0.094) | 132.0 | 25.0 | 200 |
| Isotonic | 50 | 0.031 (0.011-0.061) | 264.0 | 50.0 | 200 |
| Isotonic | 100 | 0.024 (0.010-0.045) | 527.0 | 100.0 | 200 |
| Isotonic | 200 | 0.018 (0.008-0.032) | 1055.0 | 200.0 | 200 |

## MIMIC-IV ICU -> eICU

| Method | Local outcome events | ECE (95% empirical interval) | Mean calibration N | Mean local events | Repeats |
| --- | --- | --- | --- | --- | --- |
| Raw transport | 0 | 0.059 | 0.0 | 0.0 | 1 |
| Intercept-only | 25 | 0.032 (0.031-0.034) | 261.0 | 25.0 | 200 |
| Intercept-only | 50 | 0.032 (0.031-0.033) | 522.0 | 50.0 | 200 |
| Intercept-only | 100 | 0.031 (0.031-0.033) | 1043.0 | 100.0 | 200 |
| Intercept-only | 200 | 0.031 (0.031-0.032) | 2087.0 | 200.0 | 200 |
| Platt | 25 | 0.016 (0.007-0.033) | 261.0 | 25.0 | 200 |
| Platt | 50 | 0.013 (0.007-0.028) | 522.0 | 50.0 | 200 |
| Platt | 100 | 0.012 (0.007-0.021) | 1043.0 | 100.0 | 200 |
| Platt | 200 | 0.011 (0.007-0.015) | 2087.0 | 200.0 | 200 |
| Isotonic | 25 | 0.023 (0.010-0.041) | 261.0 | 25.0 | 200 |
| Isotonic | 50 | 0.018 (0.009-0.030) | 522.0 | 50.0 | 200 |
| Isotonic | 100 | 0.014 (0.007-0.024) | 1043.0 | 100.0 | 200 |
| Isotonic | 200 | 0.011 (0.006-0.018) | 2087.0 | 200.0 | 200 |

## eICU -> MIMIC-IV ICU

| Method | Local outcome events | ECE (95% empirical interval) | Mean calibration N | Mean local events | Repeats |
| --- | --- | --- | --- | --- | --- |
| Raw transport | 0 | 0.016 | 0.0 | 0.0 | 1 |
| Intercept-only | 25 | 0.015 (0.014-0.017) | 214.0 | 25.0 | 200 |
| Intercept-only | 50 | 0.015 (0.014-0.016) | 429.0 | 50.0 | 200 |
| Intercept-only | 100 | 0.014 (0.014-0.015) | 857.0 | 100.0 | 200 |
| Intercept-only | 200 | 0.014 (0.014-0.015) | 1714.0 | 200.0 | 200 |
| Platt | 25 | 0.020 (0.006-0.049) | 214.0 | 25.0 | 200 |
| Platt | 50 | 0.017 (0.006-0.049) | 429.0 | 50.0 | 200 |
| Platt | 100 | 0.016 (0.006-0.033) | 857.0 | 100.0 | 200 |
| Platt | 200 | 0.014 (0.006-0.028) | 1714.0 | 200.0 | 200 |
| Isotonic | 25 | 0.027 (0.011-0.051) | 214.0 | 25.0 | 200 |
| Isotonic | 50 | 0.020 (0.010-0.036) | 429.0 | 50.0 | 200 |
| Isotonic | 100 | 0.016 (0.008-0.026) | 857.0 | 100.0 | 200 |
| Isotonic | 200 | 0.013 (0.007-0.021) | 1714.0 | 200.0 | 200 |

## Notes

- Primary model is unweighted logistic regression.
- Raw transport is a point estimate before local recalibration; interval columns apply to repeated recalibration samples.