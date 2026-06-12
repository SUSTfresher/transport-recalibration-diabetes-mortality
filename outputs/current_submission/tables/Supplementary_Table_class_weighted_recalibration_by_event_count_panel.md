# Table_4_recalibration_by_event_count_panel

## NHANES -> MIMIC-IV

| Method | Local outcome events | ECE (95% empirical interval) | Mean calibration N | Mean local events | Repeats |
| --- | --- | --- | --- | --- | --- |
| Raw transport | 0 | 0.286 | 0.0 | 0.0 | 1 |
| Intercept-only | 25 | 0.074 (0.066-0.087) | 132.0 | 25.0 | 200 |
| Intercept-only | 50 | 0.073 (0.067-0.084) | 264.0 | 50.0 | 200 |
| Intercept-only | 100 | 0.073 (0.068-0.079) | 527.0 | 100.0 | 200 |
| Intercept-only | 200 | 0.073 (0.069-0.077) | 1055.0 | 200.0 | 200 |
| Platt | 25 | 0.027 (0.010-0.065) | 132.0 | 25.0 | 200 |
| Platt | 50 | 0.019 (0.010-0.044) | 264.0 | 50.0 | 200 |
| Platt | 100 | 0.016 (0.010-0.032) | 527.0 | 100.0 | 200 |
| Platt | 200 | 0.013 (0.009-0.024) | 1055.0 | 200.0 | 200 |
| Isotonic | 25 | 0.045 (0.014-0.089) | 132.0 | 25.0 | 200 |
| Isotonic | 50 | 0.031 (0.011-0.064) | 264.0 | 50.0 | 200 |
| Isotonic | 100 | 0.024 (0.010-0.045) | 527.0 | 100.0 | 200 |
| Isotonic | 200 | 0.018 (0.009-0.031) | 1055.0 | 200.0 | 200 |

## MIMIC-IV ICU -> eICU

| Method | Local outcome events | ECE (95% empirical interval) | Mean calibration N | Mean local events | Repeats |
| --- | --- | --- | --- | --- | --- |
| Raw transport | 0 | 0.349 | 0.0 | 0.0 | 1 |
| Intercept-only | 25 | 0.023 (0.022-0.027) | 261.0 | 25.0 | 200 |
| Intercept-only | 50 | 0.023 (0.022-0.026) | 522.0 | 50.0 | 200 |
| Intercept-only | 100 | 0.023 (0.022-0.025) | 1043.0 | 100.0 | 200 |
| Intercept-only | 200 | 0.022 (0.022-0.023) | 2087.0 | 200.0 | 200 |
| Platt | 25 | 0.015 (0.007-0.037) | 261.0 | 25.0 | 200 |
| Platt | 50 | 0.013 (0.007-0.025) | 522.0 | 50.0 | 200 |
| Platt | 100 | 0.011 (0.007-0.018) | 1043.0 | 100.0 | 200 |
| Platt | 200 | 0.009 (0.007-0.015) | 2087.0 | 200.0 | 200 |
| Isotonic | 25 | 0.023 (0.010-0.042) | 261.0 | 25.0 | 200 |
| Isotonic | 50 | 0.019 (0.009-0.033) | 522.0 | 50.0 | 200 |
| Isotonic | 100 | 0.015 (0.009-0.024) | 1043.0 | 100.0 | 200 |
| Isotonic | 200 | 0.012 (0.007-0.018) | 2087.0 | 200.0 | 200 |

## eICU -> MIMIC-IV ICU

| Method | Local outcome events | ECE (95% empirical interval) | Mean calibration N | Mean local events | Repeats |
| --- | --- | --- | --- | --- | --- |
| Raw transport | 0 | 0.348 | 0.0 | 0.0 | 1 |
| Intercept-only | 25 | 0.010 (0.008-0.016) | 214.0 | 25.0 | 200 |
| Intercept-only | 50 | 0.009 (0.008-0.012) | 429.0 | 50.0 | 200 |
| Intercept-only | 100 | 0.009 (0.008-0.010) | 857.0 | 100.0 | 200 |
| Intercept-only | 200 | 0.009 (0.008-0.010) | 1714.0 | 200.0 | 200 |
| Platt | 25 | 0.019 (0.007-0.044) | 214.0 | 25.0 | 200 |
| Platt | 50 | 0.018 (0.006-0.043) | 429.0 | 50.0 | 200 |
| Platt | 100 | 0.016 (0.006-0.038) | 857.0 | 100.0 | 200 |
| Platt | 200 | 0.016 (0.006-0.029) | 1714.0 | 200.0 | 200 |
| Isotonic | 25 | 0.026 (0.012-0.050) | 214.0 | 25.0 | 200 |
| Isotonic | 50 | 0.020 (0.010-0.033) | 429.0 | 50.0 | 200 |
| Isotonic | 100 | 0.016 (0.008-0.027) | 857.0 | 100.0 | 200 |
| Isotonic | 200 | 0.013 (0.007-0.021) | 1714.0 | 200.0 | 200 |

## Notes

- This panel version contains the same ECE values as Table 4 but separates the three transport directions for easier reading.
- Raw transport is a point estimate before local recalibration; interval columns apply to repeated recalibration samples.