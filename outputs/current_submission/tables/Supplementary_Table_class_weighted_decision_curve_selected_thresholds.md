# Table_6_decision_curve_selected_thresholds

| Direction | Threshold | Treat none | Treat all | Raw transport logistic | Platt 100 events | Internal HGB benchmark |
| --- | --- | --- | --- | --- | --- | --- |
| NHANES -> MIMIC-IV | 0.20 | 0.0000 | -0.0129 | 0.0208 | 0.0410 | 0.0697 |
| NHANES -> MIMIC-IV | 0.25 | 0.0000 | -0.0805 | -0.0201 | 0.0259 | 0.0514 |
| NHANES -> MIMIC-IV | 0.30 | 0.0000 | -0.1576 | -0.0603 | 0.0157 | 0.0367 |
| MIMIC-IV ICU -> eICU | 0.20 | 0.0000 | -0.1370 | -0.1030 | 0.0075 | 0.0144 |
| MIMIC-IV ICU -> eICU | 0.25 | 0.0000 | -0.2128 | -0.1433 | 0.0048 | 0.0097 |
| MIMIC-IV ICU -> eICU | 0.30 | 0.0000 | -0.2994 | -0.1776 | 0.0030 | 0.0063 |
| eICU -> MIMIC-IV ICU | 0.20 | 0.0000 | -0.1108 | -0.0964 | 0.0184 | 0.0298 |
| eICU -> MIMIC-IV ICU | 0.25 | 0.0000 | -0.1849 | -0.1471 | 0.0111 | 0.0236 |
| eICU -> MIMIC-IV ICU | 0.30 | 0.0000 | -0.2695 | -0.1893 | 0.0074 | 0.0170 |

## Notes

- Net benefit is reported at thresholds 0.20, 0.25, and 0.30.
- Internal HGB benchmark is the target-site histogram-gradient-boosting model.
