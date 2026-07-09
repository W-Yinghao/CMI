TTA_MECH_01S - Baseline Mechanism Matrix

This matrix is derived only from TTA_MECH_01 outputs. It is not a method
ranking, not a deployment selection, and not a new experiment.

Full machine-readable table:

```text
results/tta_mech/tta_mech01_bnci2014_001_seed0/mechanism_matrix.csv
```

Matrix

| Backbone | Baseline | Mechanism categories | bAcc delta | NLL delta | ECE delta | Confidence delta | Entropy delta | Accuracy/calibration relation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| EEGConformerMini | ERM_NO_ADAPT | NO_CLEAR_MECHANISM | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | NO_CLEAR_GAIN |
| EEGConformerMini | MATCHED_CORAL | GEOMETRY_ALIGNMENT; ACCURACY_CALIBRATION_TRADEOFF | 0.010995 | 1.155042 | 0.010901 | 0.021298 | -0.056884 | ACCURACY_CALIBRATION_TRADEOFF |
| EEGConformerMini | SPDIM | GEOMETRY_ALIGNMENT | 0.015625 | -0.745098 | -0.036137 | -0.022033 | 0.047969 | ACCURACY_AND_CALIBRATION_ALIGNED |
| EEGConformerMini | T3A | CLASSIFIER_TEMPLATE | -0.017168 | 0.092810 | 0.012264 | -0.004702 | 0.006117 | ACCURACY_DECREASE |
| EEGConformerMini | TTA_CONTROL_REPLAY | CALIBRATION_ONLY; ENTROPY_CONFIDENCE; BALANCE_PRIOR | 0.000000 | -0.330446 | -0.007496 | -0.007463 | 0.019210 | CALIBRATION_ONLY |
| EEGNetMini | ERM_NO_ADAPT | NO_CLEAR_MECHANISM | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | NO_CLEAR_GAIN |
| EEGNetMini | MATCHED_CORAL | GEOMETRY_ALIGNMENT | 0.057870 | -0.509384 | -0.063217 | -0.005805 | 0.020250 | ACCURACY_AND_CALIBRATION_ALIGNED |
| EEGNetMini | SPDIM | GEOMETRY_ALIGNMENT | 0.052276 | -0.828787 | -0.128432 | -0.078733 | 0.172520 | ACCURACY_AND_CALIBRATION_ALIGNED |
| EEGNetMini | T3A | CLASSIFIER_TEMPLATE; ACCURACY_CALIBRATION_TRADEOFF | 0.013696 | 0.907804 | 0.049068 | 0.063232 | -0.154461 | ACCURACY_CALIBRATION_TRADEOFF |
| EEGNetMini | TTA_CONTROL_REPLAY | CALIBRATION_ONLY; BALANCE_PRIOR | 0.000000 | -0.048965 | -0.002501 | -0.002667 | 0.006796 | CALIBRATION_ONLY |

Axis availability

| Axis | Status |
| --- | --- |
| Source replay / retention | NOT_AVAILABLE_IN_THIS_REPLAY |
| BN | NOT_TESTED_IN_FROZEN_FEATURE_REPLAY |
| Normalization | NOT_AVAILABLE_IN_FROZEN_FEATURE_REPLAY |

Interpretation guard

Accuracy and calibration are separated in this matrix. Rows marked
`ACCURACY_CALIBRATION_TRADEOFF` must not be summarized as clean wins.
Unavailable axes must not be interpreted as negative evidence.
