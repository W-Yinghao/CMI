# C30 — source rank signal audit

> [RED-TEAM] the within-target rank's largest single carrier is the source_risk family (strength 0.124), but the score-vs-best-family gap (0.034) is WITHIN 9-target bootstrap noise -> NOT 'beats any family'. Distributedness holds in the RESIDUAL sense (score retains strength 0.113 after removing R_src). The MULTIVARIATE score is direction-CONSISTENT across targets (sign_consistency 1.00, transfers) while the top single family is target-LOCAL (sign-flips) -> the transferable within-target rank is DISTRIBUTED, not a single source family.

probe-score rank strength +0.159

| family | feature | within-target AUC | rank strength |
|---|---|---:|---:|
| source_risk | R_src | +0.376 | +0.124 |
| source_risk | train_surrogate | +0.593 | +0.093 |
| source_calibration | feat__source_guard_ece | +0.589 | +0.089 |
| source_calibration | feat__source_audit_ece | +0.531 | +0.031 |
| source_calibration | feat__source_guard_entropy | +0.491 | +0.009 |
| source_calibration | feat__source_audit_entropy | +0.433 | +0.067 |
| source_calibration | feat__source_guard_conf_on_wrong | +0.448 | +0.052 |
| source_calibration | feat__source_audit_conf_on_wrong | +0.572 | +0.072 |
| source_leakage | feat__selection_leakage_point | +0.539 | +0.039 |
| source_leakage | feat__audit_leakage_point | +0.522 | +0.022 |
| source_logit_geometry | feat__source_guard_confidence | +0.505 | +0.005 |
| source_logit_geometry | feat__source_guard_margin | +0.513 | +0.013 |
| source_logit_geometry | feat__source_guard_logit_norm | +0.510 | +0.010 |
| source_logit_geometry | feat__source_guard_nll | +0.377 | +0.123 |
| source_logit_geometry | feat__source_audit_confidence | +0.567 | +0.067 |
| source_logit_geometry | feat__source_audit_margin | +0.570 | +0.070 |
| source_logit_geometry | feat__source_audit_logit_norm | +0.567 | +0.067 |
| source_logit_geometry | feat__source_audit_nll | +0.489 | +0.011 |