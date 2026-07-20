# C34 - Source Objective Direction

| component | family | pairwise AUC | corr | wrong | flat | random |
|---|---|---:|---:|---:|---:|---:|
| source_score | OACI_selector_score | +0.534 | +0.162 | +0.391 | +0.150 | +0.500 |
| selection_leakage | leakage_component | +0.469 | +0.017 | +0.438 | +0.185 | +0.500 |
| audit_leakage | leakage_component | +0.497 | +0.049 | +0.130 | +0.746 | +0.500 |
| R_src | source_risk | +0.510 | +0.066 | +0.231 | +0.518 | +0.500 |
| source_guard_nll | source_endpoint | +0.510 | +0.056 | +0.232 | +0.514 | +0.500 |
| source_audit_nll | source_endpoint | +0.480 | -0.082 | +0.433 | +0.174 | +0.500 |
| source_guard_ece | source_endpoint | +0.492 | -0.044 | +0.174 | +0.668 | +0.500 |
| source_audit_ece | source_endpoint | +0.475 | -0.050 | +0.323 | +0.404 | +0.500 |
| source_audit_confidence | calibration_softness | +0.516 | +0.069 | +0.226 | +0.516 | +0.500 |
| robust_core_score | C19_robust_core | +0.508 | +0.068 | +0.426 | +0.133 | +0.500 |
| c30_source_rank | C30_source_rank | +0.546 | +0.136 | +0.454 | +0.000 | +0.500 |
| source_audit_worst_bacc | source_endpoint | n/a | n/a | n/a | n/a | +0.500 |

## Selected-pair component conflict

| component | available | wrong | flat | correct | mean delta |
|---|---:|---:|---:|---:|---:|
| source_score | True | +0.238 | +0.099 | +0.663 | +0.083 |
| selection_leakage | True | +0.016 | +0.107 | +0.877 | +0.169 |
| audit_leakage | True | +0.063 | +0.679 | +0.258 | +0.009 |
| R_src | True | +0.167 | +0.500 | +0.333 | +0.015 |
| source_guard_nll | True | +0.155 | +0.492 | +0.353 | +0.015 |
| source_audit_nll | True | +0.444 | +0.115 | +0.440 | -0.013 |
| source_guard_ece | True | +0.187 | +0.655 | +0.159 | +0.001 |
| source_audit_ece | True | +0.385 | +0.325 | +0.290 | -0.007 |
| source_audit_confidence | True | +0.190 | +0.440 | +0.369 | +0.011 |
| robust_core_score | True | +0.147 | +0.155 | +0.698 | +0.120 |
| c30_source_rank | True | +0.270 | +0.000 | +0.730 | +0.253 |
| source_audit_worst_bacc | False | n/a | n/a | n/a | n/a |

All rows are diagnostic decompositions of existing scores, not selector definitions.
