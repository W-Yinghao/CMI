# Step 16 — oracle-only benefit anatomy

Scope: oracle-only benefit anatomy; not SOTA. oracle-only benefit anatomy; NOT deployment-observable under R0/R1; used only to characterise the problem, never as a predictor or deployable signal.

- runs: **54** · beneficial **5** · harmful **43** · near-zero **6** · benefit-rate **0.0926**
- per dataset benefit-rate: **{'BNCI2014_001': 0.1111, 'BNCI2014_004': 0.0741}**
- target sign-consistency rate (same sign across seeds): **0.6111**
- beneficial gain dist (bAcc): mean **0.010553** q10 **0.006307** q50 **0.010417** q90 **0.015281** max **0.017361**

| dataset:target | n_seeds | benefit_count | sign_consistent | mean_gain_bacc |
|---|---:|---:|---|---:|
| BNCI2014_001:1 | 3 | 0 | True | -0.118055 |
| BNCI2014_001:2 | 3 | 0 | True | -0.038773 |
| BNCI2014_001:3 | 3 | 0 | True | -0.07581 |
| BNCI2014_001:4 | 3 | 0 | True | -0.050347 |
| BNCI2014_001:5 | 3 | 1 | False | -0.0 |
| BNCI2014_001:6 | 3 | 2 | False | 0.001736 |
| BNCI2014_001:7 | 3 | 0 | True | -0.021412 |
| BNCI2014_001:8 | 3 | 0 | True | -0.141782 |
| BNCI2014_001:9 | 3 | 0 | False | -0.031829 |
| BNCI2014_004:1 | 3 | 0 | True | -0.063889 |
| BNCI2014_004:2 | 3 | 1 | False | -0.003431 |
| BNCI2014_004:3 | 3 | 0 | False | -0.014815 |
| BNCI2014_004:4 | 3 | 1 | False | 0.003153 |
| BNCI2014_004:5 | 3 | 0 | True | -0.044144 |
| BNCI2014_004:6 | 3 | 0 | False | -0.042593 |
| BNCI2014_004:7 | 3 | 0 | True | -0.027778 |
| BNCI2014_004:8 | 3 | 0 | True | -0.061403 |
| BNCI2014_004:9 | 3 | 0 | True | -0.030093 |

> oracle-only benefit anatomy; NOT deployment-observable under R0/R1; used only to characterise the problem, never as a predictor or deployable signal.
