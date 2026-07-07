# C25 Q3 — R4 source-interference audit

> R4 collapse reproduced by RANDOM noise dims of the same count -> the mechanism is small-N / high-dimensional nuisance: at 9 targets, adding ~34 non-informative dims overfits LOTO and drowns the 12-dim target-unlabeled signal; source features are generic nuisance for the offset (not a specific anti-aligned direction).

- R3 gap +0.491 → R4 gap -0.485 (source+target-unlabeled collapses)
- source features 34 vs target-unlabeled 12; source coef-norm share +0.496 (hijacks ridge: False)
- condition number R3 +64.506 → R4 +152.873
- source↔target offset-prediction alignment: -0.213
- RANDOM-DIM control (34 noise dims): mean gap -0.039; random dims also collapse: True

**Mechanism: `small_N_high_dim_noise`**
