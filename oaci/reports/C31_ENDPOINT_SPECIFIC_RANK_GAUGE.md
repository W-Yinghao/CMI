# C31 — endpoint-specific rank & gauge

> the source rank is accuracy-ALIGNED BY CONSTRUCTION (probe trained on label==accuracy_good, 0/3804 mismatches) and NOT distinguishably accuracy-specific vs calibration: acc strength 0.159 vs calibration_good 0.120, gap 0.039 95% CI [-0.028, 0.100] INCLUDES 0; the only distinguishable contrast is accuracy vs ECE (gap 0.093, CI [0.013, 0.160], excludes 0: True)
> the gauge is a GENERAL per-target offset: between-target variance fraction is near-equal across endpoints (bAcc 0.88 / NLL 0.84 / ECE 0.83); accuracy pooled-vs-within gap 0.115 is only mildly above calibration 0.085 (tilt inherited from the accuracy-aligned rank, not a distinct calibration gauge)

| factor | endpoint | within-target AUC | pooled AUC | rank strength | sign-consistency |
|---|---|---:|---:|---:|---:|
| score | accuracy_good | +0.659 | +0.543 | +0.159 | +1.000 |
| score | nll_good | +0.613 | +0.472 | +0.113 | +0.889 |
| score | ece_good | +0.565 | +0.444 | +0.065 | +0.667 |
| score | calibration_good | +0.620 | +0.515 | +0.120 | +1.000 |
| score | joint_good | +0.672 | +0.541 | +0.172 | +1.000 |
| score | pareto_good | +0.574 | +0.591 | +0.074 | +0.667 |
| R_src | accuracy_good | +0.376 | +0.454 | +0.124 | +0.778 |
| R_src | nll_good | +0.543 | +0.530 | +0.043 | +0.556 |
| R_src | ece_good | +0.595 | +0.592 | +0.095 | +0.778 |
| R_src | calibration_good | +0.556 | +0.547 | +0.056 | +0.556 |
| R_src | joint_good | +0.396 | +0.464 | +0.104 | +0.778 |
| R_src | pareto_good | +0.497 | +0.500 | +0.003 | +0.556 |

## gauge (between-target variance fraction per metric)

| metric | between-target variance fraction |
|---|---:|
| bacc | +0.882 |
| nll | +0.842 |
| ece | +0.834 |