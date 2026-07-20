# C31 — accuracy-calibration trade-off audit

> accuracy improvement is NOT negatively correlated with calibration improvement (mean corr 0.597); accuracy-good checkpoints largely overlap calibration-good -> trade-off NOT confirmed at the population level

- accuracy_good_calibration_bad rate: +0.049
- P(calibration-good | accuracy-good): +0.897
- raw bAcc↔NLL / bAcc↔ECE improvement corr: +0.590 / +0.605
- epoch-residualized (within-traj): +0.504 / +0.554 (survives epoch control: True)
- **trade-off confirmed: False** — the C16 barrier is NOT a population-level accuracy-calibration Pareto conflict.
