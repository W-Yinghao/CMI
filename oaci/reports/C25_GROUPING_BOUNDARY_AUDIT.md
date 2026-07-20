# C25 Q4 — grouping problem-class boundary

> The pooled cross-target estimand is recoverable by 0-LABEL transductive within-target centering (target grouping + the target's OWN candidate scores) — a distinct problem class from source-only DG (C19/C23: offset source-unobservable) and from target-label calibration (R5). Target-unlabeled MARGINAL geometry (R3) recovers only a weak part; target GROUPING adds the rest (value over marginal = R6 - R3). Target grouping is NOT source-only, and the target-centered oracle is NOT a deployable selector.

R3 (target-unlabeled transductive) uses a CROSS-TARGET model on the held-out target's unlabeled geometry (no held-out scores); R6 (target-grouped zero-label) uses the held-out target's OWN candidate scores' mean. Both are 0-label; they differ in whether the held-out target's own score aggregate is used.

- value of grouping over marginal (R6−R3): +0.509; within-target ceiling +0.659

| problem class | target inputs | grouping | labels | uses held-out scores | gap closed |
|---|:--:|:--:|:--:|:--:|---:|
| source_only_DG | False | False | False | False | -0.825 |
| target_unlabeled_transductive | True | True | False | False | +0.491 |
| target_grouped_transductive_zero_label | True | True | False | True | +1.000 |
| few_label_target_calibration | True | True | True | True | +1.415 |
| target_label_oracle | True | True | True | True | +1.415 |