# C25 — Target-Unlabeled Gauge Mechanism + Grouping Boundary Audit (frozen C19 `664007686afb520f`)

> Read-only MECHANISM audit of the C24 result (R1 source HURTS, R3 target-unlabeled PARTIALLY recovers, R4 source+target-unlabeled COLLAPSES, R6 target-grouping FULLY recovers). No re-inference, no probe tuning, no feature selection, no selector. DIAGNOSTIC-ONLY.

- **PRIMARY: `U2_predicted_class_mix_carries_gauge`** — predicted-class proportions carry most of the weak R3 offset recovery (Shapley-dominant + necessary: leave-one-family-out on it destroys recovery), though NOT sufficient alone (needs a confidence/margin-geometry scaffold) and ENTANGLED with target identity (same family is most identity-predictive) -- credited as a transferable marginal relationship only because it survives the LOTO permutation control.
- established: **U2_predicted_class_mix_carries_gauge, U6_source_interference_confirmed, U7_grouping_is_separate_problem_class**

## HARD GATE — target-identity signature (reported before any positive mechanism claim)

- R3-feature 9-way target-id accuracy **+0.671** vs chance **+0.111** (source ref +0.541) → identity-separable: **True** (expected for per-target moments).
- recovery SURVIVES the LOTO offset-permutation control: **True** → identity signature dominates: **False**. R3 features are target-id separable (expected for per-target moments), BUT the recovery SURVIVES the LOTO offset-permutation null -> it is target-marginal geometry, not a pure identity fingerprint (which cannot help a held-out unseen target).
- **DISCLOSED entanglement**: the carrying family (**pred_class_prop**) is ALSO the most identity-predictive family (**pred_class_prop**) → dissociated: **False**. The recovery is entangled with target-identity structure; the permutation control is what distinguishes a transferable marginal relationship (survives) from a pure fingerprint (would not). Not over-claimed as identity-free.

## Q1 — which target-unlabeled family carries the weak R3 recovery?

- full R3 gap closed **+0.491** (perm p +0.024).

| family | family-only gap | perm p | survives | Shapley gap | share |
|---|---:|---:|:--:|---:|---:|
| confidence_entropy | -0.580 | +0.908 | False | -0.172 | +0.000 |
| margin_logitnorm | -0.265 | +0.796 | False | +0.011 | +0.017 |
| pred_class_prop | +0.003 | +0.427 | False | +0.651 | +0.983 |

- Shapley dominant family **pred_class_prop** (share +0.983); single family dominates (≥0.6): **True**.
- leave-one-family-out: −confidence_entropy→+0.502, −margin_logitnorm→+0.451, −pred_class_prop→-0.561
- NOT-computed families (out of scope; would need target-Z re-inference): target_feature_moments, source_target_distance, finite_feature_availability.

## Q3 — why do source features DESTROY R3 in R4?

- R3 gap **+0.491** → R4 (source+target-unlabeled) gap **-0.485** (collapse).
- source coef-norm share **+0.496** (hijacks ridge: False); cond# R3 +64.506 → R4 +152.873; source↔target offset alignment -0.213.
- **RANDOM-DIM control**: adding 34 random noise dims to R3 → mean gap **-0.039** → random dims also collapse: **True** → mechanism: **`small_N_high_dim_noise`**. R4 collapse reproduced by RANDOM noise dims of the same count -> the mechanism is small-N / high-dimensional nuisance: at 9 targets, adding ~34 non-informative dims overfits LOTO and drowns the 12-dim target-unlabeled signal; source features are generic nuisance for the offset (not a specific anti-aligned direction).

## Q4 — grouping problem-class boundary

| problem class | tgt inputs | grouping | labels | uses held-out scores | gap | deployable-transductive |
|---|:--:|:--:|:--:|:--:|---:|:--:|
| source_only_DG | False | False | False | False | -0.825 | False |
| target_unlabeled_transductive | True | True | False | False | +0.491 | True |
| target_grouped_transductive_zero_label | True | True | False | True | +1.000 | True |
| few_label_target_calibration | True | True | True | True | +1.415 | False |
| target_label_oracle | True | True | True | True | +1.415 | False |

- value of GROUPING beyond target-unlabeled marginal geometry (R6−R3): **+0.509**; within-target ceiling +0.659.
- The pooled cross-target estimand is recoverable by 0-LABEL transductive within-target centering (target grouping + the target's OWN candidate scores) — a distinct problem class from source-only DG (C19/C23: offset source-unobservable) and from target-label calibration (R5). Target-unlabeled MARGINAL geometry (R3) recovers only a weak part; target GROUPING adds the rest (value over marginal = R6 - R3). Target grouping is NOT source-only, and the target-centered oracle is NOT a deployable selector.

## Boundary of the claim

> DIAGNOSTIC-ONLY mechanism audit. Families are FROZEN (no feature selection). Target grouping is NOT source-only DG and the target-centered oracle is NOT a deployable selector. Target labels never entered the R3 feature construction.