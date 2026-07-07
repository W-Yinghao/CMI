# C22 — Estimand Transport Mechanism Audit (frozen C19 `664007686afb520f`)

> Read-only mechanism audit: WHY does the C19 in-regime source-only competence signal not transport as a pooled cross-regime estimand? No probe tuning, no selector, no external dataset. Normalization is diagnostic-only.

- **CASE: `T1_rank_signal_score_not_calibrated`**  ·  secondary: ['T4_feature_offset_dominated']
- within-target ranking signal survives; the pooled estimand fails because score offsets/scales differ by target/regime (rank-like signal, score NOT calibrated). Post-hoc target normalization recovers pooled AUC (diagnostic only).  Source-observable overlap (disclosed, NOT a trajectory downgrade): a single source-risk scalar (R_src) matches the 16-feature probe within-target -> the weak signal is low-dimensional / risk-family (echoes C17), not a novel multivariate competence dimension.
- next: future work (diagnostic-only): a target-free score-calibration ESTIMAND. NOT a selector; needs the offset removed WITHOUT target identity, which is unsolved.

## Q2 — epoch / trajectory-position confound (reported FIRST, gates everything)

- probe within-target strength **+0.659** vs best TRAJECTORY (epoch/order) baseline **+0.555** → probe beats trajectory: **True**  (trajectory strengths: {'epoch': 0.555, 'order': 0.555})
- residual (partial Spearman score~label | epoch) = **+0.225** → residual signal present: **True**
- **epoch_confounded (trajectory only): False**  (→ within-target signal SURVIVES epoch/trajectory control)
- SEPARATE source-observable overlap: best TRAINING-LOG baseline **+0.669** ({'R_src': 0.669, 'train_surrogate': 0.6}); source_risk_overlap: **True**, probe adds over R_src (partial|R_src=+0.163): **True**  → a single source-risk scalar matches the probe (low-dimensional/risk-family; NOT a trajectory proxy).

## Q1 — pooled vs within-target decomposition

- mean pooled AUC **+0.530** vs mean within-target AUC **+0.650** (gap -0.120); within exceeds pooled everywhere: **True**

| group | pooled | within-target mean | gap |
|---|---:|---:|---:|
| in_regime:S0_full_support | +0.561 | +0.663 | -0.103 |
| in_regime:S2_rare_cells | +0.537 | +0.671 | -0.135 |
| in_regime:S3_nonestimable_cells | +0.533 | +0.653 | -0.119 |
| cross_regime:S4_missing_cells | +0.500 | +0.641 | -0.141 |
| cross_regime:S5_block_class_by_domain | +0.511 | +0.634 | -0.123 |
| cross_regime:S6_boundary_aligned_mask | +0.532 | +0.652 | -0.120 |
| cross_regime:S7_random_matched_mask | +0.536 | +0.632 | -0.096 |

## Q3 — post-hoc normalization diagnostics (MECHANISM only, NON-deployable)

| mode | pooled (none) | best target-normalized pooled | target-normalization recovers |
|---|---:|---:|:--:|
| in_regime | +0.543 | +0.648 | True |
| cross_regime | +0.518 | +0.628 | True |

> Target/regime-wise normalization needs the target/regime identity at score time -> NON-deployable; reported as mechanism only. Recovery => rank-like signal / score-offset problem; no recovery => regime-specific relationship shift.

## Q4 — feature-level offset vs ranking

- 16/16 robust-core features carry within-target ranking; 12/16 are offset-dominated (fraction +0.750).

## Boundary of the claim

> DIAGNOSTIC-ONLY mechanism audit. Normalization diagnostics are NOT deployable procedures (they use target/regime identity). No selector is produced; the C19/C20 estimand boundary is explained, not rescued.