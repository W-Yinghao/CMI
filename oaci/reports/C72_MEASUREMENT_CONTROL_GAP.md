# C72 - Extreme-Order Rank-Gauge Intervention / Measurement-Control Gap Decomposition (frozen C19 `664007686afb520f`)

## Executive Verdict

Primary: `C72-E_mixed_noise_margin_gauge_mechanism`

Active: `C72-E_mixed_noise_margin_gauge_mechanism ; C72-S1_common_utility_offset_identity_confirmed ; C72-S2_common_logit_scalar_identity_confirmed ; C72-S4_candidate_specific_intervention_reproduces_rank_flips ; C72-S7_synthetic_phase_diagram_validated ; C72-S9_representation_intervention_unavailable ; C72-S10_independent_target_dataset_replication_now_justified ; C72-S11_new_training_not_yet_justified`

Inactive: `C72-A_extreme_order_geometry_explains_measurement_control_gap ; C72-B_residual_candidate_specific_gauge_dominates_gap ; C72-C_finite_label_noise_dominates_gap ; C72-D_construction_utility_mismatch_dominates_gap ; C72-F_C71_measurement_control_gap_not_mechanistically_resolved ; C72-G_rank_gauge_model_contradicted_by_intervention ; C72-S3_shared_target_calibration_insufficient ; C72-S5_construction_estimated_gauge_partial_only ; C72-S6_multi_candidate_model_bound_nontrivial ; C72-S8_no_strict_source_escape_hatch`

Final gate: `MEASUREMENT_CONTROL_GAP_PARTIALLY_RESOLVED`

## Gate-First Result

The no-intervention full-construction measurement has mean within-target Spearman `0.557457` but target-universe top-1 `0.222222`. The median best-minus-second held-out bAcc gap is `0.012916` across nine frozen targets.

At 8 labels/class, Shapley-style gap fractions are finite-label noise `0.090873`, endpoint mismatch `0.000000`, extreme order `0.355918`, and residual candidate gauge `0.553209`. At full construction the corresponding fractions are `0.000000`, `0.000000`, `0.440150`, and `0.559850`.

The zero primary mismatch fraction is specific to the registered bAcc-to-bAcc control endpoint. In the secondary joint-utility audit, construction bAcc has mean Spearman `0.486599`, endpoint-matched construction utility has `0.763830`, and mean top-1 gain is `0.333333`. Utility mismatch therefore remains material for the joint endpoint but does not explain the primary bAcc control gap.

The T2-fitted source-plus-construction model reaches T3-HO R2 `0.568422`, leaving residual variance fraction `0.431578`. This residual is candidate-specific after within-target centering; it is not a target-common offset.

## Controlled Interventions

I1 utility-common offsets produce `0` rank flips. I2 all-class logit scalars produce `0` rank flips with maximum probability delta `0.000000000000`.

The T2-locked shared class-vector intervention has T3-HO top-1 `0.111111`. T2 selected `alpha=0.0` for the construction-estimated candidate-specific class-gradient intervention; with the observed zero lock, its T3-HO top-1 `0.222222` is the no-intervention result, not partial closure. Random matched candidate perturbations still test rank-flip sensitivity but do not constitute recovery.

## Primary Hypotheses

- `H1_common_utility_offset_identity`: `pass`; effect `0.000000`; raw p `0.000000`; Holm p `0.000000`.
- `H2_common_logit_scalar_identity`: `pass`; effect `0.000000`; raw p `0.000000`; Holm p `0.000000`.
- `H3_shared_calibration_insufficient`: `not_confirmed`; effect `-0.006119`; raw p `0.785200`; Holm p `1.000000`.
- `H4_candidate_perturbation_margin_ratio`: `pass`; effect `0.782624`; raw p `0.001800`; Holm p `0.007200`.
- `H5_construction_gauge_partial`: `not_confirmed`; effect `0.000000`; raw p `1.000000`; Holm p `1.000000`.
- `H6_extreme_order_candidate_count`: `pass`; effect `-0.822222`; raw p `0.003400`; Holm p `0.010200`.

## Theory and Synthetic Stress

The exact class-stratified finite-population paired calculation covers `9` T3-HO target/budget-8 fields. The Gaussian rank-gauge union bound is nontrivial in `0/54` target-budget cells; trivial cells are disclosed rather than promoted.

The registered synthetic grid contains `2592` cells and `408` high-reliability/poor-top1 cells. Target-common offsets have zero synthetic rank-flip rate, while candidate-specific perturbations produce mean flip rate `0.748861`. The extreme-order Shapley term uses an oracle-best-conditioned two-arm counterfactual only to quantify multiplicity pressure; it is not an action rule.

## Boundary

C72 is a read-only diagnostic mechanism audit. No forward pass, re-inference, training, GPU work, BNCI2014_004, reserved seed, control artifact, or checkpoint identity is emitted. Strict source-domain trial logits are unavailable, so that route was not tested and is not reported as a failed escape hatch. Representation intervention is unsupported because neither representation nor Wdotz fields exist. Conditional observability remains a block-aware proxy. Target-population generalization remains unresolved.
