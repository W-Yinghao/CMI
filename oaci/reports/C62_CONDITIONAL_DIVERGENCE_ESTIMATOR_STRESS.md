# C62 - Conditional-Divergence Estimator Stress / Discrete-to-Kernel Bridge Audit (frozen C19 `664007686afb520f`)

## Primary Decision

`C62-A_C61_ladder_reproduced`

Active: `C62-A_C61_ladder_reproduced ; C62-B_partition_cod_ladder_stable_under_smoothing_and_support_stress ; C62-D_summary_kernel_proxy_only_not_full_conditional_cs ; C62-E_kernel_or_cs_proxy_unstable_but_partition_metrics_stable ; C62-F_endpoint_scalar_dominates_incremental_observability_across_estimators ; C62-G_template_partial_observability_but_no_screen_off ; C62-I_no_source_observable_estimator_escape_hatch_found ; C62-J_synthetic_rank_gauge_estimator_validation_successful ; C62-K_trial_level_or_atom_instrumentation_needed_for_full_cs_or_split_label ; C62-L_no_new_training_authorized`

Inactive: `C62-C_full_conditional_cs_estimator_supported_and_matches_ladder ; C62-H_source_observable_estimator_escape_hatch_found ; C62-M_claim_or_availability_inconsistency_found`

## Result

C62 reproduces the C61 conditional-observability ladder exactly and stress-tests the estimator layer. The stable evidence remains the finite-partition and binary-Y divergence family, not a sample-level conditional-CS estimator.

- source -> key: `0.506173` -> `0.487654` (`-0.018519`)
- source -> template: `0.506173` -> `0.703704` (`+0.197531`)
- source -> label diagnostic: `0.506173` -> `0.812757` (`+0.306584`)
- source -> endpoint scalar: `0.506173` -> `0.944444` (`+0.438272`)
- source + template -> endpoint scalar: `0.703704` -> `0.944444` (`+0.240741`)

Partition smoothing and support stress preserve the ordering: endpoint > label diagnostic > template > source > key. The summary-kernel proxy agrees on endpoint dominance, but it is bandwidth-sensitive and proxy-only because current artifacts are summary-level.

## Estimator Boundary

Full sample-level conditional CS remains unsupported by the current artifact set. Missing items are per-trial paired variables, logits/probabilities, split-label cache, representation tensors, and atom traces. C62 therefore activates `C62-D` and `C62-E`, not `C62-C`.

## Null Boundary

Template-only remains below the max null boundary (`0.703704` < `0.771296`), while the endpoint scalar remains above it (`0.944444` > `0.771296`). The endpoint scalar is still a same-label target endpoint oracle and unavailable at selection time.

## Gate

`TRAINING_NOT_AUTHORIZED_IN_C62`

C62 does not train, re-infer, use GPU, add BNCI2014_004, run seeds [3,4], create selector artifacts, or start manuscript drafting.
