# C61 - Conditional Observability Divergence / Information-Ladder Audit (frozen C19 `664007686afb520f`)

## Primary Decision

`C61-A_conditional_observability_divergence_ladder_established`

Active: `C61-A_conditional_observability_divergence_ladder_established ; C61-B_conditional_observability_matches_hit_and_partition_bound_ladder ; C61-C_endpoint_scalar_dominates_incremental_observability ; C61-D_template_partial_observability_but_not_sufficient ; C61-E_source_key_conditional_sufficiency_fails ; C61-F_conditional_cs_estimator_unstable_but_partition_metrics_stable ; C61-H_synthetic_rank_gauge_cod_validation_successful ; C61-I_hard_theorem_to_eeg_bridge_not_required_for_framework`

Inactive: `C61-G_source_observable_cod_escape_hatch_found ; C61-J_future_instrumentation_needed_for_split_label_or_atom_trace ; C61-K_claim_or_availability_inconsistency_found`

## Result

C61 establishes a diagnostic conditional-observability ladder over the frozen C50-C60 universe. The primary evidence is a finite-partition binary-Y plug-in COD family: conditional TV, JS, Hellinger, CS proxy, and Bayes-hit gain.

The ladder aligns with the existing hit and finite-population boundary:

- source -> key: `0.506173` -> `0.487654` (`-0.018519`), so key-only information does not improve source endpoint observability.
- source -> template: `0.506173` -> `0.703704` (`+0.197531`), so the template is partial target-label-derived observability.
- source -> label diagnostic: `0.506173` -> `0.812757` (`+0.306584`), a strong diagnostic gain that is not source-available.
- source -> endpoint scalar: `0.506173` -> `0.944444` (`+0.438272`), the dominant same-label endpoint-oracle boundary.
- source + template -> endpoint scalar: `0.703704` -> `0.944444` (`+0.240741`), so template does not screen off endpoint.

The C55/C56/C60 null boundary is preserved: template-only `0.703704` is not claimed to beat max null p95 `0.771296`, while endpoint scalar `0.944444` does beat it. The endpoint scalar remains a same-label oracle and is unavailable at selection time.

## Estimator Boundary

The Conditional CS paper (`https://arxiv.org/abs/2301.08970`) is used as structural inspiration only: formal diagnostic -> estimator/nulls -> synthetic validation -> application-style audit -> limitations.

C61 does not claim a KDE/Gram conditional CS estimator from summary artifacts; raw conditional samples, per-trial logits, and direct gauge traces are missing. That is why `C61-F` is active: the CS-style estimator family is not the primary evidence, but the partition and binary-Y divergence metrics are stable.

## Synthetic Check

The synthetic rank-gauge COD validation keeps C60's repair intact. Candidate-specific gauge gaps can change endpoint observability and induce pair flips; pure target-local common offsets are carried as a negative control because they cannot flip within-target pair ranking.

## Availability Boundary

C61 keeps the information classes separate: strict source inputs are source-only, key/template/label-diagnostic rows are diagnostic-only, and same-label endpoint scalar rows read candidate target endpoint content. No target-label diagnostic row is marked source-only.

## Training Gate

`NO_TRAINING_C61_FROZEN_ARTIFACT_DIAGNOSTIC_ONLY`

C61 does not train, re-infer, use GPU, add BNCI2014_004, run seeds [3,4], start manuscript drafting, or create selector artifacts.
