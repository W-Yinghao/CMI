# C70 - Split-Label Information Budget / Gauge-Recovery Phase Diagram (frozen C19 `664007686afb520f`)

## 1. Executive Verdict

Primary: `C70-D_c69_signal_collapses_under_hierarchical_controls`

Active: `C70-D_c69_signal_collapses_under_hierarchical_controls ; C70-S1_finite_population_label_budget_bound_established ; C70-S2_paired_model_bound_nontrivial ; C70-S4_conditional_cs_proxy_only_or_bandwidth_sensitive ; C70-S5_no_strict_source_trial_escape_hatch ; C70-S7_t3_disjoint_confirmatory_protocol_locked ; C70-S8_target_population_generalization_unresolved ; C70-S9_new_training_not_justified`

Inactive: `C70-A_small_budget_split_label_gauge_recovery_candidate ; C70-B_medium_or_dense_label_recovery_only ; C70-C_split_label_reliability_without_actionability ; C70-E_claim_or_masking_inconsistency_requires_repair ; C70-S3_block_conditional_cs_stable_diagnostic ; C70-S6_strict_source_trial_escape_hatch_found`

Final gate: `C69_SIGNAL_COLLAPSES_UNDER_HIERARCHICAL_CONTROLS`

## 2. Read-Only Boundary

C70 consumes the manifested C69 T1/T2 external caches read-only. It does not run EEG forward passes, re-inference, training, GPU work, or T3 cache consumption.

## 3. Split Contract

The T2 cache contains `9` targets, `216` checkpoint-target units, and `5184` unique target trial IDs. Construction/evaluation trial IDs are shared across candidates within each target and disjoint.

## 4. Information-Budget Curve

At 8 labels/class: gauge recovery `0.176672`, actionability coverage `0.298611`, top1 hit `0.179583`.

At 16 labels/class: gauge recovery `0.245291`, actionability coverage `0.364583`, top1 hit `0.221648`.

At full construction: mean within-target Spearman `0.547343`, gauge recovery `0.392519`, regret `0.018161`.

The C70-D call is a collapse of the strong pooled/actionability interpretation, not a null-signal claim: within-target centered blocked permutation remains significant, but the effect is much smaller and does not support small-budget gauge recovery.

These are diagnostic information-cost curves. C70 does not claim few-label sufficiency, deployability, source-only rescue, or checkpoint selection.

## 5. Hierarchical / Permutation Controls

The primary blocked permutation shuffles held-out scores within target: observed `0.650508`, permutations `4999`, exceedances `0`, p `0.0002`, floor `0.0002`.

Target-cluster bootstrap bands are emitted. C70 remains conditional on the frozen BNCI2014_001 targets and does not make target-population p-value claims.

## 6. Conditional-CS and Strict Source

Conditional-CS remains `proxy_only_directionally_stable` with full conditional-CS claimed `0`. Strict source-domain trial logits/probs are absent from the C69 cache; metadata is not relabelled as strict source evidence.

## 7. T3-HO Protocol

C70 locks but does not execute a C71 T3-HO protocol. Protocol SHA-256: `9075e13d86192c48677b167457b765854db4f7d77781474753212b62d480e611`.
