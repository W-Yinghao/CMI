# C71 - T3-HO Hierarchical Confirmation / Measurement-Control Separation Audit (frozen C19 `664007686afb520f`)

## 1. Executive Verdict

Primary: `C71-A_within_target_split_label_reliability_confirmed_actionability_weak`

Active: `C71-A_within_target_split_label_reliability_confirmed_actionability_weak ; C71-S1_T3_HO_disjointness_confirmed ; C71-S2_physical_view_isolation_passed ; C71-S3_candidate_specific_gauge_recovery_partial ; C71-S4_common_offset_not_explanatory ; C71-S5_no_strict_source_escape_hatch ; C71-S8_conditional_cs_proxy_only ; C71-S9_target_population_generalization_unresolved ; C71-S10_new_training_not_justified`

Inactive: `C71-B_small_budget_split_label_actionability_confirmed ; C71-C_dense_label_partial_recovery_confirmed ; C71-D_C70_effect_not_replicated_on_T3_HO ; C71-E_hierarchical_signal_replication_but_measurement_control_gap_narrows ; C71-F_protocol_masking_or_dependency_blocker ; C71-G_T3_HO_ready_but_not_authorized ; C71-S6_strict_source_escape_hatch_found ; C71-S7_conditional_observability_stable_diagnostic ; C71-S11_independent_target_or_dataset_replication_now_justified`

Final gate: `T3_HO_CONFIRMS_MEASUREMENT_CONTROL_SEPARATION`

## 2. Authorization Boundary

C71 exact CLI authorization status: `present`. This run executed the authorized frozen-checkpoint T3-HO re-inference path.

Observed execution counters: forward/re-inference `1`, training `0`, GPU `0`, T3 cache consumption `1`, raw cache rows emitted externally `605952`.

T3-HO external cache rows: `605952`; path hash `8e9d36764c8621fe791070cdd9d92eea69adcb4c3ba91544be833e58b949ec7f`. Raw rows are external-only and content-addressed.

## 3. Protocol Lock

C70 parent protocol SHA-256: `9075e13d86192c48677b167457b765854db4f7d77781474753212b62d480e611`.

C71 prospective protocol SHA-256: `984f4ca5a2ab57e679cbb07ab42c6ac0ccb1d937655e6fdab035b670ff19e800`.

C71 protocol lock timestamp: `2026-07-10T02:03:58Z`.

First T3-HO manifest/path access timestamp: `2026-07-10T02:03:59Z`.

First T3-HO outcome access timestamp: `2026-07-10T02:03:59Z`.

## 4. Readiness Ledger

Parent C70 records `1268` full physical units, `216` T2 consumed units, and `1052` T3-HO disjoint units.

C71 emits the risk register, disjointness ledger, overlap matrix, split contract, physical-view manifest, dependency summary, hypothesis table, hierarchical inference outputs, conditional-observability contracts, feature provenance, and failure ledger for the authorized T3-HO run.

## 5. Confirmatory Results

H1 blocked permutation: observed `0.637483`, permutations `4999`, exceedances `0`, p `0.0002`.

At 8 labels/class: Spearman `0.30747`, gauge recovery `0.17053`, coverage `0.099392`, top1 `0.050749`.

At 64 labels/class: Spearman `0.55075`, gauge recovery `0.423187`, coverage `0.434028`, top1 `0.236762`.

At full construction: Spearman `0.557457`, gauge recovery `0.43452`, coverage `0.444444`, top1 `0.222222`.

These are diagnostic split-label measurement results. They are not a selector, not checkpoint recommendations, not source-only rescue, and not target-population generalization.
