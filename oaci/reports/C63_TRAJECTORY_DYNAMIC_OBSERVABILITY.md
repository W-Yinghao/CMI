# C63 - Trajectory-Dynamic Conditional Observability / Hankel-Ladder Audit (frozen C19 `664007686afb520f`)

## Primary Decision

`C63-A_dynamic_conditional_observability_ladder_established`

Active: `C63-A_dynamic_conditional_observability_ladder_established ; C63-C_source_dynamic_history_near_static_source_only ; C63-D_source_dynamic_template_partial_but_no_screen_off_endpoint ; C63-E_endpoint_scalar_still_dominates_after_dynamic_conditioning ; C63-G_no_dynamic_source_observable_escape_hatch_found ; C63-I_trajectory_fragmentation_not_explained_by_source_dynamics ; C63-J_synthetic_dynamic_rank_gauge_validation_successful ; C63-K_full_time_series_conditional_cs_requires_trial_level_cache ; C63-L_training_not_authorized`

Inactive: `C63-B_source_dynamic_history_adds_stable_observability ; C63-F_dynamic_source_observable_escape_hatch_found ; C63-H_trajectory_fragmentation_explained_by_source_dynamics ; C63-M_claim_or_availability_inconsistency_found`

## Result

C63 establishes a compact Hankel-style dynamic conditional-observability ladder over frozen trajectory summaries. It does not construct raw trajectory windows or a full time-series conditional-CS estimator.

- static source -> source dynamic history: `0.506173` -> `0.574074` (`+0.067901`)
- static source -> source delta history: `0.506173` -> `0.481481` (`-0.024691`)
- static source + template -> source dynamic history: `0.703704` -> `0.720679` (`+0.016975`)
- source dynamic + template -> endpoint scalar: `0.720679` -> `0.944444` (`+0.223765`)

The source-dynamic increment is weak and stays below the reliability boundary. Dynamic+template remains partial and does not screen off the endpoint scalar.

## Fragmentation

C51 q20/min1 trajectory actionability fail fraction remains `0.432099`. C63 attributes this to residual target/trajectory gauge and source-score underuse, not to recoverable source-dynamic history in the committed summaries.

## Boundary

Template-only remains below max null p95 (`0.703704` < `0.771296`), while endpoint scalar remains above it (`0.944444` > `0.771296`). The endpoint scalar is a same-label target endpoint oracle and unavailable at selection time.

## Gate

`TRAINING_NOT_AUTHORIZED_IN_C63`

C63 does not train, re-infer, use GPU, add BNCI2014_004, run seeds [3,4], create selector artifacts, or start manuscript drafting.
