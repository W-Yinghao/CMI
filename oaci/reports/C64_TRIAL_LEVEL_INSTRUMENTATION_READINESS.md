# C64 - Trial-Level Instrumentation Readiness / Split-Label-CS Evidence Gate (frozen C19 `664007686afb520f`)

## Primary Decision

`C64-A_frozen_summary_artifact_paths_saturated`

Active: `C64-A_frozen_summary_artifact_paths_saturated ; C64-C_new_training_required_for_trial_cache_or_atom_trace ; C64-D_split_label_cache_protocol_ready_but_not_authorized ; C64-E_full_time_series_conditional_cs_protocol_ready_but_not_authorized ; C64-F_atom_trace_protocol_ready_but_not_authorized ; C64-G_independent_checkpoint_replication_protocol_ready_but_not_authorized`

Inactive: `C64-B_reinference_only_trial_cache_campaign_sufficient ; C64-H_instrumentation_not_scientifically_justified_yet ; C64-I_source_observable_escape_hatch_remaining ; C64-J_claim_or_availability_inconsistency_found`

## Gate

`TRIAL_LEVEL_CACHE_CAMPAIGN_READY_BUT_NOT_AUTHORIZED`

Re-inference subgate: `CONDITIONALLY_SUFFICIENT_IF_FROZEN_CHECKPOINTS_AND_PREPROCESSING_ARE_RECOVERABLE`.

Current checkout checkpoint weight files found: `0`. Re-inference-only is the preferred low-confound path if frozen checkpoints, preprocessing artifacts, and labels are recoverable, but C64 does not authorize or execute it.

## Result

C64 finds frozen summary artifact paths saturated. The next meaningful evidence is trial-level instrumentation: split-label cache, sample-level conditional-CS/Hankel variables, atom trace schema, and independent replication protocol.

Split-label and full conditional-CS protocols are ready but not authorized. Atom trace and independent replication protocols are also ready but not authorized.

New training is not needed for split-label or full conditional-CS if frozen checkpoints can be re-inferred. New instrumented training would be needed for atom trace or independent checkpoint-field replication, but it is not authorized.

## Boundary

Template-only remains below max null p95 (`0.703704` < `0.771296`), while endpoint scalar remains above it (`0.944444` > `0.771296`). The endpoint scalar remains a same-label target endpoint oracle and unavailable at selection time.

## Execution Gate

`TRAINING_NOT_AUTHORIZED_IN_C64`

C64 does not train, re-infer, use GPU, add BNCI2014_004, run seeds [3,4], create selector artifacts, or start manuscript drafting.
