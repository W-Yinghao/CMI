# C67 - Dual-Mode C66 Provenance / Masked Trial-Cache Consumption (frozen C19 `664007686afb520f`)

## 1. Executive Verdict

Primary: `C67-A_c66_dual_mode_provenance_reconciled`

Active: `C67-A_c66_dual_mode_provenance_reconciled ; C67-B_authorized_cache_integrity_validated ; C67-C_masked_view_contract_validated ; C67-D_split_label_smoke_feasible_not_sufficiency ; C67-E_split_label_smoke_underpowered_or_unstable ; C67-F_sample_level_conditional_cs_smoke_feasible ; C67-G_sample_level_conditional_cs_underpowered_or_unstable ; C67-H_endpoint_oracle_boundary_preserved ; C67-J_larger_reinference_only_cache_campaign_ready_but_not_authorized ; C67-K_new_training_still_not_justified`

Inactive: `C67-I_label_leakage_or_availability_violation_found`

Final gate: `C67_DUAL_MODE_MICROCACHE_VALID_BUT_UNDERPOWERED_FOR_SPLIT_LABEL_CS`

## 2. C66 Provenance

C66 is treated as a dual-mode milestone, not a science conflict. The no-auth baseline commit `635ccbc` records guard evidence: gate `MICROCAMPAIGN_READY_BUT_NOT_AUTHORIZED`, forward `0`, cache rows `0`.

The authorized microcampaign commit `b369f59` is the only mode consumed by C67: gate `REINFERENCE_ONLY_MICROCAMPAIGN_EXECUTED_AND_CACHE_MANIFESTED`, forward `1`, cache rows `3456`.

External cache SHA-256: `aaef9df53eed0c4ac2a38aa701f35f345958722ae9180ae3838084986a4c5e0d`. Only compact manifests and aggregate ledgers are committed.

## 3. Masked Cache Consumption

C67 reads the C66 external cache read-only. It does not train, run a new forward pass, use GPU, touch BNCI2014_004, use seeds [3,4], or emit selector/checkpoint recommendation artifacts.

The raw external CSV contains quarantined target labels, but C67 validates source-only, construction, evaluation, same-label-oracle, and conditional-CS diagnostic views through the C66 masking contract.

The source-only, construction, and evaluation views are enforced masked paths. The same-label-oracle and conditional-CS diagnostic views intentionally use the unmasked diagnostic cache, but are marked `policy_boundary_only=1`, `selection_path_enforced=0`, `available_at_selection_time=0`, and `diagnostic_only=1`.

## 4. Smoke Results

Split-label smoke: status `completed_underpowered`, checkpoint units `6`, hit `1.0`. This is diagnostic-only and does not establish few-label sufficiency.

Conditional-CS smoke: status `underpowered_or_unstable`, paired rows `1768`, independent checkpoint units `6`. This is a proxy smoke, not a full conditional-CS claim.

## 5. Boundary

C67 validates the microcache for diagnostic consumption, but the split-label and conditional-CS signals are underpowered at six checkpoint units. A larger re-inference-only cache campaign may be scientifically useful, but it is not authorized here.

## 6. Red-Team Verification

Red-team failures: `0`.
