# C74 Red-Team Verification

- Final status: `PASS`
- Checks: `33/33` passed
- Independent external payload rehash: `216/216 units`
- Main C74 report existed before red-team: `false`
- T3-HO z/Wz touched: `false`
- Same-label oracle in primary smoke input: `false`

## Repairs retained in provenance

The ledger records the cancelled initial P0 attempt, MNE lock isolation, protocol-aligned softmax identity, cross-node float32 drift audit, oracle-metadata isolation, and the nested incremental-null repair. Superseded smoke output is not used.

## Claim boundary

The cache validates instrumentation and makes representation/projection constructs analyzable. It does not validate a representation mechanism, a target gauge, a source-only escape hatch, a selector, or target-population generalization.

## Check ledger

| Check | Pass | Observed | Expected |
|---|---:|---|---|
| protocol_hash | 1 | 76d0f4d9d96d012856d72934a720c52e067e7f0ae27081ad5c97d6b4eea1acb6 | 76d0f4d9d96d012856d72934a720c52e067e7f0ae27081ad5c97d6b4eea1acb6 |
| protocol_commit_precedes_cache | 1 | 1783687468 | < 1783689027.1505842 |
| T2_exact_universe | 1 | 216 | 216 |
| T3_HO_zero_overlap | 1 | 0 | 0 |
| T3_HO_generation_flags_zero | 1 | 0 | 0 |
| P0_pilot_gate_self_hash | 1 | True | True |
| P0_pilot_gate | 1 | 54:P0_PILOT_ALL_GATES_PASSED | 54:P0_PILOT_ALL_GATES_PASSED |
| P0_pilot_payload_rehash_recorded | 1 | True | True |
| P1_expansion_gate_self_hash | 1 | True | True |
| P1_expansion_gate | 1 | 162:P1_EXPANSION_ALL_GATES_PASSED | 162:P1_EXPANSION_ALL_GATES_PASSED |
| P1_expansion_payload_rehash_recorded | 1 | True | True |
| independent_full_payload_rehash | 1 | 0 | 0 |
| label_partition_all_units | 1 | 0 | 0 |
| row_counts | 1 | 995328:124416 | 995328:124416 |
| identity_exact | 1 | 0.0 | 0.0 |
| execution_guards | 1 | 216 unit manifests | CPU eval no-grad no-update |
| restricted_input_hash | 1 | 9fd4a7c5aee0883a2edacf89a3beba6f5e7beb3dc9341f7d90980a31d14a263d | 9fd4a7c5aee0883a2edacf89a3beba6f5e7beb3dc9341f7d90980a31d14a263d |
| oracle_absent_primary_input | 1 | False | False |
| restricted_view_set | 1 | 216 | 216 units x 5 allowed views |
| cross_node_preprocessing_drift | 1 | input_max=4.656612873077393e-10;logit_max=0.0;pred_disagree=0 | within locked tolerances; zero prediction disagreements |
| analysis_information_boundary | 1 | oracle=0;T3=0;mechanism=0;escape=0 | all false |
| nested_incremental_null_semantics | 1 | 5 | 5 rows with nested new-block null |
| no_source_or_unlabeled_incremental_escape | 1 | [] | [] |
| projection_variance_accounting | 1 | 36 | 36 rows; sum=1 |
| projection_split_table | 1 | 36 | 36; oracle=0 |
| counterfactual_identity | 1 | rows=9;max_reconstruction=4.3958425521850586e-07 | 9 exact rank/utility controls; reconstruction<=1e-06 |
| risk_register_no_blocker | 1 | [] | [] |
| C73_metric_semantics_reconciled | 1 | 6 | all reconciled |
| raw_cache_not_in_git | 1 | 0 | 0 |
| git_payload_hygiene | 1 | 494087 | <50000000 |
| authorization_direct_CLI | 1 | direct literal CLI | direct literal CLI |
| forbidden_dataset_and_seeds_absent | 1 | [0, 1, 2] | [0, 1, 2] |
| main_report_not_preexisting | 1 | False | False |
