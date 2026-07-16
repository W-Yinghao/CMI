# C84A Final Report Red Team

Final gate: `C84_POST_SCIENTIFIC_HETEROGENEITY_AND_TPAMI_THEORY_BRIDGE_AUDIT_COMPLETE_C85_PROTOCOL_REVIEW_REQUIRED`

| Check | Pass | Evidence |
|---|---:|---|
| identity_replay | 1 | 10/10 |
| result_manifest_replay | 1 | 18/18 |
| result_table_count | 1 | 18 |
| result_table_rows | 1 | all nonempty |
| lifecycle_replay | 1 | Stage A/B/C |
| protected_counters | 1 | 11/11 |
| historical_verification | 1 | 6/6 |
| full_gate_rows | 1 | 18 |
| level_gate_rows | 1 | 36 |
| cott_target_rows | 1 | 118 |
| cott_Lee_worst | 1 | -0.1078732437869526 |
| cott_Lee_target | 1 | target 8 |
| cott_Cho_target | 1 | target 3 |
| cott_Physionet_floor_count | 1 | 9 |
| cott_LOTO_category_change | 1 | Lee target 8 only |
| mano_Cho_Q1_Q2 | 1 | PASS/PASS |
| mano_Cho_ERM | 1 | 160/160 |
| mano_action_density_unidentified | 1 | not in compact tables |
| frontier_rows | 1 | 15 |
| Lee_FULL_tail_failure | 1 | worst component |
| Cho_B8_qualification | 1 | B*=8 |
| Physionet_FULL_failures | 1 | maxT/tail |
| transport_rows | 1 | 45 |
| theory_gaps | 1 | 7 |
| next_experiments_unauthorized | 1 | all |
| tag_c84s_identity_replay.csv | 1 | 10 |
| source_keys_c84s_identity_replay.csv | 1 | 10 |
| tag_result_table_manifest_replay.csv | 1 | 18 |
| source_keys_result_table_manifest_replay.csv | 1 | 18 |
| tag_lifecycle_stage_replay.csv | 1 | 3 |
| source_keys_lifecycle_stage_replay.csv | 1 | 3 |
| tag_protected_counter_replay.csv | 1 | 11 |
| source_keys_protected_counter_replay.csv | 1 | 11 |
| tag_regression_redteam_replay.csv | 1 | 6 |
| source_keys_regression_redteam_replay.csv | 1 | 6 |
| tag_full_panel_gate_component_matrix.csv | 1 | 18 |
| source_keys_full_panel_gate_component_matrix.csv | 1 | 18 |
| tag_level_specific_gate_component_matrix.csv | 1 | 36 |
| source_keys_level_specific_gate_component_matrix.csv | 1 | 36 |
| tag_near_boundary_gate_failures.csv | 1 | 387 |
| source_keys_near_boundary_gate_failures.csv | 1 | 387 |
| tag_cott_target_effect_distribution.csv | 1 | 118 |
| source_keys_cott_target_effect_distribution.csv | 1 | 118 |
| tag_cott_target_influence.csv | 1 | 118 |
| source_keys_cott_target_influence.csv | 1 | 118 |
| tag_cott_average_tail_separation.csv | 1 | 3 |
| source_keys_cott_average_tail_separation.csv | 1 | 3 |
| tag_cott_cross_cohort_recurrence.csv | 1 | 3 |
| source_keys_cott_cross_cohort_recurrence.csv | 1 | 3 |
| tag_mano_cross_dataset_decision_profile.csv | 1 | 3 |
| source_keys_mano_cross_dataset_decision_profile.csv | 1 | 3 |
| tag_rank_topk_regret_separation.csv | 1 | 18 |
| source_keys_rank_topk_regret_separation.csv | 1 | 18 |
| tag_label_frontier_component_matrix.csv | 1 | 15 |
| source_keys_label_frontier_component_matrix.csv | 1 | 15 |
| tag_label_frontier_closure_failures.csv | 1 | 15 |
| source_keys_label_frontier_closure_failures.csv | 1 | 15 |
| tag_level_label_complexity_interaction.csv | 1 | 15 |
| source_keys_level_label_complexity_interaction.csv | 1 | 15 |
| tag_c82_c84_method_transport_matrix.csv | 1 | 45 |
| source_keys_c82_c84_method_transport_matrix.csv | 1 | 45 |
| tag_heterogeneity_axis_matrix.csv | 1 | 7 |
| source_keys_heterogeneity_axis_matrix.csv | 1 | 7 |
| tag_information_policy_action_geometry_matrix.csv | 1 | 6 |
| source_keys_information_policy_action_geometry_matrix.csv | 1 | 6 |
| tag_theory_gap_registry.csv | 1 | 7 |
| source_keys_theory_gap_registry.csv | 1 | 7 |
| tag_next_experiment_decision_matrix.csv | 1 | 6 |
| source_keys_next_experiment_decision_matrix.csv | 1 | 6 |
| static_no_scientific_import | 1 | ['__future__', 'argparse', 'ast', 'csv', 'hashlib', 'json', 'math', 'pathlib', 'subprocess', 'typing'] |
| no_array_loader | 1 | no array/checkpoint load call |
| no_new_pvalue_engine | 1 | frozen p-values copied only |
| immutable_gate | 1 | C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous |
| immutable_frontier | 1 | C84-L4 |
| git_payload_hygiene | 1 | no generated file >50 MiB |

All checks passed. C84-D and C84-L4 remain immutable. C85 and manuscript work remain unauthorized.

## Post-Regression Closeout

| Check | Pass | Evidence |
|---|---:|---|
| focused_regression | 1 | job 898806; 256 passed; stderr empty |
| C65_regression | 1 | job 898807; 867 passed, 1 skipped, 3 deselected; stderr empty |
| C23_regression | 1 | job 898808; 1,278 passed, 1 skipped, 3 deselected; stderr empty |
| full_regression | 1 | job 898809; 2,202 passed, 1 skipped, 3 deselected; stderr empty |
| initial_failure_preserved | 1 | wrong Python 3.9.13 attempt disclosed and rejected |
| result_report_checksum | 1 | Markdown and JSON sidecar replay after regression closeout |
| active_job_count | 1 | squeue reports zero C84/C85 jobs |
| sacct_claim_guard | 1 | sacct not used or claimed |
| C85_authorization_guard | 1 | false |
| manuscript_authorization_guard | 1 | false |
