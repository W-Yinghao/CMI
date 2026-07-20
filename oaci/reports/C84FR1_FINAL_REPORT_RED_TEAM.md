# C84FR1 Final Report Red Team

- RT01 `failed_job_preserved`: PASS
- RT02 `authorization_not_reused`: PASS
- RT03 `model_manifest_exact`: PASS
- RT04 `7776_model_artifacts_replayed`: PASS
- RT05 `target_artifacts_absent_in_failed_root`: PASS
- RT06 `2430_canary_artifacts_replayed_before_target_access`: PASS
- RT07 `canonical_field_set`: PASS
- RT08 `canonical_tuple_sort`: PASS
- RT09 `insertion_order_irrelevant`: PASS
- RT10 `missing_field_fails`: PASS
- RT11 `unknown_field_fails`: PASS
- RT12 `target_only_entrypoint`: PASS
- RT13 `no_training_import`: PASS
- RT14 `no_training_callable`: PASS
- RT15 `fresh_output_root`: PASS
- RT16 `raw_manifest_exact_replay`: PASS
- RT17 `target_registry_before_forward`: PASS
- RT18 `target_y_forbidden`: PASS
- RT19 `scientific_metrics_forbidden`: PASS
- RT20 `oracle_forbidden`: PASS
- RT21 `C84S_forbidden`: PASS
- RT22 `numerical_gates_unchanged`: PASS
- RT23 `candidate_scope_unchanged`: PASS
- RT24 `target_scope_unchanged`: PASS
- RT25 `protocol_precedes_implementation`: PASS
- RT26 `fresh_authorization_required`: PASS
- RT27 `focused_regression_242_passed`: PASS
- RT28 `C65_regression_728_passed_1_skip_3_deselected`: PASS
- RT29 `C23_regression_1139_passed_1_skip_3_deselected`: PASS
- RT30 `full_regression_2063_passed_1_skip_3_deselected`: PASS
- RT31 `all_regression_stderr_empty`: PASS
- RT32 `skip_and_deselection_reasons_exact`: PASS
- RT33 `no_active_C84FR1_job_by_squeue`: PASS
- RT34 `Git_payload_and_artifact_hygiene`: PASS

Gate: `C84F_TARGET_STAGE_CANONICAL_REGISTRY_REPAIR_LOCKED_READY_FOR_PI_REAUTHORIZATION`. Result: 34/34 PASS.
