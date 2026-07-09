"""C65 Frozen Checkpoint Recovery / Trial-Level Cache ABI Gate tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c65_frozen_checkpoint_recovery_trial_cache_gate as c65
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C65_FROZEN_CHECKPOINT_RECOVERY_TRIAL_CACHE_GATE.json"
TABLE_DIR = "oaci/reports/c65_tables"


def _summary() -> dict:
    with open(REPORT_JSON) as f:
        return json.load(f)


def _rows(name: str) -> list[dict]:
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_c65_decision_scope_and_gates_are_frozen():
    assert c65._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert set(c65.DECISIONS) == {
        "C65-A_frozen_checkpoint_weights_recovered_and_manifested",
        "C65-B_preprocessing_pipeline_recovered_and_manifested",
        "C65-C_frozen_checkpoint_universe_mapping_complete",
        "C65-D_reinference_only_trial_cache_campaign_ready_but_not_authorized",
        "C65-E_reinference_only_blocked_by_missing_weights",
        "C65-F_reinference_only_blocked_by_missing_preprocessing_or_data_contract",
        "C65-G_new_training_campaign_needed_if_recovery_fails_but_not_authorized",
        "C65-H_trial_cache_schema_and_split_label_contract_ready",
        "C65-I_full_conditional_cs_sample_schema_ready_but_cache_missing",
        "C65-J_atom_trace_requires_new_forward_hooks_or_training_instrumentation",
        "C65-K_reserved_holdout_boundary_preserved",
        "C65-L_claim_or_availability_inconsistency_found",
    }
    d = _summary()
    assert d["milestone"] == "C65"
    assert d["config_hash"] == "664007686afb520f"
    assert d["c64_commit"] == "4c06fff"
    assert d["c64_decision"] == "C64-A_frozen_summary_artifact_paths_saturated"
    assert d["decision"]["primary"] == "C65-D_reinference_only_trial_cache_campaign_ready_but_not_authorized"
    assert d["final_gate"] == "REINFERENCE_ONLY_CAMPAIGN_READY_BUT_NOT_AUTHORIZED"
    assert d["training_gate"] == c65.TRAINING_GATE
    assert d["reinference_gate"] == c65.REINFERENCE_GATE
    assert d["gpu_gate"] == c65.GPU_GATE
    assert d["decision"]["red_team_failure_count"] == 0
    for active in (
        "C65-A_frozen_checkpoint_weights_recovered_and_manifested",
        "C65-B_preprocessing_pipeline_recovered_and_manifested",
        "C65-C_frozen_checkpoint_universe_mapping_complete",
        "C65-D_reinference_only_trial_cache_campaign_ready_but_not_authorized",
        "C65-H_trial_cache_schema_and_split_label_contract_ready",
        "C65-I_full_conditional_cs_sample_schema_ready_but_cache_missing",
        "C65-J_atom_trace_requires_new_forward_hooks_or_training_instrumentation",
        "C65-K_reserved_holdout_boundary_preserved",
    ):
        assert active in d["decision"]["active"]
    for inactive in (
        "C65-E_reinference_only_blocked_by_missing_weights",
        "C65-F_reinference_only_blocked_by_missing_preprocessing_or_data_contract",
        "C65-G_new_training_campaign_needed_if_recovery_fails_but_not_authorized",
        "C65-L_claim_or_availability_inconsistency_found",
    ):
        assert inactive in d["decision"]["inactive"]


def test_c65_recovered_checkpoint_store_is_manifested_without_accepting_adjacent_weights():
    d = _summary()
    gate = d["gate_decision"]
    assert gate["checkout_checkpoint_weight_files_found"] == 0
    assert gate["oaci_checkpoint_weight_files_found"] == 5454
    assert gate["checkpoint_json_sidecars_found"] == 5454
    assert gate["checkpoint_artifact_index_count"] == 27
    assert gate["external_weight_sightings_found"] >= 6
    assert gate["training_authorized"] is False
    assert gate["reinference_authorized"] is False
    assert gate["gpu_authorized"] is False
    assert gate["reinference_only_ready"] is True

    candidates = _rows("checkpoint_candidate_manifest.csv")
    primary = candidates[0]
    assert primary["path"] == c65.PRIMARY_OACI_STORE
    assert primary["exists"] == "1"
    assert primary["checkpoint_id_candidates"] == "checkpoint_pt=5454;checkpoint_json=5454"
    assert primary["metadata_inferable"] == "artifact_index_count=27"
    assert primary["safe_to_load_cpu_metadata"] == "0"
    assert primary["oaci_frozen_universe_candidate"] == "1"
    assert "no_bulk_rehash" in primary["sha256_if_reasonable"]
    assert all(r["oaci_frozen_universe_candidate"] == "0" for r in candidates[1:])

    missing = {r["missing_item"]: r for r in _rows("checkpoint_missing_manifest.csv")}
    assert missing["checkout_weight_files"]["present"] == "0"
    assert missing["checkout_weight_files"]["blocks_reinference_only"] == "1"
    assert missing["oaci_frozen_weight_files"]["present"] == "1"
    assert missing["oaci_frozen_weight_files"]["blocks_reinference_only"] == "0"


def test_c65_abi_validation_is_sidecar_only_and_shape_compatible():
    abi = {r["check"]: r for r in _rows("checkpoint_abi_validation.csv")}
    assert abi["checkpoint_store_abi_code_present"]["status"] == "pass"
    assert abi["train_checkpoint_record_abi_present"]["status"] == "pass"
    assert abi["recovered_weight_cpu_metadata_load"]["status"] == "sidecar_metadata_present_not_torch_loaded"
    assert abi["model_class_and_head_dim_match"]["status"] == "sidecar_model_spec_matches_shallowconvnet_22x385_4class"
    assert abi["normalization_state_buffers_match"]["status"] == "batchnorm_buffers_listed_in_sidecar"
    assert {r["cpu_load_attempted"] for r in abi.values()} == {"0"}
    assert {r["blocks_campaign"] for r in abi.values()} == {"0"}

    state = {r["source"]: r for r in _rows("checkpoint_state_dict_key_summary.csv")}
    sample = state["recovered_checkpoint_sidecar_sample"]
    assert sample["loaded"] == "0"
    assert sample["key_count"] == "10"
    assert "classifier.weight[4,800]" in sample["shape_signature"]
    assert "spatial.weight[40,40,22,1]" in sample["shape_signature"]
    assert "metadata_sidecar_only_no_torch_load" == sample["status"]


def test_c65_preprocess_dataset_and_holdout_contracts_are_recovered_and_reserved():
    contracts = {r["contract_item"]: r for r in _rows("preprocess_contract_manifest.csv")}
    assert contracts["bandpass"]["value"] == "fmin=4.0;fmax=38.0"
    assert contracts["epoch_window"]["value"] == "tmin=0.5;tmax=3.5;expected_n_times=385"
    assert contracts["channel_order"]["value"] == "22_frozen_BNCI001_channels"
    assert contracts["class_order"]["value"] == "left_hand|right_hand|feet|tongue"
    assert contracts["exact_model_runtime_config"]["value"] == "shallow_convnet_input_22x385_nclasses4"
    assert {r["blocks_campaign"] for r in contracts.values()} == {"0"}

    split = {r["contract"]: r for r in _rows("dataset_split_contract.csv")}
    assert split["dataset"]["value"] == "BNCI2014_001"
    assert split["historical_model_seeds"]["value"] == "0,1,2"
    assert split["reserved_seeds"]["value"] == "3,4"
    assert split["reserved_dataset"]["value"] == "BNCI2014_004"
    holdout = {r["resource"]: r for r in _rows("reserved_holdout_policy.csv")}
    for resource in ("BNCI2014_004", "seed_3", "seed_4"):
        assert holdout[resource]["used_in_c65"] == "0"
        assert holdout[resource]["released"] == "0"


def test_c65_frozen_universe_checkpoint_mapping_is_complete_and_singleton_safe():
    mapping = _rows("frozen_universe_checkpoint_map.csv")
    assert len(mapping) == 3804
    assert {r["file_status"] for r in mapping} == {"pt+json_verified"}
    assert {r["pt_exists"] for r in mapping} == {"1"}
    assert {r["json_exists"] for r in mapping} == {"1"}
    assert len({r["checkpoint_id"] for r in mapping}) == 1268
    first = mapping[0]
    assert first["row_id"] == "c50q_0000"
    assert first["candidate_id"] == "s0_t001_l000_S0_full_support_o000"
    assert first["checkpoint_prefix"] == "4b4247ebfcfb"
    assert first["pt_path"].startswith(c65.PRIMARY_OACI_STORE)
    assert first["json_path"].startswith(c65.PRIMARY_OACI_STORE)
    assert len(first["pt_file_sha256"]) == 64

    summary = {r["metric"]: r for r in _rows("mapping_completeness_summary.csv")}
    assert summary["c17_model_hash_rows"]["value"] == "1268"
    assert summary["c50_singleton_candidate_rows"]["value"] == "3804"
    assert summary["verified_c50_singleton_paths"]["value"] == "3804"
    assert summary["unique_checkpoint_ids_in_c50"]["value"] == "1268"
    assert summary["recovered_oaci_weight_files"]["value"] == "5454"
    assert summary["summary_to_weight_file_mapping_complete"]["passed"] == "1"
    assert summary["repo_only_checkpoint_files"]["value"] == "0"
    assert summary["repo_only_checkpoint_files"]["passed"] == "0"

    unmapped = {r["unmapped_group"]: r for r in _rows("unmapped_checkpoint_rows.csv")}
    assert unmapped["C50_singleton_candidate_rows"]["row_count"] == "0"
    assert unmapped["C50_singleton_candidate_rows"]["blocks_mapping_complete"] == "0"
    assert unmapped["repo_only_checkpoint_files"]["row_count"] == "0"


def test_c65_trial_cache_split_label_cs_and_atom_boundaries_are_protocol_only():
    fields = {r["field"]: r for r in _rows("trial_cache_minimal_fields.csv")}
    assert len(fields) == 26
    for required in (
        "dataset_id",
        "trial_id",
        "class_label",
        "checkpoint_id",
        "logits",
        "probabilities",
        "predicted_class",
        "representation_z_path",
        "projection_Wz",
        "split_label_role",
    ):
        assert required in fields
    assert fields["class_label"]["target_label_dependent"] == "1"
    assert fields["logits"]["requires_forward_output"] == "1"
    assert fields["representation_z_path"]["large_payload_ref_only"] == "1"

    split = _rows("split_label_budget_grid.csv")
    assert len(split) == 4
    assert {r["same_label_reuse_allowed"] for r in split} == {"0"}
    same_label = {r["guard"]: r for r in _rows("same_label_oracle_guard.csv")}
    assert same_label["same_candidate_endpoint_scalar"]["forbidden_in_split_label_feature"] == "1"
    assert same_label["few_label_claim"]["allowed_as_diagnostic_oracle"] == "0"

    cs = {r["audit"]: r for r in _rows("conditional_cs_variable_map.csv")}
    assert {r["supported_now"] for r in cs.values()} == {"0"}
    assert cs["split_label_increment"]["future_reinference_cache_support"] == "1"
    assert cs["target_unlabeled_geometry_increment"]["requires_new_training"] == "0"
    assert cs["atom_trace_identity"]["requires_new_training"] == "1"
    atom = {r["trace"]: r for r in _rows("atom_trace_requires_forward_or_training.csv")}
    assert atom["per_trial_logits_probabilities"]["recovered_by_reinference_if_weights"] == "1"
    assert atom["domain_class_leakage_atom_identity"]["new_training_required"] == "1"


def test_c65_mock_interfaces_do_not_use_real_eeg_or_real_checkpoints():
    cache = _rows("mock_trial_cache_writer_test.csv")
    assert len(cache) == 3
    assert {r["uses_real_eeg"] for r in cache} == {"0"}
    assert {r["uses_real_checkpoint"] for r in cache} == {"0"}
    assert {r["passed"] for r in cache} == {"1"}
    cs = _rows("mock_conditional_cs_interface_test.csv")
    assert len(cs) == 3
    assert {r["uses_real_eeg"] for r in cs} == {"0"}
    assert {r["passed"] for r in cs} == {"1"}
    fixture = _rows("synthetic_rank_gauge_cache_fixture.csv")
    assert len(fixture) == 3
    assert {r["checkpoint_id"].startswith("mock_") for r in fixture} == {True}


def test_c65_artifact_hygiene_red_team_and_manifest_hashes_pass():
    d = _summary()
    assert d["table_row_counts"] == {
        "artifact_manifest": 39,
        "atom_trace_requires_forward_or_training": 4,
        "campaign_cost_risk_matrix": 4,
        "checkpoint_abi_validation": 5,
        "checkpoint_candidate_manifest": 10,
        "checkpoint_missing_manifest": 6,
        "checkpoint_state_dict_key_summary": 3,
        "conditional_cs_variable_map": 5,
        "dataset_split_contract": 7,
        "forbidden_claim_scan": 16,
        "frozen_universe_checkpoint_map": 3804,
        "instrumentation_value_of_information": 6,
        "large_artifact_scan": 39,
        "mapping_completeness_summary": 7,
        "mock_conditional_cs_interface_test": 3,
        "mock_trial_cache_writer_test": 3,
        "preprocess_contract_manifest": 10,
        "red_team_failure_ledger": 16,
        "reserved_holdout_policy": 4,
        "same_label_oracle_guard": 4,
        "schema_validation_summary": 25,
        "search_scope_manifest": 6,
        "split_label_budget_grid": 4,
        "synthetic_rank_gauge_cache_fixture": 3,
        "test_command_manifest": 4,
        "trial_cache_minimal_fields": 26,
        "unmapped_checkpoint_rows": 5,
    }
    red = _rows("red_team_failure_ledger.csv")
    assert len(red) == 16
    assert {r["failed"] for r in red} == {"0"}
    tests = {r["test_scope"]: r for r in _rows("test_command_manifest.csv")}
    assert set(tests) == {"focused_c65", "c50_c65_slice", "c23_c65_regression", "full_oaci_tests"}
    assert {r["status"] for r in tests.values()} == {"green"}
    assert {r["slurm_partition"] for r in tests.values()} == {"cpu-high"}
    assert {r["environment"] for r in tests.values()} == {"eeg2025"}
    forbidden = _rows("forbidden_claim_scan.csv")
    assert len(forbidden) == 16
    assert {r["affirmative_hits"] for r in forbidden} == {"0"}
    assert {r["passed"] for r in forbidden} == {"1"}
    large = _rows("large_artifact_scan.csv")
    assert len(large) == 39
    assert {r["over_50mb"] for r in large} == {"0"}
    assert {r["passed"] for r in large} == {"1"}

    manifest = _rows("artifact_manifest.csv")
    assert len(manifest) == 39
    for row in manifest:
        assert os.path.exists(row["path"])
        assert int(row["size_bytes"]) == os.path.getsize(row["path"])
        assert row["sha256"] == _sha256(row["path"])

    report_text = open("oaci/reports/C65_FROZEN_CHECKPOINT_RECOVERY_TRIAL_CACHE_GATE.md").read()
    assert "if weights are restored" not in report_text
    assert "C65 does not train, re-infer, use GPU" in report_text
    red_report = open("oaci/reports/C65_RED_TEAM_VERIFICATION.md").read()
    assert "full_oaci_tests job" in red_report
    assert "1344 passed" in red_report


def test_c65_generator_replays_current_primary_gate_without_writing_payloads():
    res = c65.run(test_status="planned")
    assert res["decision"]["primary"] == "C65-D_reinference_only_trial_cache_campaign_ready_but_not_authorized"
    assert res["c65_gate_decision"]["oaci_checkpoint_weight_files_found"] == 5454
    assert res["c65_gate_decision"]["checkout_checkpoint_weight_files_found"] == 0
    assert res["c65_gate_decision"]["reinference_only_ready"] is True
    assert len(res["frozen_universe_checkpoint_map_rows"]) == 3804
    assert res["decision"]["red_team_failure_count"] == 0
