"""C66 Re-inference-only trial-level cache microcampaign gate tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c66_reinference_only_trial_cache_microcampaign as c66
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C66_REINFERENCE_ONLY_TRIAL_CACHE_MICROCAMPAIGN.json"
TABLE_DIR = "oaci/reports/c66_tables"


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


def test_c66_decision_scope_authorization_and_primary_gate_are_frozen():
    assert c66._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert set(c66.DECISIONS) == {
        "C66-A_reinference_only_microcampaign_authorized_and_executed",
        "C66-B_no_authorization_protocol_only",
        "C66-C_cpu_torchload_abi_validated_no_forward",
        "C66-D_checkpoint_abi_or_state_dict_mismatch_found",
        "C66-E_preprocessing_dataset_contract_validated",
        "C66-F_preprocessing_dataset_contract_blocked",
        "C66-G_trial_level_cache_schema_validated",
        "C66-H_minimal_trial_cache_emitted_and_manifested",
        "C66-I_split_label_protocol_feasible_on_cache",
        "C66-J_sample_level_conditional_cs_feasible_on_cache",
        "C66-K_atom_trace_forward_hooks_feasible_without_training",
        "C66-L_reinference_only_path_blocked_new_training_may_be_needed_but_not_authorized",
        "C66-M_claim_or_availability_inconsistency_found",
    }
    d = _summary()
    assert d["milestone"] == "C66"
    assert d["config_hash"] == "664007686afb520f"
    assert d["c65_commit"] == "192a82d"
    assert d["c65_decision"] == "C65-D_reinference_only_trial_cache_campaign_ready_but_not_authorized"
    assert d["authorization_phrase_required"] == c66.AUTH_PHRASE
    assert d["authorization_present"] is False
    assert d["decision"]["primary"] == "C66-B_no_authorization_protocol_only"
    assert d["final_gate"] == "MICROCAMPAIGN_READY_BUT_NOT_AUTHORIZED"
    assert d["decision"]["red_team_failure_count"] == 0
    for active in (
        "C66-B_no_authorization_protocol_only",
        "C66-C_cpu_torchload_abi_validated_no_forward",
        "C66-E_preprocessing_dataset_contract_validated",
        "C66-G_trial_level_cache_schema_validated",
        "C66-K_atom_trace_forward_hooks_feasible_without_training",
    ):
        assert active in d["decision"]["active"]
    for inactive in (
        "C66-A_reinference_only_microcampaign_authorized_and_executed",
        "C66-D_checkpoint_abi_or_state_dict_mismatch_found",
        "C66-F_preprocessing_dataset_contract_blocked",
        "C66-H_minimal_trial_cache_emitted_and_manifested",
        "C66-I_split_label_protocol_feasible_on_cache",
        "C66-J_sample_level_conditional_cs_feasible_on_cache",
        "C66-L_reinference_only_path_blocked_new_training_may_be_needed_but_not_authorized",
        "C66-M_claim_or_availability_inconsistency_found",
    ):
        assert inactive in d["decision"]["inactive"]

    auth = {r["gate"]: r for r in _rows("authorization_ledger.csv")}
    assert auth["authorization_phrase_required"]["observed"] == "0"
    assert auth["forward_reinference_authorized"]["allowed"] == "0"
    assert auth["forward_reinference_authorized"]["observed"] == "0"
    assert auth["cpu_torchload_metadata_authorized"]["allowed"] == "1"
    assert auth["cpu_torchload_metadata_authorized"]["observed"] == "1"
    assert auth["training_authorized"]["allowed"] == "0"
    assert auth["gpu_authorized"]["allowed"] == "0"
    assert auth["cache_emission_authorized"]["observed"] == "0"


def test_c66_frozen_store_and_mapping_replay_match_c65():
    frozen = {r["check"]: r for r in _rows("frozen_store_integrity.csv")}
    assert frozen["c65_commit"]["value"] == "192a82d"
    assert frozen["primary_store_exists"]["passed"] == "1"
    assert frozen["c65_oaci_pt_count"]["value"] == "5454"
    assert frozen["c65_sidecar_count"]["value"] == "5454"
    assert frozen["c65_artifact_index_count"]["value"] == "27"
    assert frozen["c50_singleton_mapping_rows"]["value"] == "3804"
    assert frozen["unique_checkpoint_ids"]["value"] == "1268"
    assert frozen["all_mapped_paths_exist"]["value"] == "3804"
    assert frozen["checkout_weight_files"]["value"] == "0"
    assert {r["passed"] for r in frozen.values()} == {"1"}

    replay = _rows("checkpoint_mapping_replay.csv")
    assert len(replay) == 162
    assert sum(int(r["singleton_rows"]) for r in replay) == 3804
    assert sum(int(r["unique_checkpoint_ids"]) for r in replay) == 3804
    assert {r["all_pt_json_verified"] for r in replay} == {"1"}
    assert len({(r["seed"], r["target"], r["level"]) for r in replay}) == 54
    assert {r["regime"] for r in replay} == {"S0_full_support", "S2_rare_cells", "S3_nonestimable_cells"}


def test_c66_cpu_torchload_sample_validates_state_hashes_without_forward_or_payload_rehash():
    sample = _rows("cpu_torchload_abi_sample.csv")
    assert len(sample) == 6
    assert {r["sample_id"] for r in sample} == {f"abi_sample_{i:02d}" for i in range(1, 7)}
    assert {r["seed"] for r in sample} == {"0", "1", "2"}
    assert {r["target"] for r in sample} == {"1", "5", "9"}
    assert {r["level"] for r in sample} == {"0", "1"}
    assert {r["regime"] for r in sample} == {"S0_full_support"}
    assert {r["torch_load_attempted"] for r in sample} == {"1"}
    assert {r["forward_attempted"] for r in sample} == {"0"}
    assert {r["training_attempted"] for r in sample} == {"0"}
    assert {r["payload_file_sha256_rehashed"] for r in sample} == {"0"}
    assert {r["load_status"] for r in sample} == {"pass"}
    assert {r["state_hash_matches_checkpoint_id"] for r in sample} == {"1"}
    assert {r["sidecar_tensor_schema_matches"] for r in sample} == {"1"}
    assert {r["key_count"] for r in sample} == {"10"}
    assert {r["tensor_count"] for r in sample} == {"10"}
    assert {r["dtype_set"] for r in sample} == {"torch.float32;torch.int64"}
    assert {r["total_elements"] for r in sample} == {"39605"}

    keys = {r["state_key"]: r for r in _rows("state_dict_key_shape_summary.csv")}
    assert set(keys) == {
        "bn.bias",
        "bn.num_batches_tracked",
        "bn.running_mean",
        "bn.running_var",
        "bn.weight",
        "classifier.bias",
        "classifier.weight",
        "spatial.weight",
        "temporal.bias",
        "temporal.weight",
    }
    assert keys["classifier.weight"]["shape_set"] == "[4, 800]"
    assert keys["spatial.weight"]["shape_set"] == "[40, 40, 22, 1]"
    assert keys["temporal.weight"]["shape_set"] == "[40, 1, 1, 25]"
    assert {r["required_by_shallowconvnet_abi"] for r in keys.values()} == {"1"}


def test_c66_sidecar_and_model_abi_are_consistent_without_constructor_forward():
    sidecars = _rows("sidecar_schema_summary.csv")
    assert len(sidecars) == 1
    sig = sidecars[0]
    assert sig["sidecar_count"] == "1268"
    assert sig["tensor_count"] == "10"
    assert sig["writer_versions"] == "oaci-ckpt-v1:1268"
    assert sig["all_model_hash_match"] == "1"
    assert "classifier.weight" in sig["tensor_keys"]
    assert "classifier.weight[4, 800]" in sig["shape_signature"]

    model = {r["check"]: r for r in _rows("model_abi_compatibility_ledger.csv")}
    assert model["model_factory"]["observed"] == "shallow_convnet"
    assert model["input_shape"]["observed"] == "[22, 385]"
    assert model["n_classes"]["observed"] == "4"
    assert model["feature_dim_formula"]["observed"] == "800"
    assert model["load_state_dict_execution"]["observed"] == "not_run"
    assert {r["passed"] for r in model.values()} == {"1"}
    assert {r["forward_attempted"] for r in model.values()} == {"0"}


def test_c66_preprocess_split_and_label_quarantine_contracts_hold():
    pp = {r["item"]: r for r in _rows("preprocess_contract_inventory.csv")}
    assert pp["dataset"]["value"] == "BNCI2014_001"
    assert pp["channels"]["value"] == "22"
    assert pp["classes"]["value"] == "left_hand|right_hand|feet|tongue"
    assert pp["bandpass"]["value"] == "fmin=4.0;fmax=38.0"
    assert pp["resample"]["value"] == "128.0"
    assert pp["epoch_window"]["value"] == "tmin=0.5;tmax=3.5;n_times=385"
    assert pp["normalization"]["value"] == "zscore_sample;eps=1e-08"
    assert {r["validated"] for r in pp.values()} == {"1"}
    assert {r["blocks_campaign"] for r in pp.values()} == {"0"}

    splits = _rows("dataset_split_contract.csv")
    assert len(splits) == 9
    assert {r["dataset_id"] for r in splits} == {"BNCI2014_001"}
    assert {r["historical_seeds"] for r in splits} == {"0;1;2"}
    assert {r["reserved_seeds"] for r in splits} == {"3;4"}
    assert {r["reserved_dataset"] for r in splits} == {"BNCI2014_004"}
    assert {r["roles_reconstructable"] for r in splits} == {"1"}

    labels = {r["label_source"]: r for r in _rows("label_quarantine_contract.csv")}
    assert labels["target_construct_labels"]["allowed_for_selection_rule"] == "0"
    assert labels["target_eval_labels"]["same_label_reuse_allowed"] == "0"
    assert labels["same_candidate_endpoint_scalar"]["allowed_for_cache_field"] == "0"
    assert labels["same_candidate_endpoint_scalar"]["future_split_role"] == "forbidden_oracle_boundary"


def test_c66_trial_cache_schema_and_payload_plan_emit_no_real_cache():
    schema = {r["field"]: r for r in _rows("trial_cache_schema.csv")}
    assert len(schema) == 29
    for field in (
        "trial_cache_id",
        "checkpoint_id",
        "dataset_id",
        "trial_id",
        "class_label_quarantined",
        "y_pred",
        "logits",
        "probabilities",
        "confidence",
        "margin",
        "entropy",
        "representation_z",
        "Wz",
    ):
        assert field in schema
    assert schema["logits"]["requires_forward"] == "1"
    assert schema["probabilities"]["available_now"] == "0"
    assert schema["class_label_quarantined"]["target_label_dependent"] == "1"
    assert schema["representation_z"]["large_payload_ref_only"] == "1"

    availability = {r["field"]: r for r in _rows("cache_field_availability_ledger.csv")}
    assert availability["checkpoint_id"]["available_in_committed_summary_artifacts"] == "1"
    assert availability["logits"]["available_in_current_c66_cache"] == "0"
    assert availability["logits"]["available_after_authorized_microcampaign"] == "1"

    external = _rows("cache_external_manifest.csv")
    assert len(external) == 2
    assert {r["created_in_c66"] for r in external} == {"0"}
    assert {r["real_trial_rows"] for r in external} == {"0"}
    assert {r["status"] for r in external} == {"not_authorized_not_created"}
    payload = {r["artifact"]: r for r in _rows("cache_payload_plan.csv")}
    assert payload["trial_logits_probs_cache"]["git_tracked"] == "0"
    assert payload["representation_z_cache"]["large_payload_policy"] == "external_only"
    assert payload["cache_manifest_summary"]["git_tracked"] == "1"
    assert payload["checkpoint_payloads"]["large_payload_policy"] == "reuse_existing_no_copy"


def test_c66_split_label_conditional_cs_and_atom_protocols_are_claim_blocked_until_cache_exists():
    split = {r["check"]: r for r in _rows("split_label_feasibility_on_cache.csv")}
    assert split["real_trial_cache_present"]["feasible_now"] == "0"
    assert split["construct_eval_disjointness_defined"]["feasible_after_authorized_cache"] == "1"
    assert split["same_label_endpoint_oracle_blocked"]["feasible_now"] == "1"
    assert split["few_label_sufficiency_claim"]["blocks_current_claim"] == "1"
    protocol = _rows("split_label_protocol.csv")
    assert len(protocol) == 4
    assert {r["same_label_reuse_allowed"] for r in protocol} == {"0"}
    assert {r["claim_allowed_now"] for r in protocol} == {"0"}

    cs = {r["check"]: r for r in _rows("sample_level_cs_feasibility.csv")}
    assert cs["toy_paired_sample_interface"]["toy_interface_pass"] == "1"
    assert cs["gram_matrix_inputs"]["feasible_now"] == "0"
    assert cs["hankel_window_inputs"]["feasible_after_authorized_cache"] == "1"
    assert cs["full_conditional_cs_claim"]["feasible_after_authorized_cache"] == "0"
    varmap = _rows("conditional_cs_variable_map.csv")
    assert len(varmap) == 4
    assert {r["paired_sample_vars_available_now"] for r in varmap} == {"0"}
    assert {r["available_after_authorized_cache"] for r in varmap} == {"1"}

    atom = {r["trace"]: r for r in _rows("atom_trace_hook_feasibility.csv")}
    assert atom["logits_probabilities"]["requires_new_training"] == "0"
    assert atom["representation_z"]["recoverable_by_reinfer_only"] == "1"
    assert atom["projection_Wz"]["requires_forward_hook"] == "1"
    assert atom["optimizer_step_atom_contribution"]["requires_new_training"] == "1"
    hooks = {r["hook"]: r for r in _rows("representation_hook_contract.csv")}
    assert hooks["model_output_z"]["large_payload_ref_only"] == "1"
    assert hooks["atom_training_step"]["requires_training"] == "1"


def test_c66_artifact_hygiene_red_team_and_manifest_hashes_pass():
    d = _summary()
    assert d["table_row_counts"] == {
        "artifact_manifest": 31,
        "atom_trace_hook_feasibility": 6,
        "authorization_ledger": 7,
        "cache_external_manifest": 2,
        "cache_field_availability_ledger": 29,
        "cache_payload_plan": 4,
        "checkpoint_mapping_replay": 162,
        "conditional_cs_variable_map": 4,
        "cpu_torchload_abi_sample": 6,
        "dataset_split_contract": 9,
        "forbidden_claim_scan": 15,
        "frozen_store_integrity": 9,
        "label_quarantine_contract": 5,
        "large_artifact_scan": 31,
        "microcampaign_expected_payload_size": 3,
        "microcampaign_sampling_plan": 6,
        "model_abi_compatibility_ledger": 7,
        "preprocess_contract_inventory": 9,
        "red_team_failure_ledger": 16,
        "representation_hook_contract": 4,
        "sample_level_cs_feasibility": 4,
        "schema_validation_summary": 26,
        "sidecar_schema_summary": 1,
        "split_label_feasibility_on_cache": 4,
        "split_label_protocol": 4,
        "state_dict_key_shape_summary": 10,
        "test_command_manifest": 4,
        "trial_cache_schema": 29,
    }
    red = _rows("red_team_failure_ledger.csv")
    assert len(red) == 16
    assert {r["failed"] for r in red} == {"0"}
    forbidden = _rows("forbidden_claim_scan.csv")
    assert len(forbidden) == 15
    assert {r["affirmative_hits"] for r in forbidden} == {"0"}
    assert {r["passed"] for r in forbidden} == {"1"}
    large = _rows("large_artifact_scan.csv")
    assert len(large) == 31
    assert {r["over_50mb"] for r in large} == {"0"}
    assert {r["passed"] for r in large} == {"1"}

    manifest = _rows("artifact_manifest.csv")
    assert len(manifest) == 31
    for row in manifest:
        assert os.path.exists(row["path"])
        assert int(row["size_bytes"]) == os.path.getsize(row["path"])
        assert row["sha256"] == _sha256(row["path"])

    tests = {r["test_scope"]: r for r in _rows("test_command_manifest.csv")}
    assert set(tests) == {"focused_c66", "c50_c66_slice", "c23_c66_regression", "full_oaci_tests"}
    assert {r["environment"] for r in tests.values()} == {"eeg2025"}
    assert {r["slurm_partition"] for r in tests.values()} == {"cpu-high"}
    assert {r["status"] for r in tests.values()} == {"green"}

    report = open("oaci/reports/C66_REINFERENCE_ONLY_TRIAL_CACHE_MICROCAMPAIGN.md").read()
    assert "C66 stayed in no-forward / no-reinference mode" in report
    assert "No real trial-level cache is emitted" in report
    assert "MICROCAMPAIGN_READY_BUT_NOT_AUTHORIZED" in report
    red_report = open("oaci/reports/C66_RED_TEAM_VERIFICATION.md").read()
    assert "full_oaci_tests job" in red_report
    assert "1352 passed" in red_report
