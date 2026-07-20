"""C64 Trial-Level Instrumentation Readiness tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c64_trial_level_instrumentation_readiness as c64
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C64_TRIAL_LEVEL_INSTRUMENTATION_READINESS.json"
TABLE_DIR = "oaci/reports/c64_tables"


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_c64_decision_scope_and_gates_are_frozen():
    assert c64._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert set(c64.DECISIONS) == {
        "C64-A_frozen_summary_artifact_paths_saturated",
        "C64-B_reinference_only_trial_cache_campaign_sufficient",
        "C64-C_new_training_required_for_trial_cache_or_atom_trace",
        "C64-D_split_label_cache_protocol_ready_but_not_authorized",
        "C64-E_full_time_series_conditional_cs_protocol_ready_but_not_authorized",
        "C64-F_atom_trace_protocol_ready_but_not_authorized",
        "C64-G_independent_checkpoint_replication_protocol_ready_but_not_authorized",
        "C64-H_instrumentation_not_scientifically_justified_yet",
        "C64-I_source_observable_escape_hatch_remaining",
        "C64-J_claim_or_availability_inconsistency_found",
    }
    d = _summary()
    assert d["milestone"] == "C64"
    assert d["config_hash"] == "664007686afb520f"
    assert d["c63_commit"] == "25626fa"
    assert d["c63_decision"] == "C63-A_dynamic_conditional_observability_ladder_established"
    assert d["decision"]["primary"] == "C64-A_frozen_summary_artifact_paths_saturated"
    for active in (
        "C64-C_new_training_required_for_trial_cache_or_atom_trace",
        "C64-D_split_label_cache_protocol_ready_but_not_authorized",
        "C64-E_full_time_series_conditional_cs_protocol_ready_but_not_authorized",
        "C64-F_atom_trace_protocol_ready_but_not_authorized",
        "C64-G_independent_checkpoint_replication_protocol_ready_but_not_authorized",
    ):
        assert active in d["decision"]["active"]
    for inactive in (
        "C64-B_reinference_only_trial_cache_campaign_sufficient",
        "C64-H_instrumentation_not_scientifically_justified_yet",
        "C64-I_source_observable_escape_hatch_remaining",
        "C64-J_claim_or_availability_inconsistency_found",
    ):
        assert inactive in d["decision"]["inactive"]
    assert d["final_gate"] == c64.FINAL_GATE
    assert d["reinference_subgate"] == c64.REINFERENCE_SUBGATE
    assert d["training_gate"] == c64.TRAINING_GATE
    assert d["decision"]["red_team_failure_count"] == 0


def test_c64_table_shapes_reports_and_payloads_are_complete():
    d = _summary()
    assert d["table_row_counts"] == {
        "artifact_manifest": 37,
        "atom_trace_claim_boundary": 4,
        "availability_tag_definitions": 7,
        "checkpoint_inventory_summary": 7,
        "conditional_cs_feasibility_matrix": 5,
        "forbidden_claim_scan": 21,
        "frozen_path_closure_ledger": 6,
        "frozen_path_saturation_decision": 3,
        "hankel_variable_mapping": 5,
        "instrumentation_power_simulation": 5,
        "large_artifact_scan": 37,
        "minimum_sample_requirements": 5,
        "red_team_failure_ledger": 12,
        "reinference_only_feasibility": 6,
        "reinference_resource_estimate": 5,
        "reinference_risk_ledger": 5,
        "remaining_summary_escape_hatches": 6,
        "replication_pass_fail_criteria": 5,
        "replication_protocol_options": 4,
        "sample_level_missing_data_ledger": 6,
        "schema_validation_summary": 26,
        "split_label_forbidden_claims": 4,
        "split_label_power_requirements": 4,
        "test_command_manifest": 4,
        "training_necessity_decision": 6,
        "training_vs_reinference_tradeoff": 4,
        "trial_level_cache_minimal_columns": 26,
        "value_of_information_summary": 4,
    }
    expected_reports = {
        "C64_ATOM_IDENTITY_GATE_PROTOCOL.md",
        "C64_NEW_TRAINING_MINIMAL_PROTOCOL.md",
        "C64_RED_TEAM_VERIFICATION.md",
        "C64_RESERVED_HOLDOUT_POLICY.md",
        "C64_SPLIT_LABEL_PROTOCOL.md",
        "C64_TRIAL_LEVEL_INSTRUMENTATION_READINESS.json",
        "C64_TRIAL_LEVEL_INSTRUMENTATION_READINESS.md",
    }
    assert expected_reports <= {p for p in os.listdir("oaci/reports") if p.startswith("C64_")}
    assert os.path.exists(os.path.join(TABLE_DIR, "trial_level_cache_schema.json"))
    assert os.path.exists(os.path.join(TABLE_DIR, "full_cs_supported_flag.json"))
    assert os.path.exists(os.path.join(TABLE_DIR, "atom_trace_schema.json"))
    assert os.path.exists(os.path.join(TABLE_DIR, "c64_gate_decision.json"))


def test_c64_frozen_summary_paths_are_saturated_but_remaining_evidence_is_instrumented():
    frozen = {r["path_id"]: r for r in _rows("frozen_path_closure_ledger.csv")}
    assert len(frozen) == 6
    assert frozen["FP3_endpoint_scalar"]["status"] == "same_label_oracle_boundary"
    assert frozen["FP6_dynamic_cod"]["status"] == "dynamic_escape_hatch_closed"
    saturation = {r["decision_id"]: r for r in _rows("frozen_path_saturation_decision.csv")}
    assert {r["passed"] for r in saturation.values()} == {"1"}
    assert saturation["SAT3"]["decision"] == "NEXT_EVIDENCE_IS_TRIAL_LEVEL_CACHE"
    hatches = {r["hatch_id"]: r for r in _rows("remaining_summary_escape_hatches.csv")}
    assert hatches["RH1_more_source_summary_features"]["remaining"] == "0"
    assert hatches["RH3_split_label_cache"]["remaining"] == "1"
    assert hatches["RH4_full_conditional_cs"]["scientifically_justified_now"] == "1"
    assert hatches["RH6_independent_replication"]["remaining"] == "1"


def test_c64_trial_cache_schema_separates_source_available_fields_from_label_content():
    columns = {r["column"]: r for r in _rows("trial_level_cache_minimal_columns.csv")}
    assert len(columns) == 26
    for required in (
        "candidate_id",
        "checkpoint_id",
        "target_subject",
        "trajectory_id",
        "trial_id",
        "split_role",
        "class_label",
        "prediction",
        "logits",
        "probabilities",
        "representation_z",
        "projection_Wz",
        "audit_role",
    ):
        assert required in columns
    assert columns["logits"]["needed_for_split_label"] == "1"
    assert columns["probabilities"]["needed_for_split_label"] == "1"
    assert columns["class_label"]["available_at_selection_time_if_source_only"] == "0"
    assert columns["correct_flag"]["label_dependency"] == "label-derived"
    schema = json.load(open(os.path.join(TABLE_DIR, "trial_level_cache_schema.json")))
    schema_columns = {r["column"] for r in schema["columns"]}
    assert {"logits", "probabilities", "split_role", "audit_role", "representation_z", "projection_Wz"} <= schema_columns
    tags = {r["tag"]: r for r in schema["availability_tags"]}
    assert tags["same_label_oracle"]["allowed_for_source_rule"] == 0
    assert tags["split_label_allowed"]["diagnostic_only"] == 1


def test_c64_split_label_and_full_conditional_cs_are_protocol_ready_not_currently_supported():
    forbidden = _rows("split_label_forbidden_claims.csv")
    assert len(forbidden) == 4
    assert {r["forbidden"] for r in forbidden} == {"1"}
    cs = {r["estimator_component"]: r for r in _rows("conditional_cs_feasibility_matrix.csv")}
    assert cs["summary kernel proxy"]["current_supported"] == "1"
    for component in (
        "p(y|x1) vs p(y|x1,x2)",
        "Gram/KDE sample matrix",
        "Hankel past-window response",
        "split-label conditional diagnostic",
    ):
        assert cs[component]["current_supported"] == "0"
        assert cs[component]["protocol_ready"] == "1"
    full_flag = json.load(open(os.path.join(TABLE_DIR, "full_cs_supported_flag.json")))
    assert full_flag["full_conditional_cs_supported_now"] is False
    missing = {r["missing_item"]: r for r in _rows("sample_level_missing_data_ledger.csv")}
    assert missing["per_trial_logits_probabilities"]["blocks_split_label"] == "1"
    assert missing["per_trial_logits_probabilities"]["blocks_full_cs"] == "1"
    assert missing["per_trial_labels_and_split_roles"]["blocks_split_label"] == "1"


def test_c64_checkpoint_inventory_keeps_reinference_conditional_in_current_checkout():
    d = _summary()
    gate = d["gate_decision"]
    assert gate["checkpoint_weights_found_in_checkout"] == 0
    assert gate["reinference_only_sufficient_from_current_checkout"] is False
    assert gate["training_authorized"] is False
    assert gate["reinference_authorized"] is False
    assert gate["gpu_authorized"] is False
    inventory = {r["inventory_item"]: r for r in _rows("checkpoint_inventory_summary.csv")}
    assert inventory["checkpoint_weight_files"]["present"] == "0"
    assert inventory["checkpoint_weight_files"]["blocks_reinference_now"] == "1"
    assert inventory["checkpoint_abi_code"]["present"] == "1"
    assert inventory["training_checkpoint_record_code"]["present"] == "1"
    assert inventory["split_label_cache"]["present"] == "0"
    feasibility = {r["requirement"]: r for r in _rows("reinference_only_feasibility.csv")}
    assert feasibility["frozen_checkpoint_weights_loadable"]["status"] == "blocking_in_current_checkout"
    assert feasibility["overall_reinference_only_decision"]["status"] == c64.REINFERENCE_SUBGATE
    assert feasibility["overall_reinference_only_decision"]["present_in_checkout"] == "0"


def test_c64_training_atom_and_replication_boundaries_are_not_authorized():
    training = {r["need"]: r for r in _rows("training_necessity_decision.csv")}
    assert training["split_label_trial_cache"]["new_training_required"] == "0"
    assert training["full_time_series_conditional_cs"]["new_training_required"] == "0"
    assert training["atom_trace"]["new_training_required"] == "1"
    assert training["independent_checkpoint_field_replication"]["new_training_required"] == "1"
    assert training["overall_training_gate"]["decision"] == c64.TRAINING_GATE
    tradeoff = {r["path"]: r for r in _rows("training_vs_reinference_tradeoff.csv")}
    assert {r["authorized"] for r in tradeoff.values()} == {"0"}
    assert tradeoff["re_inference_only"]["supports_atom_trace"] == "0"
    atom = {r["claim"]: r for r in _rows("atom_trace_claim_boundary.csv")}
    assert atom["atom sums reproduce aggregate leakage"]["current_supported"] == "0"
    assert atom["atom branch closed under current artifacts"]["current_supported"] == "1"
    assert atom["atom trace selector/action-rule claim"]["future_status"] == "forbidden_action_claim"
    replication = {r["option"]: r for r in _rows("replication_protocol_options.csv")}
    assert replication["reserved_holdout_final_stress"]["uses_reserved_holdout"] == "1"
    assert replication["reserved_holdout_final_stress"]["authorized"] == "0"


def test_c64_key_numbers_preserve_endpoint_oracle_boundary():
    d = _summary()
    nums = d["key_numbers"]
    assert nums["strict_source"] == c64.STRICT_SOURCE_HIT
    assert nums["source_dynamic"] == c64.SOURCE_DYNAMIC_HIT
    assert nums["source_dynamic_template"] == c64.SOURCE_DYNAMIC_TEMPLATE_HIT
    assert nums["template_only"] == c64.TEMPLATE_ONLY_HIT
    assert nums["endpoint_oracle"] == c64.ENDPOINT_ORACLE_HIT
    assert nums["max_null_p95"] == c64.MAX_NULL_P95
    assert nums["template_only"] < nums["max_null_p95"]
    assert nums["endpoint_oracle"] > nums["max_null_p95"]


def test_c64_red_team_forbidden_large_and_schema_gates_pass():
    red = {r["gate"]: r for r in _rows("red_team_failure_ledger.csv")}
    assert len(red) == 12
    assert {r["failed"] for r in red.values()} == {"0"}
    forbidden = _rows("forbidden_claim_scan.csv")
    assert len(forbidden) == 21
    assert {r["affirmative_hits"] for r in forbidden} == {"0"}
    assert {r["passed"] for r in forbidden} == {"1"}
    large = _rows("large_artifact_scan.csv")
    assert len(large) == 37
    assert {r["over_50mb"] for r in large} == {"0"}
    assert {r["passed"] for r in large} == {"1"}
    schema = _rows("schema_validation_summary.csv")
    assert len(schema) == 26
    assert {r["passed"] for r in schema} == {"1"}


def test_c64_manifest_hashes_match_and_run_recomputes_decision():
    manifest = _rows("artifact_manifest.csv")
    assert len(manifest) == 37
    for row in manifest:
        assert os.path.exists(row["path"])
        assert row["sha256"] == _sha256(row["path"])
        if row["path"].endswith(".csv"):
            assert row["artifact_class"] == "table"
            assert row["row_count"] != ""
    res = c64.run(test_status="unit")
    assert res["decision"]["primary"] == "C64-A_frozen_summary_artifact_paths_saturated"
    assert res["c64_gate_decision"]["training_authorized"] is False
    assert res["c64_gate_decision"]["reinference_authorized"] is False
    assert res["c64_gate_decision"]["checkpoint_weights_found_in_checkout"] == 0
