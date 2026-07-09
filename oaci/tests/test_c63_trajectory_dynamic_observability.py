"""C63 Trajectory-Dynamic Conditional Observability tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c63_trajectory_dynamic_observability as c63
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C63_TRAJECTORY_DYNAMIC_OBSERVABILITY.json"
TABLE_DIR = "oaci/reports/c63_tables"


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


def test_c63_decision_scope_and_training_gate_are_frozen():
    assert c63._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert set(c63.DECISIONS) == {
        "C63-A_dynamic_conditional_observability_ladder_established",
        "C63-B_source_dynamic_history_adds_stable_observability",
        "C63-C_source_dynamic_history_near_static_source_only",
        "C63-D_source_dynamic_template_partial_but_no_screen_off_endpoint",
        "C63-E_endpoint_scalar_still_dominates_after_dynamic_conditioning",
        "C63-F_dynamic_source_observable_escape_hatch_found",
        "C63-G_no_dynamic_source_observable_escape_hatch_found",
        "C63-H_trajectory_fragmentation_explained_by_source_dynamics",
        "C63-I_trajectory_fragmentation_not_explained_by_source_dynamics",
        "C63-J_synthetic_dynamic_rank_gauge_validation_successful",
        "C63-K_full_time_series_conditional_cs_requires_trial_level_cache",
        "C63-L_training_not_authorized",
        "C63-M_claim_or_availability_inconsistency_found",
    }
    d = _summary()
    assert d["milestone"] == "C63"
    assert d["config_hash"] == "664007686afb520f"
    assert d["c62_commit"] == "d914e44"
    assert d["c62_decision"] == "C62-A_C61_ladder_reproduced"
    assert d["decision"]["primary"] == "C63-A_dynamic_conditional_observability_ladder_established"
    for active in (
        "C63-C_source_dynamic_history_near_static_source_only",
        "C63-D_source_dynamic_template_partial_but_no_screen_off_endpoint",
        "C63-E_endpoint_scalar_still_dominates_after_dynamic_conditioning",
        "C63-G_no_dynamic_source_observable_escape_hatch_found",
        "C63-I_trajectory_fragmentation_not_explained_by_source_dynamics",
        "C63-K_full_time_series_conditional_cs_requires_trial_level_cache",
        "C63-L_training_not_authorized",
    ):
        assert active in d["decision"]["active"]
    assert "C63-B_source_dynamic_history_adds_stable_observability" in d["decision"]["inactive"]
    assert "C63-F_dynamic_source_observable_escape_hatch_found" in d["decision"]["inactive"]
    assert "C63-H_trajectory_fragmentation_explained_by_source_dynamics" in d["decision"]["inactive"]
    assert "C63-M_claim_or_availability_inconsistency_found" in d["decision"]["inactive"]
    assert d["decision"]["training_gate"] == c63.TRAINING_GATE
    assert d["decision"]["instrumentation_gate"] == c63.INSTRUMENTATION_GATE
    assert d["decision"]["red_team_failure_count"] == 0


def test_c63_table_shapes_reports_and_gate_json_are_complete():
    d = _summary()
    assert d["table_row_counts"] == {
        "artifact_manifest": 26,
        "dynamic_availability_ledger": 5,
        "dynamic_cod_ladder_summary": 7,
        "dynamic_estimator_stress_summary": 24,
        "dynamic_null_summary": 6,
        "dynamic_screening_off_summary": 5,
        "dynamic_source_escape_hatch_audit": 6,
        "dynamic_source_feature_inventory": 7,
        "dynamic_support_sensitivity": 16,
        "forbidden_claim_scan": 24,
        "hankel_window_support_summary": 4,
        "large_artifact_scan": 26,
        "missing_dynamic_fields": 5,
        "red_team_failure_ledger": 13,
        "schema_validation_summary": 20,
        "synthetic_dynamic_cod_ladder": 12,
        "synthetic_dynamic_rank_gauge_summary": 6,
        "test_command_manifest": 4,
        "trajectory_artifact_inventory": 5,
        "trajectory_failure_reason_update": 5,
        "trajectory_fragmentation_dynamic_ledger": 6,
        "trajectory_sequence_schema": 7,
    }
    required_reports = {
        "C63_TRAJECTORY_DYNAMIC_OBSERVABILITY.json",
        "C63_TRAJECTORY_DYNAMIC_OBSERVABILITY.md",
        "C63_DYNAMIC_ESTIMATOR_NOTES.md",
        "C63_DYNAMIC_INSTRUMENTATION_GATE.md",
        "C63_RED_TEAM_VERIFICATION.md",
    }
    assert required_reports <= {p for p in os.listdir("oaci/reports") if p.startswith("C63_")}
    assert os.path.exists(os.path.join(TABLE_DIR, "dynamic_instrumentation_gate.json"))
    gate = json.load(open(os.path.join(TABLE_DIR, "dynamic_instrumentation_gate.json")))
    assert gate["full_time_series_conditional_cs_supported"] is False
    assert gate["authorized_in_c63"] is False


def test_c63_trajectory_artifacts_and_hankel_support_are_compact():
    inventory = {r["artifact"]: r for r in _rows("trajectory_artifact_inventory.csv")}
    assert inventory["C50_island_morphology"]["row_count"] == "3804"
    assert inventory["C51_trajectory_failure_ledger"]["row_count"] == "162"
    assert inventory["C51_trajectory_failure_ledger"]["supports_hankel_proxy"] == "1"
    schema = {r["schema_item"]: r for r in _rows("trajectory_sequence_schema.csv")}
    assert schema["trajectory_id"]["value"] == "162"
    assert schema["candidate_rows"]["value"] == "3804"
    assert schema["source_score_vectors"]["available"] == "0"
    support = {r["window_k"]: r for r in _rows("hankel_window_support_summary.csv")}
    assert set(support) == {"1", "2", "3", "5"}
    assert {r["support_fraction"] for r in support.values()} == {"1.0"}
    assert {r["emits_row_payload"] for r in support.values()} == {"0"}
    assert {r["full_time_series_cs_supported"] for r in support.values()} == {"0"}


def test_c63_dynamic_availability_and_missing_fields_keep_source_boundary():
    features = {r["feature_id"]: r for r in _rows("dynamic_source_feature_inventory.csv")}
    assert features["D1"]["uses_target_labels"] == "0"
    assert features["D1"]["uses_endpoint_scalar"] == "0"
    assert features["D1"]["summary_proxy_hit"] == "0.5740740740740741"
    assert features["D5"]["summary_proxy_hit"] == "0.4074074074074074"
    availability = {r["information_class"]: r for r in _rows("dynamic_availability_ledger.csv")}
    assert availability["D_source_dynamic_proxy"]["uses_source_only_inputs"] == "1"
    assert availability["D_source_dynamic_proxy"]["uses_target_labels"] == "0"
    assert availability["D_source_dynamic_proxy"]["uses_endpoint_scalar"] == "0"
    assert availability["I7_endpoint_scalar"]["available_at_selection_time"] == "0"
    missing = {r["field"]: r for r in _rows("missing_dynamic_fields.csv")}
    assert missing["raw_source_score_vector_by_checkpoint"]["present"] == "0"
    assert missing["per_trial_logits_probabilities"]["blocks_full_time_series_cs"] == "1"
    assert missing["split_label_cache"]["present"] == "0"


def test_c63_dynamic_cod_ladder_keeps_endpoint_dominance():
    ladder = {r["comparison_id"]: r for r in _rows("dynamic_cod_ladder_summary.csv")}
    assert float(ladder["DYN_static_to_source_history"]["hit_gain"]) == 0.5740740740740741 - 0.5061728395061729
    assert ladder["DYN_static_to_source_history"]["beats_max_null_p95"] == "0"
    assert float(ladder["DYN_static_to_source_delta"]["hit_gain"]) < 0.0
    assert float(ladder["DYN_static_template_to_source_history"]["hit_gain"]) < 0.02
    endpoint = ladder["DYN_dynamic_template_to_endpoint"]
    assert float(endpoint["hit_gain"]) == 0.9444444444444444 - 0.720679012345679
    assert endpoint["beats_max_null_p95"] == "1"
    assert endpoint["screen_off_endpoint"] == "0"
    assert float(endpoint["hit_gain"]) > 0.20


def test_c63_dynamic_estimator_stress_and_nulls_do_not_promote_dynamic_source():
    stress = _rows("dynamic_estimator_stress_summary.csv")
    assert len(stress) == 24
    dynamic_rows = [r for r in stress if r["comparison"] == "source_dynamic"]
    assert {r["dynamic_near_static"] for r in dynamic_rows} == {"1"}
    endpoint_rows = [r for r in stress if r["comparison"] == "endpoint_after_dynamic_template"]
    assert "1" in {r["endpoint_dominates"] for r in endpoint_rows}
    nulls = {r["null_id"]: r for r in _rows("dynamic_null_summary.csv")}
    assert nulls["N1_within_trajectory_time_shuffle"]["passes"] == "0"
    assert nulls["N3_source_dynamic_feature_permutation"]["passes"] == "0"
    assert nulls["N4_endpoint_label_permutation"]["passes"] == "1"
    assert nulls["N6_template_vs_max_null"]["passes"] == "0"


def test_c63_fragmentation_remains_residual_and_support_sensitivity_preserves_boundary():
    frag = {r["ledger_row"]: r for r in _rows("trajectory_fragmentation_dynamic_ledger.csv")}
    assert float(frag["C51_trajectory_fail_fraction"]["observed_value"]) == 0.43209876543209874
    assert frag["C51_trajectory_fail_fraction"]["dynamic_explained"] == "0"
    assert float(frag["source_dynamic_closure_fraction"]["observed_value"]) < 0.20
    assert float(frag["endpoint_after_dynamic_template"]["observed_value"]) > 0.20
    update = {r["failure_code"]: r for r in _rows("trajectory_failure_reason_update.csv")}
    assert update["TRAJECTORY_FRAGMENTED"]["explained_by_source_dynamics"] == "0"
    support = _rows("dynamic_support_sensitivity.csv")
    assert len(support) == 16
    assert {r["dynamic_source_beats_max_null"] for r in support} == {"0"}
    assert {r["dynamic_template_beats_max_null"] for r in support} == {"0"}


def test_c63_dynamic_source_escape_hatch_is_closed():
    adversary = {r["candidate_id"]: r for r in _rows("dynamic_source_escape_hatch_audit.csv")}
    assert len(adversary) == 6
    assert {r["reliable_escape_hatch"] for r in adversary.values()} == {"0"}
    assert adversary["DADV63-1"]["uses_target_labels"] == "0"
    assert adversary["DADV63-1"]["uses_endpoint_scalar"] == "0"
    assert float(adversary["DADV63-1"]["hit"]) == 0.5740740740740741
    assert adversary["DADV63-5"]["hit"] == ""


def test_c63_synthetic_dynamic_rank_gauge_validation_is_conservative():
    summary = {r["scenario"]: r for r in _rows("synthetic_dynamic_rank_gauge_summary.csv")}
    assert len(summary) == 6
    assert {r["expected_behavior_pass"] for r in summary.values()} == {"1"}
    assert summary["S3_hidden_gauge_dynamics_independent"]["source_dynamic_hit"] == str(c63.STRICT_SOURCE_HIT)
    assert summary["S4_gauge_partially_source_coupled"]["gauge_source_coupled"] == "1"
    ladder = _rows("synthetic_dynamic_cod_ladder.csv")
    assert len(ladder) == 12
    assert {r["synthetic_model_only"] for r in ladder} == {"1"}
    endpoint_rows = [r for r in ladder if r["comparison_id"].endswith("dynamic_template_to_endpoint")]
    assert all(float(r["hit_gain"]) >= 0.0 for r in endpoint_rows)


def test_c63_red_team_manifest_hashes_and_large_artifact_scan_pass():
    red = {r["gate"]: r for r in _rows("red_team_failure_ledger.csv")}
    assert len(red) == 13
    assert {r["failed"] for r in red.values()} == {"0"}
    forbidden = _rows("forbidden_claim_scan.csv")
    assert len(forbidden) == 24
    assert {r["affirmative_hits"] for r in forbidden} == {"0"}
    tests = _rows("test_command_manifest.csv")
    assert {r["status"] for r in tests} <= {"planned", "green"}
    large = _rows("large_artifact_scan.csv")
    assert len(large) == 26
    assert {r["passed"] for r in large} == {"1"}
    manifest = _rows("artifact_manifest.csv")
    assert len(manifest) == 26
    for row in manifest:
        assert os.path.exists(row["path"])
        assert int(row["size_bytes"]) == os.path.getsize(row["path"])
        assert row["sha256"] == _sha256(row["path"])
        if row["artifact_class"] == "table":
            assert row["row_count"] != ""


def test_c63_run_recomputes_core_decision_without_writing():
    res = c63.run(test_status="unit")
    assert res["decision"]["primary"] == "C63-A_dynamic_conditional_observability_ladder_established"
    assert res["decision"]["training_gate"] == c63.TRAINING_GATE
    assert len(res["dynamic_cod_ladder_summary_rows"]) == 7
    endpoint = {
        r["comparison_id"]: r for r in res["dynamic_cod_ladder_summary_rows"]
    }["DYN_dynamic_template_to_endpoint"]
    assert endpoint["hit_gain"] == c63.ENDPOINT_ORACLE_HIT - c63.SOURCE_DYNAMIC_TEMPLATE_HIT
