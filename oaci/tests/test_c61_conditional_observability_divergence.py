"""C61 Conditional Observability Divergence / Information-Ladder tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c61_conditional_observability_divergence as c61
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C61_CONDITIONAL_OBSERVABILITY_DIVERGENCE.json"
TABLE_DIR = "oaci/reports/c61_tables"


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


def test_c61_decision_scope_and_training_gate_are_frozen():
    assert c61._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert set(c61.DECISIONS) == {
        "C61-A_conditional_observability_divergence_ladder_established",
        "C61-B_conditional_observability_matches_hit_and_partition_bound_ladder",
        "C61-C_endpoint_scalar_dominates_incremental_observability",
        "C61-D_template_partial_observability_but_not_sufficient",
        "C61-E_source_key_conditional_sufficiency_fails",
        "C61-F_conditional_cs_estimator_unstable_but_partition_metrics_stable",
        "C61-G_source_observable_cod_escape_hatch_found",
        "C61-H_synthetic_rank_gauge_cod_validation_successful",
        "C61-I_hard_theorem_to_eeg_bridge_not_required_for_framework",
        "C61-J_future_instrumentation_needed_for_split_label_or_atom_trace",
        "C61-K_claim_or_availability_inconsistency_found",
    }
    d = _summary()
    assert d["milestone"] == "C61"
    assert d["config_hash"] == "664007686afb520f"
    assert d["c60_commit"] == "2e6ae07"
    assert d["c60_decision"] == "C60-B_rank_gauge_proof_repaired_or_strengthened"
    assert d["decision"]["primary"] == "C61-A_conditional_observability_divergence_ladder_established"
    assert "C61-B_conditional_observability_matches_hit_and_partition_bound_ladder" in d["decision"]["active"]
    assert "C61-C_endpoint_scalar_dominates_incremental_observability" in d["decision"]["active"]
    assert "C61-G_source_observable_cod_escape_hatch_found" in d["decision"]["inactive"]
    assert "C61-K_claim_or_availability_inconsistency_found" in d["decision"]["inactive"]
    assert d["decision"]["training_gate"] == c61.TRAINING_GATE
    assert d["decision"]["red_team_failure_count"] == 0
    assert d["cod_status"]["finite_partition_plugin"] == "established"
    assert d["cod_status"]["conditional_cs_kde"] == "not_primary_summary_artifacts_missing_raw_samples"


def test_c61_table_shapes_and_reports_are_complete():
    d = _summary()
    assert d["table_row_counts"] == {
        "artifact_manifest": 29,
        "cod_availability_ledger": 6,
        "cod_bandwidth_support_sensitivity": 5,
        "cod_estimator_summary": 6,
        "cod_null_calibration": 5,
        "conditional_cs_to_oaci_mapping": 7,
        "conditional_observability_spec": 8,
        "conditional_screening_summary": 4,
        "eeg_cod_cell_ledger": 4,
        "eeg_cod_ladder_summary": 6,
        "eeg_cod_vs_hit_ladder": 5,
        "endpoint_after_template_screening": 3,
        "forbidden_claim_scan": 20,
        "future_training_need_matrix": 6,
        "information_class_variable_ledger": 7,
        "large_artifact_scan": 29,
        "red_team_failure_ledger": 16,
        "same_label_oracle_boundary_checks": 5,
        "schema_validation_summary": 23,
        "source_observable_cod_adversary": 4,
        "subagent_audit_manifest": 10,
        "synthetic_rank_gauge_cod_summary": 7,
        "synthetic_rank_gauge_parameter_grid": 7,
        "test_command_manifest": 4,
        "training_gate_decision_matrix": 7,
    }
    required_reports = {
        "C61_CONDITIONAL_OBSERVABILITY_DIVERGENCE.json",
        "C61_CONDITIONAL_OBSERVABILITY_DIVERGENCE.md",
        "C61_FRAMEWORK_TEMPLATE.md",
        "C61_INSTRUMENTATION_IMPLICATIONS.md",
        "C61_RED_TEAM_VERIFICATION.md",
        "C61_SOURCE_ESCAPE_HATCH_RED_TEAM.md",
    }
    assert required_reports <= {p for p in os.listdir("oaci/reports") if p.startswith("C61_")}
    expected_tables = {
        "artifact_manifest.csv",
        "cod_availability_ledger.csv",
        "cod_bandwidth_support_sensitivity.csv",
        "cod_estimator_summary.csv",
        "cod_null_calibration.csv",
        "conditional_cs_to_oaci_mapping.csv",
        "conditional_observability_spec.csv",
        "conditional_screening_summary.csv",
        "eeg_cod_cell_ledger.csv",
        "eeg_cod_ladder_summary.csv",
        "eeg_cod_vs_hit_ladder.csv",
        "endpoint_after_template_screening.csv",
        "forbidden_claim_scan.csv",
        "future_training_need_matrix.csv",
        "information_class_variable_ledger.csv",
        "large_artifact_scan.csv",
        "red_team_failure_ledger.csv",
        "same_label_oracle_boundary_checks.csv",
        "schema_validation_summary.csv",
        "source_observable_cod_adversary.csv",
        "subagent_audit_manifest.csv",
        "synthetic_rank_gauge_cod_summary.csv",
        "synthetic_rank_gauge_parameter_grid.csv",
        "test_command_manifest.csv",
        "training_gate_decision_matrix.csv",
    }
    assert expected_tables == {p for p in os.listdir(TABLE_DIR) if p.endswith(".csv")}


def test_c61_cod_ladder_orders_information_without_overclaiming_template():
    ladder = {r["comparison_id"]: r for r in _rows("eeg_cod_ladder_summary.csv")}
    assert float(ladder["COD_key_given_source"]["bayes_hit_gain"]) < 0.0
    assert ladder["COD_key_given_source"]["beats_max_null_p95"] == "0"
    assert float(ladder["COD_template_given_source"]["bayes_hit_gain"]) == (
        0.7037037037037037 - 0.5061728395061729
    )
    assert ladder["COD_template_given_source"]["beats_max_null_p95"] == "0"
    assert ladder["COD_endpoint_given_source"]["beats_max_null_p95"] == "1"
    assert float(ladder["COD_endpoint_given_source"]["conditional_cs_binary_proxy"]) > float(
        ladder["COD_template_given_source"]["conditional_cs_binary_proxy"]
    )
    assert float(ladder["COD_endpoint_given_source_template"]["bayes_hit_gain"]) == (
        0.9444444444444444 - 0.7037037037037037
    )


def test_c61_null_and_same_label_oracle_boundary_are_explicit():
    checks = {r["check"]: r for r in _rows("same_label_oracle_boundary_checks.csv")}
    assert float(checks["endpoint_hit_beats_max_null_p95"]["observed"]) == 0.9444444444444444
    assert float(checks["endpoint_hit_beats_max_null_p95"]["reference"]) == 0.7712962962962961
    assert checks["endpoint_hit_beats_max_null_p95"]["passed"] == "1"
    assert float(checks["template_hit_not_above_max_null_p95"]["observed"]) == 0.7037037037037037
    assert checks["template_hit_not_above_max_null_p95"]["passed"] == "1"
    assert checks["endpoint_unavailable_at_selection_time"]["passed"] == "1"
    nulls = {r["null_id"]: r for r in _rows("cod_null_calibration.csv")}
    assert nulls["N4_template_vs_max_null_p95"]["template_beats_null"] == "0"
    assert nulls["N5_source_scalarization_vs_max_null_p95"]["template_beats_null"] == "0"


def test_c61_availability_ledger_separates_source_key_template_and_endpoint():
    availability = {r["score_name"]: r for r in _rows("cod_availability_ledger.csv")}
    assert availability["source_rank"]["uses_source_only_inputs"] == "1"
    assert availability["source_rank"]["available_at_selection_time"] == "1"
    assert availability["key_only"]["uses_key_only_inputs"] == "1"
    assert availability["key_only"]["diagnostic_only"] == "1"
    assert availability["matched_template"]["uses_target_label_diagnostic"] == "1"
    assert availability["matched_template"]["available_at_selection_time"] == "0"
    endpoint = availability["same_label_endpoint_scalar"]
    assert endpoint["uses_test_candidate_endpoint_scalar"] == "1"
    assert endpoint["uses_same_cell_target_labels"] == "1"
    assert endpoint["available_at_selection_time"] == "0"
    assert endpoint["diagnostic_only"] == "1"


def test_c61_screening_off_and_source_escape_hatch_red_team_close():
    screening = {r["screening_test"]: r for r in _rows("conditional_screening_summary.csv")}
    assert screening["key_screens_endpoint_after_source"]["screens_off"] == "0"
    assert float(screening["key_screens_endpoint_after_source"]["endpoint_remaining_gain"]) > 0.4
    assert screening["template_screens_endpoint_after_source"]["screens_off"] == "0"
    assert float(screening["template_screens_endpoint_after_source"]["endpoint_remaining_gain"]) > 0.2
    assert screening["endpoint_self_redundancy"]["screens_off"] == "1"
    adversary = {r["candidate_id"]: r for r in _rows("source_observable_cod_adversary.csv")}
    assert len(adversary) == 4
    assert {r["reliable_escape_hatch"] for r in adversary.values()} == {"0"}
    assert float(adversary["CODADV2"]["hit"]) == 0.5740740740740741
    assert adversary["CODADV4"]["reason"] == "not executable without new nested probe design"


def test_c61_synthetic_rank_gauge_validation_keeps_negative_control():
    synthetic = {r["comparison_id"]: r for r in _rows("synthetic_rank_gauge_cod_summary.csv")}
    assert synthetic["RG-COD1_weak_rank_candidate_gauge"]["candidate_specific_gauge_gap"] == "1"
    assert synthetic["RG-COD1_weak_rank_candidate_gauge"]["pair_flip_possible"] == "1"
    assert synthetic["RG-COD5_target_local_common_offset"]["candidate_specific_gauge_gap"] == "0"
    assert synthetic["RG-COD5_target_local_common_offset"]["pair_flip_possible"] == "0"
    assert synthetic["RG-COD6_endpoint_oracle"]["hit_after"] == "0.9444444444444444"
    grid = {r["gamma"]: r for r in _rows("synthetic_rank_gauge_parameter_grid.csv")}
    assert float(grid["0.0"]["tail_error"]) == 0.5
    assert float(grid["1.0"]["tail_error"]) == 0.15865525393145707
    assert {r["model_bound_only"] for r in grid.values()} == {"1"}


def test_c61_future_instrumentation_and_training_gates_remain_locked():
    future = {r["need_id"]: r for r in _rows("future_training_need_matrix.csv")}
    assert {r["authorized_in_c61"] for r in future.values()} == {"0"}
    assert future["FT1"]["needed_for_c61_claim"] == "0"
    assert future["FT6"]["training_or_inference_required"] == "blocked"
    gates = {r["gate"]: r for r in _rows("training_gate_decision_matrix.csv")}
    assert gates["C61_training"]["decision"] == "not_executed"
    assert gates["re_inference"]["decision"] == "not_authorized"
    assert gates["GPU"]["decision"] == "not_authorized"
    assert gates["BNCI2014_004"]["decision"] == "reserved"
    assert gates["seeds_3_4"]["decision"] == "reserved"
    assert gates["selector_search"]["decision"] == "forbidden"
    assert gates["manuscript_drafting"]["decision"] == "not_authorized"


def test_c61_red_team_manifest_hashes_and_large_artifact_scan_pass():
    red = {r["gate"]: r for r in _rows("red_team_failure_ledger.csv")}
    assert len(red) == 16
    assert {r["failed"] for r in red.values()} == {"0"}
    forbidden = _rows("forbidden_claim_scan.csv")
    assert len(forbidden) == 20
    assert {r["affirmative_hits"] for r in forbidden} == {"0"}
    large = _rows("large_artifact_scan.csv")
    assert len(large) == 29
    assert {r["passed"] for r in large} == {"1"}
    manifest = _rows("artifact_manifest.csv")
    assert len(manifest) == 29
    for row in manifest:
        assert os.path.exists(row["path"])
        assert int(row["size_bytes"]) == os.path.getsize(row["path"])
        assert row["sha256"] == _sha256(row["path"])
        if row["artifact_class"] == "table":
            assert row["row_count"] != ""


def test_c61_run_recomputes_core_decision_without_writing():
    res = c61.run(test_status="unit")
    assert res["decision"]["primary"] == "C61-A_conditional_observability_divergence_ladder_established"
    assert res["decision"]["training_gate"] == c61.TRAINING_GATE
    assert len(res["eeg_cod_ladder_summary_rows"]) == 6
    endpoint = {
        r["comparison_id"]: r for r in res["eeg_cod_ladder_summary_rows"]
    }["COD_endpoint_given_source_template"]
    assert endpoint["bayes_hit_gain"] == 0.9444444444444444 - 0.7037037037037037
