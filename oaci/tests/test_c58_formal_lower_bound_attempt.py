"""C58 Formal Lower-Bound Attempt / Training Gate tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c58_formal_lower_bound_attempt as c58
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C58_FORMAL_LOWER_BOUND_ATTEMPT.json"
TABLE_DIR = "oaci/reports/c58_tables"
FORMAL_DIR = "oaci/reports/c58_formal_lower_bound"
TRAINING_DIR = "oaci/reports/c58_training_gate"


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


def test_c58_config_decision_and_non_m1_scope_are_frozen():
    assert c58._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert c58.MILESTONE == "C58"
    assert set(c58.DECISIONS) == {
        "C58-A_finite_population_lower_bound_established",
        "C58-B_lecam_style_two_point_bound_established_under_empirical_assumptions",
        "C58-C_fano_assouad_packing_bound_nontrivial",
        "C58-D_empirical_boundary_only_formal_bound_not_yet_supported",
        "C58-E_source_observable_escape_hatch_found",
        "C58-F_formalization_requires_new_instrumented_real_eeg_training",
        "C58-G_new_training_campaign_scientifically_authorized",
        "C58-H_new_training_not_justified_yet",
    }
    d = _summary()
    assert d["milestone"] == "C58"
    assert d["config_hash"] == "664007686afb520f"
    assert d["diagnostic_only_non_deployable"] is True
    assert d["c57_decision"] == "C57-A_manuscript_scaffold_ready"
    assert d["decision"]["primary"] == "C58-A_finite_population_lower_bound_established"
    assert "C58-D_empirical_boundary_only_formal_bound_not_yet_supported" in d["decision"]["active"]
    assert "C58-F_formalization_requires_new_instrumented_real_eeg_training" in d["decision"]["active"]
    assert "C58-H_new_training_not_justified_yet" in d["decision"]["active"]
    assert "C58-E_source_observable_escape_hatch_found" in d["decision"]["inactive"]
    assert d["decision"]["training_gate"] == "TRAINING_NEEDED_BUT_NOT_AUTHORIZED"
    assert d["decision"]["red_team_failure_count"] == 0
    assert d["recommended_next_step"] == "wait for remote review; do not start M1 or training without explicit user instruction"


def test_c58_table_shapes_and_required_files_are_complete():
    d = _summary()
    assert d["finite_population_reference"] == {
        "endpoint_oracle_hit": 0.9444444444444444,
        "n_candidates": 3804,
        "n_cells": 162,
        "random_tie_hit": 0.4297233780360411,
    }
    assert d["table_row_counts"] == {
        "adversarial_source_rule_ledger": 8,
        "artifact_manifest": 43,
        "artifact_sufficiency_for_lower_bounds": 10,
        "assumption_gap_ledger": 6,
        "cellwise_bayes_error_ledger": 8,
        "conditional_entropy_or_ambiguity_summary": 5,
        "conditional_sufficiency_summary": 5,
        "empirical_to_synthetic_parameter_map": 6,
        "failed_and_successful_escape_hatches": 8,
        "fano_packing_summary": 3,
        "finite_population_bound_summary": 10,
        "forbidden_claim_scan": 13,
        "large_artifact_scan": 43,
        "lecam_witness_summary": 3,
        "markov_boundary_candidate_ledger": 5,
        "missing_instrumentation_ledger": 5,
        "mutual_information_proxy_summary": 4,
        "packing_cell_ledger": 4,
        "packing_null_calibration": 3,
        "partition_bayes_error_by_information_class": 10,
        "red_team_failure_ledger": 12,
        "regret_to_endpoint_oracle": 10,
        "schema_validation_summary": 31,
        "selector_measurability_contract": 8,
        "sigma_field_ladder": 8,
        "source_equivalent_target_divergent_pairs": 4,
        "source_escape_hatch_attack_summary": 6,
        "subagent_audit_manifest": 10,
        "synthetic_simulation_summary": 5,
        "test_command_manifest": 4,
        "training_gate_decision_table": 5,
        "two_point_bound_candidates": 3,
        "utility_and_loss_definitions": 6,
    }
    expected_tables = {
        "adversarial_source_rule_ledger.csv",
        "artifact_manifest.csv",
        "artifact_sufficiency_for_lower_bounds.csv",
        "assumption_gap_ledger.csv",
        "cellwise_bayes_error_ledger.csv",
        "conditional_entropy_or_ambiguity_summary.csv",
        "conditional_sufficiency_summary.csv",
        "empirical_to_synthetic_parameter_map.csv",
        "failed_and_successful_escape_hatches.csv",
        "fano_packing_summary.csv",
        "finite_population_bound_summary.csv",
        "forbidden_claim_scan.csv",
        "large_artifact_scan.csv",
        "lecam_witness_summary.csv",
        "markov_boundary_candidate_ledger.csv",
        "missing_instrumentation_ledger.csv",
        "mutual_information_proxy_summary.csv",
        "packing_cell_ledger.csv",
        "packing_null_calibration.csv",
        "partition_bayes_error_by_information_class.csv",
        "red_team_failure_ledger.csv",
        "regret_to_endpoint_oracle.csv",
        "schema_validation_summary.csv",
        "selector_measurability_contract.csv",
        "sigma_field_ladder.csv",
        "source_equivalent_target_divergent_pairs.csv",
        "source_escape_hatch_attack_summary.csv",
        "subagent_audit_manifest.csv",
        "synthetic_simulation_summary.csv",
        "test_command_manifest.csv",
        "training_gate_decision_table.csv",
        "two_point_bound_candidates.csv",
        "utility_and_loss_definitions.csv",
    }
    assert expected_tables == {p for p in os.listdir(TABLE_DIR) if p.endswith(".csv")}
    assert {"formal_problem_spec.md", "rank_gauge_synthetic_model_spec.md", "synthetic_lower_bound_derivation.md"} == {
        p for p in os.listdir(FORMAL_DIR) if p.endswith(".md")
    }
    assert {
        "atom_trace_schema.md",
        "instrumented_training_gate_decision.md",
        "instrumented_training_protocol.yaml",
        "split_label_cache_schema.md",
        "training_artifact_schema.md",
    } == set(os.listdir(TRAINING_DIR))


def test_c58_finite_population_bound_numbers_are_locked():
    rows = {r["bound_id"]: r for r in _rows("finite_population_bound_summary.csv")}
    assert len(rows) == 10
    expected_hits = {
        "B0": 0.4297233780360411,
        "B1": 0.5061728395061729,
        "B2": 0.5740740740740741,
        "B3": 0.4876543209876543,
        "B4": 0.4297233780360411,
        "B5": 0.8127572016460904,
        "B6": 0.7037037037037037,
        "B7": 0.9444444444444444,
        "B8": 0.9259259259259259,
        "B9": 1.0,
    }
    for bid, hit in expected_hits.items():
        assert float(rows[bid]["measured_hit"]) == hit
        assert float(rows[bid]["empirical_miss_lower_bound"]) == 1.0 - hit
    assert rows["B0"]["exact_partition_bound"] == "1"
    assert rows["B4"]["exact_partition_bound"] == "1"
    assert rows["B7"]["exact_partition_bound"] == "1"
    assert rows["B1"]["exact_partition_bound"] == "0"
    assert float(rows["B1"]["regret_to_endpoint_oracle"]) == 0.43827160493827155
    assert float(rows["B6"]["regret_to_endpoint_oracle"]) == 0.2407407407407407
    assert rows["B7"]["note"] == "endpoint-oracle reference; unavailable at selection time"


def test_c58_sigma_field_and_measurability_contract_separate_oracle_from_selector():
    ladder = {r["sigma_field"]: r for r in _rows("sigma_field_ladder.csv")}
    assert set(ladder) == {
        "G0_random_or_tie",
        "G1_strict_source_observables",
        "G2_source_plus_key",
        "G3_target_unlabeled_geometry",
        "G4_target_grouped_zero_label_structure",
        "G5_split_label_or_few_label",
        "G6_target_label_diagnostic_content",
        "G7_same_label_endpoint_oracle",
    }
    assert ladder["G1_strict_source_observables"]["available_at_selection_time"] == "1"
    assert ladder["G7_same_label_endpoint_oracle"]["available_at_selection_time"] == "0"
    assert ladder["G7_same_label_endpoint_oracle"]["uses_same_label_endpoint"] == "1"
    contract = {r["rule_family"]: r for r in _rows("selector_measurability_contract.csv")}
    assert contract["strict_source_score_rule"]["allowed_for_original_source_only_DG"] == "1"
    assert contract["same_label_endpoint_scalar_rule"]["outputs_action_rule"] == "0"
    assert contract["same_label_endpoint_scalar_rule"]["forbidden_interpretation"] == "not available at selection time"
    assert contract["future_instrumented_training_rule"]["c58_bound_scope"] == "protocol specification only"


def test_c58_lecam_fano_and_synthetic_are_candidates_not_theorems():
    lecam = {r["witness_id"]: r for r in _rows("lecam_witness_summary.csv")}
    assert len(lecam) == 3
    assert float(lecam["LC1_within_target_q10"]["source_neighborhood_divergent_q10"]) == 0.004842615012106538
    assert float(lecam["LC2_within_trajectory_q10"]["source_neighborhood_divergent_q10"]) == 0.13287671232876713
    assert float(lecam["LC3_cross_target_q10"]["source_neighborhood_divergent_q10"]) == 0.9369369369369369
    assert {r["formal_status"] for r in lecam.values()} == {
        "empirical_proxy_not_TV",
        "shows_cross_target_comparability_break",
    }
    fano = {r["packing_id"]: r for r in _rows("fano_packing_summary.csv")}
    assert fano["F1_source_rank_partition"]["bound_status"] == "candidate_only"
    assert fano["F2_key_only_cells"]["bound_status"] == "not_established"
    assert fano["F3_endpoint_oracle"]["bound_status"] == "tautological_oracle"
    synth = {r["scenario_id"]: r for r in _rows("synthetic_simulation_summary.csv")}
    assert synth["RG1_weak_rank_strong_gauge"]["matches_empirical"] == "1"
    formal = open(os.path.join(FORMAL_DIR, "synthetic_lower_bound_derivation.md")).read()
    assert "rather than a universal theorem" in formal


def test_c58_conditional_sufficiency_training_gate_and_missing_instrumentation():
    suff = {r["information_set"]: r for r in _rows("conditional_sufficiency_summary.csv")}
    assert float(suff["S"]["best_hit"]) == 0.5061728395061729
    assert suff["S"]["sufficient"] == "0"
    assert float(suff["S+K"]["best_hit"]) == 0.4876543209876543
    assert suff["S+K+U"]["best_hit"] == ""
    assert float(suff["D"]["best_hit"]) == 0.8127572016460904
    assert suff["D"]["diagnostic_only"] == "1"
    assert float(suff["E"]["best_hit"]) == 0.9444444444444444
    assert suff["E"]["sufficient"] == "1"
    artifacts = {r["artifact_need"]: r for r in _rows("artifact_sufficiency_for_lower_bounds.csv")}
    for need in ("split_label_cache", "per_trial_logits_probabilities", "atom_trace_table", "independent_checkpoint_field_replication"):
        assert artifacts[need]["present"] == "0"
        assert artifacts[need]["needed_for_future_training"] == "1"
    gate = {r["gate"]: r for r in _rows("training_gate_decision_table.csv")}
    assert gate["C58_training_run"]["decision"] == "blocked"
    assert gate["C58_training_run"]["required_for_c58"] == "0"
    assert gate["manuscript_drafting"]["decision"] == "blocked_by_user_discipline"
    protocol = open(os.path.join(TRAINING_DIR, "instrumented_training_protocol.yaml")).read()
    assert "run_training_in_c58: false" in protocol
    assert "BNCI2014_004" in protocol
    assert "reserved_until_user_release" in protocol


def test_c58_red_team_manifest_and_forbidden_scan_pass():
    red = _rows("red_team_failure_ledger.csv")
    assert len(red) == 12
    assert {r["failed"] for r in red} == {"0"}
    scan = _rows("forbidden_claim_scan.csv")
    assert len(scan) == len(c58.FORBIDDEN_PATTERNS) == 13
    assert {r["affirmative_hits"] for r in scan} == {"0"}
    assert {r["passed"] for r in scan} == {"1"}
    schema = _rows("schema_validation_summary.csv")
    assert len(schema) == 31
    assert {r["passed"] for r in schema} == {"1"}
    large = _rows("large_artifact_scan.csv")
    assert len(large) == 43
    assert {r["over_50mb"] for r in large} == {"0"}
    manifest = _rows("artifact_manifest.csv")
    assert len(manifest) == 43
    assert all(r["sha256"] for r in manifest)
    for row in manifest:
        assert os.path.exists(row["path"])
        assert int(row["size_bytes"]) == os.path.getsize(row["path"])
        assert row["sha256"] == _sha256(row["path"])
    manifest_paths = {r["path"] for r in manifest}
    assert "oaci/reports/c58_tables/artifact_manifest.csv" not in manifest_paths
    assert "oaci/reports/c58_tables/large_artifact_scan.csv" not in manifest_paths
    assert os.path.getsize(REPORT_JSON) < 50_000
    subagents = _rows("subagent_audit_manifest.csv")
    assert len(subagents) == 10
    assert {r["integration_status"] for r in subagents} == {"launched_integrated"}


def test_c58_outputs_do_not_emit_selected_or_checkpoint_artifacts():
    roots = ("oaci/reports", FORMAL_DIR, TRAINING_DIR, TABLE_DIR)
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            if "c58" not in dirpath.lower() and dirpath != "oaci/reports":
                continue
            for name in filenames:
                if not (name.startswith("C58_") or dirpath.endswith("c58_tables") or dirpath.endswith("c58_formal_lower_bound") or dirpath.endswith("c58_training_gate")):
                    continue
                text = open(os.path.join(dirpath, name), errors="ignore").read().lower()
                assert "selected_candidate_id" not in text
                assert "checkpoint_hash" not in text
                assert "chosen checkpoint" not in text
    report = open("oaci/reports/C58_FORMAL_LOWER_BOUND_ATTEMPT.md").read()
    assert "does not start M1 manuscript drafting" in report
    assert "does not claim a formal theorem" in report


def test_c58_run_recomputes_without_training_or_reinference():
    res = c58.run()
    assert res["c57_decision"] == "C57-A_manuscript_scaffold_ready"
    assert res["training_gate_decision"] == "TRAINING_NEEDED_BUT_NOT_AUTHORIZED"
    assert len(res["finite_population_bound_summary_rows"]) == 10
    assert len(res["lecam_witness_summary_rows"]) == 3
    assert len(res["fano_packing_summary_rows"]) == 3
    assert len(res["subagent_audit_manifest_rows"]) == 10
