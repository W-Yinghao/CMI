"""C59 Rank-Gauge Theorem Factory tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c59_formal_lower_bound_theory_factory as c59
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C59_FORMAL_LOWER_BOUND_THEORY_FACTORY.json"
TABLE_DIR = "oaci/reports/c59_tables"


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


def test_c59_decision_scope_and_training_gate_are_frozen():
    assert c59._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert set(c59.DECISIONS) == {
        "C59-A_registered_partition_bound_formalized_as_theorem",
        "C59-B_rank_gauge_synthetic_lower_bound_proved",
        "C59-C_empirical_lecam_witness_bound_nontrivial",
        "C59-D_fano_assouad_bound_still_trivial_or_unstable",
        "C59-E_conditional_sufficiency_boundary_formalized",
        "C59-F_source_observable_counterexample_found",
        "C59-G_training_blueprint_ready_but_not_authorized",
        "C59-H_theory_blocked_requires_new_instrumented_data",
        "C59-I_theory_blocked_by_definition_or_claim_inconsistency",
    }
    d = _summary()
    assert d["milestone"] == "C59"
    assert d["config_hash"] == "664007686afb520f"
    assert d["c58_commit"] == "5132193"
    assert d["c58_decision"] == "C58-A_finite_population_lower_bound_established"
    assert d["decision"]["primary"] == "C59-B_rank_gauge_synthetic_lower_bound_proved"
    assert "C59-A_registered_partition_bound_formalized_as_theorem" in d["decision"]["active"]
    assert "C59-E_conditional_sufficiency_boundary_formalized" in d["decision"]["active"]
    assert "C59-F_source_observable_counterexample_found" in d["decision"]["inactive"]
    assert d["decision"]["training_gate"] == "TRAINING_BLUEPRINT_READY_BUT_NOT_AUTHORIZED"
    assert d["decision"]["red_team_failure_count"] == 0
    assert d["recommended_next_step"] == "wait for remote review; do not train or start manuscript drafting"


def test_c59_table_shapes_and_reports_are_complete():
    d = _summary()
    assert d["table_row_counts"] == {
        "artifact_manifest": 42,
        "assouad_cube_attempts": 3,
        "conditional_entropy_ladder": 5,
        "conditional_mi_diagnostics": 5,
        "fano_failure_reasons": 4,
        "fano_packing_summary": 3,
        "forbidden_claim_scan": 14,
        "formal_object_spec": 12,
        "information_sigma_field_ladder": 8,
        "large_artifact_scan": 42,
        "lecam_bound_candidates": 3,
        "lecam_failure_reasons": 4,
        "lecam_witness_candidates": 3,
        "markov_boundary_candidate_summary": 5,
        "missing_data_for_atom_trace": 5,
        "missing_data_for_split_label": 5,
        "missing_data_for_theory": 5,
        "partition_function_class_limits": 5,
        "rank_gauge_bound_grid": 7,
        "rank_gauge_empirical_constant_map": 8,
        "rank_gauge_theorem_assumptions": 6,
        "red_team_failure_ledger": 11,
        "registered_partition_bound_constants": 10,
        "schema_validation_summary": 30,
        "source_adversary_candidates": 6,
        "source_adversary_red_team": 4,
        "source_adversary_results": 6,
        "subagent_audit_manifest": 10,
        "test_command_manifest": 4,
        "training_authorization_gate": 5,
        "training_campaign_options": 6,
        "training_instrumentation_schema": 9,
    }
    required_reports = {
        "C59_ARTIFACT_SUFFICIENCY_AUDIT.md",
        "C59_CONDITIONAL_SUFFICIENCY_BOUNDARY.md",
        "C59_FANO_ASSOUAD_ATTEMPT.md",
        "C59_FORMAL_LOWER_BOUND_THEORY_FACTORY.json",
        "C59_FORMAL_LOWER_BOUND_THEORY_FACTORY.md",
        "C59_FORMAL_SPEC.md",
        "C59_INSTRUMENTED_EEG_BLUEPRINT.md",
        "C59_LECAM_WITNESS_ATTEMPT.md",
        "C59_RANK_GAUGE_THEOREM_MODEL.md",
        "C59_RED_TEAM_VERIFICATION.md",
        "C59_REGISTERED_PARTITION_BOUND.md",
        "C59_SOURCE_ESCAPE_HATCH_ADVERSARY.md",
    }
    assert required_reports <= {p for p in os.listdir("oaci/reports") if p.startswith("C59_")}
    expected_tables = {
        "artifact_manifest.csv",
        "assouad_cube_attempts.csv",
        "conditional_entropy_ladder.csv",
        "conditional_mi_diagnostics.csv",
        "fano_failure_reasons.csv",
        "fano_packing_summary.csv",
        "forbidden_claim_scan.csv",
        "formal_object_spec.csv",
        "information_sigma_field_ladder.csv",
        "large_artifact_scan.csv",
        "lecam_bound_candidates.csv",
        "lecam_failure_reasons.csv",
        "lecam_witness_candidates.csv",
        "markov_boundary_candidate_summary.csv",
        "missing_data_for_atom_trace.csv",
        "missing_data_for_split_label.csv",
        "missing_data_for_theory.csv",
        "partition_function_class_limits.csv",
        "rank_gauge_bound_grid.csv",
        "rank_gauge_empirical_constant_map.csv",
        "rank_gauge_theorem_assumptions.csv",
        "red_team_failure_ledger.csv",
        "registered_partition_bound_constants.csv",
        "schema_validation_summary.csv",
        "source_adversary_candidates.csv",
        "source_adversary_red_team.csv",
        "source_adversary_results.csv",
        "subagent_audit_manifest.csv",
        "test_command_manifest.csv",
        "training_authorization_gate.csv",
        "training_campaign_options.csv",
        "training_instrumentation_schema.csv",
    }
    assert expected_tables == {p for p in os.listdir(TABLE_DIR) if p.endswith(".csv")}


def test_c59_registered_partition_theorem_constants_are_c58_consistent():
    rows = {r["partition_id"]: r for r in _rows("registered_partition_bound_constants.csv")}
    assert len(rows) == 10
    assert rows["B0"]["finite_population_theorem_status"] == "proved_for_registered_partition"
    assert rows["B4"]["finite_population_theorem_status"] == "proved_for_registered_partition"
    assert rows["B7"]["finite_population_theorem_status"] == "proved_for_registered_partition"
    assert float(rows["B0"]["H_star_pi"]) == 0.4297233780360411
    assert float(rows["B0"]["miss_lower_bound"]) == 0.5702766219639589
    assert float(rows["B7"]["H_star_pi"]) == 0.9444444444444444
    assert float(rows["B1"]["H_star_pi"]) == 0.5061728395061729
    assert rows["B1"]["finite_population_theorem_status"] == "empirical_surrogate_not_full_partition_theorem"
    limits = {r["limit_id"]: r for r in _rows("partition_function_class_limits.csv")}
    assert limits["L1_registered_partition_only"]["covered"] == "1"
    assert "arbitrary nonlinear source functions" in limits["L1_registered_partition_only"]["not_covered"]
    assert limits["L5_distributional_generalization"]["covered"] == "0"


def test_c59_rank_gauge_model_bound_is_theorem_scoped_not_eeg_theorem():
    assumptions = {r["assumption_id"]: r for r in _rows("rank_gauge_theorem_assumptions.csv")}
    assert assumptions["RG-A6"]["statement"] == "The theorem does not assert a theorem about EEG distributions"
    grid = {r["grid_id"]: r for r in _rows("rank_gauge_bound_grid.csv")}
    assert len(grid) == 7
    assert float(grid["gamma_0.00"]["two_candidate_error_lower_bound_normal_gauge"]) == 0.5
    assert float(grid["gamma_1.00"]["two_candidate_error_lower_bound_normal_gauge"]) == 0.15865525393145707
    assert float(grid["gamma_2.00"]["two_candidate_hit_upper_bound"]) == 0.9772498680518208
    assert {r["proof_status"] for r in grid.values()} == {"analytic_for_two_candidate_normal_gauge"}
    mapping = {r["constant"]: r for r in _rows("rank_gauge_empirical_constant_map.csv")}
    assert float(mapping["strict_source_hit"]["empirical_value"]) == 0.5061728395061729
    assert float(mapping["endpoint_oracle_hit"]["empirical_value"]) == 0.9444444444444444
    report = open("oaci/reports/C59_RANK_GAUGE_THEOREM_MODEL.md").read()
    assert "synthetic/model-bound" in report
    assert "not an EEG distribution theorem" in report


def test_c59_lecam_fano_conditional_boundaries_are_conservative():
    lecam = {r["bound_id"]: r for r in _rows("lecam_bound_candidates.csv")}
    assert float(lecam["LC1_within_target"]["lecam_error_candidate"]) == 0.4975786924939467
    assert float(lecam["LC2_within_trajectory"]["lecam_error_candidate"]) == 0.43356164383561646
    assert {r["theorem_status"] for r in lecam.values()} == {"empirical_candidate_not_distributional_theorem"}
    fano = {r["packing_id"]: r for r in _rows("fano_packing_summary.csv")}
    assert fano["F1_source_rank_binary"]["status"] == "trivial_due_log2_term"
    assert fano["F2_key_cells"]["status"] == "missing_stable_MI"
    assert fano["F3_endpoint_oracle_binary"]["status"] == "tautological_endpoint_oracle"
    mb = {r["candidate"]: r for r in _rows("markov_boundary_candidate_summary.csv")}
    assert mb["source_only"]["markov_boundary_status"] == "insufficient"
    assert mb["source_plus_key"]["markov_boundary_status"] == "insufficient"
    assert mb["same_label_endpoint_scalar"]["markov_boundary_status"] == "oracle_boundary"
    assert float(mb["same_label_endpoint_scalar"]["hit"]) == 0.9444444444444444


def test_c59_source_adversary_and_training_blueprint_keep_guardrails():
    adv = {r["candidate_id"]: r for r in _rows("source_adversary_results.csv")}
    assert len(adv) == 6
    assert {r["reliable_escape_hatch"] for r in adv.values()} == {"0"}
    assert float(adv["SADV2"]["hit"]) == 0.5740740740740741
    assert float(adv["SADV6"]["hit"]) == 0.7037037037037037
    gates = {r["gate"]: r for r in _rows("training_authorization_gate.csv")}
    assert gates["C59_training"]["decision"] == "not_authorized"
    assert gates["BNCI2014_004"]["decision"] == "reserved"
    assert gates["seeds_3_4"]["decision"] == "reserved"
    assert gates["GPU"]["decision"] == "not_authorized"
    campaigns = {r["option_id"]: r for r in _rows("training_campaign_options.csv")}
    assert campaigns["P0"]["minimal_runs"] == "0"
    assert campaigns["P5"]["touches_BNCI2014_004"] == "1"
    assert campaigns["P5"]["touches_seeds_3_4"] == "1"
    assert campaigns["P5"]["minimal_runs"] == "not authorized"
    schema = {r["schema_field"] for r in _rows("training_instrumentation_schema.csv")}
    assert {"representation_z", "Wz_projection", "leakage_atom_trace", "logits_probabilities"} <= schema


def test_c59_red_team_manifest_and_forbidden_scan_pass():
    red = _rows("red_team_failure_ledger.csv")
    assert len(red) == 11
    assert {r["failed"] for r in red} == {"0"}
    scan = _rows("forbidden_claim_scan.csv")
    assert len(scan) == len(c59.FORBIDDEN_PATTERNS) == 14
    assert {r["affirmative_hits"] for r in scan} == {"0"}
    assert {r["passed"] for r in scan} == {"1"}
    schema = _rows("schema_validation_summary.csv")
    assert len(schema) == 30
    assert {r["passed"] for r in schema} == {"1"}
    large = _rows("large_artifact_scan.csv")
    assert len(large) == 42
    assert {r["over_50mb"] for r in large} == {"0"}
    for row in large:
        assert os.path.exists(row["path"])
        assert int(row["size_bytes"]) == os.path.getsize(row["path"])
    manifest = _rows("artifact_manifest.csv")
    assert len(manifest) == 42
    for row in manifest:
        assert os.path.exists(row["path"])
        assert int(row["size_bytes"]) == os.path.getsize(row["path"])
        assert row["sha256"] == _sha256(row["path"])
    manifest_paths = {r["path"] for r in manifest}
    assert "oaci/reports/c59_tables/artifact_manifest.csv" not in manifest_paths
    assert "oaci/reports/c59_tables/large_artifact_scan.csv" not in manifest_paths
    assert os.path.getsize(REPORT_JSON) < 50_000
    subagents = _rows("subagent_audit_manifest.csv")
    assert len(subagents) == 10
    assert {r["integration_status"] for r in subagents} == {"launched_integrated"}


def test_c59_outputs_do_not_emit_selector_training_or_m1_artifacts():
    for dirpath, _, filenames in os.walk("oaci/reports"):
        if "c59" not in dirpath.lower() and dirpath != "oaci/reports":
            continue
        for name in filenames:
            if not (name.startswith("C59_") or dirpath.endswith("c59_tables")):
                continue
            if name in {"forbidden_claim_scan.csv", "red_team_failure_ledger.csv", "training_authorization_gate.csv"}:
                continue
            text = open(os.path.join(dirpath, name), errors="ignore").read().lower()
            assert "selected_candidate_id" not in text
            assert "chosen checkpoint" not in text
            assert "new eeg training run" not in text
            assert "manuscript drafting starts" not in text
    main = open("oaci/reports/C59_FORMAL_LOWER_BOUND_THEORY_FACTORY.md").read()
    assert "C59 does not run training" in main
    assert "No distribution-free minimax theorem is claimed" in main


def test_c59_run_recomputes_without_training():
    res = c59.run()
    assert res["c58_decision"] == "C58-A_finite_population_lower_bound_established"
    assert res["training_gate"] == "TRAINING_BLUEPRINT_READY_BUT_NOT_AUTHORIZED"
    assert len(res["rank_gauge_bound_grid_rows"]) == 7
    assert len(res["source_adversary_results_rows"]) == 6
    assert len(res["training_campaign_options_rows"]) == 6
