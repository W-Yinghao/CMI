"""C60 Rank-Gauge Proof Stress / Empirical-Theory Bridge tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c60_rank_gauge_proof_stress_empirical_bridge as c60
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C60_RANK_GAUGE_PROOF_STRESS_EMPIRICAL_BRIDGE.json"
TABLE_DIR = "oaci/reports/c60_tables"


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


def test_c60_decision_scope_and_training_gate_are_frozen():
    assert c60._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert set(c60.DECISIONS) == {
        "C60-A_rank_gauge_proof_validated_without_change",
        "C60-B_rank_gauge_proof_repaired_or_strengthened",
        "C60-C_empirical_assumption_bridge_supported",
        "C60-D_empirical_assumption_bridge_partial_or_weak",
        "C60-E_rank_gauge_assumptions_fail_on_frozen_eeg_artifacts",
        "C60-F_source_observable_theory_counterexample_found",
        "C60-G_no_source_observable_counterexample_found",
        "C60-H_theorem_to_eeg_bridge_requires_instrumented_data",
        "C60-I_training_blueprint_refined_but_not_authorized",
        "C60-J_training_not_scientifically_justified_yet",
        "C60-K_claim_or_definition_inconsistency_found",
    }
    d = _summary()
    assert d["milestone"] == "C60"
    assert d["config_hash"] == "664007686afb520f"
    assert d["c59_commit"] == "828adb3"
    assert d["c59_decision"] == "C59-B_rank_gauge_synthetic_lower_bound_proved"
    assert d["decision"]["primary"] == "C60-B_rank_gauge_proof_repaired_or_strengthened"
    assert "C60-D_empirical_assumption_bridge_partial_or_weak" in d["decision"]["active"]
    assert "C60-G_no_source_observable_counterexample_found" in d["decision"]["active"]
    assert "C60-F_source_observable_theory_counterexample_found" in d["decision"]["inactive"]
    assert d["decision"]["training_gate"] == "TRAINING_BLUEPRINT_REFINED_BUT_NOT_AUTHORIZED"
    assert d["decision"]["red_team_failure_count"] == 0
    assert d["theorem_status"]["rank_gauge"] == "synthetic_model_bound_repaired"
    assert d["recommended_next_step"] == "wait for remote review; C61 may request instrumented training approval but C60 does not authorize execution"


def test_c60_table_shapes_and_reports_are_complete():
    d = _summary()
    assert d["table_row_counts"] == {
        "artifact_manifest": 39,
        "assumption_gap_ledger": 7,
        "assumption_support_scores": 8,
        "conditional_sufficiency_summary": 6,
        "coverage_abstention_bound_summary": 4,
        "fano_packing_repair_summary": 4,
        "forbidden_claim_scan": 19,
        "gauge_tail_bound_curve": 8,
        "gauge_tail_empirical_summary": 6,
        "information_ladder_formal_status": 9,
        "instrumentation_blueprint_v2": 7,
        "large_artifact_scan": 39,
        "lecam_witness_repair_summary": 4,
        "lower_bound_failure_reason_ledger": 7,
        "markov_boundary_probe_summary": 6,
        "missing_data_to_theorem_gap": 8,
        "partition_bound_provenance": 5,
        "proof_audit_checklist": 12,
        "proof_repair_log": 6,
        "rank_gauge_theorem_variants": 9,
        "rank_margin_gauge_scale_summary": 6,
        "red_team_failure_ledger": 16,
        "registered_partition_bound_extensions": 7,
        "schema_validation_summary": 32,
        "source_error_vs_gauge_tail_ledger": 7,
        "source_escape_hatch_red_team": 5,
        "source_observable_adversary_results": 8,
        "source_observable_availability_ledger": 8,
        "subagent_audit_manifest": 10,
        "test_command_manifest": 4,
        "theorem_assumption_inventory": 10,
        "theorem_stress_failure_modes": 9,
        "theorem_to_eeg_assumption_map": 8,
        "training_gate_decision_matrix": 13,
    }
    required_reports = {
        "C60_EMPIRICAL_THEORY_BRIDGE.md",
        "C60_INSTRUMENTATION_BLUEPRINT.md",
        "C60_PROOF_STRESS_AUDIT.md",
        "C60_RANK_GAUGE_PROOF_STRESS_EMPIRICAL_BRIDGE.json",
        "C60_RANK_GAUGE_PROOF_STRESS_EMPIRICAL_BRIDGE.md",
        "C60_RED_TEAM_VERIFICATION.md",
        "C60_SOURCE_ADVERSARY_AND_LOWER_BOUND_REPAIR.md",
    }
    assert required_reports <= {p for p in os.listdir("oaci/reports") if p.startswith("C60_")}
    expected_tables = {
        "artifact_manifest.csv",
        "assumption_gap_ledger.csv",
        "assumption_support_scores.csv",
        "conditional_sufficiency_summary.csv",
        "coverage_abstention_bound_summary.csv",
        "fano_packing_repair_summary.csv",
        "forbidden_claim_scan.csv",
        "gauge_tail_bound_curve.csv",
        "gauge_tail_empirical_summary.csv",
        "information_ladder_formal_status.csv",
        "instrumentation_blueprint_v2.csv",
        "large_artifact_scan.csv",
        "lecam_witness_repair_summary.csv",
        "lower_bound_failure_reason_ledger.csv",
        "markov_boundary_probe_summary.csv",
        "missing_data_to_theorem_gap.csv",
        "partition_bound_provenance.csv",
        "proof_audit_checklist.csv",
        "proof_repair_log.csv",
        "rank_gauge_theorem_variants.csv",
        "rank_margin_gauge_scale_summary.csv",
        "red_team_failure_ledger.csv",
        "registered_partition_bound_extensions.csv",
        "schema_validation_summary.csv",
        "source_error_vs_gauge_tail_ledger.csv",
        "source_escape_hatch_red_team.csv",
        "source_observable_adversary_results.csv",
        "source_observable_availability_ledger.csv",
        "subagent_audit_manifest.csv",
        "test_command_manifest.csv",
        "theorem_assumption_inventory.csv",
        "theorem_stress_failure_modes.csv",
        "theorem_to_eeg_assumption_map.csv",
        "training_gate_decision_matrix.csv",
    }
    assert expected_tables == {p for p in os.listdir(TABLE_DIR) if p.endswith(".csv")}


def test_c60_proof_audit_repairs_rank_gauge_scope():
    audit = {r["gate"]: r for r in _rows("proof_audit_checklist.csv")}
    assert audit["selector_measurability"]["status"] == "repaired"
    assert audit["error_hit_regret_separated"]["status"] == "repaired"
    assert audit["two_candidate_scope"]["status"] == "repaired"
    repairs = {r["repair_id"]: r for r in _rows("proof_repair_log.csv")}
    assert "source sigma-field" in repairs["PR1"]["repair"]
    assert "general CDF formula" in repairs["PR2"]["repair"]
    assert "demote to corollary" in repairs["PR3"]["repair"]
    assumptions = {r["assumption_id"]: r for r in _rows("theorem_assumption_inventory.csv")}
    assert "W" in assumptions["RG60-A2"]["statement"]
    assert assumptions["RG60-A6"]["proof_status"] == "optional_corollary_only"
    assert assumptions["RG60-A10"]["proof_status"] == "scope_guardrail"


def test_c60_rank_gauge_stress_keeps_multicandidate_and_offset_limits():
    variants = {r["variant_id"]: r for r in _rows("rank_gauge_theorem_variants.csv")}
    assert variants["RGV1_general_two_candidate_cdf"]["claim_status"] == "proved_repaired"
    assert "high-rank-rule error" in variants["RGV1_general_two_candidate_cdf"]["formula_or_bound"]
    assert variants["RGV4_multi_candidate_top1"]["claim_status"] == "proxy_not_theorem"
    assert variants["RGV9_target_local_offset"]["claim_status"] == "does_not_support_pairwise_flip"
    failures = {r["failure_id"]: r for r in _rows("theorem_stress_failure_modes.csv")}
    assert failures["TSF1"]["breaks_extension"] == "1"
    assert failures["TSF9"]["breaks_core_theorem"] == "1"
    curve = {r["gamma_abs_rank_margin_over_gauge_scale"]: r for r in _rows("gauge_tail_bound_curve.csv")}
    assert float(curve["0.0"]["two_candidate_error_tail"]) == 0.5
    assert float(curve["1.0"]["two_candidate_error_tail"]) == 0.15865525393145707
    assert {r["theorem_status"] for r in curve.values()} == {"two_candidate_theorem_multi_candidate_proxy"}


def test_c60_empirical_bridge_is_partial_and_traceable():
    support = {r["bridge_axis"]: r for r in _rows("assumption_support_scores.csv")}
    assert support["source_visible_rank"]["support_class"] == "partial"
    assert support["endpoint_oracle"]["support_class"] == "oracle"
    assert support["direct_gauge_distribution"]["support_class"] == "missing"
    gap = {r["gap_id"]: r for r in _rows("assumption_gap_ledger.csv")}
    assert gap["BG1"]["requires_new_instrumentation"] == "1"
    assert gap["BG3"]["missing_item"] == "source transcript observation law"
    tail = {r["quantity"]: r for r in _rows("gauge_tail_empirical_summary.csv")}
    assert float(tail["strict_source_miss"]["value"]) == 1.0 - 0.5061728395061729
    margin = {r["field"]: r for r in _rows("rank_margin_gauge_scale_summary.csv")}
    assert float(margin["strict_source"]["error_tail_proxy"]) == 1.0 - 0.5061728395061729
    assert float(margin["endpoint_oracle"]["hit"]) == 0.9444444444444444


def test_c60_lower_bound_repairs_remain_conservative():
    lecam = {r["attempt_id"]: r for r in _rows("lecam_witness_repair_summary.csv")}
    assert lecam["LC60-1_within_target"]["repair_status"] == "empirical_witness_only"
    assert lecam["LC60-4_synthetic_rank_gauge"]["theorem_grade"] == "1"
    assert lecam["LC60-4_synthetic_rank_gauge"]["blocker"] == "not EEG distribution law"
    fano = {r["attempt_id"]: r for r in _rows("fano_packing_repair_summary.csv")}
    assert {r["theorem_grade"] for r in fano.values()} == {"0"}
    assert fano["F60-2_key_cells"]["mi_or_kl_status"] == "missing_stable_matrix"
    failures = {r["failure_id"]: r for r in _rows("lower_bound_failure_reason_ledger.csv")}
    assert failures["LB1"]["branch"] == "LeCam"
    assert failures["LB7"]["branch"] == "TrainingGate"


def test_c60_source_adversary_and_conditional_ladder_keep_information_boundary():
    adv = {r["candidate_id"]: r for r in _rows("source_observable_adversary_results.csv")}
    assert len(adv) == 8
    assert {r["reliable_escape_hatch"] for r in adv.values()} == {"0"}
    assert float(adv["SA60-2"]["hit"]) == 0.5740740740740741
    assert adv["SA60-6"]["uses_target_labels"] == "1"
    avail = {r["input_family"]: r for r in _rows("source_observable_availability_ledger.csv")}
    assert avail["target_joint_margin_raw"]["allowed"] == "0"
    suff = {r["candidate_added"]: r for r in _rows("conditional_sufficiency_summary.csv")}
    assert suff["label_diagnostic"]["status"] == "large_diagnostic_gain"
    assert suff["same_label_endpoint"]["status"] == "endpoint_adds_after_template"
    assert suff["same_label_endpoint"]["screens_off_endpoint"] == "0"
    assert suff["endpoint_self_redundancy"]["screens_off_endpoint"] == "1"
    mb = {r["variable"]: r for r in _rows("markov_boundary_probe_summary.csv")}
    assert mb["same_label_endpoint"]["available_at_selection_time"] == "0"
    assert mb["same_label_endpoint"]["empirical_boundary_role"] == "empirical_oracle_boundary"
    ladder = {r["information_class"]: r for r in _rows("information_ladder_formal_status.csv")}
    assert ladder["I1_strict_source"]["source_only"] == "1"
    assert ladder["I5_split_label_or_few_label"]["formal_status"] == "future_unresolved_missing_cache"
    assert ladder["I7_same_label_endpoint"]["label_content"] == "1"


def test_c60_training_blueprint_refined_but_not_authorized():
    gates = {r["gate"]: r for r in _rows("training_gate_decision_matrix.csv")}
    assert gates["C60_training"]["decision"] == "not_executed"
    assert gates["P0_no_training"]["decision"] == "allowed_read_only"
    assert gates["P1_split_label_cache"]["decision"] == "proposal_only"
    assert gates["P6_reserved_holdout"]["decision"] == "reserved"
    assert gates["BNCI2014_004"]["decision"] == "reserved"
    assert gates["seeds_3_4"]["decision"] == "reserved"
    assert gates["GPU"]["decision"] == "not_authorized"
    assert gates["re_inference"]["decision"] == "not_authorized"
    blueprint = {r["protocol_id"]: r for r in _rows("instrumentation_blueprint_v2.csv")}
    assert blueprint["P0"]["training_or_inference_needed"] == "none"
    assert blueprint["P6"]["approval_required"] == "1"
    assert "BNCI2014_004" in blueprint["P6"]["new_data_required"]
    missing = {r["gap_id"]: r for r in _rows("missing_data_to_theorem_gap.csv")}
    assert missing["MD1"]["needed_for"] == "EEG distribution-level bridge"
    assert missing["MD7"]["data_item"] == "stable KL/MI matrix"


def test_c60_red_team_manifest_and_forbidden_scan_pass():
    red = _rows("red_team_failure_ledger.csv")
    assert len(red) == 16
    assert {r["failed"] for r in red} == {"0"}
    scan = _rows("forbidden_claim_scan.csv")
    assert len(scan) == len(c60.FORBIDDEN_PATTERNS) == 19
    assert {r["affirmative_hits"] for r in scan} == {"0"}
    assert {r["passed"] for r in scan} == {"1"}
    schema = _rows("schema_validation_summary.csv")
    assert len(schema) == 32
    assert {r["passed"] for r in schema} == {"1"}
    large = _rows("large_artifact_scan.csv")
    assert len(large) == 39
    assert {r["over_50mb"] for r in large} == {"0"}
    for row in large:
        assert os.path.exists(row["path"])
        assert int(row["size_bytes"]) == os.path.getsize(row["path"])
    manifest = _rows("artifact_manifest.csv")
    assert len(manifest) == 39
    for row in manifest:
        assert os.path.exists(row["path"])
        assert int(row["size_bytes"]) == os.path.getsize(row["path"])
        assert row["sha256"] == _sha256(row["path"])
    manifest_paths = {r["path"] for r in manifest}
    assert "oaci/reports/c60_tables/artifact_manifest.csv" not in manifest_paths
    assert "oaci/reports/c60_tables/large_artifact_scan.csv" not in manifest_paths
    assert os.path.getsize(REPORT_JSON) < 50_000
    subagents = _rows("subagent_audit_manifest.csv")
    assert len(subagents) == 10
    assert {r["integration_status"] for r in subagents} == {"launched_or_locally_integrated"}


def test_c60_outputs_do_not_emit_selector_training_or_manuscript_artifacts():
    for dirpath, _, filenames in os.walk("oaci/reports"):
        if "c60" not in dirpath.lower() and dirpath != "oaci/reports":
            continue
        for name in filenames:
            if not (name.startswith("C60_") or dirpath.endswith("c60_tables")):
                continue
            if name in {
                "forbidden_claim_scan.csv",
                "red_team_failure_ledger.csv",
                "training_gate_decision_matrix.csv",
                "source_escape_hatch_red_team.csv",
            }:
                continue
            text = open(os.path.join(dirpath, name), errors="ignore").read().lower()
            assert "selected_candidate_id" not in text
            assert "chosen checkpoint" not in text
            assert "new real eeg training" not in text
            assert "manuscript drafting starts" not in text
    main = open("oaci/reports/C60_RANK_GAUGE_PROOF_STRESS_EMPIRICAL_BRIDGE.md").read()
    assert "C60 does not run training" in main
    assert "C60 itself authorizes no execution" in main


def test_c60_run_recomputes_without_training():
    res = c60.run()
    assert res["c59_decision"] == "C59-B_rank_gauge_synthetic_lower_bound_proved"
    assert res["training_gate"] == "TRAINING_BLUEPRINT_REFINED_BUT_NOT_AUTHORIZED"
    assert len(res["rank_gauge_theorem_variants_rows"]) == 9
    assert len(res["source_observable_adversary_results_rows"]) == 8
    assert len(res["instrumentation_blueprint_v2_rows"]) == 7
