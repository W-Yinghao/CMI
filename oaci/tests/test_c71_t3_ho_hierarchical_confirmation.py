"""C71 T3-HO authorized confirmation tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c71_t3_ho_hierarchical_confirmation as c71
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C71_T3_HO_HIERARCHICAL_CONFIRMATION.json"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c71_tables"
PROTOCOL_JSON = "oaci/reports/C71_T3_HO_CONFIRMATORY_PROTOCOL.json"
PROTOCOL_SHA = "oaci/reports/C71_T3_HO_CONFIRMATORY_PROTOCOL.sha256"


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


def test_c71_taxonomy_cli_auth_and_authorized_gate():
    assert c71._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert c71._auth_present("") is False
    assert c71._auth_present("handoff says C71_T3_HO_REINFERENCE_ONLY_AUTHORIZED") is False
    assert c71._auth_present(c71.AUTH_TOKEN) is True
    assert set(c71.DECISIONS) == {
        "C71-A_within_target_split_label_reliability_confirmed_actionability_weak",
        "C71-B_small_budget_split_label_actionability_confirmed",
        "C71-C_dense_label_partial_recovery_confirmed",
        "C71-D_C70_effect_not_replicated_on_T3_HO",
        "C71-E_hierarchical_signal_replication_but_measurement_control_gap_narrows",
        "C71-F_protocol_masking_or_dependency_blocker",
        "C71-G_T3_HO_ready_but_not_authorized",
        "C71-S1_T3_HO_disjointness_confirmed",
        "C71-S2_physical_view_isolation_passed",
        "C71-S3_candidate_specific_gauge_recovery_partial",
        "C71-S4_common_offset_not_explanatory",
        "C71-S5_no_strict_source_escape_hatch",
        "C71-S6_strict_source_escape_hatch_found",
        "C71-S7_conditional_observability_stable_diagnostic",
        "C71-S8_conditional_cs_proxy_only",
        "C71-S9_target_population_generalization_unresolved",
        "C71-S10_new_training_not_justified",
        "C71-S11_independent_target_or_dataset_replication_now_justified",
    }

    d = _summary()
    assert d["milestone"] == "C71"
    assert d["diagnostic_only_non_deployable"] is True
    assert d["authorization_present"] is True
    assert d["no_forward_readiness_only"] is False
    assert d["forward_or_reinference_executed"] == 1
    assert d["training_attempted"] == 0
    assert d["gpu_used"] == 0
    assert d["t3_cache_consumed"] == 1
    assert d["raw_cache_rows_emitted"] == 605952
    assert d["selector_artifact_emitted"] == 0
    assert d["checkpoint_recommendation_artifact_emitted"] == 0
    assert d["final_gate"] == "T3_HO_CONFIRMS_MEASUREMENT_CONTROL_SEPARATION"
    assert d["decision"]["red_team_failure_count"] == 0
    assert d["decision"]["primary"] == "C71-A_within_target_split_label_reliability_confirmed_actionability_weak"
    assert d["key_numbers"]["t3_full_physical_units"] == 1268
    assert d["key_numbers"]["t2_consumed_units"] == 216
    assert d["key_numbers"]["t3_ho_disjoint_units"] == 1052
    assert d["key_numbers"]["t3_ho_cache_rows"] == 605952
    assert d["key_numbers"]["blocked_perm_p"] == 0.0002
    assert d["key_numbers"]["budget_8_gauge_recovery"] == 0.17053
    assert d["key_numbers"]["full_construction_gauge_recovery"] == 0.43452

    noauth = c71.run(test_status="unit_noauth")
    assert noauth["authorization_present"] is False
    assert noauth["decision"]["final_gate"] == "T3_HO_READY_BUT_NOT_AUTHORIZED"
    assert noauth["forward_or_reinference_executed"] == 0
    assert noauth["t3_cache_consumed"] == 0


def test_c71_protocol_timing_parent_sha_and_authorized_t3_access():
    assert os.path.exists(PROTOCOL_JSON)
    assert os.path.exists(PROTOCOL_SHA)
    protocol = json.load(open(PROTOCOL_JSON))
    summary = _summary()
    assert protocol["schema_version"] == "c71_t3_ho_confirmatory_protocol_v1"
    assert protocol["authorization_token_status"] == "present"
    assert protocol["t3_ho_cache_generation_authorized"] == 1
    assert protocol["t3_ho_cache_generation_executed"] == 0
    assert protocol["t3_ho_units_from_parent"] == 1052
    assert protocol["t3_full_physical_units_from_parent"] == 1268
    assert protocol["t2_consumed_units_from_parent"] == 216
    assert open(PROTOCOL_SHA).read().strip() == _sha256(PROTOCOL_JSON)
    assert summary["c71_protocol_sha256"] == _sha256(PROTOCOL_JSON)
    assert summary["parent_c70_protocol_sha256"] == "9075e13d86192c48677b167457b765854db4f7d77781474753212b62d480e611"
    assert summary["parent_c70_protocol_sha256"] == summary["parent_c70_protocol_sha256_replayed"]

    timing = {r["event"]: r for r in _rows("protocol_timing.csv")}
    assert timing["c71_protocol_lock"]["status"] == "created_before_t3_access"
    assert timing["first_t3_ho_manifest_path_read"]["timestamp_utc"] >= timing["c71_protocol_lock"]["timestamp_utc"]
    assert timing["first_t3_ho_manifest_path_read"]["status"] == "after_protocol_lock"
    assert timing["first_t3_ho_outcome_read"]["timestamp_utc"] >= timing["c71_protocol_lock"]["timestamp_utc"]
    assert timing["first_t3_ho_outcome_read"]["status"] == "after_protocol_lock"


def test_c71_required_authorized_tables_and_view_contracts():
    risk = {r["risk_id"]: r for r in _rows("risk_register.csv")}
    assert set(risk) == set(c71.RISK_ROWS)
    assert {r["blocking"] for r in risk.values()} == {"0"}

    disjoint = {r["check"]: r for r in _rows("t3_ho_disjointness_ledger.csv")}
    assert disjoint["parent_protocol_sha_match"]["passed"] == "1"
    assert disjoint["t3_ho_units"]["observed"] == "1052"
    assert disjoint["t3_ho_units"]["status"] == "executed"
    assert disjoint["t2_t3_ho_overlap"]["observed"] == "0"
    assert disjoint["cache_hashes_match"]["passed"] == "1"

    overlap = {(r["left"], r["right"]): r for r in _rows("t1_t2_t3_overlap_matrix.csv")}
    assert overlap[("T1", "T2")]["overlap_units"] == "64"
    assert overlap[("T1", "T2")]["independent_confirmation"] == "0"
    assert overlap[("T2", "T3-HO")]["overlap_units"] == "0"
    assert overlap[("T2", "T3-HO")]["independent_confirmation"] == "1"

    views = {r["view_name"]: r for r in _rows("physical_view_manifest.csv")}
    assert set(views) == {"source_only_view", "key_template_view", "construction_label_view", "evaluation_label_view", "same_label_oracle_view"}
    assert all(os.path.exists(r["path"]) and _sha256(r["path"]) == r["sha256"] for r in views.values())
    assert views["source_only_view"]["uses_target_labels"] == "0"
    assert views["source_only_view"]["available_at_selection_time"] == "1"
    assert views["construction_label_view"]["uses_evaluation_labels"] == "0"
    assert views["evaluation_label_view"]["uses_evaluation_labels"] == "1"
    assert views["same_label_oracle_view"]["uses_target_labels"] == "1"
    assert views["same_label_oracle_view"]["available_at_selection_time"] == "0"

    hypotheses = _rows("primary_hypothesis_summary.csv")
    assert len(hypotheses) == 5
    by_h = {r["hypothesis"]: r for r in hypotheses}
    assert by_h["H1_within_target_reliability"]["status"] == "pass"
    assert by_h["H2_small_budget_weakness"]["status"] == "fail_actionability_gate"
    assert by_h["H3_dense_partial_recovery"]["status"] == "partial"
    assert by_h["H4_measurement_control_separation"]["status"] == "pass"
    assert {r["budget"] for r in _rows("reliability_actionability_separation.csv")} == set(c71.PRIMARY_BUDGETS)

    cache = {r["cache_kind"]: r for r in _rows("t3_ho_external_cache_manifest.csv")}
    assert cache["minimal_logits_probs_metadata"]["row_count"] == "605952"
    assert cache["minimal_logits_probs_metadata"]["sha256_match"] == "1"
    assert cache["minimal_logits_probs_metadata"]["git_tracked"] == "0"

    schema = _rows("t3_ho_cache_schema_audit.csv")
    assert schema
    assert {r["passed"] for r in schema} == {"1"}


def test_c71_inference_contracts_feature_provenance_and_failure_ledger():
    split = {r["contract"]: r for r in _rows("shared_trial_split_contract.csv")}
    assert split["unique_trial_budget"]["passed"] == "1"
    assert split["shared_construction_ids"]["passed"] == "1"
    assert split["disjoint_construction_evaluation"]["passed"] == "1"

    unique = _rows("unique_label_budget_ledger.csv")
    assert {r["checkpoint_scaled_cost_allowed"] for r in unique} == {"0"}
    assert {r["labels_counted_as"] for r in unique} == {"unique_target_trial_ids_per_class_from_construction_view"}

    blocked = _rows("blocked_permutation_summary.csv")
    assert len(blocked) == 1
    assert blocked[0]["observed"] == "0.637483"
    assert blocked[0]["permutations"] == "4999"
    assert blocked[0]["exceedances"] == "0"
    assert blocked[0]["p_value"] == "0.0002"
    assert blocked[0]["minimum_p"] == "0.0002"
    assert blocked[0]["row_iid_used"] == "0"

    boot = _rows("cluster_bootstrap_summary.csv")
    assert boot
    assert {r["row_iid_used"] for r in boot} == {"0"}
    assert all(r["status"] == "actual_t3_ho_conditional_on_frozen_targets" for r in boot)

    cs = _rows("conditional_cs_estimator_contract.csv")[0]
    assert cs["assumptions_met_now"] == "0"
    assert cs["faithfulness_claim_allowed"] == "0"

    features = {r["feature_family"]: r for r in _rows("feature_availability_ledger.csv")}
    assert features["strict_source_domain_trial_logits"]["available_now"] == "0"
    assert features["strict_source_domain_trial_logits"]["uses_target_labels"] == "0"
    assert features["same_label_endpoint_oracle"]["available_at_selection_time"] == "0"

    adversary = _rows("strict_source_adversary_summary.csv")[0]
    assert adversary["target_labels_used"] == "0"
    assert adversary["escape_hatch_found"] == "0"

    failures = {r["reason"]: r for r in _rows("failure_reason_ledger.csv")}
    assert failures["authorization"]["status"] == "pass"
    assert failures["protocol_locked_before_t3_access"]["status"] == "pass"
    assert failures["t3_ho_consumed"]["status"] == "pass"
    assert failures["target_population_generalization"]["status"] == "unresolved"
    assert failures["measurement_control_separation"]["status"] == "pass"

    curve = {r["budget"]: r for r in _rows("label_budget_curve.csv")}
    assert curve["8"]["mean_gauge_residual_recovery"] == "0.17053"
    assert curve["64"]["mean_gauge_residual_recovery"] == "0.423187"
    assert curve[c71.FULL_BUDGET_LABEL]["mean_gauge_residual_recovery"] == "0.43452"
    assert all(r["few_label_sufficiency_claimed"] == "0" for r in curve.values())


def test_c71_red_team_artifact_hygiene_and_reports_are_clean():
    red = _rows("red_team_failure_ledger.csv")
    assert red
    assert {r["failed"] for r in red} == {"0"}

    forbidden = _rows("forbidden_claim_scan.csv")
    assert forbidden
    assert {r["affirmative_hits"] for r in forbidden} == {"0"}
    assert {r["passed"] for r in forbidden} == {"1"}

    large = _rows("large_artifact_scan.csv")
    assert large
    assert {r["over_50mb"] for r in large} == {"0"}
    assert {r["passed"] for r in large} == {"1"}

    manifest = _rows("artifact_manifest.csv")
    assert manifest
    for row in manifest:
        assert os.path.exists(row["path"])
        assert _sha256(row["path"]) == row["sha256"]
    paths = {r["path"] for r in manifest}
    assert os.path.join(REPORT_DIR, "C71_T3_HO_HIERARCHICAL_CONFIRMATION.md") in paths
    assert REPORT_JSON in paths
    assert os.path.join(REPORT_DIR, "C71_RED_TEAM_VERIFICATION.md") in paths
    assert os.path.join(REPORT_DIR, "C71_PROTOCOL_TIMING_AUDIT.md") in paths
    assert PROTOCOL_JSON in paths
    assert PROTOCOL_SHA in paths
