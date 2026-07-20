"""C67 dual-mode C66 provenance and masked cache-consumption tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c67_c66_dual_mode_cache_consumption as c67
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C67_C66_DUAL_MODE_CACHE_CONSUMPTION.json"
TABLE_DIR = "oaci/reports/c67_tables"


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


def test_c67_taxonomy_lock_and_final_gate_are_dual_mode():
    assert c67._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert set(c67.DECISIONS) == {
        "C67-A_c66_dual_mode_provenance_reconciled",
        "C67-B_authorized_cache_integrity_validated",
        "C67-C_masked_view_contract_validated",
        "C67-D_split_label_smoke_feasible_not_sufficiency",
        "C67-E_split_label_smoke_underpowered_or_unstable",
        "C67-F_sample_level_conditional_cs_smoke_feasible",
        "C67-G_sample_level_conditional_cs_underpowered_or_unstable",
        "C67-H_endpoint_oracle_boundary_preserved",
        "C67-I_label_leakage_or_availability_violation_found",
        "C67-J_larger_reinference_only_cache_campaign_ready_but_not_authorized",
        "C67-K_new_training_still_not_justified",
    }
    d = _summary()
    assert d["milestone"] == "C67"
    assert d["config_hash"] == "664007686afb520f"
    assert d["diagnostic_only_non_deployable"] is True
    assert d["phase0_gate"] == "C66_DUAL_MODE_PROVENANCE_RECONCILED"
    assert d["decision"]["primary"] == "C67-A_c66_dual_mode_provenance_reconciled"
    assert d["final_gate"] == "C67_DUAL_MODE_MICROCACHE_VALID_BUT_UNDERPOWERED_FOR_SPLIT_LABEL_CS"
    assert d["decision"]["red_team_failure_count"] == 0
    for active in (
        "C67-A_c66_dual_mode_provenance_reconciled",
        "C67-B_authorized_cache_integrity_validated",
        "C67-C_masked_view_contract_validated",
        "C67-D_split_label_smoke_feasible_not_sufficiency",
        "C67-E_split_label_smoke_underpowered_or_unstable",
        "C67-F_sample_level_conditional_cs_smoke_feasible",
        "C67-G_sample_level_conditional_cs_underpowered_or_unstable",
        "C67-H_endpoint_oracle_boundary_preserved",
        "C67-J_larger_reinference_only_cache_campaign_ready_but_not_authorized",
        "C67-K_new_training_still_not_justified",
    ):
        assert active in d["decision"]["active"]
    assert "C67-I_label_leakage_or_availability_violation_found" in d["decision"]["inactive"]


def test_c67_dual_mode_provenance_ledger_keeps_noauth_and_authorized_modes_separate():
    ledger = {r["mode"]: r for r in _rows("c66_dual_mode_provenance_ledger.csv")}
    assert set(ledger) == {"no_auth_baseline", "authorized_microcampaign"}

    noauth = ledger["no_auth_baseline"]
    assert noauth["commit_id"] == "635ccbc"
    assert noauth["gate"] == "MICROCAMPAIGN_READY_BUT_NOT_AUTHORIZED"
    assert noauth["forward_attempted"] == "0"
    assert noauth["cache_rows"] == "0"
    assert noauth["authoritative_for_consumption"] == "0"
    assert noauth["external_cache_path"] == ""

    auth = ledger["authorized_microcampaign"]
    assert auth["commit_id"] == "b369f59"
    assert auth["gate"] == "REINFERENCE_ONLY_MICROCAMPAIGN_EXECUTED_AND_CACHE_MANIFESTED"
    assert auth["forward_attempted"] == "1"
    assert auth["cache_rows"] == "3456"
    assert auth["authoritative_for_consumption"] == "1"
    assert auth["external_cache_path"].endswith("trial_logits_probs_cache.csv")
    assert len(auth["cache_sha256"]) == 64

    replay = {r["cache_id"]: r for r in _rows("authorized_cache_manifest_replay.csv")}
    assert replay["c66_trial_cache_v1"]["sha256_match"] == "1"
    assert replay["c66_trial_cache_v1"]["exists"] == "1"
    assert replay["c66_trial_cache_v1"]["row_count_manifest"] == "3456"
    assert replay["c66_trial_cache_manifest_v1"]["sha256_match"] == "1"
    assert replay["c66_trial_cache_manifest_v1"]["exists"] == "1"
    assert _sha256(auth["external_cache_path"]) == auth["cache_sha256"]


def test_c67_authorized_cache_integrity_validates_schema_counts_and_holdouts():
    checks = {r["check"]: r for r in _rows("cache_integrity_summary.csv")}
    assert checks["phase0_gate"]["value"] == "C66_DUAL_MODE_PROVENANCE_RECONCILED"
    assert checks["row_count"]["value"] == "3456"
    assert checks["checkpoint_count"]["value"] == "6"
    assert checks["target_count"]["value"] == "3"
    assert checks["dataset_ids"]["value"] == "BNCI2014_001"
    assert checks["seed_set"]["value"] == "0;1;2"
    assert checks["duplicate_checkpoint_trial_keys"]["value"] == "0"
    assert checks["probabilities_sum_to_one"]["value"] == "3456"
    assert checks["pred_matches_argmax"]["value"] == "3456"
    assert checks["margin_matches_prob_gap"]["value"] == "3456"
    assert checks["raw_cache_not_git_tracked"]["value"] == "0"
    assert {r["passed"] for r in checks.values()} == {"1"}

    schema = {r["field"]: r for r in _rows("cache_schema_inventory.csv")}
    for field in (
        "trial_cache_id",
        "checkpoint_id",
        "dataset_id",
        "target_id",
        "seed",
        "trial_id",
        "class_label_quarantined",
        "y_true_quarantined",
        "y_pred",
        "logits",
        "probabilities",
        "split_role_for_future_split_label",
    ):
        assert schema[field]["present"] == "1"
        assert schema[field]["nonempty_count"] == "3456"

    mapping = _rows("checkpoint_trial_mapping_audit.csv")
    assert len(mapping) == 6
    assert sum(int(r["trial_rows"]) for r in mapping) == 3456
    assert {r["trial_rows"] for r in mapping} == {"576"}
    assert {r["seed"] for r in mapping} <= {"0", "1", "2"}
    assert {r["target"] for r in mapping} == {"1", "5", "9"}
    assert {r["status"] for r in mapping} == {"pass"}


def test_c67_masked_view_contract_blocks_label_leakage_and_oracle_availability():
    views = {r["view"]: r for r in _rows("masked_view_contract.csv")}
    assert set(views) == {
        "source_only_view",
        "target_construction_view",
        "target_evaluation_view",
        "same_label_oracle_view",
        "conditional_cs_diagnostic_view",
    }
    assert views["source_only_view"]["label_visible_rows"] == "0"
    assert views["source_only_view"]["prediction_visible_rows"] == "0"
    assert views["source_only_view"]["selection_path_enforced"] == "1"
    assert views["source_only_view"]["policy_boundary_only"] == "0"
    assert views["target_construction_view"]["selection_path_enforced"] == "1"
    assert views["target_evaluation_view"]["selection_path_enforced"] == "1"
    assert views["same_label_oracle_view"]["uses_same_label_endpoint_scalar"] == "1"
    assert views["same_label_oracle_view"]["available_at_selection_time"] == "0"
    assert views["same_label_oracle_view"]["selection_path_enforced"] == "0"
    assert views["same_label_oracle_view"]["policy_boundary_only"] == "1"
    assert views["conditional_cs_diagnostic_view"]["selection_path_enforced"] == "0"
    assert views["conditional_cs_diagnostic_view"]["policy_boundary_only"] == "1"
    assert {r["allowed_for_selection_rule"] for r in views.values()} == {"0"}
    assert {r["diagnostic_only"] for r in views.values()} == {"1"}
    assert {r["status"] for r in views.values()} == {"pass"}

    unit = {r["test"]: r for r in _rows("label_view_unit_test_summary.csv")}
    assert unit["source_only_masks_labels_and_predictions"]["passed"] == "1"
    assert unit["construction_view_masks_eval_labels"]["passed"] == "1"
    assert unit["evaluation_view_masks_construct_labels"]["passed"] == "1"
    assert unit["same_label_oracle_unavailable_at_selection_time"]["passed"] == "1"

    fields = {r["field"]: r for r in _rows("label_dependency_ledger.csv")}
    assert fields["y_true_quarantined"]["source_only_view"] == "masked"
    assert fields["class_label_quarantined"]["source_only_view"] == "masked"
    assert fields["probabilities"]["source_only_view"] == "masked"
    assert fields["y_pred"]["source_only_view"] == "masked"


def test_c67_split_label_smoke_is_feasible_but_not_sufficiency():
    summary = _rows("split_label_smoke_summary.csv")
    assert len(summary) == 1
    row = summary[0]
    assert row["analysis"] == "construct_bacc_predicts_eval_bacc_high"
    assert row["checkpoint_units"] == "6"
    assert row["status"] == "completed_underpowered"
    assert row["claim"] == "diagnostic_smoke_not_sufficiency"
    assert float(row["same_label_oracle_hit"]) == 1.0
    assert 0.0 <= float(row["hit_rate"]) <= 1.0

    failures = {r["risk"]: r for r in _rows("split_label_failure_ledger.csv")}
    assert failures["independent_checkpoint_units"]["blocks_sufficiency_claim"] == "1"
    assert failures["same_label_oracle_boundary"]["blocks_sufficiency_claim"] == "1"
    assert failures["source_only_selector_claim"]["value"] == "0"


def test_c67_sample_level_cs_smoke_is_underpowered_and_keeps_variable_availability_flags():
    summary = _rows("sample_level_cs_smoke_summary.csv")
    assert len(summary) == 1
    row = summary[0]
    assert row["estimator"] == "ridge_increment_proxy_not_full_cs"
    assert row["independent_checkpoint_units"] == "6"
    assert row["status"] == "underpowered_or_unstable"
    assert row["claim"] == "conditional_cs_smoke_not_full_claim"

    bandwidth = _rows("cs_bandwidth_stress.csv")
    assert len(bandwidth) == 4
    assert {r["independent_units"] for r in bandwidth} == {"6"}
    assert {r["status"] for r in bandwidth} == {"underpowered_n6"}

    feasible = {r["check"]: r for r in _rows("cs_estimator_feasibility_ledger.csv")}
    assert feasible["uses_eval_labels_in_x2"]["passed"] == "1"
    assert feasible["full_conditional_cs_claim"]["value"] == "0"
    assert feasible["full_conditional_cs_claim"]["passed"] == "1"
    assert feasible["independent_units_sufficient"]["value"] == "6"
    assert feasible["independent_units_sufficient"]["passed"] == "0"

    variables = {r["audit"]: r for r in _rows("cs_variable_availability_ledger.csv")}
    assert variables["split_label_increment"]["uses_eval_labels_in_x2"] == "0"
    assert variables["split_label_increment"]["diagnostic_only"] == "1"
    assert variables["same_label_endpoint_oracle"]["uses_same_label_endpoint_scalar"] == "1"
    assert variables["same_label_endpoint_oracle"]["available_at_selection_time"] == "0"


def test_c67_atom_scope_and_runtime_guards_preserve_no_new_forward_boundary():
    runtime = {r["check"]: r for r in _rows("device_runtime_audit.csv")}
    assert runtime["c67_new_forward_pass"]["observed"] == "0"
    assert runtime["c67_training"]["observed"] == "0"
    assert runtime["c67_gpu"]["observed"] == "0"
    assert runtime["c66_execution_gpu_used"]["observed"] == "0"
    assert {r["passed"] for r in runtime.values()} == {"1"}

    atoms = {r["trace"]: r for r in _rows("atom_trace_feasibility_from_c66_cache.csv")}
    assert atoms["logits"]["present_in_c66_cache"] == "1"
    assert atoms["probabilities"]["present_in_c66_cache"] == "1"
    assert atoms["predictions"]["present_in_c66_cache"] == "1"
    assert atoms["representation_z"]["present_in_c66_cache"] == "0"
    assert atoms["representation_z"]["requires_new_forward_in_c67"] == "1"
    assert atoms["Wz"]["present_in_c66_cache"] == "0"
    assert atoms["Wz"]["requires_new_forward_in_c67"] == "1"


def test_c67_red_team_artifact_hygiene_and_reports_are_clean():
    summary = _summary()
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
    paths = {r["path"]: r for r in manifest}
    for path, row in paths.items():
        assert os.path.exists(path)
        assert _sha256(path) == row["sha256"]
    assert "oaci/reports/C67_C66_DUAL_MODE_CACHE_CONSUMPTION.md" in paths
    assert "oaci/reports/C67_C66_DUAL_MODE_CACHE_CONSUMPTION.json" in paths

    emitted_csvs = {
        os.path.splitext(name)[0]
        for name in os.listdir(TABLE_DIR)
        if name.endswith(".csv") and name not in {"large_artifact_scan.csv"}
    }
    assert emitted_csvs <= set(summary["table_row_counts"])

    with open("oaci/reports/C67_C66_DUAL_MODE_CACHE_CONSUMPTION.md") as f:
        report = f.read()
    assert "dual-mode milestone" in report
    assert "not a science conflict" in report
    assert "not a full conditional-CS claim" in report
