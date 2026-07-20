"""C70 split-label information-budget tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c70_split_label_information_budget as c70
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C70_SPLIT_LABEL_INFORMATION_BUDGET.json"
TABLE_DIR = "oaci/reports/c70_tables"


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


def test_c70_taxonomy_and_read_only_boundary():
    assert c70._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert set(c70.DECISIONS) == {
        "C70-A_small_budget_split_label_gauge_recovery_candidate",
        "C70-B_medium_or_dense_label_recovery_only",
        "C70-C_split_label_reliability_without_actionability",
        "C70-D_c69_signal_collapses_under_hierarchical_controls",
        "C70-E_claim_or_masking_inconsistency_requires_repair",
        "C70-S1_finite_population_label_budget_bound_established",
        "C70-S2_paired_model_bound_nontrivial",
        "C70-S3_block_conditional_cs_stable_diagnostic",
        "C70-S4_conditional_cs_proxy_only_or_bandwidth_sensitive",
        "C70-S5_no_strict_source_trial_escape_hatch",
        "C70-S6_strict_source_trial_escape_hatch_found",
        "C70-S7_t3_disjoint_confirmatory_protocol_locked",
        "C70-S8_target_population_generalization_unresolved",
        "C70-S9_new_training_not_justified",
    }
    d = _summary()
    assert d["milestone"] == "C70"
    assert d["diagnostic_only_non_deployable"] is True
    assert d["read_only_c69_t1_t2_cache"] is True
    assert d["forward_or_reinference_executed"] == 0
    assert d["training_attempted"] == 0
    assert d["gpu_used"] == 0
    assert d["t3_cache_consumed"] == 0
    assert d["decision"]["red_team_failure_count"] == 0
    assert d["key_numbers"]["target_count"] == 9
    assert d["key_numbers"]["checkpoint_unit_count"] == 216
    assert d["key_numbers"]["unique_target_trial_ids"] == 5184
    assert "C70-S7_t3_disjoint_confirmatory_protocol_locked" in d["decision"]["active"]
    assert "C70-S8_target_population_generalization_unresolved" in d["decision"]["active"]
    assert "C70-S9_new_training_not_justified" in d["decision"]["active"]


def test_c70_c69_dependency_split_contract_and_overlap_are_explicit():
    units = _rows("c69_unit_dependency_graph.csv")
    assert len(units) == 216
    assert {r["target_trial_count"] for r in units} == {"576"}
    assert {r["source_domain_trial_logits_available"] for r in units} == {"0"}

    split = _rows("c69_split_contract_audit.csv")
    assert len(split) == 9
    assert {r["checkpoint_units"] for r in split} == {"24"}
    assert {r["unique_target_trial_ids"] for r in split} == {"576"}
    assert {r["construction_eval_disjoint"] for r in split} == {"1"}
    assert {r["trial_split_shared_across_candidates"] for r in split} == {"1"}
    assert {r["fixed_budget_claim_allowed"] for r in split} == {"1"}

    overlap = {r["comparison"]: r for r in _rows("c69_t1_t2_overlap.csv")}
    assert overlap["t1_vs_t2"]["overlap_units"] == "64"
    assert overlap["t1_vs_t2"]["left_subset_of_right"] == "1"
    assert overlap["t1_vs_t2"]["independent_confirmation_claimed"] == "0"
    assert overlap["t2_vs_t3_ho"]["right_units"] == "1052"
    assert overlap["t2_vs_t3_ho"]["overlap_units"] == "0"


def test_c70_permutation_resolution_and_budget_curve_contract():
    perm = {r["test"]: r for r in _rows("c69_permutation_resolution_audit.csv")}
    assert perm["c69_split_label_spearman"]["reported_p"] == "0.004975"
    assert perm["c69_split_label_spearman"]["inferred_permutations"] == "200"
    assert perm["c69_split_label_spearman"]["inferred_exceedances"] == "0"
    assert perm["c69_binary_y_cod_proxy"]["reported_p"] == "0.015385"
    assert perm["c69_binary_y_cod_proxy"]["inferred_permutations"] == "64"
    assert perm["c69_binary_y_cod_proxy"]["inferred_exceedances"] == "0"

    curve = {r["budget"]: r for r in _rows("label_budget_curve.csv")}
    expected = {str(b) for b in c70.BUDGETS} | {c70.FULL_BUDGET_LABEL}
    assert set(curve) == expected
    assert curve["0"]["mean_unique_construct_trials"] == "0.0"
    assert curve["0"]["mean_pairwise_order_accuracy"] == "0.5"
    assert curve["0"]["few_label_sufficiency_claimed"] == "0"
    assert int(curve["8"]["repeat_count"]) >= 200
    assert int(curve[c70.FULL_BUDGET_LABEL]["target_count"]) == 9
    assert all(r["few_label_sufficiency_claimed"] == "0" for r in curve.values())

    action = {r["budget"]: r for r in _rows("actionability_budget_curve.csv")}
    assert set(action) == expected
    assert all(float(r["coverage_regret_le_0p02"]) <= 1.0 for r in action.values())


def test_c70_hierarchical_bounds_feature_availability_and_t3_protocol():
    blocked = _rows("blocked_permutation_summary.csv")
    assert len(blocked) == 1
    assert blocked[0]["test"] == "full_construction_within_target_centered_spearman"
    assert int(blocked[0]["permutations"]) >= 4999
    assert float(blocked[0]["monte_carlo_floor"]) <= 0.0002
    assert blocked[0]["row_iid_interpretation_used"] == "0"

    bootstrap = _rows("cluster_bootstrap_summary.csv")
    assert bootstrap
    assert {r["target_population_generalization_claimed"] for r in bootstrap} == {"0"}

    finite = _rows("finite_population_budget_bound.csv")
    assert finite
    assert {r["finite_population_scope"] for r in finite} == {"fixed_C69_T2_targets_and_checkpoint_units"}

    paired = _rows("paired_candidate_sample_complexity.csv")
    assert paired
    assert {r["bound_scope"] for r in paired} == {"paired_finite_population_normal_proxy_not_eeg_minimax"}

    features = {r["feature_family"]: r for r in _rows("feature_availability_ledger.csv")}
    assert features["strict_source_domain_trial_logits"]["available_in_c69_cache"] == "0"
    assert features["strict_source_domain_trial_logits"]["status"] == "absent_not_tested"
    assert features["checkpoint_metadata_seed_level_order"]["status"] == "non_label_metadata_not_strict_source_trial_signal"
    assert features["same_label_endpoint_oracle"]["available_at_selection_time"] == "0"

    protocol_path = os.path.join(TABLE_DIR, "C71_T3_CONFIRMATORY_PROTOCOL.json")
    sha_path = os.path.join(TABLE_DIR, "C71_T3_CONFIRMATORY_PROTOCOL.sha256")
    assert os.path.exists(protocol_path)
    assert os.path.exists(sha_path)
    protocol = json.load(open(protocol_path))
    assert protocol["t3_ho_units"] == 1052
    assert protocol["t2_consumed_units"] == 216
    assert protocol["diagnostic_only_non_deployable"] is True
    assert open(sha_path).read().strip() == _sha256(protocol_path)


def test_c70_red_team_artifact_hygiene_and_reports_are_clean():
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
    assert "oaci/reports/C70_SPLIT_LABEL_INFORMATION_BUDGET.md" in paths
    assert "oaci/reports/C70_SPLIT_LABEL_INFORMATION_BUDGET.json" in paths
    assert "oaci/reports/C70_RED_TEAM_VERIFICATION.md" in paths
