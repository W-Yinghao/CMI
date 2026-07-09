"""C54 Endpoint-Scalar Tautology / Bit-Budget Boundary Audit tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.conditioned_ceiling_coverage import c54_endpoint_scalar_tautology_bit_budget as c54
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C54_ENDPOINT_SCALAR_TAUTOLOGY_BIT_BUDGET.json"
TABLE_DIR = "oaci/reports/c54_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c54_config_taxonomy_and_guards_are_frozen():
    assert c54._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert c54.MILESTONE == "C54"
    assert c54.LABEL == "primary_joint_good"
    assert c54.BEST_SCALAR == "target_joint_margin_raw"
    assert c54.BEST_ORIENTATION == "high"
    assert c54.NULL_REPS == 64
    assert c54.NULL_SEED == 54054
    assert c54.HIT_GATE == 0.70
    assert set(c54.DECISIONS) == {
        "C54-A_direct_joint_endpoint_tautology",
        "C54-B_single_endpoint_component_sufficiency",
        "C54-C_low_bit_endpoint_oracle_sufficiency",
        "C54-D_near_continuous_endpoint_margin_required",
        "C54-E_cross_cell_endpoint_template_sufficiency",
        "C54-F_same_cell_candidate_endpoint_oracle_only",
        "C54-G_split_label_boundary_unresolved",
        "C54-H_artifact_or_null_like",
    }


def test_c54_decision_is_direct_joint_endpoint_tautology():
    d = _summary()
    assert d["milestone"] == "C54"
    assert d["inherits_from"] == ["C49", "C50", "C51", "C52", "C53"]
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    dec = d["decision"]
    assert dec["primary"] == "C54-A_direct_joint_endpoint_tautology"
    assert dec["primary"] in c54.DECISIONS
    assert dec["secondary"] == [
        "C54-S1_nontransferable_cell_local_endpoint_content",
        "C54-S2_component_endpoint_asymmetry",
        "C54-S3_joint_margin_dominates_components",
        "C54-S4_binary_threshold_already_sufficient",
        "C54-S5_transfer_partially_but_not_fully_closes",
        "C54-S6_no_split_label_budget_available",
    ]
    assert dec["artifact_or_null_like"] is False
    assert dec["direct_joint_label_tautology"] is True
    assert dec["binary_threshold_sufficient"] is True
    assert dec["best_single_endpoint_component"] == "target_bacc_delta:high"
    assert dec["best_single_endpoint_component_hit"] == 0.9259259259259259
    assert dec["best_cross_cell_transfer_hit"] == 0.7037037037037037
    assert dec["same_cell_minus_best_transfer_gap"] == 0.2407407407407407


def test_c54_replays_c53_identity_metrics():
    d = _summary()
    assert d["c53_replay"] == {
        "best_key_only_hit": 0.4876543209876543,
        "best_scalar_field": "target_joint_margin_raw:high",
        "best_strict_source_hit": 0.5061728395061729,
        "c52_trajectory_diagnostic_hit": 0.8127572016460904,
        "c53_best_scalar_endpoint_hit": 0.9444444444444444,
        "random_tie_hit": 0.429723378036041,
    }
    rows = _rows("c53_replay_identity.csv")
    assert len(rows) == 8
    assert {r["pass"] for r in rows} == {"1"}
    replay = {r["metric"]: r for r in rows}
    assert replay["best_scalar_field"]["c54_replayed_value"] == "target_joint_margin_raw:high"
    assert float(replay["c53_best_scalar_endpoint_hit"]["abs_diff"]) == 0.0
    assert replay["cell_count"]["c54_replayed_value"] == "162"


def test_c54_classifies_joint_margin_as_same_label_oracle():
    rows = {r["scalar_name"]: r for r in _rows("endpoint_scalar_inventory.csv")}
    assert len(rows) == 20
    joint = rows["target_joint_margin_raw:high"]
    assert joint["semantic_class"] == "S5_same_label_oracle_margin"
    assert joint["target_label_derived"] == "1"
    assert joint["same_label_derived"] == "1"
    assert joint["requires_target_endpoint"] == "1"
    assert joint["available_at_selection_time"] == "0"
    assert joint["directly_contains_joint_good_threshold"] == "1"
    assert joint["diagnostic_only"] == "1"
    assert float(joint["hit"]) == 0.9444444444444444
    assert int(joint["cell_hit_ge_0_7_count"]) == 153
    bacc = rows["target_bacc_delta:high"]
    assert bacc["semantic_class"] == "S3_target_endpoint_component_label"
    assert bacc["target_label_derived"] == "1"
    assert bacc["same_label_derived"] == "0"
    assert bacc["available_at_selection_time"] == "0"
    assert rows["cell_prior:high"]["semantic_class"] == "S1_key_or_cell_prior_only"
    assert rows["cell_prior:high"]["target_label_derived"] == "0"
    assert rows["split_label_constructed_scalar"]["semantic_class"] == "S6_split_label_constructed_scalar"
    assert rows["split_label_constructed_scalar"]["notes"] == (
        "required per-trial target prediction/label cache unavailable"
    )


def test_c54_tautology_distance_marks_direct_threshold_overlap():
    rows = {r["scalar_name"]: r for r in _rows("endpoint_tautology_distance.csv")}
    assert len(rows) == 19
    joint = rows["target_joint_margin_raw:high"]
    assert float(joint["hit"]) == 0.9444444444444444
    assert float(joint["closed_fraction_vs_c53_gap"]) == 1.0
    assert float(joint["auc_vs_joint_good"]) == 1.0
    assert float(joint["threshold_overlap_with_joint_good"]) == 1.0
    assert joint["direct_joint_label_field"] == "1"
    assert joint["near_endpoint_oracle"] == "1"
    assert joint["same_label_tautology"] == "1"
    label = rows["primary_joint_good:high"]
    assert float(label["spearman_vs_joint_good"]) == 1.0
    bacc = rows["target_bacc_delta:high"]
    assert float(bacc["hit"]) == 0.9259259259259259
    assert float(bacc["closed_fraction_vs_c53_gap"]) == 0.9594594594594595
    assert bacc["same_label_tautology"] == "0"
    assert float(rows["target_nll_delta:high"]["hit"]) == 0.8333333333333334
    assert float(rows["target_ece_delta:high"]["hit"]) == 0.7037037037037037


def test_c54_component_ablation_emits_joint_and_component_rows():
    rows = {r["component_family"]: r for r in _rows("endpoint_component_ablation.csv")}
    assert len(rows) == 11
    assert float(rows["accuracy_component"]["hit"]) == 0.9259259259259259
    assert float(rows["accuracy_component"]["closed_fraction_vs_best_scalar"]) == 0.9594594594594595
    assert int(rows["accuracy_component"]["cell_hit_ge_0_7_count"]) == 150
    assert float(rows["nll_component"]["hit"]) == 0.8333333333333334
    assert float(rows["ece_component"]["hit"]) == 0.7037037037037037
    assert float(rows["joint_margin"]["hit"]) == 0.9444444444444444
    assert rows["joint_margin"]["semantic_class"] == "S5_same_label_oracle_margin"
    assert float(rows["joint_good_label"]["hit"]) == 0.9444444444444444
    assert float(rows["min_endpoint_margin"]["hit"]) == 0.8518518518518519


def test_c54_bit_budget_curve_shows_one_bit_sufficiency():
    d = _summary()
    assert d["bit_budget"] == {
        "binary_sufficient": True,
        "minimal_bits_for_90pct_gap_closure": 1,
        "minimal_bits_for_cell_close_count_ge_114": 1,
        "minimal_bits_for_hit_ge_0_90": 1,
    }
    rows = _rows("endpoint_bit_budget_curve.csv")
    assert len(rows) == 76
    lookup = {(r["scalar_name"], r["bit_budget_mode"], r["threshold_scope"]): r for r in rows}
    binary = lookup[("target_joint_margin_raw:high", "binary_sign", "global_threshold")]
    assert float(binary["hit"]) == 0.9444444444444444
    assert float(binary["closed_fraction_vs_c53_gap"]) == 1.0
    assert int(binary["cell_close_count"]) == 153
    assert float(lookup[("target_joint_margin_raw:high", "tertile", "global")]["hit"]) == 0.8775940623162846
    assert float(lookup[("target_joint_margin_raw:high", "quartile", "global")]["hit"]) == 0.8433979147214441
    assert float(lookup[("target_joint_margin_raw:high", "decile", "global")]["hit"]) == 0.9351851851851852
    assert float(lookup[("target_joint_margin_raw:high", "rank_only", "within_target_trajectory_cell")]["hit"]) == 0.9444444444444444
    assert float(lookup[("target_joint_margin_raw:high", "continuous_raw", "raw")]["hit"]) == 0.9444444444444444


def test_c54_transfer_templates_are_partial_not_full():
    rows = {r["transfer_mode"]: r for r in _rows("endpoint_transfer_template_summary.csv")}
    assert len(rows) == 6
    assert float(rows["T0_same_cell"]["transfer_hit"]) == 0.9444444444444444
    assert float(rows["T1_leave_target_out"]["transfer_hit"]) == 0.5740740740740741
    assert float(rows["T2_leave_trajectory_out"]["transfer_hit"]) == 0.5740740740740741
    assert float(rows["T3_leave_target_trajectory_cell_out"]["transfer_hit"]) == 0.5740740740740741
    assert float(rows["T4_global_template"]["transfer_hit"]) == 0.5740740740740741
    assert float(rows["T5_matched_source_geometry_template"]["transfer_hit"]) == 0.7037037037037037
    assert float(rows["T5_matched_source_geometry_template"]["same_cell_minus_transfer_gap"]) == 0.2407407407407407
    assert int(rows["T5_matched_source_geometry_template"]["cells_improved"]) == 102
    ledger = _rows("endpoint_transfer_cell_ledger.csv")
    assert len(ledger) == 972
    assert {r["diagnostic_only"] for r in ledger} == {"1"}
    assert {r["no_selection_artifact"] for r in ledger} == {"1"}


def test_c54_label_nulls_include_cell_preserving_controls():
    rows = {r["null_name"]: r for r in _rows("endpoint_label_null_summary.csv")}
    assert len(rows) == 7
    assert float(rows["N0_random_tie_within_cell"]["null_mean_hit"]) == 0.4306520061728395
    assert float(rows["N1_permute_scalar_within_cell"]["null_mean_hit"]) == 0.4296682098765432
    assert float(rows["N1_permute_scalar_within_cell"]["null_p95_hit"]) == 0.4592592592592593
    assert float(rows["N1_permute_scalar_within_cell"]["observed_minus_null_mean"]) == 0.5147762345679012
    assert float(rows["N2_permute_scalar_within_target"]["null_mean_hit"]) == 0.430224408436214
    assert float(rows["N3_permute_scalar_within_trajectory"]["null_mean_hit"]) == 0.42363040123456797
    assert float(rows["N4_permute_joint_good_labels_within_cell"]["null_mean_hit"]) == 0.4376350308641976
    assert float(rows["N5_sign_flip_or_reverse_endpoint_scalar"]["null_mean_hit"]) == 0.05555555555555555
    assert float(rows["N6_quantile_label_shuffle_preserving_cell_histogram"]["null_mean_hit"]) == 0.4274948429728884
    assert {int(r["num_repeats"]) for r in rows.values()} == {1, 64}


def test_c54_cell_ledger_has_162_endpoint_oracle_cells():
    rows = _rows("endpoint_oracle_cell_ledger.csv")
    assert len(rows) == 162
    assert sum(r["reason_code"] == "endpoint_joint_margin_oracle_closes" for r in rows) == 153
    assert sum(r["reason_code"] == "null_like_or_unstable" for r in rows) == 9
    assert sum(r["minimal_bit_budget"] == "binary_sign" for r in rows) == 153
    assert sum(int(r["same_label_tautology"]) for r in rows) == 153
    assert sum(int(r["null_like"]) for r in rows) == 9
    assert sum(int(r["component_sufficient"]) for r in rows) == 150
    assert sum(int(r["joint_margin_required"]) for r in rows) == 3
    first = next(r for r in rows if r["trajectory_id"] == "0|1|0|S0_full_support")
    assert float(first["random_hit"]) == 0.05
    assert float(first["c53_same_cell_scalar_hit"]) == 1.0
    assert first["best_endpoint_scalar_semantic_class"] == "S5_same_label_oracle_margin"
    assert first["available_at_selection_time"] == "0"
    assert first["diagnostic_only"] == "1"


def test_c54_tables_have_expected_shape_and_gates():
    d = _summary()
    assert d["table_row_counts"] == {
        "c53_replay_identity": 8,
        "endpoint_bit_budget_curve": 76,
        "endpoint_component_ablation": 11,
        "endpoint_label_null_summary": 7,
        "endpoint_oracle_cell_ledger": 162,
        "endpoint_scalar_inventory": 20,
        "endpoint_tautology_distance": 19,
        "endpoint_transfer_cell_ledger": 972,
        "endpoint_transfer_template_summary": 6,
        "no_selector_artifact_gate": 10,
        "red_team_verification": 8,
    }
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    red = {r["check"]: r["passed"] for r in _rows("red_team_verification.csv")}
    assert red and set(red.values()) == {"1"}
    assert "endpoint_oracle_cell_ledger_rows" not in d
    assert os.path.getsize(REPORT_JSON) < 50_000


def test_c54_run_loads_committed_artifacts_without_recompute():
    res = c54.run()
    assert res["decision"]["primary"] == "C54-A_direct_joint_endpoint_tautology"
    assert res["bit_budget"]["binary_sufficient"] is True
    assert len(res["endpoint_oracle_cell_ledger_rows"]) == 162
    assert all(g["passed"] for g in c54.no_selector_gate(res))


def test_c54_outputs_do_not_emit_selector_artifacts_or_forbidden_claims():
    for name in (
        "c53_replay_identity.csv",
        "endpoint_scalar_inventory.csv",
        "endpoint_tautology_distance.csv",
        "endpoint_component_ablation.csv",
        "endpoint_bit_budget_curve.csv",
        "endpoint_transfer_template_summary.csv",
        "endpoint_transfer_cell_ledger.csv",
        "endpoint_label_null_summary.csv",
        "endpoint_oracle_cell_ledger.csv",
    ):
        text = open(os.path.join(TABLE_DIR, name)).read().lower()
        assert "model_hash" not in text
        assert "checkpoint_hash" not in text
        assert "selected_candidate_id" not in text
        assert "checkpoint recommendation" not in text
    forbidden = (
        "deployable selector",
        "source-only selector",
        "few-label method",
        "oaci rescue",
        "target labels can be used at deployment",
        "actionable rule",
        "checkpoint recommendation",
        "production selection",
        "solves target selection",
        "deployable target-aware selector",
    )
    for name in ("C54_ENDPOINT_SCALAR_TAUTOLOGY_BIT_BUDGET.md",
                 "C54_ENDPOINT_SCALAR_TAUTOLOGY_BIT_BUDGET.json",
                 "C54_RED_TEAM_VERIFICATION.md"):
        text = open(os.path.join("oaci/reports", name)).read().lower()
        for phrase in forbidden:
            assert phrase not in text
    md = open("oaci/reports/C54_ENDPOINT_SCALAR_TAUTOLOGY_BIT_BUDGET.md").read()
    assert "C54-A_direct_joint_endpoint_tautology" in md
    assert "same-label target endpoint oracle" in md
