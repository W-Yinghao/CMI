"""C55 Cross-Cell Endpoint-Template Transfer / Information-Boundary tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.conditioned_ceiling_coverage import c55_cross_cell_endpoint_template_boundary as c55
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C55_CROSS_CELL_ENDPOINT_TEMPLATE_BOUNDARY.json"
TABLE_DIR = "oaci/reports/c55_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c55_config_taxonomy_and_protocols_are_frozen():
    assert c55._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert c55.MILESTONE == "C55"
    assert c55.LABEL == "primary_joint_good"
    assert c55.BEST_SCALAR == "target_joint_margin_raw"
    assert c55.BEST_ORIENTATION == "high"
    assert c55.NULL_REPS == 64
    assert c55.NULL_SEED == 55055
    assert c55.HIT_GATE == 0.70
    assert set(c55.DECISIONS) == {
        "C55-A_global_endpoint_template_transfer_sufficiency",
        "C55-B_leave_cell_only_partial_transfer",
        "C55-C_leave_target_out_transfer_failure",
        "C55-D_leave_trajectory_out_transfer_failure",
        "C55-E_component_endpoint_transfer_sufficiency",
        "C55-F_same_cell_endpoint_oracle_required",
        "C55-G_transfer_requires_unavailable_test_endpoint_scalar",
        "C55-H_null_like_transfer_artifact",
        "C55-I_inconclusive_due_to_support_or_artifact",
    }
    assert c55.TEMPLATE_PROTOCOLS == (
        "leave_cell_out",
        "leave_target_out",
        "leave_trajectory_out",
        "same_target_cross_trajectory",
        "matched_source_geometry",
    )
    assert c55.TRANSFER_PROTOCOLS == (
        "leave_cell_out",
        "leave_target_out",
        "leave_trajectory_out",
    )


def test_c55_decision_is_unavailable_test_endpoint_scalar_boundary():
    d = _summary()
    assert d["milestone"] == "C55"
    assert d["inherits_from"] == ["C49", "C50", "C51", "C52", "C53", "C54"]
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    dec = d["decision"]
    assert dec["primary"] == "C55-G_transfer_requires_unavailable_test_endpoint_scalar"
    assert dec["primary"] in c55.DECISIONS
    assert dec["secondary"] == [
        "C55-S1_joint_margin_transfer_dominates",
        "C55-S3_threshold_transfers_but_scalar_unavailable",
        "C55-S5_target_local_transfer_only",
        "C55-S7_no_split_label_budget_available",
        "C55-S8_source_only_escape_hatch_still_closed",
    ]
    assert dec["artifact_or_null_like"] is False
    assert dec["requires_test_endpoint_scalar_for_full_close"] is True
    assert dec["best_template_only_score"] == "matched_source_geometry_template_only"
    assert dec["best_endpoint_scalar_transfer_score"] == "leave_cell_out_endpoint_scalar_threshold"
    assert dec["best_template_only_hit"] == 0.7037037037037037
    assert dec["best_endpoint_scalar_transfer_hit"] == 0.9444444444444444
    assert dec["same_cell_minus_best_template_gap"] == 0.2407407407407407
    assert dec["same_cell_minus_best_endpoint_scalar_transfer_gap"] == 0.0


def test_c55_replays_c54_identity_metrics():
    d = _summary()
    assert d["c54_replay"] == {
        "auc_vs_joint_good": 1.0,
        "best_key_only_hit": 0.4876543209876543,
        "best_scalar_field": "target_joint_margin_raw:high",
        "best_strict_source_hit": 0.5061728395061729,
        "binary_sign_hit": 0.9444444444444444,
        "c52_trajectory_diagnostic_hit": 0.8127572016460904,
        "cross_cell_transfer_hit": 0.7037037037037037,
        "random_tie_hit": 0.429723378036041,
        "same_cell_scalar_hit": 0.9444444444444444,
        "threshold_overlap": 1.0,
    }
    rows = _rows("c54_replay_identity.csv")
    assert len(rows) == 9
    assert {r["pass"] for r in rows} == {"1"}
    replay = {r["metric"]: r for r in rows}
    assert replay["best_scalar_field"]["c55_replayed_value"] == "target_joint_margin_raw:high"
    assert float(replay["binary_sign_hit"]["c55_replayed_value"]) == 0.9444444444444444
    assert float(replay["threshold_overlap"]["c55_replayed_value"]) == 1.0
    assert replay["split_label_budget_available"]["c55_replayed_value"] == "0"


def test_c55_protocol_summary_separates_template_only_from_endpoint_scalar_transfer():
    rows = {r["score_name"]: r for r in _rows("cross_cell_protocol_summary.csv")}
    assert len(rows) == 12
    assert float(rows["best_key_only"]["hit"]) == 0.4876543209876543
    assert rows["best_key_only"]["uses_key_only_inputs"] == "1"
    assert rows["best_key_only"]["requires_test_endpoint_scalar"] == "0"
    assert float(rows["same_cell_endpoint_scalar"]["hit"]) == 0.9444444444444444
    assert rows["same_cell_endpoint_scalar"]["uses_same_cell_target_labels_for_template"] == "1"
    assert rows["same_cell_endpoint_scalar"]["uses_target_endpoint_scalar_on_test_candidate"] == "1"
    assert float(rows["leave_cell_out_template_only"]["hit"]) == 0.5740740740740741
    assert float(rows["leave_target_out_template_only"]["hit"]) == 0.5740740740740741
    assert float(rows["leave_trajectory_out_template_only"]["hit"]) == 0.5740740740740741
    assert float(rows["same_target_cross_trajectory_template_only"]["hit"]) == 0.6481481481481481
    matched = rows["matched_source_geometry_template_only"]
    assert float(matched["hit"]) == 0.7037037037037037
    assert matched["requires_test_endpoint_scalar"] == "0"
    assert matched["uses_other_cell_target_labels_for_template"] == "1"
    assert matched["full_close_gate"] == "0"
    endpoint = rows["leave_cell_out_endpoint_scalar_threshold"]
    assert float(endpoint["hit"]) == 0.9444444444444444
    assert endpoint["requires_test_endpoint_scalar"] == "1"
    assert endpoint["uses_same_cell_target_labels_for_template"] == "0"
    assert endpoint["uses_other_cell_target_labels_for_template"] == "1"
    assert endpoint["full_close_gate"] == "1"
    assert endpoint["available_under_original_source_only_DG"] == "0"


def test_c55_availability_ledger_marks_unavailable_endpoint_scalar_use():
    rows = {r["score_name"]: r for r in _rows("endpoint_template_availability_ledger.csv")}
    assert len(rows) == 13
    assert rows["best_strict_source"]["available_under_original_source_only_DG"] == "1"
    assert rows["best_strict_source"]["diagnostic_only"] == "0"
    assert rows["best_key_only"]["uses_key_only_inputs"] == "1"
    assert rows["best_key_only"]["available_under_original_source_only_DG"] == "0"
    assert rows["matched_source_geometry_template_only"]["availability_class"] == "cross_cell_label_template"
    assert rows["matched_source_geometry_template_only"]["uses_target_endpoint_scalar_on_test_candidate"] == "0"
    assert rows["matched_source_geometry_template_only"]["uses_other_cell_target_labels_for_template"] == "1"
    scalar = rows["leave_target_out_endpoint_scalar_threshold"]
    assert scalar["availability_class"] == "endpoint_scalar_on_test_candidate"
    assert scalar["uses_target_endpoint_scalar_on_test_candidate"] == "1"
    assert scalar["uses_same_cell_target_labels_for_template"] == "0"
    assert scalar["uses_other_cell_target_labels_for_template"] == "1"
    assert scalar["available_under_original_source_only_DG"] == "0"
    assert scalar["diagnostic_only"] == "1"
    assert rows["split_label_constructed_endpoint_template"]["availability_class"] == "split_label_budget_unavailable"


def test_c55_field_family_shows_joint_margin_endpoint_transfer_dominates():
    rows = {
        (r["field_family"], r["transfer_mode"]): r
        for r in _rows("field_family_transfer_summary.csv")
    }
    assert len(rows) == 16
    assert float(rows[("joint_margin", "leave_cell_endpoint_scalar_binary")]["hit"]) == 0.9444444444444444
    assert float(rows[("bacc_component", "leave_cell_endpoint_scalar_binary")]["hit"]) == 0.8195484483887477
    assert float(rows[("nll_component", "leave_cell_endpoint_scalar_binary")]["hit"]) == 0.6040451279445861
    assert float(rows[("ece_component", "leave_cell_endpoint_scalar_binary")]["hit"]) == 0.5850532135995119
    assert float(rows[("bacc_component", "same_cell_continuous")]["hit"]) == 0.9259259259259259
    assert float(rows[("bacc_component", "matched_geometry_template_only")]["hit"]) == 0.7222222222222222
    assert rows[("joint_margin", "matched_geometry_template_only")]["requires_test_endpoint_scalar"] == "0"
    assert rows[("joint_margin", "leave_cell_endpoint_scalar_binary")]["requires_test_endpoint_scalar"] == "1"


def test_c55_threshold_transfer_curve_keeps_endpoint_scalar_requirement_visible():
    rows = {
        (r["protocol"], r["threshold_mode"]): r
        for r in _rows("threshold_transfer_curve.csv")
    }
    assert len(rows) == 24
    for protocol in c55.TRANSFER_PROTOCOLS:
        assert float(rows[(protocol, "binary_sign")]["hit"]) == 0.9444444444444444
        assert rows[(protocol, "binary_sign")]["requires_test_endpoint_scalar"] == "1"
        assert rows[(protocol, "binary_sign")]["uses_same_cell_target_labels_for_template"] == "0"
        assert float(rows[(protocol, "continuous_raw")]["hit"]) == 0.9444444444444444
        assert float(rows[(protocol, "rank_only_within_cell")]["hit"]) == 0.9444444444444444
    assert float(rows[("leave_cell_out", "decile_bins")]["hit"]) == 0.9351851851851852
    assert float(rows[("leave_target_out", "decile_bins")]["hit"]) == 0.9296296296296296
    assert float(rows[("leave_cell_out", "train_median_threshold")]["hit"]) == 0.7825776866656639


def test_c55_cell_and_reason_ledgers_explain_template_gap():
    rows = _rows("transfer_cell_ledger.csv")
    assert len(rows) == 162
    assert sum(r["template_matches_same_cell"] == "1" for r in rows) == 135
    assert sum(r["requires_test_endpoint_scalar"] == "1" for r in rows) == 27
    assert sum(r["failure_reason_code"] == "test_endpoint_scalar_required" for r in rows) == 27
    assert sum(r["failure_reason_code"] == "template_matches_same_cell" for r in rows) == 135
    first = next(r for r in rows if r["trajectory_id"] == "0|1|0|S0_full_support")
    assert float(first["same_cell_scalar_hit"]) == 1.0
    assert float(first["best_template_hit"]) == 0.0
    assert float(first["best_endpoint_scalar_transfer_hit"]) == 1.0
    assert first["requires_test_endpoint_scalar"] == "1"
    reasons = {r["reason_code"]: r for r in _rows("transfer_failure_reason_ledger.csv")}
    assert reasons["template_matches_same_cell"]["cell_count"] == "135"
    assert reasons["test_endpoint_scalar_required"]["cell_count"] == "27"


def test_c55_nulls_are_below_endpoint_scalar_transfer():
    d = _summary()
    assert d["nulls"] == {
        "max_null_p95_hit": 0.7712962962962961,
        "max_null_p95_name": "N5_trajectory_block_shuffle",
        "observed_gt_all_null_p95": True,
    }
    rows = {r["null_name"]: r for r in _rows("transfer_null_summary.csv")}
    assert len(rows) == 6
    assert {int(r["num_repeats"]) for r in rows.values()} == {64}
    assert {r["observed_gt_null_p95"] for r in rows.values()} == {"1"}
    assert float(rows["N1_cell_preserving_label_shuffle"]["null_mean_hit"]) == 0.431794293843376
    assert float(rows["N2_field_identity_shuffle"]["null_p95_hit"]) == 0.7620855461500601
    assert float(rows["N5_trajectory_block_shuffle"]["null_p95_hit"]) == 0.7712962962962961
    assert float(rows["N6_scalar_value_permutation_within_cell"]["observed_minus_null_mean"]) == 0.5132619598765432


def test_c55_tables_have_expected_shapes_and_gates():
    d = _summary()
    assert d["table_row_counts"] == {
        "artifact_hygiene_gate": 13,
        "c54_replay_identity": 9,
        "cross_cell_protocol_summary": 12,
        "endpoint_template_availability_ledger": 13,
        "field_family_transfer_summary": 16,
        "leave_cell_out_transfer_summary": 2,
        "leave_target_out_transfer_summary": 2,
        "leave_trajectory_out_transfer_summary": 2,
        "red_team_verification": 8,
        "threshold_transfer_curve": 24,
        "transfer_cell_ledger": 162,
        "transfer_failure_reason_ledger": 2,
        "transfer_null_summary": 6,
    }
    gates = {r["check"]: r["passed"] for r in _rows("artifact_hygiene_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    red = {r["check"]: r["passed"] for r in _rows("red_team_verification.csv")}
    assert red and set(red.values()) == {"1"}
    assert "transfer_cell_ledger_rows" not in d
    assert os.path.getsize(REPORT_JSON) < 50_000


def test_c55_run_loads_committed_artifacts_without_recompute():
    res = c55.run()
    assert res["decision"]["primary"] == "C55-G_transfer_requires_unavailable_test_endpoint_scalar"
    assert res["transfer_boundary"]["best_template_only_hit"] == 0.7037037037037037
    assert res["transfer_boundary"]["best_endpoint_scalar_transfer_hit"] == 0.9444444444444444
    assert all(g["passed"] for g in c55.gate_rows(res))


def test_c55_outputs_do_not_emit_forbidden_claims_or_chosen_checkpoint_artifacts():
    for name in (
        "c54_replay_identity.csv",
        "endpoint_template_availability_ledger.csv",
        "cross_cell_protocol_summary.csv",
        "field_family_transfer_summary.csv",
        "threshold_transfer_curve.csv",
        "transfer_cell_ledger.csv",
        "transfer_failure_reason_ledger.csv",
        "transfer_null_summary.csv",
    ):
        text = open(os.path.join(TABLE_DIR, name)).read().lower()
        assert "model_hash" not in text
        assert "checkpoint_hash" not in text
        assert "selected_candidate_id" not in text
        assert "checkpoint recommendation" not in text
    forbidden = (
        "checkpoint selector",
        "deployable rule",
        "oaci rescue",
        "source-only target control",
        "few-label sufficiency",
        "calibration method",
        "held-out target method",
        "usable without target labels",
        "target labels can be used at deployment",
        "checkpoint recommendation",
        "production selection",
        "solves target selection",
    )
    for name in ("C55_CROSS_CELL_ENDPOINT_TEMPLATE_BOUNDARY.md",
                 "C55_CROSS_CELL_ENDPOINT_TEMPLATE_BOUNDARY.json",
                 "C55_RED_TEAM_VERIFICATION.md"):
        text = open(os.path.join("oaci/reports", name)).read().lower()
        for phrase in forbidden:
            assert phrase not in text
    md = open("oaci/reports/C55_CROSS_CELL_ENDPOINT_TEMPLATE_BOUNDARY.md").read()
    assert "C55-G_transfer_requires_unavailable_test_endpoint_scalar" in md
    assert "endpoint-scalar availability gap" in md
