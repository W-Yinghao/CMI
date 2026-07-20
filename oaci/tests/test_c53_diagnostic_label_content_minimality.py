"""C53 Diagnostic-Label Content Minimality / Split-Label Boundary Audit tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.conditioned_ceiling_coverage import c53_diagnostic_label_content_minimality as c53
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C53_DIAGNOSTIC_LABEL_CONTENT_MINIMALITY.json"
TABLE_DIR = "oaci/reports/c53_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c53_config_taxonomy_and_guards_are_frozen():
    assert c53._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert c53.MILESTONE == "C53"
    assert c53.NULL_REPS == 64
    assert c53.NULL_SEED == 53053
    assert c53.GAP_CLOSE_GATE == 0.05
    assert c53.WEAK_CLOSE_GATE == 0.50
    assert c53.STRONG_CLOSE_GATE == 0.80
    assert c53.NEAR_FULL_GAP == 0.02
    assert c53.LABEL == "primary_joint_good"
    assert c53.BEST_SCALAR_FIELD == "target_joint_margin_raw"
    assert set(c53.DECISIONS) == {
        "C53-A_cell_prior_label_sufficiency",
        "C53-B_scalar_endpoint_label_sufficiency",
        "C53-C_class_conditioned_label_content_required",
        "C53-D_pairwise_or_rank_label_content_required",
        "C53-E_near_full_diagnostic_label_content_required",
        "C53-F_nontransferable_cell_local_label_content",
        "C53-G_split_label_budget_sufficiency",
        "C53-H_same_label_diagnostic_only",
        "C53-I_null_like_or_artifact",
    }


def test_c53_decision_is_scalar_endpoint_sufficiency_with_same_label_guard():
    d = _summary()
    assert d["milestone"] == "C53"
    assert d["inherits_from"] == ["C49", "C50", "C51", "C52"]
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    dec = d["decision"]
    assert dec["decision"] == "C53-B_scalar_endpoint_label_sufficiency"
    assert dec["decision"] in c53.DECISIONS
    assert dec["secondary_tags"] == [
        "C53-H_same_label_diagnostic_only",
        "C53-F_nontransferable_cell_local_label_content",
    ]
    assert dec["cell_prior_strong_close"] is False
    assert dec["scalar_endpoint_strong_close"] is True
    assert dec["scalar_endpoint_near_or_exceeds_l7"] is True
    assert dec["pairwise_rank_strong_close"] is True
    assert dec["split_label_budget_available"] is False
    assert dec["same_label_diagnostic_only"] is True
    assert dec["null_like_or_artifact"] is False
    assert dec["best_transfer_hit"] == 0.7037037037037037
    assert dec["best_transfer_closed_fraction"] == 0.6645569620253167


def test_c53_replays_c52_baselines_exactly():
    replay = _summary()["c52_replay"]
    assert replay["trajectory_random_tie_hit"] == 0.429723378036041
    assert replay["best_strict_source_hit"] == 0.5061728395061729
    assert replay["best_key_only_hit"] == 0.4876543209876543
    assert replay["c51_trajectory_local_bayes_oracle_hit"] == 0.808641975308642
    assert replay["trajectory_centered_diagnostic_hit"] == 0.8127572016460904
    assert replay["key_only_closes_cell_count"] == 12
    assert replay["label_diagnostic_closes_cell_count"] == 131
    assert replay["cell_count"] == 162


def test_c53_label_content_ladder_identifies_scalar_endpoint_content():
    rows = {r["level"]: r for r in _rows("label_content_ladder_summary.csv")}
    assert len(rows) == 8
    assert rows["L0_random_tie"]["diagnostic_label_content"] == "0"
    assert float(rows["L0_random_tie"]["hit"]) == 0.429723378036041
    assert float(rows["L1_strict_source"]["hit"]) == 0.5061728395061729
    assert rows["L2_key_only"]["key_only"] == "1"
    assert rows["L2_key_only"]["diagnostic_label_content"] == "0"
    assert float(rows["L2_key_only"]["hit"]) == 0.4876543209876543
    assert rows["L3_cell_prior_label_content"]["diagnostic_label_content"] == "1"
    assert rows["L3_cell_prior_label_content"]["candidate_specific_label_content"] == "0"
    assert rows["L3_cell_prior_label_content"]["strong_close"] == "0"
    scalar = rows["L4_scalar_endpoint_label_content"]
    assert scalar["comparison_source"] == "target_joint_margin_raw:high"
    assert scalar["diagnostic_label_content"] == "1"
    assert scalar["same_label_diagnostic"] == "1"
    assert scalar["split_label_evaluated"] == "0"
    assert scalar["candidate_specific_label_content"] == "1"
    assert scalar["strong_close"] == "1"
    assert scalar["near_or_exceeds_l7"] == "1"
    assert float(scalar["hit"]) == 0.9444444444444444
    assert float(scalar["closed_fraction_from_key_only"]) == 1.4050632911392409
    assert rows["L5_class_conditioned_label_content"]["available"] == "0"
    assert rows["L6_pairwise_or_rank_label_content"]["strong_close"] == "1"
    assert rows["L7_full_trajectory_centered_diagnostic"]["hit"] == "0.8127572016460904"


def test_c53_best_scalar_registry_and_nulls_are_reported():
    d = _summary()
    scalar = d["best_scalar_endpoint"]
    assert scalar == {
        "field": "target_joint_margin_raw",
        "hit": 0.9444444444444444,
        "mean_top_tie_count": 1.1111111111111112,
        "orientation": "high",
    }
    fields = {r["field"]: r for r in d["scalar_endpoint_field_results"]}
    assert float(fields["target_bacc_delta"]["hit"]) == 0.9259259259259259
    assert float(fields["target_nll_delta"]["hit"]) == 0.8333333333333334
    assert float(fields["target_ece_delta"]["hit"]) == 0.7037037037037037

    rows = {r["null_name"]: r for r in _rows("label_null_calibration_summary.csv")}
    assert len(rows) == 7
    assert float(rows["N0_random_tie_within_trajectory"]["null_mean_hit"]) == 0.4331597222222222
    assert float(rows["N0_random_tie_within_trajectory"]["null_p95_hit"]) == 0.4876543209876543
    assert float(rows["N1_key_preserving_label_shuffle"]["null_mean_hit"]) == 0.4371624228395061
    assert float(rows["N1_key_preserving_label_shuffle"]["null_p95_hit"]) == 0.48077160493827165
    assert rows["N2_cell_prior_preserving_shuffle"]["status"] == "analytical_cell_prior"
    assert rows["N3_class_marginal_preserving_shuffle"]["status"] == "unavailable"
    assert rows["N3_class_marginal_preserving_shuffle"]["unavailable_reason"] == (
        "candidate-level per-class target prediction/label cache unavailable"
    )
    assert float(rows["N4_rank_permutation_within_cell"]["null_mean_hit"]) == 0.43178047839506173
    assert rows["N5_cross_cell_label_transfer"]["status"] == "reported_in_transferability_tables"
    assert float(rows["N6_source_geometry_preserving_label_null"]["observed_hit"]) == 0.4876543209876543
    assert float(rows["N6_source_geometry_preserving_label_null"]["null_mean_hit"]) == 0.4263117283950617
    assert int(rows["N6_source_geometry_preserving_label_null"]["n_permutations"]) == 64


def test_c53_transferability_is_partial_not_full_closure():
    rows = {r["transfer_test"]: r for r in _rows("label_transferability_summary.csv")}
    assert len(rows) == 5
    assert float(rows["T1_leave_one_target_out_label_summary_transfer"]["transfer_hit"]) == 0.5740740740740741
    assert float(rows["T2_leave_one_trajectory_out_label_summary_transfer"]["transfer_hit"]) == 0.5740740740740741
    assert float(rows["T3_same_target_cross_trajectory_transfer"]["transfer_hit"]) == 0.6481481481481481
    assert float(rows["T4_same_seed_level_regime_cross_target_transfer"]["transfer_hit"]) == 0.4444444444444444
    best = rows["T5_support_matched_cross_cell_transfer"]
    assert float(best["transfer_hit"]) == 0.7037037037037037
    assert float(best["transfer_closed_fraction"]) == 0.6645569620253167
    assert float(best["transfer_gap_to_local_scalar"]) == 0.2407407407407407
    assert int(best["cells_improved"]) == 102
    assert int(best["cells_degraded"]) == 30

    ledger = _rows("label_transferability_ledger.csv")
    assert len(ledger) == 810
    assert {r["target_labels_diagnostic_only"] for r in ledger} == {"1"}
    assert {r["no_selection_artifact"] for r in ledger} == {"1"}


def test_c53_cell_ledger_and_reason_codes_cover_all_cells():
    cells = _rows("cell_label_content_ledger.csv")
    assert len(cells) == 162
    assert sum(float(r["scalar_endpoint_hit"]) >= 0.7 for r in cells) == 153
    assert sum(float(r["full_diag_hit"]) >= 0.7 for r in cells) == 131
    assert sum(float(r["cell_prior_hit"]) >= 0.7 for r in cells) == 39
    assert sum(int(r["key_only_closes"]) for r in cells) == 12
    assert sum(int(r["label_diag_closes"]) for r in cells) == 131
    assert sum(int(r["source_underuse_cell"]) for r in cells) == 84
    assert sum(int(r["low_local_bayes_cell"]) for r in cells) == 31
    assert sum(float(r["scalar_endpoint_hit"]) >= 0.7 and float(r["transfer_best_hit"]) < 0.7
               for r in cells) == 12
    reasons = {r["final_reason_code"]: sum(x["final_reason_code"] == r["final_reason_code"] for x in cells)
               for r in cells}
    assert reasons == {
        "cell_prior_sufficient": 39,
        "null_like_or_unstable": 9,
        "scalar_endpoint_sufficient": 114,
    }
    first = next(r for r in cells if r["trajectory_id"] == "0|1|0|S0_full_support")
    assert float(first["random_tie_hit"]) == 0.05
    assert float(first["scalar_endpoint_hit"]) == 1.0
    assert float(first["full_diag_hit"]) == 0.0
    assert first["final_reason_code"] == "scalar_endpoint_sufficient"

    ledger = {r["reason_code"]: r for r in _rows("failure_reason_ledger.csv")}
    assert int(ledger["key_only_sufficient"]["n_cells"]) == 12
    assert int(ledger["cell_prior_sufficient"]["n_cells"]) == 39
    assert int(ledger["scalar_endpoint_sufficient"]["n_cells"]) == 114
    assert int(ledger["nontransferable_cell_local_content"]["n_cells"]) == 12
    assert int(ledger["null_like_or_unstable"]["n_cells"]) == 9
    assert int(ledger["insufficient_artifact_for_split_label"]["n_cells"]) == 162


def test_c53_split_label_budget_is_explicitly_unavailable():
    curve = _rows("split_label_budget_curve.csv")
    assert len(curve) == 1
    assert curve[0]["budget"] == "unavailable"
    assert curve[0]["available"] == "0"
    assert curve[0]["reason"] == "required per-trial target prediction/label cache unavailable"
    ledger = _rows("split_label_budget_ledger.csv")
    assert len(ledger) == 1
    assert ledger[0]["construction_eval_disjoint"] == "0"
    assert ledger[0]["available"] == "0"


def test_c53_tables_have_expected_shape_and_gates():
    d = _summary()
    assert d["table_row_counts"] == {
        "cell_label_content_ledger": 162,
        "failure_reason_ledger": 10,
        "label_content_ladder_summary": 8,
        "label_null_calibration_summary": 7,
        "label_transferability_ledger": 810,
        "label_transferability_summary": 5,
        "no_selector_artifact_gate": 10,
        "red_team_verification": 8,
        "split_label_budget_curve": 1,
        "split_label_budget_ledger": 1,
    }
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    red = {r["check"]: r["passed"] for r in _rows("red_team_verification.csv")}
    assert red and set(red.values()) == {"1"}
    assert "cell_label_content_ledger_rows" not in d
    assert os.path.getsize(REPORT_JSON) < 50_000


def test_c53_run_loads_committed_artifacts_without_recompute():
    res = c53.run()
    assert res["decision"]["decision"] == "C53-B_scalar_endpoint_label_sufficiency"
    assert res["split_label_budget_available"] is False
    assert len(res["cell_label_content_ledger_rows"]) == 162
    assert all(g["passed"] for g in c53.no_selector_gate(res))


def test_c53_outputs_do_not_emit_selector_artifacts_or_forbidden_claims():
    for name in (
        "label_content_ladder_summary.csv",
        "label_null_calibration_summary.csv",
        "cell_label_content_ledger.csv",
        "label_transferability_summary.csv",
        "label_transferability_ledger.csv",
        "split_label_budget_curve.csv",
        "split_label_budget_ledger.csv",
        "failure_reason_ledger.csv",
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
    )
    for name in ("C53_DIAGNOSTIC_LABEL_CONTENT_MINIMALITY.md",
                 "C53_DIAGNOSTIC_LABEL_CONTENT_MINIMALITY.json",
                 "C53_RED_TEAM_VERIFICATION.md"):
        text = open(os.path.join("oaci/reports", name)).read().lower()
        for phrase in forbidden:
            assert phrase not in text
    md = open("oaci/reports/C53_DIAGNOSTIC_LABEL_CONTENT_MINIMALITY.md").read()
    assert "C53-B_scalar_endpoint_label_sufficiency" in md
    assert "same-label diagnostic evidence" in md
