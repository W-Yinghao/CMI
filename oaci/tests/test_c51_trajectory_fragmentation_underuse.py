"""C51 Trajectory Fragmentation / Source-Describability Audit tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.conditioned_ceiling_coverage import c51_trajectory_fragmentation_underuse as c51
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C51_TRAJECTORY_FRAGMENTATION_UNDERUSE.json"
TABLE_DIR = "oaci/reports/c51_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c51_config_and_locked_c50_witness_are_frozen():
    assert c51._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert c51.LOCKED_SCOPE == "within_target"
    assert c51.LOCKED_SOURCE_SPACE == "all_source_objectives"
    assert c51.LOCKED_NEIGHBORHOOD == "eps_q20"
    assert c51.LOCKED_MIN_N == 1
    assert c51.LOCKED_LABEL == "primary_joint_good"
    assert c51.EPS_QUANTILES == (0.10, 0.20, 0.30, 0.40)
    assert c51.MIN_N_GRID == (1, 2, 3, 5)
    assert c51.NULL_REPS == 64
    assert c51.NULL_SEED == 51051
    assert set(c51.NULL_NAMES) == {
        "N0_global_label_shuffle",
        "N1_within_target_label_shuffle",
        "N2_within_target_trajectory_label_shuffle",
        "N3_degree_preserving_neighbor_randomization",
        "N4_source_geometry_permutation_within_target",
    }


def test_c51_decision_is_target_trajectory_gauge_residual_with_caveat():
    d = _summary()
    assert d["milestone"] == "C51"
    assert d["inherits_from"] == ["C49", "C50"]
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    dec = d["decision"]
    assert dec["decision"] == "C51-E_target_trajectory_gauge_residual"
    assert dec["decision"] in c51.DECISIONS
    assert dec["support_material"] is False
    assert dec["null_like"] is False
    assert dec["residual_stronger_than_null"] is False
    assert dec["orientation_closes_gap"] is False
    assert dec["monotone_closes_gap"] is False
    assert dec["diagnostic_gauge_control_closes_gap"] is True
    assert dec["max_raw_underuse_gap"] == 0.40123456790123463
    assert dec["best_sign_flip_gap"] == 0.3827160493827161
    assert dec["best_monotone_gap"] == 0.33827260702260703
    assert dec["best_target_centered_gap"] == 0.24383834514553482
    assert dec["best_trajectory_centered_gap"] == -0.00411522633744843
    assert dec["n2_fail_fraction_percentile"] == 0.0
    assert dec["n3_fail_fraction_percentile"] == 0.0
    assert dec["n4_enrichment_null_mean"] == 0.8342243865476644


def test_c51_replays_c50_locked_witness_exactly():
    d = _summary()
    w = d["locked_witness"]
    assert w == {
        "condition": "within_target",
        "source_objectives": "all_source_objectives",
        "eps_quantile": "q20",
        "min_n": 1,
        "epsilon": 3.2532662364835945,
    }
    r = d["observed_c50_replay"]
    assert r["hit"] == 1.0
    assert r["coverage"] == 1.0
    assert r["enrichment"] == 2.3597078708023864
    assert r["target_min_hit"] == 1.0
    assert r["target_min_coverage"] == 1.0
    assert r["trajectory_min_hit"] == 0.0
    assert r["trajectory_min_coverage"] == 1.0
    assert r["trajectory_actionability_fail_fraction"] == 0.43209876543209874
    assert r["replayed_target_hit"] == 1.0
    assert r["replayed_trajectory_min_hit"] == 0.0


def test_c51_support_ablation_grid_rules_out_singleton_artifact():
    rows = _rows("support_ablation_grid.csv")
    assert len(rows) == 16
    assert {(r["eps_quantile"], int(r["min_n"])) for r in rows} == {
        (f"q{q:02d}", m) for q in (10, 20, 30, 40) for m in (1, 2, 3, 5)
    }
    q20_m1 = next(r for r in rows if r["eps_quantile"] == "q20" and r["min_n"] == "1")
    assert float(q20_m1["hit"]) == 1.0
    assert float(q20_m1["coverage"]) == 1.0
    assert float(q20_m1["trajectory_actionability_fail_fraction"]) == 0.43209876543209874
    assert float(q20_m1["p10_neighbor_count"]) == 5.0
    assert float(q20_m1["singleton_fraction"]) == 0.0002628811777076761
    q20_m3 = next(r for r in rows if r["eps_quantile"] == "q20" and r["min_n"] == "3")
    assert float(q20_m3["coverage"]) == 0.9220324781574136
    assert float(q20_m3["trajectory_actionability_fail_fraction"]) == 0.5
    assert float(q20_m3["trajectory_min_coverage"]) == 0.75


def test_c51_null_calibration_emits_n0_to_n4_and_source_geometry_signal():
    rows = _rows("null_calibration_summary.csv")
    assert len(rows) == 35
    assert {r["null_name"] for r in rows} == set(c51.NULL_NAMES)
    assert {r["statistic"] for r in rows} == set(c51.NULL_STATISTICS)
    assert {int(r["n_permutations"]) for r in rows} == {64}
    lookup = {(r["null_name"], r["statistic"]): r for r in rows}
    n2_fail = lookup[("N2_within_target_trajectory_label_shuffle",
                      "trajectory_actionability_fail_fraction")]
    assert float(n2_fail["observed"]) == 0.43209876543209874
    assert float(n2_fail["null_mean"]) == 0.8055555555555556
    assert float(n2_fail["percentile"]) == 0.0
    n4_enrich = lookup[("N4_source_geometry_permutation_within_target",
                        "covered_island_enrichment")]
    assert float(n4_enrich["observed"]) == 2.3597078708023864
    assert float(n4_enrich["null_mean"]) == 0.8342243865476644
    assert float(n4_enrich["percentile"]) == 1.0
    n3_under = lookup[("N3_degree_preserving_neighbor_randomization",
                       "max_mean_existing_score_underuse_gap")]
    assert float(n3_under["observed"]) == 0.4308510638297872
    assert float(n3_under["null_mean"]) == -0.006990176415989915


def test_c51_source_score_underuse_attribution_covers_available_scores():
    rows = {r["score_name"]: r for r in _rows("source_score_underuse_attribution.csv")}
    assert set(rows) == {
        "selection_leakage",
        "R_src",
        "C30_source_rank",
        "C19_robust_core",
        "C43_best_hindsight_scalarization",
    }
    assert rows["C43_best_hindsight_scalarization"]["hindsight_diagnostic_only"] == "1"
    assert rows["selection_leakage"]["primary_attribution"] == "score_trajectory_gauge_misaligned"
    assert float(rows["selection_leakage"]["raw_score_hit"]) == 0.4074074074074074
    assert float(rows["selection_leakage"]["mean_underuse_gap_against_c50_local_bayes"]) == 0.40123456790123463
    assert float(rows["selection_leakage"]["best_trajectory_centered_diagnostic_hit"]) == 0.7947530864197531
    assert float(rows["C30_source_rank"]["trajectory_centered_gap"]) == -0.00411522633744843
    assert set(r["target_labels_diagnostic_only"] for r in rows.values()) == {"1"}
    assert set(r["no_selection_artifact"] for r in rows.values()) == {"1"}


def test_c51_trajectory_failure_ledger_covers_all_trajectories():
    rows = _rows("trajectory_failure_ledger.csv")
    assert len(rows) == 162
    assert sum(int(r["actionability_fail"]) for r in rows) == 70
    assert max(float(r["singleton_fraction"]) for r in rows) == 0.038461538461538464
    assert sum(float(r["singleton_fraction"]) > 0 for r in rows) == 1
    first = next(r for r in rows if r["trajectory_id"] == "0|1|0|S0_full_support")
    assert float(first["coverage"]) == 1.0
    assert float(first["hit"]) == 0.0
    assert int(first["actionability_fail"]) == 1
    assert float(first["null_percentile_N1"]) == 0.578125
    assert float(first["null_percentile_N2"]) == 0.953125
    assert first["primary_failure_code"] == "LOW_TRAJECTORY_HIT"
    assert first["secondary_failure_code"] == "TRAJECTORY_FRAGMENTED"


def test_c51_tables_have_expected_shape_and_gates():
    d = _summary()
    assert d["table_row_counts"] == {
        "no_selector_artifact_gate": 9,
        "null_calibration_summary": 35,
        "red_team_verification": 8,
        "source_score_underuse_attribution": 5,
        "support_ablation_grid": 16,
        "trajectory_failure_ledger": 162,
    }
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    red = {r["check"]: r["passed"] for r in _rows("red_team_verification.csv")}
    assert red and set(red.values()) == {"1"}
    assert d["self_neighbor_excluded"] is True
    assert "support_ablation_rows" not in d
    assert "trajectory_failure_ledger_rows" not in d


def test_c51_report_run_loads_committed_artifacts_without_recompute():
    res = c51.run()
    assert res["decision"]["decision"] == "C51-E_target_trajectory_gauge_residual"
    assert res["n_candidate_rows"] == 3804
    assert all(g["passed"] for g in c51.no_selector_gate(res))


def test_c51_outputs_do_not_emit_selector_artifacts_or_forbidden_claims():
    for name in (
        "support_ablation_grid.csv",
        "null_calibration_summary.csv",
        "trajectory_failure_ledger.csv",
        "source_score_underuse_attribution.csv",
    ):
        text = open(os.path.join(TABLE_DIR, name)).read().lower()
        assert "model_hash" not in text
        assert "checkpoint_hash" not in text
        assert "selected_candidate_id" not in text
        assert "checkpoint recommendation" not in text
    forbidden = (
        "deployable selector",
        "oaci rescue",
        "source-only rescue",
        "target-unlabeled method",
        "target-grouped method",
        "checkpoint recommendation",
        "production rule",
        "action rule",
    )
    for name in ("C51_TRAJECTORY_FRAGMENTATION_UNDERUSE.md",
                 "C51_TRAJECTORY_FRAGMENTATION_UNDERUSE.json",
                 "C51_RED_TEAM_VERIFICATION.md"):
        text = open(os.path.join("oaci/reports", name)).read().lower()
        for phrase in forbidden:
            assert phrase not in text
    md = open("oaci/reports/C51_TRAJECTORY_FRAGMENTATION_UNDERUSE.md").read()
    assert "C51-E_target_trajectory_gauge_residual" in md
    assert "diagnostic source-describability boundary" in md
