"""C50 Conditioned-Island Morphology / Fragmentation Audit tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.conditioned_ceiling_coverage import c50_conditioned_island_morphology as c50
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C50_CONDITIONED_ISLAND_MORPHOLOGY.json"
TABLE_DIR = "oaci/reports/c50_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c50_locks_c49_broad_witness_not_grid_search():
    assert c50._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert c50.WITNESS_SCOPE == "within_target"
    assert c50.WITNESS_SOURCE_SPACE == "all_source_objectives"
    assert c50.WITNESS_NEIGHBORHOOD == "eps_q20"
    assert c50.WITNESS_MIN_NEIGHBOR_COUNT == 1
    assert c50.WITNESS_LABEL == "primary_joint_good"
    assert c50.COVERAGE_GATE == 0.50
    assert c50.HIT_GATE == 0.70
    assert c50.ENRICHMENT_GATE == 1.50
    assert c50.PERMUTATION_REPS == 64
    assert c50.PERMUTATION_SEED == 50050


def test_c50_decision_is_mixed_fragmentation_plus_underuse():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    dec = d["decision"]
    assert dec["outcome"] == "C50-C_mixed_fragmentation_plus_underuse"
    assert dec["fragmentation_material"] is True
    assert dec["underuse_material"] is True
    assert dec["target_min_hit"] == 1.0
    assert dec["target_min_coverage"] == 1.0
    assert dec["trajectory_min_hit"] == 0.0
    assert dec["trajectory_min_coverage"] == 1.0
    assert dec["target_actionability_fail_fraction"] == 0.0
    assert dec["trajectory_actionability_fail_fraction"] == 0.43209876543209874
    assert dec["max_mean_underuse_gap"] == 0.4308510638297872


def test_c50_locked_witness_reconstructs_c49_broad_row():
    w = _summary()["locked_witness"]
    assert w["condition_scope"] == "within_target"
    assert w["source_space"] == "all_source_objectives"
    assert w["neighborhood"] == "eps_q20"
    assert w["neighborhood_kind"] == "epsilon"
    assert w["epsilon_radius"] == 3.2532662364835945
    assert w["min_neighbor_count"] == 1
    assert w["label"] == "primary_joint_good"
    assert w["c49_hit"] == 1.0
    assert w["c49_coverage"] == 1.0
    assert w["c49_enrichment"] == 2.3597078708023864
    assert w["c49_mean_neighbor_count"] == 53.25221499477471
    assert w["c49_covered_base_rate"] == 0.42378127071295635
    assert w["inherited_from_c49_commit"] == "b0d7831"


def test_c50_group_fragmentation_explains_broad_coverage_not_actionability():
    rows = _rows("group_fragmentation.csv")
    targets = [r for r in rows if r["group_type"] == "target"]
    trajectories = [r for r in rows if r["group_type"] == "trajectory"]
    assert len(targets) == 9
    assert len(trajectories) == 162
    assert min(float(r["coverage"]) for r in targets) == 1.0
    assert min(float(r["hit_rate_if_covered"]) for r in targets) == 1.0
    assert sum(int(r["actionability_pass"]) for r in targets) == 9
    assert min(float(r["coverage"]) for r in trajectories) == 1.0
    assert min(float(r["hit_rate_if_covered"]) for r in trajectories) == 0.0
    assert sum(1 - int(r["actionability_pass"]) for r in trajectories) == 70
    traj0 = next(r for r in trajectories if r["group_key"] == "0|1|0|S0_full_support")
    assert float(traj0["hit_rate_if_covered"]) == 0.0
    assert float(traj0["max_neighbor_positive_rate"]) == 0.8983050847457628


def test_c50_existing_score_underuse_is_diagnostic_and_score_specific():
    rows = {r["score_name"]: r for r in _rows("existing_score_underuse_summary.csv")}
    assert set(rows) == {"selection_leakage", "R_src", "C30_source_rank", "C19_robust_core"}
    assert float(rows["selection_leakage"]["mean_oracle_hit_or_ceiling"]) == 0.8351063829787234
    assert float(rows["selection_leakage"]["mean_score_hit_within_covered_set"]) == 0.40425531914893614
    assert float(rows["selection_leakage"]["mean_underuse_gap"]) == 0.4308510638297872
    assert float(rows["selection_leakage"]["max_underuse_gap"]) == 1.0
    assert float(rows["R_src"]["mean_underuse_gap"]) == 0.3617021276595745
    assert float(rows["C30_source_rank"]["mean_underuse_gap"]) == 0.34574468085106386
    assert float(rows["C19_robust_core"]["mean_underuse_gap"]) == 0.34574468085106386
    assert set(r["diagnostic_underuse_only"] for r in rows.values()) == {"1"}


def test_c50_reason_codes_and_baselines_are_emitted():
    reason = {r["reason_code"]: int(r["n_groups"]) for r in _rows("reason_code_counts.csv")}
    assert reason == {
        "COVERED_BUT_LOW_HIT": 31,
        "DIAGNOSTIC_ONLY_TARGET_CONDITIONING": 70,
        "GROUP_BASE_RATE_DOMINATES": 39,
        "SCORE_UNDERUSES_AVAILABLE_POSITIVES": 3,
        "TRAJECTORY_FRAGMENTED": 70,
    }
    baselines = _rows("baseline_sanity.csv")
    assert len(baselines) == 18
    local_target = next(
        r for r in baselines
        if r["baseline_name"] == "locked_local_bayes_ceiling" and r["group_type"] == "target")
    assert float(local_target["hit"]) == 1.0
    assert float(local_target["coverage"]) == 1.0
    assert float(local_target["permutation_adjusted_gap"]) == 0.658445295540911
    assert float(local_target["permutation_hit"]) == 0.341554704459089
    assert local_target["permutation_reps"] == "64"


def test_c50_tables_have_expected_shape_and_gates():
    d = _summary()
    assert d["table_row_counts"] == {
        "actionability_failure_ledger": 70,
        "baseline_sanity": 18,
        "existing_score_underuse": 752,
        "existing_score_underuse_summary": 4,
        "group_fragmentation": 188,
        "island_morphology": 3804,
        "locked_witness": 1,
        "reason_code_counts": 5,
    }
    assert len(_rows("locked_witness.csv")) == 1
    assert len(_rows("island_morphology.csv")) == 3804
    assert len(_rows("group_fragmentation.csv")) == 188
    assert len(_rows("existing_score_underuse_by_group.csv")) == 752
    assert len(_rows("actionability_failure_ledger.csv")) == 70
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    red = {r["check"]: r["passed"] for r in _rows("red_team_verification.csv")}
    assert red and set(red.values()) == {"1"}


def test_c50_outputs_do_not_emit_selector_or_checkpoint_artifacts():
    for name in (
        "island_morphology.csv", "group_fragmentation.csv", "existing_score_underuse_by_group.csv",
        "actionability_failure_ledger.csv", "baseline_sanity.csv",
    ):
        text = open(os.path.join(TABLE_DIR, name)).read()
        assert "model_hash" not in text
        assert "checkpoint_hash" not in text
        assert "selected_candidate_id" not in text
    island_header = open(os.path.join(TABLE_DIR, "island_morphology.csv")).readline()
    assert "query_id" in island_header
    assert "no_checkpoint_recommendation" in island_header


def test_c50_report_run_loads_committed_artifacts_without_recompute():
    res = c50.run()
    assert res["decision"]["outcome"] == "C50-C_mixed_fragmentation_plus_underuse"
    assert res["n_candidate_rows"] == 3804
    assert all(g["passed"] for g in c50.no_selector_gate(res))


def test_c50_reports_do_not_overclaim():
    forbidden = (
        "deployable selector",
        "actionable selector",
        "target-free detector",
        "oaci rescue",
        "external validation success",
        "target-unlabeled dg success",
        "target-grouped oracle as method",
        "source-only control is restored",
        "target-conditioned local bayes estimates are deployable",
    )
    for name in ("C50_CONDITIONED_ISLAND_MORPHOLOGY.md",
                 "C50_CONDITIONED_ISLAND_MORPHOLOGY.json",
                 "C50_RED_TEAM_VERIFICATION.md"):
        text = open(os.path.join("oaci/reports", name)).read().lower()
        for phrase in forbidden:
            assert phrase not in text
    md = open("oaci/reports/C50_CONDITIONED_ISLAND_MORPHOLOGY.md").read()
    assert "C50-C_mixed_fragmentation_plus_underuse" in md
    assert "diagnostic-only" in md
