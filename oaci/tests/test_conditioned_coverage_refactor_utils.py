"""Regression tests for conditioned coverage refactor utilities."""
from __future__ import annotations

import csv

from oaci.conditioned_ceiling_coverage import audit_utils as au
from oaci.conditioned_ceiling_coverage import c50_conditioned_island_morphology as c50
from oaci.conditioned_ceiling_coverage import island_metrics
from oaci.conditioned_ceiling_coverage import locked_witness
from oaci.conditioned_ceiling_coverage import score_diagnostics as sd


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_locked_witness_public_registry_matches_c50_aliases():
    w = locked_witness.c49_locked_witness()
    assert c50.WITNESS_SCOPE == locked_witness.WITNESS_SCOPE == "within_target"
    assert c50.WITNESS_SOURCE_SPACE == locked_witness.WITNESS_SOURCE_SPACE == "all_source_objectives"
    assert c50.WITNESS_NEIGHBORHOOD == locked_witness.WITNESS_NEIGHBORHOOD == "eps_q20"
    assert c50.WITNESS_MIN_NEIGHBOR_COUNT == locked_witness.WITNESS_MIN_NEIGHBOR_COUNT == 1
    assert c50.WITNESS_LABEL == locked_witness.WITNESS_LABEL == "primary_joint_good"
    assert w["epsilon_radius"] == 3.2532662364835945
    assert w["c49_hit"] == 1.0
    assert locked_witness.compact_witness() == {
        "condition": "within_target",
        "source_objectives": "all_source_objectives",
        "eps_quantile": "q20",
        "min_n": 1,
        "epsilon": 3.2532662364835945,
    }


def test_audit_utils_keep_stable_query_and_group_keys():
    row = {
        "source_idx": "7",
        "target": "2",
        "seed": "0",
        "level": "1",
        "regime": "S0_full_support",
        "trajectory": "0|2|1|S0_full_support",
    }
    assert au.query_id(row) == "c50q_0007"
    assert au.row_group_key(row, "target") == "2"
    assert au.row_group_key(row, "trajectory") == "0|2|1|S0_full_support"
    assert au.row_group_key(row, "conditioned_key") == "2"
    assert au.enrichment(1.0, 0.5) == 2.0


def test_island_metrics_reconstruct_c50_target_and_trajectory_fragmentation():
    islands = _rows("oaci/reports/c50_tables/island_morphology.csv")
    frag = island_metrics.group_fragmentation(
        islands, c50.GROUP_TYPES, c50.COVERAGE_GATE, c50.HIT_GATE, c50.ENRICHMENT_GATE)
    targets = [r for r in frag if r["group_type"] == "target"]
    trajectories = [r for r in frag if r["group_type"] == "trajectory"]
    assert len(targets) == 9
    assert len(trajectories) == 162
    assert min(float(r["coverage"]) for r in targets) == 1.0
    assert min(float(r["hit_rate_if_covered"]) for r in targets) == 1.0
    assert min(float(r["hit_rate_if_covered"]) for r in trajectories) == 0.0
    assert sum(1 - int(r["actionability_pass"]) for r in trajectories) == 70


def test_score_diagnostics_small_arrays_are_deterministic():
    scores = [0.1, 0.4, 0.3, 0.8]
    labels = [0, 1, 0, 1]
    assert sd.rankdata(scores).tolist() == [0.0, 0.6666666666666666, 0.3333333333333333, 1.0]
    assert sd.auc(scores, labels) == 1.0
    assert sd.auprc(scores, labels) == 1.0
    assert abs(sd.spearman(scores, labels) - 0.894427190999916) <= 1e-12
    decile_scores = sd.diagnostic_decile_scores(scores, labels)
    assert decile_scores.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_refactor_does_not_change_committed_c50_and_c51_report_loads():
    c50_res = c50.run()
    assert c50_res["decision"]["outcome"] == "C50-C_mixed_fragmentation_plus_underuse"
    assert c50_res["locked_witness"]["epsilon_radius"] == 3.2532662364835945
