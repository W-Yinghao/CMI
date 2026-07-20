"""C49 Sparse Local-Bayes Ceiling / Coverage-Actionability tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.conditioned_ceiling_coverage import report, schema


REPORT_JSON = "oaci/reports/C49_SPARSE_LOCAL_BAYES_COVERAGE_AUDIT.json"
TABLE_DIR = "oaci/reports/c49_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c49_config_grid_and_coverage_gates_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.GROUP_SCOPES == (
        "global", "within_target", "within_trajectory",
        "within_target_seed", "within_target_level", "within_regime")
    assert schema.K_VALUES == (1, 3, 5, 10, 20)
    assert schema.EPSILON_QUANTILES == (0.01, 0.02, 0.05, 0.10, 0.20)
    assert schema.MIN_NEIGHBOR_COUNTS == (1, 2, 3, 5)
    assert schema.COVERAGE_THRESHOLDS == (0.25, 0.50, 0.75)
    assert schema.STABILITY_GROUPINGS == ("target", "seed", "level", "trajectory", "regime")
    assert schema.RELIABLE_TOP1_HIT_GATE == 0.70
    assert schema.RELIABLE_ENRICHMENT_GATE == 1.50
    assert [s[0] for s in schema.SOURCE_SPACES] == [
        "all_source_objectives", "rank_only", "leakage_only",
        "risk_only", "rank_risk", "leakage_rank"]


def test_c49_taxonomy_is_broad_but_unstable_diagnostic_ceiling():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    cases = set(d["taxonomy"]["cases"])
    assert cases == {schema.SC2, schema.SC3, schema.SC5, schema.SC8}
    for inactive in (schema.SC1, schema.SC4, schema.SC6, schema.SC7, schema.SC9, schema.SC10):
        assert inactive not in cases


def test_c49_best_ceiling_row_remains_sparse_after_coverage_audit():
    d = _summary()
    m = d["taxonomy"]["primary_metrics"]
    assert m["best_conditioned_scope"] == "within_target"
    assert m["best_conditioned_source_space"] == "all_source_objectives"
    assert m["best_conditioned_neighborhood"] == "eps_q01"
    assert m["best_conditioned_min_neighbor_count"] == 5
    assert m["best_conditioned_hit"] == 1.0
    assert m["best_conditioned_enrichment"] == 2.656126482213439
    assert m["best_conditioned_coverage"] == 0.022697831804226423
    assert m["best_conditioned_mean_neighbor_count"] == 0.47852850669011243
    best = d["best_conditioned_setup"]
    assert best["mean_empty_fraction"] == 0.8579774309313367
    assert best["mean_c47_actual_strict_source_top1_hit"] == 0.4444444444444444
    assert best["target_labels_diagnostic_only"] == 1


def test_c49_broad_coverage_witness_is_fixed_and_label_diagnostic_only():
    d = _summary()
    m = d["taxonomy"]["primary_metrics"]
    assert m["coverage50_reliable"] is True
    assert m["coverage75_reliable"] is True
    assert m["coverage50_best_scope"] == "within_target"
    assert m["coverage50_best_source_space"] == "all_source_objectives"
    assert m["coverage50_best_neighborhood"] == "eps_q20"
    assert m["coverage50_best_min_neighbor_count"] == 1
    assert m["coverage50_best_hit"] == 1.0
    assert m["coverage50_best_enrichment"] == 2.3597078708023864
    assert m["coverage50_best_coverage"] == 1.0
    assert m["coverage75_best_scope"] == m["coverage50_best_scope"]
    assert m["coverage75_best_source_space"] == m["coverage50_best_source_space"]
    assert m["coverage75_best_neighborhood"] == m["coverage50_best_neighborhood"]
    assert m["coverage75_best_min_neighbor_count"] == m["coverage50_best_min_neighbor_count"]
    assert m["coverage75_best_hit"] == m["coverage50_best_hit"]
    assert m["coverage75_best_coverage"] == m["coverage50_best_coverage"]


def test_c49_stability_and_existing_score_underuse_numbers():
    d = _summary()
    m = d["taxonomy"]["primary_metrics"]
    assert m["target_stability_min_hit"] == 1.0
    assert m["target_stability_min_coverage"] == 0.0
    assert m["trajectory_stability_min_hit"] == 0.0
    assert m["trajectory_stability_min_coverage"] == 0.0
    assert m["max_underuse_score"] == "C19_robust_core"
    assert m["max_underuse_gap"] == 1.0
    assert m["reliable_min_neighbor_ge_2_count"] == 270
    under = {r["score"]: r for r in _rows("existing_score_underuse_summary.csv")}
    assert float(under["C19_robust_core"]["mean_local_bayes_top1_hit"]) == 1.0
    assert float(under["C19_robust_core"]["mean_source_score_top_hit"]) == 0.0
    assert float(under["C19_robust_core"]["mean_underuse_gap"]) == 1.0
    stab = {r["stability_grouping"]: r for r in _rows("stability_summary.csv")}
    assert float(stab["target"]["min_hit"]) == 1.0
    assert float(stab["target"]["min_coverage"]) == 0.0
    assert float(stab["trajectory"]["min_hit"]) == 0.0
    assert float(stab["trajectory"]["min_coverage"]) == 0.0


def test_c49_inherits_c48_sparse_ceiling_reference():
    d = _summary()
    ref = d["c48_reference"]
    assert ref["cases"] == [
        "LC1_conditioned_source_space_ceiling_high",
        "LC3_existing_scores_underuse_source_space",
    ]
    assert ref["best_conditioned_top1_hit"] == 1.0
    assert ref["best_conditioned_enrichment"] == 2.3597078708023864
    assert ref["best_conditioned_gap_vs_permutation"] == 0.5743454359971853
    assert float(ref["mean_local_purity"]) == 0.42537373669020523
    assert float(ref["mean_neighbor_count"]) == 1.2747740581303715
    assert float(ref["mean_empty_neighborhood_fraction"]) == 0.6549140933100052


def test_c49_tables_have_expected_shape_and_gates():
    assert len(_rows("source_space_registry.csv")) == 6
    assert len(_rows("epsilon_radius_registry.csv")) == 30
    assert len(_rows("coverage_accuracy_curve.csv")) == 4320
    assert len(_rows("coverage_best_by_scope.csv")) == 18
    assert len(_rows("reliability_under_coverage.csv")) == 12960
    assert len(_rows("stability_rows.csv")) == 179
    assert len(_rows("stability_summary.csv")) == 5
    assert len(_rows("source_space_island_audit.csv")) == 3
    assert len(_rows("existing_score_underuse_rows.csv")) == 12
    assert len(_rows("existing_score_underuse_summary.csv")) == 4
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    red = {r["check"]: r["passed"] for r in _rows("red_team_verification.csv")}
    assert red and set(red.values()) == {"1"}
    tax = {r["case"]: r["established"] for r in _rows("c49_case_taxonomy.csv")}
    assert tax[schema.SC2] == "1"
    assert tax[schema.SC3] == "1"
    assert tax[schema.SC5] == "1"
    assert tax[schema.SC8] == "1"
    assert tax[schema.SC1] == "0"
    assert tax[schema.SC6] == "0"
    assert tax[schema.SC9] == "0"


def test_c49_outputs_do_not_emit_method_artifacts():
    for name in (
        "coverage_accuracy_curve.csv", "coverage_best_by_scope.csv",
        "reliability_under_coverage.csv", "source_space_island_audit.csv",
        "existing_score_underuse_rows.csv",
    ):
        text = open(os.path.join(TABLE_DIR, name)).read()
        assert "model_hash" not in text
        assert "checkpoint_hash" not in text
        assert "selected_candidate_id" not in text
    assert "no_candidate_id_emitted" in open(
        os.path.join(TABLE_DIR, "source_space_island_audit.csv")).read()
    assert "no_candidate_id_emitted" in open(
        os.path.join(TABLE_DIR, "existing_score_underuse_rows.csv")).read()


def test_c49_report_run_loads_committed_artifacts_without_recompute():
    res = report.run()
    assert res["taxonomy"]["cases"] == [schema.SC2, schema.SC3, schema.SC5, schema.SC8]
    assert res["n_candidate_rows"] == 3804
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c49_reports_do_not_overclaim():
    forbidden = (
        "conditioned local bayes selector",
        "deployable selector",
        "target-free detector",
        "oaci rescue",
        "external validation success",
        "target-unlabeled dg success",
        "target-grouped oracle as method",
    )
    for name in ("C49_SPARSE_LOCAL_BAYES_COVERAGE_AUDIT.md",
                 "C49_SPARSE_LOCAL_BAYES_COVERAGE_AUDIT.json",
                 "C49_RED_TEAM_VERIFICATION.md"):
        text = open(os.path.join("oaci/reports", name)).read().lower()
        for phrase in forbidden:
            assert phrase not in text
    md = open("oaci/reports/C49_SPARSE_LOCAL_BAYES_COVERAGE_AUDIT.md").read()
    assert "same-group random baselines" in md
    assert "diagnostic-only" in md
