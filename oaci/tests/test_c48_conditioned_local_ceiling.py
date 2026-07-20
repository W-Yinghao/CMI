"""C48 Conditioned Source-Space Ceiling / Local Bayes tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.conditioned_local_ceiling import report, schema


REPORT_JSON = "oaci/reports/C48_CONDITIONED_LOCAL_BAYES_CEILING.json"
TABLE_DIR = "oaci/reports/c48_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c48_config_spaces_neighborhoods_and_permutation_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.GROUP_SCOPES == (
        "global", "within_target", "within_trajectory",
        "within_target_seed", "within_target_level", "within_regime")
    assert schema.K_VALUES == (3, 5, 10)
    assert schema.EPSILON_QUANTILES == (0.01, 0.02, 0.05, 0.10)
    assert schema.PERMUTATION_REPS == 64
    assert schema.PERMUTATION_SEED == 48048
    assert [s[0] for s in schema.SOURCE_SPACES] == [
        "all_source_objectives", "rank_only", "leakage_only",
        "risk_only", "rank_risk", "leakage_rank"]


def test_c48_taxonomy_finds_high_sparse_local_ceiling_and_score_underuse():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    cases = set(d["taxonomy"]["cases"])
    assert cases == {schema.LC1, schema.LC3}
    for inactive in (schema.LC2, schema.LC4, schema.LC5, schema.LC6, schema.LC7):
        assert inactive not in cases


def test_c48_primary_ceiling_numbers_are_permutation_adjusted():
    d = _summary()
    m = d["taxonomy"]["primary_metrics"]
    assert m["best_conditioned_scope"] == "within_target"
    assert m["best_conditioned_source_space"] == "all_source_objectives"
    assert m["best_conditioned_neighborhood"] == "eps_q02"
    assert m["best_conditioned_top1_hit"] == 1.0
    assert m["best_conditioned_enrichment"] == 2.3597078708023864
    assert m["best_conditioned_permutation_top1_hit"] == 0.42565456400281465
    assert m["best_conditioned_gap_vs_permutation"] == 0.5743454359971853
    assert m["best_conditioned_gap_vs_c47"] == 0.5555555555555556
    assert m["best_global_top1_hit"] == 1.0
    assert m["best_within_regime_top1_hit"] == 1.0
    assert m["hit_gap_conditioned_vs_mixed"] == 0.0
    assert m["gain_gap_conditioned_vs_mixed"] == 0.0005089501072329528


def test_c48_best_row_discloses_sparse_max_local_caveat():
    rows = [
        r for r in _rows("local_ceiling_best_by_scope.csv")
        if r["group_scope"] == "within_target" and r["label"] == "primary_joint_good"
    ]
    assert len(rows) == 1
    r = rows[0]
    assert r["source_space"] == "all_source_objectives"
    assert r["neighborhood"] == "eps_q02"
    assert float(r["mean_local_purity"]) == 0.42537373669020523
    assert float(r["mean_random_top1_baseline"]) == 0.42378127071295635
    assert float(r["mean_neighbor_count"]) == 1.2747740581303715
    assert float(r["mean_empty_neighborhood_fraction"]) == 0.6549140933100052
    assert float(r["mean_local_bayes_gap_vs_permutation"]) == 0.5743454359971853


def test_c48_c47_reference_is_unchanged():
    d = _summary()
    ref = d["c47_reference"]
    assert ref["cases"] == [
        "GCA1_conditioning_restores_source_neighborhood_homogeneity",
        "GCA2_conditioning_improves_but_not_reliable_actionability",
        "GCA5_grouped_actionability_still_base_rate_limited",
        "GCA6_global_source_only_comparability_fails",
        "GCA7_group_conditioning_is_separate_problem_class",
    ]
    assert ref["best_conditioned_strict_source_top1_hit"] == 0.5555555555555556
    assert ref["best_conditioned_strict_source_top1_enrichment"] == 1.3066262341305979


def test_c48_tables_have_expected_shape_and_gates():
    assert len(_rows("source_space_registry.csv")) == 6
    assert len(_rows("epsilon_radius_registry.csv")) == 24
    assert len(_rows("local_ceiling_group_detail.csv")) == 27720
    assert len(_rows("local_ceiling_summary.csv")) == 756
    assert len(_rows("local_ceiling_best_by_scope.csv")) == 18
    assert len(_rows("cross_group_stability.csv")) == 126
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    red = {r["check"]: r["passed"] for r in _rows("red_team_verification.csv")}
    assert red and set(red.values()) == {"1"}
    tax = {r["case"]: r["established"] for r in _rows("c48_case_taxonomy.csv")}
    assert tax[schema.LC1] == "1"
    assert tax[schema.LC3] == "1"
    assert tax[schema.LC4] == "0"
    assert tax[schema.LC6] == "0"


def test_c48_outputs_do_not_emit_method_artifacts():
    for name in ("local_ceiling_group_detail.csv", "local_ceiling_summary.csv",
                 "local_ceiling_best_by_scope.csv", "cross_group_stability.csv"):
        text = open(os.path.join(TABLE_DIR, name)).read()
        assert "model_hash" not in text
        assert "checkpoint_hash" not in text
        assert "selected_candidate_id" not in text
    assert "no_candidate_id_emitted" in open(os.path.join(TABLE_DIR, "local_ceiling_group_detail.csv")).read()


def test_c48_report_run_loads_committed_artifacts_without_recompute():
    res = report.run()
    assert res["taxonomy"]["cases"] == [schema.LC1, schema.LC3]
    assert res["n_candidate_rows"] == 3804
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c48_reports_do_not_overclaim():
    forbidden = (
        "source-space selector",
        "local bayes selector",
        "usable selector",
        "target-free detector",
        "oaci rescue",
        "external validation success",
        "target-unlabeled dg success",
        "target-grouped oracle as method",
    )
    for name in ("C48_CONDITIONED_LOCAL_BAYES_CEILING.md",
                 "C48_CONDITIONED_LOCAL_BAYES_CEILING.json",
                 "C48_RED_TEAM_VERIFICATION.md"):
        text = open(os.path.join("oaci/reports", name)).read().lower()
        for phrase in forbidden:
            assert phrase not in text
    md = open("oaci/reports/C48_CONDITIONED_LOCAL_BAYES_CEILING.md").read()
    assert "sparse max-local diagnostic ceiling" in md
    assert "not a broad purity shift" in md
