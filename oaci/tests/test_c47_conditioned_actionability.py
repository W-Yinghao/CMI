"""C47 Conditioned Source-Space Actionability Audit tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.conditioned_actionability import report, schema


REPORT_JSON = "oaci/reports/C47_CONDITIONED_SOURCE_SPACE_ACTIONABILITY.json"
TABLE_DIR = "oaci/reports/c47_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c47_config_groups_and_gates_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.GROUP_SCOPES == (
        "global", "within_target", "within_trajectory",
        "within_target_seed", "within_target_level", "within_regime")
    assert schema.TOP_KS == (1, 3, 5, 10)
    assert schema.LABELS == (
        "primary_joint_good", "pareto_good", "preference_robust_better_candidate")
    assert schema.PAIR_SAMPLE_MAX == 100000
    assert schema.PAIR_SAMPLE_SEED == 47047
    assert schema.RELIABLE_TOP1_HIT_GATE == 0.70


def test_c47_taxonomy_is_conditioned_but_not_reliable_actionability():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    cases = set(d["taxonomy"]["cases"])
    assert cases == {schema.GCA1, schema.GCA2, schema.GCA5, schema.GCA6, schema.GCA7}
    assert schema.GCA3 not in cases
    assert schema.GCA4 not in cases
    assert schema.GCA8 not in cases


def test_c47_inherits_c46_conditioning_boundary_numbers():
    d = _summary()
    b = d["inherited_c46_boundary"]
    assert b["within_target_q10_divergent"] == 0.004842615012106538
    assert b["within_trajectory_q10_divergent"] == 0.13287671232876713
    assert b["within_regime_q10_divergent"] == 0.29901423877327493
    assert b["cross_target_q10_divergent"] == 0.9369369369369369
    assert b["cross_regime_q10_divergent"] == 0.004033342296316214


def test_c47_primary_actionability_numbers_are_group_conditioned():
    d = _summary()
    m = d["taxonomy"]["primary_metrics"]
    assert m["global_strict_source_top1_gain"] == -0.4242902208201893
    assert m["within_target_strict_source_top1_gain"] == 0.02066317373148807
    assert m["within_trajectory_strict_source_top1_gain"] == 0.07644946147013171
    assert m["within_target_seed_strict_source_top1_gain"] == 0.1303723309701325
    assert m["within_target_level_strict_source_top1_gain"] == 0.01895296607851954
    assert m["within_regime_strict_source_top1_gain"] == -0.4242902208201893
    assert m["best_conditioned_strict_source_top1_hit"] == 0.5555555555555556
    assert m["best_conditioned_strict_source_top1_enrichment"] == 1.3066262341305979
    tax = {r["case"]: r for r in d["taxonomy"]["case_rows"]}
    assert "trajectory_unique_advantage=-0.0539228695000008" in tax[schema.GCA3]["evidence"]


def test_c47_score_registry_discloses_hindsight_and_target_ceiling():
    rows = {r["score"]: r for r in _rows("source_score_registry.csv")}
    assert rows["selection_leakage"]["source_only"] == "1"
    assert rows["R_src"]["orientation"] == "lower"
    assert rows["C43_best_hindsight_scalarization"]["hindsight_diagnostic_only"] == "1"
    assert rows["C43_best_hindsight_scalarization"]["target_label_used"] == "0"
    assert rows["target_utility_oracle_ceiling"]["source_only"] == "0"
    assert rows["target_utility_oracle_ceiling"]["target_label_used"] == "1"
    assert rows["target_utility_oracle_ceiling"]["diagnostic_ceiling"] == "1"


def test_c47_smoothing_and_sign_consistency_numbers():
    d = _summary()
    m = d["taxonomy"]["primary_metrics"]
    assert d["source_neighborhood_smoothing_summary"]["distance_metric"] == \
        "within_trajectory_z_euclidean"
    assert d["source_neighborhood_smoothing_summary"]["q10_radius"] == 1.269009511962666
    assert d["source_neighborhood_smoothing_summary"]["target_oracle_ceiling_smoothed"] is False
    assert m["max_primary_top1_smoothing_gain_delta"] == 0.11111111111111112
    assert m["global_max_strict_source_pairwise_auc"] == 0.5197068289601652
    assert m["within_target_max_strict_source_pairwise_auc"] == 0.5966503556474801
    assert m["within_trajectory_max_strict_source_pairwise_auc"] == 0.601322550171655
    assert d["pairwise_sign_consistency_summary"]["sampled_scopes"] == [
        "global", "within_regime", "within_target", "within_target_level", "within_target_seed"]


def test_c47_tables_have_expected_shape_and_gates():
    assert len(_rows("group_scope_registry.csv")) == 6
    assert len(_rows("source_score_registry.csv")) == 6
    assert len(_rows("group_actionability_detail.csv")) == 15840
    assert len(_rows("group_actionability_summary.csv")) == 432
    assert len(_rows("group_actionability_best_by_scope.csv")) == 216
    assert len(_rows("source_neighborhood_smoothing.csv")) == 360
    assert len(_rows("pairwise_sign_consistency.csv")) == 36
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    red = {r["check"]: r["passed"] for r in _rows("red_team_verification.csv")}
    assert red and set(red.values()) == {"1"}
    tax = {r["case"]: r["established"] for r in _rows("c47_case_taxonomy.csv")}
    assert tax[schema.GCA1] == "1"
    assert tax[schema.GCA3] == "0"
    assert tax[schema.GCA4] == "0"
    assert tax[schema.GCA8] == "0"


def test_c47_outputs_do_not_emit_method_artifacts():
    for name in ("group_actionability_detail.csv", "group_actionability_summary.csv",
                 "source_neighborhood_smoothing.csv", "pairwise_sign_consistency.csv"):
        text = open(os.path.join(TABLE_DIR, name)).read()
        assert "model_hash" not in text
        assert "checkpoint_hash" not in text
        assert "selected_candidate_id" not in text
    detail = open(os.path.join(TABLE_DIR, "group_actionability_detail.csv")).read()
    assert "no_candidate_id_emitted" in detail


def test_c47_report_run_loads_committed_artifacts_without_recompute():
    res = report.run()
    assert res["taxonomy"]["cases"] == [schema.GCA1, schema.GCA2, schema.GCA5, schema.GCA6, schema.GCA7]
    assert res["n_candidate_rows"] == 3804
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c47_reports_do_not_overclaim():
    forbidden = (
        "conditioned selector",
        "deployable selector",
        "target-free detector",
        "oaci rescue",
        "external validation success",
        "target-unlabeled dg success",
        "target-grouped oracle as method",
    )
    for name in ("C47_CONDITIONED_SOURCE_SPACE_ACTIONABILITY.md",
                 "C47_CONDITIONED_SOURCE_SPACE_ACTIONABILITY.json",
                 "C47_SOURCE_NEIGHBORHOOD_SMOOTHING_AUDIT.md",
                 "C47_RED_TEAM_VERIFICATION.md"):
        text = open(os.path.join("oaci/reports", name)).read().lower()
        for phrase in forbidden:
            assert phrase not in text
    md = open("oaci/reports/C47_CONDITIONED_SOURCE_SPACE_ACTIONABILITY.md").read()
    assert "same-group random baselines" in md
    assert "below reliability gates" in md
