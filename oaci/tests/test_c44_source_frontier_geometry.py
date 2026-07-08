"""C44 Source-Pareto Degeneracy / Objective-Geometry tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.source_frontier_geometry import report, schema


REPORT_JSON = "oaci/reports/C44_SOURCE_PARETO_DEGENERACY_AUDIT.json"
TABLE_DIR = "oaci/reports/c44_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c44_config_and_nulls_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.NULL_REPS == 50
    assert schema.NULL_SEED == 44044
    assert schema.FRONT_DEGENERATE_FRACTION == 0.85


def test_c44_summary_taxonomy_is_conservative():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    cases = set(d["taxonomy"]["cases"])
    assert cases == {schema.PF1, schema.PF2, schema.PF3, schema.PF4,
                     schema.PF5, schema.PF6, schema.PF7, schema.PF9}
    assert schema.PF8 not in cases
    assert schema.PF10 not in cases


def test_c44_pareto_front_matches_high_dimensional_null():
    d = _summary()
    n = d["source_frontier_null_summary"]
    assert n["observed_front_fraction"] == 0.9724614975980579
    assert n["gaussian_null_front_fraction"] == 0.9810450967215901
    assert n["objective_shuffled_front_fraction"] == 0.9811108958203643
    assert n["family_shuffled_front_fraction"] == 0.9624459760901136
    assert abs(n["observed_front_fraction"] - n["gaussian_null_front_fraction"]) < schema.FRONT_NULL_CLOSE_DELTA


def test_c44_effective_dimension_and_conflict_numbers():
    d = _summary()
    e = d["objective_effective_dimension_summary"]
    assert e["n_objectives"] == 10
    assert e["effective_rank"] == 5.695876335809214
    assert e["pca_var1"] == 0.41356684026343554
    assert e["pca_cum3"] == 0.7134764204246475
    assert e["negative_pair_fraction"] == 0.4444444444444444
    assert e["leakage_rank_mean_spearman"] == -0.07109398916881127


def test_c44_front_membership_is_non_discriminative():
    d = _summary()
    j = d["front_membership_summary"]["joint_good"]
    assert j["mean_p_label_given_front"] == 0.43105701988584916
    assert j["mean_trajectory_baseline"] == 0.4297233780360411
    assert j["mean_p_label_given_not_front"] == 0.5291666666666666
    assert j["mean_front_enrichment_over_trajectory"] == 1.0374830061099583
    co = d["target_good_bad_front_cooccupancy_summary"]
    assert co["front_contains_both_good_and_bad_fraction"] == 0.8888888888888888
    assert co["mean_not_front_count"] == 0.7469135802469136


def test_c44_family_frontiers_and_depth():
    fam = {r["subset"]: r for r in _rows("family_reduced_frontiers.csv")}
    assert float(fam["all_families"]["mean_front_fraction"]) == 0.9724614975980579
    assert float(fam["rank_only"]["mean_front_fraction"]) == 0.04375006002171776
    assert float(fam["rank_only"]["mean_front_joint_good_enrichment"]) == 1.1542183729112576
    assert float(fam["rank_only"]["mean_depth_auc_vs_target_utility"]) == 0.5904967774230292
    assert float(fam["leakage_only"]["mean_front_joint_good_enrichment"]) == 0.710724248421947
    depth = _summary()["dominance_depth_summary"]
    assert depth["mean_layer_auc_vs_target_utility"] == 0.49906596898123196
    assert depth["mean_n_dominators_auc_vs_target_utility"] == 0.49903229894756185
    assert depth["mean_n_dominated_auc_vs_target_utility"] == 0.5010971753227703


def test_c44_tables_have_expected_shape_and_gates():
    assert len(_rows("source_frontier_null_audit.csv")) == 5
    assert len(_rows("objective_effective_dimension.csv")) == 1
    assert len(_rows("objective_family_conflict_matrix.csv")) == 16
    assert len(_rows("family_reduced_frontiers.csv")) == 9
    assert len(_rows("front_membership_discriminativeness.csv")) == 3
    assert len(_rows("dominance_depth_target_alignment.csv")) == 162
    assert len(_rows("target_good_bad_front_cooccupancy.csv")) == 162
    assert len(_rows("source_objective_geometry_summary.csv")) == 1
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    tax = {r["case"]: r["established"] for r in _rows("c44_case_taxonomy.csv")}
    assert tax[schema.PF1] == "1"
    assert tax[schema.PF9] == "1"
    assert tax[schema.PF8] == "0"
    assert tax[schema.PF10] == "0"


def test_c44_outputs_do_not_emit_method_artifacts():
    for name in ("source_frontier_null_audit.csv", "family_reduced_frontiers.csv",
                 "dominance_depth_target_alignment.csv", "target_good_bad_front_cooccupancy.csv"):
        text = open(os.path.join(TABLE_DIR, name)).read()
        assert "model_hash" not in text
        assert "checkpoint_hash" not in text
        assert "selected_candidate_id" not in text


def test_c44_report_run_loads_committed_artifacts_without_recompute():
    res = report.run()
    assert res["taxonomy"]["cases"] == [
        schema.PF1, schema.PF2, schema.PF3, schema.PF4, schema.PF5, schema.PF6, schema.PF7, schema.PF9]
    assert res["n_candidate_rows"] == 3804
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c44_reports_do_not_overclaim():
    for name in ("C44_SOURCE_PARETO_DEGENERACY_AUDIT.md",
                 "C44_SOURCE_PARETO_DEGENERACY_AUDIT.json",
                 "C44_OBJECTIVE_GEOMETRY_DIMENSION_AUDIT.md",
                 "C44_FRONTIER_DISCRIMINATIVENESS_AUDIT.md"):
        text = open(os.path.join("oaci/reports", name)).read()
        low = text.lower()
        assert "deployable selector" not in low
        assert "oaci rescue" not in low
        assert "target-free detector" not in low
        assert "external validation success" not in low
    md = open("oaci/reports/C44_SOURCE_PARETO_DEGENERACY_AUDIT.md").read()
    assert "target-good-on-front does not imply source-side identifiability" in md
