"""C46 Conditioning Boundary / Grouping-Sensitive Non-Identifiability tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.source_conditioning_boundary import report, schema


REPORT_JSON = "oaci/reports/C46_CONDITIONING_BOUNDARY_AUDIT.json"
TABLE_DIR = "oaci/reports/c46_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c46_config_scopes_and_metrics_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.PRIMARY_DISTANCE == "within_trajectory_z_euclidean"
    assert schema.CONDITIONING_SCOPES == (
        "within_trajectory", "within_target", "within_seed", "within_level",
        "within_regime", "cross_target", "cross_regime")
    assert schema.PAIR_SAMPLE_SEED == 46046
    assert schema.PAIR_SAMPLE_MAX == 100000


def test_c46_taxonomy_is_conditioning_sensitive_not_target_identity_dominant():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    assert d["n_source_objectives"] == 17
    cases = set(d["taxonomy"]["cases"])
    assert cases == {schema.CB1, schema.CB2, schema.CB3, schema.CB4, schema.CB6}
    assert schema.CB5 not in cases
    assert schema.CB7 not in cases


def test_c46_neighbor_boundary_numbers():
    d = _summary()
    n = d["conditioning_neighbor_summary"]
    assert n["within_target"]["source_equivalent_q10_target_divergent_rate"] == 0.004842615012106538
    assert n["within_trajectory"]["source_equivalent_q10_target_divergent_rate"] == 0.13287671232876713
    assert n["within_regime"]["source_equivalent_q10_target_divergent_rate"] == 0.29901423877327493
    assert n["cross_target"]["source_equivalent_q10_target_divergent_rate"] == 0.9369369369369369
    assert n["cross_regime"]["source_equivalent_q10_target_divergent_rate"] == 0.004033342296316214
    assert n["within_target"]["target_divergent_rate"] == 0.007360672975814932
    assert n["cross_target"]["target_divergent_rate"] == 0.9400630914826499
    assert n["within_regime"]["joint_good_disagreement_rate"] == 0.3588328075709779


def test_c46_conditional_variance_and_decomposition_numbers():
    d = _summary()
    cv = d["conditional_target_variance_summary"]
    vd = d["variance_decomposition_summary"]
    assert cv["target"]["target_utility_variance_over_global"] == 0.7529306744087769
    assert cv["trajectory"]["target_utility_variance_over_global"] == 0.3850502874437927
    assert cv["seed"]["target_utility_variance_over_global"] == 0.9912246689563659
    assert cv["level"]["target_utility_variance_over_global"] == 0.9777954918316467
    assert cv["regime"]["target_utility_variance_over_global"] == 1.0
    assert cv["trajectory"]["target_gauge_variance_over_global"] == 0.40123547038769314
    assert vd["target_utility_score|target"]["eta_squared"] == 0.24706932559122305
    assert vd["target_utility_score|trajectory"]["eta_squared"] == 0.6149497125562071
    assert vd["target_utility_score|regime"]["eta_squared"] == 0.0
    assert vd["target_utility_score|residual_within_trajectory"]["eta_squared"] == 0.3850502874437927


def test_c46_distance_usefulness_numbers():
    d = _summary()
    du = d["source_distance_usefulness_summary"]
    assert du["within_trajectory"]["source_distance_target_utility_gap_spearman"] == 0.39119886096262285
    assert du["within_target"]["source_distance_target_utility_gap_spearman"] == 0.07920068556376955
    assert du["cross_target"]["source_distance_target_utility_gap_spearman"] == 0.016024950337713784
    assert du["cross_regime"]["source_distance_target_utility_gap_spearman"] == 0.022849960766657622
    assert du["within_trajectory"]["n_pairs"] == 44115
    assert du["cross_target"]["n_pairs"] == 100000


def test_c46_tables_have_expected_shape_and_gates():
    assert len(_rows("conditioning_scope_registry.csv")) == 7
    assert len(_rows("conditioning_neighbor_ambiguity.csv")) == 7
    assert len(_rows("conditioning_nearest_neighbor_witnesses.csv")) == 53256
    assert len(_rows("conditional_target_variance.csv")) == 7
    assert len(_rows("conditioning_group_variance_rows.csv")) == 207
    assert len(_rows("variance_decomposition.csv")) == 21
    assert len(_rows("source_distance_usefulness.csv")) == 7
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    tax = {r["case"]: r["established"] for r in _rows("c46_case_taxonomy.csv")}
    assert tax[schema.CB1] == "1"
    assert tax[schema.CB5] == "0"
    assert tax[schema.CB7] == "0"


def test_c46_outputs_do_not_emit_method_artifacts():
    for name in ("conditioning_neighbor_ambiguity.csv", "conditioning_nearest_neighbor_witnesses.csv",
                 "source_distance_usefulness.csv", "variance_decomposition.csv"):
        text = open(os.path.join(TABLE_DIR, name)).read()
        assert "model_hash" not in text
        assert "checkpoint_hash" not in text
        assert "selected_candidate_id" not in text


def test_c46_report_run_loads_committed_artifacts_without_recompute():
    res = report.run()
    assert res["taxonomy"]["cases"] == [schema.CB1, schema.CB2, schema.CB3, schema.CB4, schema.CB6]
    assert res["n_candidate_rows"] == 3804
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c46_reports_do_not_overclaim():
    forbidden = (
        "deployable selector",
        "target-free detector",
        "oaci rescue",
        "external validation success",
        "target-unlabeled dg success",
        "target-grouped oracle as method",
    )
    for name in ("C46_CONDITIONING_BOUNDARY_AUDIT.md",
                 "C46_CONDITIONING_BOUNDARY_AUDIT.json",
                 "C46_GROUPING_SENSITIVE_NONIDENTIFIABILITY.md",
                 "C46_VARIANCE_DECOMPOSITION_AUDIT.md"):
        text = open(os.path.join("oaci/reports", name)).read().lower()
        for phrase in forbidden:
            assert phrase not in text
    md = open("oaci/reports/C46_CONDITIONING_BOUNDARY_AUDIT.md").read()
    assert "Regime alone is not the break" in md
    assert "cross-regime same-target neighborhoods remain homogeneous" in md
