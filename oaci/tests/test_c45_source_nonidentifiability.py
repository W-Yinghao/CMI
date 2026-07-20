"""C45 Source-equivalence / target-divergence witness tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.source_nonidentifiability import report, schema


REPORT_JSON = "oaci/reports/C45_SOURCE_EQUIVALENCE_NONIDENTIFIABILITY.json"
TABLE_DIR = "oaci/reports/c45_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c45_config_and_distance_metrics_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.PRIMARY_DISTANCE == "within_trajectory_z_euclidean"
    assert schema.RANK_DISTANCE == "within_trajectory_rank_l1"
    assert schema.FAMILY_BLOCK_DISTANCE == "family_block_z_euclidean"
    assert schema.EPSILON_QUANTILES == (0.01, 0.02, 0.05, 0.10)
    assert schema.SOURCE_EQUIVALENT_Q == 0.10


def test_c45_taxonomy_is_conservative_after_red_team():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    cases = set(d["taxonomy"]["cases"])
    assert cases == {schema.N1, schema.N6, schema.N7}
    for inactive in (schema.N2, schema.N3, schema.N4, schema.N5, schema.N8, schema.N9):
        assert inactive not in cases


def test_c45_witness_summary_numbers():
    d = _summary()
    w = d["nearest_source_neighbor_summary"]
    assert w["within_trajectory"]["source_equivalent_q10_target_divergent_rate"] == 0.13287671232876713
    assert w["within_trajectory"]["joint_good_disagreement_rate"] == 0.19137749737118823
    assert w["within_trajectory"]["baseline_joint_good_disagreement_rate"] == 0.3107255520504732
    assert w["cross_target"]["source_equivalent_q10_target_divergent_rate"] == 0.9369369369369369
    assert w["same_regime"]["source_equivalent_q10_target_divergent_rate"] == 0.29901423877327493
    assert w["within_target"]["source_equivalent_q10_target_divergent_rate"] == 0.004842615012106538
    assert w["source_equivalent_target_divergent_pair_count"] == 632
    assert w["trajectories_with_source_equivalent_divergent_pair_fraction"] == 0.2222222222222222


def test_c45_local_variance_and_lower_bound_do_not_overstate():
    d = _summary()
    v = d["epsilon_radius_target_variance_summary"]
    lb = d["selector_lower_bound_summary"]
    assert v["q05"]["target_utility_variance_over_baseline"] == 0.0002558406501751236
    assert v["q05"]["joint_entropy_over_baseline"] == 0.005105781801103029
    assert v["q10"]["joint_entropy_over_baseline"] == 0.03562075943766549
    assert lb["q10"]["ambiguous_neighborhood_fraction"] == 0.02996845425867508
    assert lb["q10"]["minimum_unavoidable_ambiguity_rate"] == 0.00707238971759291


def test_c45_family_reduced_spaces_remain_ambiguous():
    d = _summary()
    fam = d["family_space_witness_summary"]
    assert fam["all_source_objectives"]["n_objectives"] == 17
    assert fam["all_source_objectives"]["source_equivalent_q10_target_divergent_rate"] == 0.13287671232876713
    assert fam["leakage_rank"]["source_equivalent_q10_target_divergent_rate"] == 0.26990692864529475
    assert fam["rank_risk"]["source_equivalent_q10_target_divergent_rate"] == 0.27740492170022374
    assert fam["rank_only"]["source_equivalent_q10_target_divergent_rate"] == 0.5945807770961146
    assert fam["rank_only"]["joint_disagreement_reduction_vs_baseline"] == 0.07308096740273398


def test_c45_gauge_is_diagnostic_but_not_primary_driver():
    d = _summary()
    g = d["target_gauge_residual_summary"]
    assert g["n_source_equivalent_divergent_pairs"] == 632
    assert g["n_source_equivalent_gauge_witnesses"] == 17
    assert g["source_equivalent_gauge_witness_fraction"] == 0.02689873417721519
    assert g["gauge_gap_target_utility_gap_corr"] == 0.7255185224457221
    assert g["target_gauge_candidate_level_source_available"] is False
    tax = {r["case"]: r["established"] for r in _rows("c45_case_taxonomy.csv")}
    assert tax[schema.N5] == "0"


def test_c45_tables_have_expected_shape_and_gates():
    assert len(_rows("source_objective_space_registry.csv")) == 18
    assert len(_rows("nearest_source_neighbor_witnesses.csv")) == 15216
    assert len(_rows("source_equivalent_target_divergent_pairs.csv")) == 632
    assert len(_rows("epsilon_radius_target_variance.csv")) == 4
    assert len(_rows("source_neighborhood_label_entropy.csv")) == 648
    assert len(_rows("family_space_witness_comparison.csv")) == 7
    assert len(_rows("target_gauge_residual_witnesses.csv")) == 632
    assert len(_rows("empirical_selector_lower_bound.csv")) == 4
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}


def test_c45_outputs_do_not_emit_method_artifacts():
    for name in ("nearest_source_neighbor_witnesses.csv", "source_equivalent_target_divergent_pairs.csv",
                 "target_gauge_residual_witnesses.csv", "family_space_witness_comparison.csv"):
        text = open(os.path.join(TABLE_DIR, name)).read()
        assert "model_hash" not in text
        assert "checkpoint_hash" not in text
        assert "selected_candidate_id" not in text


def test_c45_report_run_loads_committed_artifacts_without_recompute():
    res = report.run()
    assert res["taxonomy"]["cases"] == [schema.N1, schema.N6, schema.N7]
    assert res["n_candidate_rows"] == 3804
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c45_reports_do_not_overclaim():
    forbidden = (
        "deployable selector",
        "target-free detector",
        "oaci rescue",
        "external validation success",
        "target-unlabeled dg success",
        "target-grouped oracle as method",
    )
    for name in ("C45_SOURCE_EQUIVALENCE_NONIDENTIFIABILITY.md",
                 "C45_SOURCE_EQUIVALENCE_NONIDENTIFIABILITY.json",
                 "C45_TARGET_DIVERGENT_WITNESSES.md",
                 "C45_SELECTOR_LOWER_BOUND_AUDIT.md"):
        text = open(os.path.join("oaci/reports", name)).read().lower()
        for phrase in forbidden:
            assert phrase not in text
    md = open("oaci/reports/C45_SOURCE_EQUIVALENCE_NONIDENTIFIABILITY.md").read()
    assert "Full all-source within-trajectory q10 neighborhoods are comparatively homogeneous" in md
    assert "N2/N3/N8 and gauge-driver N5 remain inactive" in md
