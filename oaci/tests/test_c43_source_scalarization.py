"""C43 Source-Objective Scalarization Frontier tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.source_scalarization import report, schema


REPORT_JSON = "oaci/reports/C43_SOURCE_OBJECTIVE_SCALARIZATION_FRONTIER.json"
TABLE_DIR = "oaci/reports/c43_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c43_config_and_grid_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.SCALARIZATION_GRID_STEP == 0.10
    assert schema.RELIABLE_TOP1_JOINT_GATE == 0.70
    assert schema.TARGET_SIGN_CONSISTENCY_GATE == 0.80


def test_c43_summary_taxonomy_closes_scalarization_escape_hatch():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    assert set(d["taxonomy"]["cases"]) == {schema.F1, schema.F3, schema.F4, schema.F5, schema.F7, schema.F8}
    assert schema.F2 not in d["taxonomy"]["cases"]
    assert schema.F6 not in d["taxonomy"]["cases"]
    assert schema.F9 not in d["taxonomy"]["cases"]
    assert schema.F10 not in d["taxonomy"]["cases"]


def test_c43_best_hindsight_ceiling_is_weak_despite_correction():
    d = _summary()
    m = d["scalarization_multiplicity_summary"]
    assert m["best_scalarization_id"] == "leakage_rank_risk__leakage_0.5__rank_0.1__risk_0.4"
    assert m["best_top1_joint_good_rate"] == 0.5740740740740741
    assert m["best_expected_random_top1_joint_good_rate"] == 0.4297233780360411
    assert m["best_top1_gain_vs_random"] == 0.14435069603803297
    assert m["best_holm_p_value"] == 9.703820315555286e-05
    assert m["best_bh_q_value"] == 9.703820315555286e-05
    assert m["best_per_target_sign_consistency"] == 1.0
    assert m["any_positive_scalarization_claim_allowed"] is False


def test_c43_source_pareto_front_is_broad_not_localizing():
    d = _summary()
    f = d["source_pareto_frontier_summary"]
    assert f["mean_front_fraction"] == 0.9724614975980579
    assert f["oaci_selected_front_rate"] == 1.0
    assert f["source_rank_top_front_rate"] == 1.0
    assert f["joint_good_front_fraction"] == 0.9875535762598431
    assert f["pareto_good_front_fraction"] == 0.9963991769547327
    assert f["preference_robust_front_fraction"] == 0.9649122807017544
    assert f["joint_good_rejected_fraction"] == 0.012446423740156923


def test_c43_leakage_rank_tradeoff_and_blocking():
    d = _summary()
    c = d["leakage_rank_frontier_summary"]
    assert c["mean_leakage_rank_spearman"] == -0.29180753712959345
    assert c["negative_leakage_rank_corr_fraction"] == 0.7962962962962963
    assert c["mean_oaci_leakage_rank_percentile"] == 0.016206140755335025
    assert c["mean_oaci_source_rank_percentile"] == 0.6854946290352651
    assert c["leakage_blocks_rank_better_fraction"] == 0.5370370370370371
    assert c["source_rank_leakage_tradeoff_real"] is True
    assert c["leakage_extreme_blocks_rank_frontier"] is True


def test_c43_tables_have_expected_shape_and_gates():
    assert len(_rows("source_objective_registry.csv")) == 18
    assert len(_rows("source_pareto_frontier_status.csv")) == 162
    assert len(_rows("scalarization_grid_registry.csv")) == 103
    assert len(_rows("scalarization_actionability_metrics.csv")) == 1236
    assert len(_rows("scalarization_multiplicity_audit.csv")) == 103
    assert len(_rows("leakage_rank_frontier_conflict.csv")) == 162
    assert len(_rows("best_hindsight_scalarization_ceiling.csv")) == 10
    assert len(_rows("per_target_scalarization_stability.csv")) == 927
    assert len(_rows("trajectory_conditioned_random_baseline.csv")) == 1944
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    tax = {r["case"]: r["established"] for r in _rows("c43_case_taxonomy.csv")}
    assert tax[schema.F1] == "1"
    assert tax[schema.F8] == "1"
    assert tax[schema.F9] == "0"
    assert tax[schema.F10] == "0"


def test_c43_registry_and_tables_do_not_emit_method_artifacts():
    obj = {r["objective"]: r for r in _rows("source_objective_registry.csv")}
    assert obj["selection_leakage_point"]["target_field"] == "0"
    assert obj["source_rank_score"]["used_for_scalarization_grid"] == "1"
    assert obj["feat__source_guard_entropy"]["family"] == "source_calibration_softness"
    for name in ("source_pareto_frontier_status.csv", "leakage_rank_frontier_conflict.csv"):
        text = open(os.path.join(TABLE_DIR, name)).read()
        assert "model_hash" not in text
        assert "checkpoint_hash" not in text
        assert "selected_candidate_id" not in text
        assert "no_candidate_id_emitted" in text


def test_c43_report_run_loads_committed_artifacts_without_recompute():
    res = report.run()
    assert res["taxonomy"]["cases"] == [schema.F1, schema.F3, schema.F4, schema.F5, schema.F7, schema.F8]
    assert res["n_candidate_rows"] == 3804
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c43_outputs_do_not_overclaim():
    for name in ("C43_SOURCE_OBJECTIVE_SCALARIZATION_FRONTIER.md",
                 "C43_SOURCE_OBJECTIVE_SCALARIZATION_FRONTIER.json",
                 "C43_SOURCE_PARETO_FRONTIER_AUDIT.md",
                 "C43_SCALARIZATION_ESCAPE_HATCH_AUDIT.md"):
        text = open(os.path.join("oaci/reports", name)).read()
        low = text.lower()
        assert "source-only selector" not in low
        assert "new selector" not in low
        assert "deployable selector" not in low
        assert "oaci rescue" not in low
        assert "target-free detector" not in low
        assert "external validation success" not in low
    md = open("oaci/reports/C43_SOURCE_OBJECTIVE_SCALARIZATION_FRONTIER.md").read()
    assert "F8_source_only_scalarization_escape_hatch_closed" in md
    assert "Best Hindsight Source Scalarization" in md
