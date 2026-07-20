"""C35 Utility-Cone / Pareto Regret Robustness Audit tests."""
from __future__ import annotations

import numpy as np

from oaci.utility_cone import (artifact_loader, endpoint_vectors, pareto_audit, report, schema, taxonomy,
                               utility_simplex)


def _toy_vectors():
    return [
        {"pair_id": "robust", "seed": "0", "target": "1", "level": "0", "regime": "r",
         "comparison": "nearest_continuous_better", "raw_delta_bacc": 0.1,
         "raw_delta_nll_improve": 0.2, "raw_delta_ece_improve": 0.3,
         "global_z_delta_bacc": 0.1, "global_z_delta_nll_improve": 0.2, "global_z_delta_ece_improve": 0.3,
         "within_z_delta_bacc": 0.1, "within_z_delta_nll_improve": 0.2, "within_z_delta_ece_improve": 0.3,
         "rank_delta_bacc": 0.1, "rank_delta_nll_improve": 0.2, "rank_delta_ece_improve": 0.3},
        {"pair_id": "tradeoff", "seed": "0", "target": "1", "level": "0", "regime": "r",
         "comparison": "nearest_continuous_better", "raw_delta_bacc": 0.1,
         "raw_delta_nll_improve": -0.3, "raw_delta_ece_improve": 0.1,
         "global_z_delta_bacc": 0.1, "global_z_delta_nll_improve": -0.3, "global_z_delta_ece_improve": 0.1,
         "within_z_delta_bacc": 0.1, "within_z_delta_nll_improve": -0.3, "within_z_delta_ece_improve": 0.1,
         "rank_delta_bacc": 0.1, "rank_delta_nll_improve": -0.3, "rank_delta_ece_improve": 0.1},
    ]


def test_weight_grid_is_frozen_simplex():
    w = utility_simplex.weight_grid()
    assert len(w) == 231
    assert np.all(w >= 0)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert schema.UTILITY_GRID_STEP == 0.05


def test_pareto_and_simplex_classify_toy_vectors():
    p = pareto_audit.pareto_audit(_toy_vectors())
    assert p["summary"]["strict_pareto_better_fraction"] == 0.5
    u = utility_simplex.simplex_audit(_toy_vectors())
    cats = {r["pair_id"]: r["utility_cone_category"] for r in u["rows"]}
    assert cats["robust"] == "preference_robust_regret"
    assert cats["tradeoff"] in {"preference_dependent_regret", "narrow_scalarization_regret"}


def test_c34s_loader_gates_and_reconstruction():
    c34s = artifact_loader.load_c34s()
    assert all(c34s["c34s_gates"].values())
    rec = c34s["c34_reconstruction"]
    assert rec["taxonomy_cases"] == [
        "M2_continuous_source_active_misranking",
        "M7_target_unlabeled_pooled_only_reconfirmed",
        "M8_continuous_endpoint_tradeoff_local",
    ]
    assert rec["n_selected_continuous_better_pairs"] == 153
    assert rec["continuous_raw_pareto_nonworse_count"] == 72
    assert rec["continuous_raw_endpoint_backward_count"] == 81
    assert rec["continuous_joint_min_negative_count"] == 33


def test_endpoint_vectors_build_from_c34_tables():
    c34s = artifact_loader.load_c34s()
    vectors = endpoint_vectors.build_endpoint_vectors(c34s["tables"])
    assert len(vectors) == 549
    primary = [r for r in vectors if r["comparison"] == "nearest_continuous_better"]
    assert len(primary) == 153
    assert {"raw_delta_bacc", "global_z_delta_bacc", "within_z_delta_bacc", "rank_delta_bacc"}.issubset(primary[0])


def test_real_report_run_and_taxonomy():
    res = report.run()
    assert res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH
    assert all(res["c34s"]["c34s_gates"].values())
    assert res["utility_simplex"]["summary"]["preference_robust_fraction"] == 0.7450980392156863
    assert res["pareto"]["summary"]["pareto_incomparable_fraction"] == 0.5294117647058824
    assert schema.U1 in res["taxonomy"]["cases"]
    assert schema.U7 in res["taxonomy"]["cases"]


def test_no_selector_gate_and_report_guard():
    res = report.run()
    gates = {r["check"]: r["passed"] for r in report.no_selector_gate(res)}
    assert gates["G0_manifest_resolves"]
    assert gates["G1_table_hashes_match"]
    assert gates["G2_key_numbers_reconstruct"]
    assert gates["G3_no_legacy_monolithic_dependency"]
    try:
        report._guard_forbidden("utility-cone selector works")
        raise AssertionError("guard failed")
    except ValueError:
        pass
    report._guard_forbidden("not a utility-cone selector; no OACI rescue is claimed.")
