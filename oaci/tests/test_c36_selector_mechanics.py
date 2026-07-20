"""C36 OACI Selector Mechanics / Feasibility-Regret Audit tests."""
from __future__ import annotations

from oaci.selector_mechanics import artifact_loader, report, schema, selector_trace


def test_c36_config_and_trace_constants_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.UTILITY_GRID_STEP == 0.05
    assert schema.ACTUAL_SELECTOR_SCORE_NAME == "selection_bootstrap_ucl"
    assert schema.POINT_PLATEAU_EPS == 0.02
    assert len(schema.SOURCE_PARETO_OBJECTIVES) == 9


def test_preference_robust_pairs_imported_from_c35_without_grid_change():
    pairs = artifact_loader.load_preference_robust_pairs()
    assert len(pairs) == 114
    assert {p["utility_cone_category"] for p in pairs} == {"preference_robust_regret"}
    assert {p["comparison"] for p in pairs} == {"nearest_continuous_better"}
    assert {p["weight_grid_step"] for p in pairs} == {"0.05"}


def test_c10_trace_resolves_robust_pairs_without_hash_emission():
    pairs = artifact_loader.load_preference_robust_pairs()
    trace = artifact_loader.load_c10_selector_trace(regimes=artifact_loader.regimes_from_pairs(pairs))
    resolved = selector_trace.robust_pair_trace_resolves(pairs, trace)
    assert resolved["all_resolved"]
    assert all("model_hash" not in r for r in trace["registry"])
    availability = {r["field"]: r for r in selector_trace.availability_audit(trace["registry"])}
    assert availability["checkpoint_hash_emitted"]["status"] == "pass_not_emitted"
    assert availability["per_candidate_selector_ucl_available"]["status"] == "unavailable"
    assert availability["actual_selector_rank_known"]["status"] == "partial"


def test_real_c36_report_taxonomy_and_gate():
    res = report.run()
    assert res["n_preference_robust_pairs"] == 114
    assert res["n_unique_selector_units"] == 38
    assert res["feasibility_regret"]["summary"]["risk_gate_regret_fraction"] == 0.0
    assert res["feasibility_regret"]["summary"]["leakage_objective_regret_fraction"] == 1.0
    assert res["selection_audit_inversion"]["summary"]["selection_to_audit_inversion_rate"] == 0.4473684210526316
    assert res["source_pareto"]["summary"]["source_pareto_conflict_fraction"] == 1.0
    assert res["source_pareto"]["summary"]["better_source_dominates_fraction"] == 0.0
    assert schema.S2 in res["taxonomy"]["cases"]
    assert schema.S4 in res["taxonomy"]["cases"]
    assert schema.S5 in res["taxonomy"]["cases"]
    assert schema.S9 in res["taxonomy"]["cases"]
    assert schema.S7 not in res["taxonomy"]["cases"]
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c36_report_guard_blocks_overclaim():
    try:
        report._guard_forbidden("deployable selector works")
        raise AssertionError("guard failed")
    except ValueError:
        pass
    report._guard_forbidden("not a deployable selector; no OACI rescue is claimed.")

