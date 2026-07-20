"""C38 Leakage-UCL Objective Geometry / Source-Target Conflict Audit tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.leakage_objective_geometry import report, schema


REPORT_JSON = "oaci/reports/C38_LEAKAGE_UCL_OBJECTIVE_GEOMETRY.json"
TABLE_DIR = "oaci/reports/c38_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c38_config_and_constants_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.ACTUAL_SELECTOR_SCORE_NAME == "selection_bootstrap_ucl"
    assert schema.UCL_CLEAR_EPS == 1e-9
    assert schema.POINT_CLEAR_EPS == 1e-12
    assert schema.UCL_PLATEAU_EPS == 0.02


def test_c38_summary_taxonomy_and_core_counts():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_preference_robust_pairs"] == 114
    u = d["ucl_point_width_summary"]
    assert u["ucl_prefers_selected_count"] == 114
    assert u["point_prefers_selected_count"] == 114
    assert u["uncertainty_driven_count"] == 0
    assert u["point_dominant_count"] == 111
    assert u["point_dominant_fraction"] == 0.9736842105263158
    assert d["selection_audit_inversion_summary"]["selection_ucl_to_audit_inversion_rate"] == (
        0.4473684210526316)
    assert d["source_target_conflict_summary"]["source_rational_target_wrong_fraction"] == 1
    assert d["gauge_conflict_summary"]["leakage_target_gauge_conflict_fraction"] == (
        0.9210526315789473)
    cases = set(d["taxonomy"]["cases"])
    assert {schema.L1, schema.L5, schema.L6, schema.L7, schema.L8, schema.L10}.issubset(cases)
    assert schema.L2 not in cases
    assert schema.L3 not in cases
    assert schema.L4 not in cases
    assert schema.L9 not in cases


def test_c38_tables_have_expected_shape_and_boundaries():
    assert len(_rows("ucl_point_width_decomposition.csv")) == 114
    assert len(_rows("selected_vs_better_leakage_components.csv")) == 114
    assert len(_rows("selection_audit_local_inversion.csv")) == 114
    assert len(_rows("source_rational_target_wrong_cases.csv")) == 114
    assert len(_rows("leakage_vs_target_gauge_conflict.csv")) == 114
    assert len(_rows("leakage_endpoint_decoupling.csv")) == 114
    assert len(_rows("leakage_atom_contribution_by_class_domain.csv")) == 6
    support = _rows("support_estimability_artifact_audit.csv")
    assert sum(1 for r in support if r["scope"] == "regime") == 3
    assert sum(1 for r in support if r["scope"] == "pair_key_across_regimes" and
               r["support_edge_driver"] == "0") == 38
    atoms = _rows("leakage_atom_contribution_by_class_domain.csv")
    assert all(r["atom_available"] == "0" for r in atoms)
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"True"}


def test_c38_report_run_matches_committed_artifacts():
    res = report.run()
    assert res["n_preference_robust_pairs"] == 114
    assert res["ucl_point_width"]["summary"]["point_dominant_count"] == 111
    assert res["selection_audit_inversion"]["summary"]["audit_prefers_better_count"] == 51
    assert res["source_target_conflict"]["summary"]["source_endpoint_majority_prefers_better_count"] == 57
    assert res["gauge_conflict"]["summary"]["leakage_target_gauge_conflict_count"] == 105
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c38_outputs_do_not_emit_hashes_or_overclaims():
    for root, _, files in os.walk(TABLE_DIR):
        for name in files:
            assert "model_hash" not in open(os.path.join(root, name)).read()
    for name in ("C38_LEAKAGE_UCL_OBJECTIVE_GEOMETRY.md",
                 "C38_LEAKAGE_UCL_OBJECTIVE_GEOMETRY.json"):
        text = open(os.path.join("oaci/reports", name)).read()
        assert "model_hash" not in text
        assert "deployable selector" not in text.lower()
        assert "oaci rescue" not in text.lower()


def test_c38_report_guard_blocks_affirmative_overclaim():
    try:
        report._guard_forbidden("target-free detector succeeds")
        raise AssertionError("guard failed")
    except ValueError:
        pass
    report._guard_forbidden("not a target-free detector; no OACI rescue is claimed.")
