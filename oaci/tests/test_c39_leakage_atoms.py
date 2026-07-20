"""C39 Leakage Atom Recovery / Support-Cell Conflict Audit tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.leakage_atoms import report, schema


REPORT_JSON = "oaci/reports/C39_LEAKAGE_ATOM_RECOVERY_AUDIT.json"
TABLE_DIR = "oaci/reports/c39_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c39_config_and_constants_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.ACTUAL_SELECTOR_SCORE_NAME == "selection_bootstrap_ucl"
    assert schema.POINT_IDENTITY_TOL == 1e-9
    assert schema.ATOM_ADDITIVE_TOL == 1e-9
    assert schema.CONCENTRATED_TOP3_SHARE_GATE == 0.75
    assert schema.BROAD_MIN_POSITIVE_ATOMS == 8


def test_c39_summary_taxonomy_and_gate_blocking():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_preference_robust_pairs"] == 114
    ident = d["selected_atom_identity_summary"]
    assert ident["n_selection_candidates"] == 76
    assert ident["n_selection_identity_pass"] == 48
    assert ident["selection_identity_pass"] is False
    assert ident["n_source_audit_additive_pass"] == 76
    assert ident["source_audit_additive_pass"] is True
    assert ident["max_selection_point_abs_diff"] == 0.00021521578246364026
    assert ident["max_selection_additive_abs_diff"] <= 1e-12
    cases = set(d["taxonomy"]["cases"])
    assert cases == {schema.A9, schema.A10}
    gates = {r["check"]: r["passed"] for r in d["no_selector_artifact_gate"]}
    assert gates["identity_failure_blocks_atom_contribution_claims"] is True
    assert set(gates.values()) == {True}


def test_c39_diagnostic_summaries_are_boundary_marked():
    d = _summary()
    conc = d["atom_concentration_summary"]
    assert conc["n_pairs"] == 114
    assert conc["broad_pair_count"] == 108
    assert conc["concentrated_pair_count"] == 0
    aud = d["selection_audit_atom_stability_summary"]
    assert aud["selection_to_audit_inversion_rate"] == 0.4473684210526316
    assert aud["mean_atom_sign_preservation_rate"] == 0.5187969924812029
    support = d["support_cell_artifact_summary"]
    assert support["support_artifact_pair_fraction"] == 0.0
    boot = d["bootstrap_atom_diagnostics_summary"]
    assert boot["point_atom_additive_identity_exact"] is True
    assert boot["persisted_point_identity_pass"] is False
    assert boot["replicate_atom_replay_available"] is False
    assert boot["per_atom_ucl_summed"] is False
    assert boot["ucl_quantile_atom_limit"] is True
    gauge = d["atom_target_gauge_conflict_summary"]
    assert gauge["atom_target_gauge_conflict_count"] == 105
    assert gauge["atom_target_gauge_conflict_fraction"] == 0.9210526315789473


def test_c39_tables_have_expected_shape_and_boundaries():
    assert len(_rows("atom_recovery_availability.csv")) == 38
    assert len(_rows("selected_atom_identity_gate.csv")) == 152
    assert len(_rows("selected_vs_better_point_atoms.csv")) == 2673
    assert len(_rows("atom_concentration_summary.csv")) == 114
    assert len(_rows("class_domain_atom_contributions.csv")) == 34
    assert len(_rows("selection_audit_atom_stability.csv")) == 114
    assert len(_rows("support_cell_artifact_audit.csv")) == 114
    assert len(_rows("bootstrap_atom_diagnostics.csv")) == 114
    assert len(_rows("atom_target_gauge_conflict.csv")) == 114
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    tax = {r["case"]: r["established"] for r in _rows("c39_case_taxonomy.csv")}
    assert tax[schema.A9] == "1"
    assert tax[schema.A10] == "1"
    assert tax[schema.A1] == "0"


def test_c39_report_run_loads_committed_artifacts_without_recompute():
    res = report.run()
    assert res["taxonomy"]["cases"] == [schema.A9, schema.A10]
    assert res["selected_atom_identity_gate"]["summary"]["n_selection_identity_pass"] == 48
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c39_outputs_do_not_emit_model_hashes_or_overclaims():
    for root, _, files in os.walk(TABLE_DIR):
        for name in files:
            text = open(os.path.join(root, name)).read()
            assert "model_hash" not in text
    for name in ("C39_LEAKAGE_ATOM_RECOVERY_AUDIT.md",
                 "C39_LEAKAGE_ATOM_RECOVERY_AUDIT.json",
                 "C39_SELECTION_AUDIT_ATOM_STABILITY.md",
                 "C39_LEAKAGE_ATOM_TARGET_GAUGE_CONFLICT.md"):
        text = open(os.path.join("oaci/reports", name)).read()
        low = text.lower()
        assert "model_hash" not in text
        assert "deployable selector" not in low
        assert "oaci rescue" not in low
    md = open("oaci/reports/C39_LEAKAGE_ATOM_RECOVERY_AUDIT.md").read()
    assert "Point Atom Diagnostics (Blocked)" in md
    assert "not elevated atom contribution claims" in md


def test_c39_report_guard_blocks_affirmative_overclaim():
    try:
        report._guard_forbidden("target-free detector succeeds")
        raise AssertionError("guard failed")
    except ValueError:
        pass
    report._guard_forbidden("not a target-free detector; no OACI rescue is claimed.")
