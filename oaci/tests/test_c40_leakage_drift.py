"""C40 Leakage Point Drift Forensics / Atom-Trace Boundary tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.leakage_drift import report, schema


REPORT_JSON = "oaci/reports/C40_LEAKAGE_POINT_DRIFT_FORENSICS.json"
TABLE_DIR = "oaci/reports/c40_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c40_config_and_constants_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.ACTUAL_SELECTOR_SCORE_NAME == "selection_bootstrap_ucl"
    assert schema.POINT_IDENTITY_TOL == 1e-9
    assert schema.TOLERANCE_LADDER == (1e-9, 1e-8, 1e-6, 1e-4, 1e-3)
    assert schema.BOUNDED_DRIFT_TOL == 1e-3


def test_c40_summary_taxonomy_and_identity_boundary():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_preference_robust_pairs"] == 114
    manifest = d["leakage_drift_manifest_summary"]
    assert manifest["n_selection_candidates"] == 76
    assert manifest["n_selection_pass_1e_9"] == 48
    assert manifest["selection_identity_pass"] is False
    assert manifest["max_abs_drift"] == 0.00021521578246364026
    assert manifest["all_additive_pass"] is True
    cases = set(d["taxonomy"]["cases"])
    assert cases == {schema.D2, schema.D4, schema.D5, schema.D6, schema.D7}
    assert schema.D1 not in cases
    assert schema.D8 not in cases


def test_c40_stagewise_and_numeric_diagnostics():
    d = _summary()
    stage = d["stagewise_drift_localization_summary"]
    assert stage["selection_first_divergent_stage_counts"] == {
        "none": 48,
        "persisted_aggregate_point_identity": 28,
    }
    assert stage["observed_semantic_mismatch_count"] == 0
    assert stage["aggregate_vs_atom_path_divergence_count"] == 28
    numeric = d["numeric_drift_diagnostics_summary"]
    assert numeric["bounded_at_1e_3"] is True
    assert numeric["numeric_only_not_proven_due_to_missing_per_fold_trace"] is True
    assert numeric["positive_signed_drift_count"] == 40
    assert numeric["negative_signed_drift_count"] == 36


def test_c40_tolerance_ladder_and_blocked_stability():
    d = _summary()
    ladder = d["tolerance_ladder_identity_summary"]
    assert ladder["pass_counts"] == {
        "1e-09": 48,
        "1e-08": 57,
        "1e-06": 68,
        "0.0001": 75,
        "0.001": 76,
    }
    assert ladder["all_pass_at_1e_3"] is True
    assert ladder["all_pass_at_frozen_1e_9"] is False
    stability = d["atom_pattern_stability_under_drift_summary"]
    assert stability["point_sign_stable_fraction"] == 1.0
    assert stability["pattern_claims_elevated"] is False
    assert stability["broad_diagnostic_count"] == 108
    assert stability["atom_gauge_conflict_diagnostic_count"] == 105


def test_c40_tables_have_expected_shape_and_gates():
    assert len(_rows("leakage_drift_manifest.csv")) == 152
    assert len(_rows("selection_vs_audit_identity_contrast.csv")) == 76
    assert len(_rows("stagewise_drift_localization.csv")) == 152
    assert len(_rows("numeric_drift_diagnostics.csv")) == 5
    assert len(_rows("tolerance_ladder_identity.csv")) == 5
    assert len(_rows("atom_pattern_stability_under_drift.csv")) == 114
    assert len(_rows("aggregate_vs_atom_path_diff.csv")) == 76
    assert len(_rows("future_trace_field_requirements.csv")) == 12
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    tax = {r["case"]: r["established"] for r in _rows("c40_case_taxonomy.csv")}
    assert tax[schema.D1] == "0"
    assert tax[schema.D8] == "0"
    assert tax[schema.D6] == "1"
    assert tax[schema.D7] == "1"


def test_c40_report_run_loads_committed_artifacts_without_recompute():
    res = report.run()
    assert res["taxonomy"]["cases"] == [schema.D2, schema.D4, schema.D5, schema.D6, schema.D7]
    assert res["leakage_drift_manifest"]["summary"]["n_selection_pass_1e_9"] == 48
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c40_outputs_do_not_emit_hashes_or_overclaims():
    for root, _, files in os.walk(TABLE_DIR):
        for name in files:
            text = open(os.path.join(root, name)).read()
            assert "model_hash" not in text
    for name in ("C40_LEAKAGE_POINT_DRIFT_FORENSICS.md",
                 "C40_LEAKAGE_POINT_DRIFT_FORENSICS.json",
                 "C40_ATOM_TRACE_BOUNDARY.md",
                 "C40_FUTURE_LEAKAGE_TRACE_INSTRUMENTATION.md"):
        text = open(os.path.join("oaci/reports", name)).read()
        low = text.lower()
        assert "model_hash" not in text
        assert "deployable selector" not in low
        assert "oaci rescue" not in low
        assert "target-free detector" not in low
        assert "small drift therefore acceptable" not in low
    md = open("oaci/reports/C40_LEAKAGE_POINT_DRIFT_FORENSICS.md").read()
    assert "These diagnostic patterns remain blocked" in md
    assert "exact identity is not restored" in md


def test_c40_report_guard_blocks_affirmative_overclaim():
    try:
        report._guard_forbidden("small drift therefore acceptable")
        raise AssertionError("guard failed")
    except ValueError:
        pass
    report._guard_forbidden("not small drift therefore acceptable; atom claims remain blocked.")
