"""C37 Exact Selector Trace Recovery / Leakage-UCL Audit tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.selector_trace_recovery import report, schema


REPORT_JSON = "oaci/reports/C37_EXACT_SELECTOR_TRACE_RECOVERY.json"
TABLE_DIR = "oaci/reports/c37_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c37_constants_and_config_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.ACTUAL_SELECTOR_SCORE_NAME == "selection_bootstrap_ucl"
    assert schema.UCL_IDENTITY_TOL == 1e-9
    assert schema.POINT_IDENTITY_TOL == 1e-9
    assert schema.UCL_CLEAR_EPS == 1e-9
    assert schema.UCL_PLATEAU_EPS == 0.02


def test_c37_committed_summary_and_taxonomy():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_preference_robust_pairs"] == 114
    assert d["n_unique_better_candidates"] == 38
    assert d["selected_ucl_identity_summary"]["p0_pass"] is True
    assert d["selected_ucl_identity_summary"]["n_pass"] == 3
    assert d["better_candidate_ucl_summary"]["all_recovered"] is True
    assert d["better_candidate_ucl_summary"]["n_recovered"] == 38
    assert d["exact_ucl_ordering_summary"]["ucl_prefers_selected_count"] == 114
    assert d["exact_ucl_ordering_summary"]["ucl_prefers_better_count"] == 0
    assert d["exact_ucl_ordering_summary"]["ucl_flat_count"] == 0
    assert d["selection_audit_reconciliation_summary"]["selection_audit_inversion_exact_rate"] == (
        0.4473684210526316)
    assert d["source_pareto_after_ucl_summary"]["source_pareto_conflict_fraction"] == 1.0
    cases = set(d["taxonomy"]["cases"])
    assert {schema.T1, schema.T5, schema.T6, schema.T8}.issubset(cases)
    assert schema.T2 not in cases
    assert schema.T3 not in cases
    assert schema.T7 not in cases
    assert schema.T9 not in cases


def test_c37_tables_have_expected_counts_and_gates():
    assert len(_rows("selected_ucl_identity_gate.csv")) == 3
    assert len(_rows("better_candidate_ucl_recovery.csv")) == 38
    assert len(_rows("selected_vs_better_exact_ucl.csv")) == 114
    manifest = _rows("selector_trace_recovery_manifest.csv")
    assert len(manifest) == 38
    assert sum(int(r["store_exists"]) for r in manifest) == 38
    assert sum(int(r["selection_design_available"]) for r in manifest) == 38
    assert sum(int(r["selection_fold_plan_available"]) for r in manifest) == 38
    assert sum(int(r["selection_bootstrap_plan_available"]) for r in manifest) == 38
    assert sum(int(r["support_graph_available"]) for r in manifest) == 38
    assert sum(r["better_source_train_feature_available"] == "1" for r in manifest) == 38
    assert sum(r["selected_source_train_feature_available"] == "1" for r in manifest) == 3
    assert sum(int(r["target_labels_loaded_for_replay"]) for r in manifest) == 0
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"True"}


def test_c37_outputs_do_not_emit_hashes_or_selector_artifacts():
    for root, _, files in os.walk(TABLE_DIR):
        for name in files:
            text = open(os.path.join(root, name)).read()
            assert "model_hash" not in text
    for name in ("C37_EXACT_SELECTOR_TRACE_RECOVERY.md", "C37_EXACT_SELECTOR_TRACE_RECOVERY.json"):
        text = open(os.path.join("oaci/reports", name)).read()
        assert "model_hash" not in text
        assert "deployable selector" not in text.lower()
        assert "oaci rescue" not in text.lower()


def test_c37_report_guard_blocks_affirmative_overclaim():
    try:
        report._guard_forbidden("deployable selector works")
        raise AssertionError("guard failed")
    except ValueError:
        pass
    report._guard_forbidden("not a deployable selector; no OACI rescue is claimed.")


def test_c37_make_worklists_is_compact_and_hash_free(tmp_path):
    meta = report.make_worklists(str(tmp_path))
    assert meta["n_preference_robust_pairs"] == 114
    assert meta["n_selected_p0_jobs"] == 3
    assert meta["n_unique_better_jobs"] == 38
    assert meta["candidate_hash_emitted"] is False
    text = "".join(open(tmp_path / f).read() for f in ("selected_worklist.csv", "better_worklist.csv",
                                                       "worklist_metadata.json"))
    assert "model_hash" not in text
