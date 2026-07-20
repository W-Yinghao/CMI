"""C41 Global Leakage-Target Utility Objective Field Audit tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.objective_field import report, schema


REPORT_JSON = "oaci/reports/C41_LEAKAGE_TARGET_OBJECTIVE_FIELD.json"
TABLE_DIR = "oaci/reports/c41_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c41_config_and_selector_boundaries_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.ACTUAL_SELECTOR_SCORE_NAME == "selection_bootstrap_ucl"
    assert schema.ALIGNMENT_AUC_LOW == 0.45
    assert schema.ALIGNMENT_AUC_HIGH == 0.55
    assert schema.LOCAL_REPRESENTATIVE_GATE == 0.8


def test_c41_summary_taxonomy_is_conservative():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    assert set(d["taxonomy"]["cases"]) == {schema.O2, schema.O5, schema.O6}
    assert schema.O4 not in d["taxonomy"]["cases"]
    assert schema.O8 not in d["taxonomy"]["cases"]
    assert schema.O10 not in d["taxonomy"]["cases"]


def test_c41_key_numbers_and_field_alignment():
    d = _summary()
    by_field = {r["field"]: r for r in d["leakage_target_alignment_summary_rows"]}
    assert by_field["selection_leakage_point"]["n_trajectories"] == 162
    assert by_field["selection_leakage_point"]["mean_pairwise_auc"] == 0.49373160501299335
    assert by_field["audit_leakage_point"]["mean_pairwise_auc"] == 0.4981768980759732
    assert d["source_audit_vs_selection_summary"]["audit_mean_auc_minus_selection"] == 0.00444529306297986
    assert d["source_audit_vs_selection_summary"]["source_audit_no_better"] is True
    comp = d["objective_field_comparison_summary"]
    assert comp["best_field"] == "C30_source_rank_score"
    assert comp["c30_source_rank_auc"] == 0.658600107451297
    assert comp["c30_rank_better_than_selection_leakage"] is True


def test_c41_enrichment_and_local_global_gates_stay_below_overclaim_thresholds():
    d = _summary()
    enrich = {(r["selection_rule"], r["label"]): r for r in d["low_leakage_enrichment_summary_rows"]}
    assert enrich[("top3", "primary_joint_good")]["mean_enrichment_ratio"] == 0.9182126539374492
    assert enrich[("top3", "preference_robust_better_candidate")]["mean_enrichment_ratio"] == 2.412280701754386
    assert all(r["significant_enriched_trajectories_bonferroni"] == 0
               for r in d["low_leakage_enrichment_summary_rows"])
    local = d["local_global_conflict_summary"]
    assert local["n_pairs"] == 114
    assert local["local_conflict_count"] == 102
    assert local["representative_fraction"] == 0.7894736842105263
    assert local["representative_fraction"] < schema.LOCAL_REPRESENTATIVE_GATE
    assert local["tail_only_fraction"] == 0.0


def test_c41_tables_have_expected_shape_and_gates():
    assert len(_rows("objective_field_availability.csv")) == 17
    assert len(_rows("candidate_objective_field_registry.csv")) == 3804
    assert len(_rows("leakage_target_rank_alignment.csv")) == 1539
    assert len(_rows("low_leakage_enrichment.csv")) == 2430
    assert len(_rows("objective_field_comparison.csv")) == 14
    assert len(_rows("source_audit_vs_selection_leakage_alignment.csv")) == 2
    assert len(_rows("local_global_conflict_consistency.csv")) == 114
    assert len(_rows("target_gauge_vs_leakage_field.csv")) == 114
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    tax = {r["case"]: r["established"] for r in _rows("c41_case_taxonomy.csv")}
    assert tax[schema.O2] == "1"
    assert tax[schema.O5] == "1"
    assert tax[schema.O6] == "1"
    assert tax[schema.O4] == "0"
    assert tax[schema.O8] == "0"


def test_c41_candidate_registry_preserves_robust_alternative_join():
    rows = _rows("candidate_objective_field_registry.csv")
    assert sum(int(r["selected_oaci"]) for r in rows) == 162
    assert sum(int(r["primary_joint_good"]) for r in rows) == 1614
    assert sum(int(r["pareto_good"]) for r in rows) == 519
    assert sum(int(r["preference_robust_better_candidate"]) for r in rows) == 114


def test_c41_report_run_loads_committed_artifacts_without_recompute():
    res = report.run()
    assert res["taxonomy"]["cases"] == [schema.O2, schema.O5, schema.O6]
    assert res["n_candidate_rows"] == 3804
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c41_outputs_do_not_overclaim():
    for name in ("C41_LEAKAGE_TARGET_OBJECTIVE_FIELD.md",
                 "C41_LEAKAGE_TARGET_OBJECTIVE_FIELD.json",
                 "C41_LOW_LEAKAGE_ENRICHMENT_AUDIT.md",
                 "C41_LOCAL_GLOBAL_CONFLICT_AUDIT.md"):
        text = open(os.path.join("oaci/reports", name)).read()
        low = text.lower()
        assert "deployable selector" not in low
        assert "oaci rescue" not in low
        assert "target-free detector" not in low
        assert "atom-level leakage mechanism established" not in low
    md = open("oaci/reports/C41_LEAKAGE_TARGET_OBJECTIVE_FIELD.md").read()
    assert "O4 status: **not active**" in md
    assert "O8 status: **not active**" in md
    assert "representative of the broader leakage-target field" not in md

