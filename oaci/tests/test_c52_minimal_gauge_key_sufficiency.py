"""C52 Minimal Gauge-Key / Conditioning-Sufficiency Audit tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.conditioned_ceiling_coverage import c52_minimal_gauge_key_sufficiency as c52
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C52_MINIMAL_GAUGE_KEY_SUFFICIENCY.json"
TABLE_DIR = "oaci/reports/c52_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c52_config_taxonomy_and_inputs_are_frozen():
    assert c52._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert c52.MILESTONE == "C52"
    assert c52.NULL_REPS == 64
    assert c52.NULL_SEED == 52052
    assert c52.GAP_CLOSE_GATE == 0.05
    assert c52.EVALUATION_SCOPE == "within_trajectory_top_hit"
    assert set(c52.DECISIONS) == {
        "C52-A_source_observable_sufficiency",
        "C52-B_target_key_sufficiency",
        "C52-C_trajectory_key_sufficiency",
        "C52-D_additive_target_plus_trajectory_sufficiency",
        "C52-E_target_x_trajectory_interaction_required",
        "C52-F_target_unlabeled_geometry_sufficiency",
        "C52-G_diagnostic_label_content_required",
        "C52-H_mixed_key_interaction_and_label_content",
    }


def test_c52_decision_requires_label_derived_diagnostic_content():
    d = _summary()
    assert d["milestone"] == "C52"
    assert d["inherits_from"] == ["C49", "C50", "C51"]
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    dec = d["decision"]
    assert dec["decision"] == "C52-G_diagnostic_label_content_required"
    assert dec["decision"] in c52.DECISIONS
    assert dec["source_observable_closes_gap"] is False
    assert dec["target_key_only_closes_gap"] is False
    assert dec["trajectory_key_only_closes_gap"] is False
    assert dec["additive_target_plus_trajectory_key_closes_gap"] is False
    assert dec["target_x_trajectory_key_only_closes_gap"] is False
    assert dec["target_unlabeled_geometry_closes_gap"] is False
    assert dec["label_derived_diagnostic_closes_gap"] is True
    assert dec["best_strict_source_hit"] == 0.5061728395061729
    assert dec["best_key_only_hit"] == 0.4876543209876543
    assert dec["best_label_derived_hit"] == 0.8127572016460904
    assert dec["trajectory_centered_diagnostic_hit"] == 0.8127572016460904
    assert dec["best_key_only_gap_to_c51_oracle"] == 0.3209876543209877


def test_c52_replays_c51_residual_caveat_numbers():
    d = _summary()
    replay = d["c51_residual_replay"]
    assert replay["decision"] == "C51-E_target_trajectory_gauge_residual"
    assert replay["max_raw_underuse_gap"] == 0.40123456790123463
    assert replay["best_trajectory_centered_gap"] == -0.00411522633744843
    assert replay["n2_fail_fraction_percentile"] == 0.0
    assert replay["n3_fail_fraction_percentile"] == 0.0
    assert replay["n4_enrichment_null_mean"] == 0.8342243865476644
    assert replay["observed_enrichment"] == 2.3597078708023864
    assert d["c51_oracle_trajectory_local_bayes_hit"] == 0.808641975308642
    assert d["trajectory_conditioned_random_tie_hit"] == 0.429723378036041
    geom = d["source_geometry_best_field"]
    assert geom["best_field"] == "source_distance_q25"
    assert geom["best_orientation"] == "high"
    assert geom["best_hit"] == 0.4876543209876543


def test_c52_ladder_separates_key_only_from_label_diagnostic_rows():
    rows = {r["candidate"]: r for r in _rows("conditioning_ladder_summary.csv")}
    assert len(rows) == 13
    assert rows["existing_source_best_raw"]["comparison_source"] == "C30_source_rank"
    assert float(rows["existing_source_best_raw"]["hit"]) == 0.5061728395061729
    assert float(rows["source_metadata_key_only_tie"]["hit"]) == 0.429723378036041
    assert float(rows["source_geometry_key_only_best_observable"]["hit"]) == 0.4876543209876543
    assert rows["source_geometry_key_only_best_observable"]["comparison_source"] == "source_distance_q25:high"
    for name in (
        "target_id_key_only",
        "trajectory_id_key_only",
        "additive_target_plus_trajectory_key_only",
        "target_x_trajectory_key_only",
    ):
        assert rows[name]["key_only"] == "1"
        assert rows[name]["target_label_derived"] == "0"
        assert rows[name]["closes_gap"] == "0"
        assert float(rows[name]["hit"]) == 0.429723378036041
    assert rows["target_centered_label_diagnostic"]["target_label_derived"] == "1"
    assert rows["target_centered_label_diagnostic"]["closes_gap"] == "0"
    assert float(rows["target_centered_label_diagnostic"]["hit"]) == 0.5648036301631072
    assert rows["trajectory_centered_label_diagnostic"]["target_label_derived"] == "1"
    assert rows["trajectory_centered_label_diagnostic"]["closes_gap"] == "1"
    assert float(rows["trajectory_centered_label_diagnostic"]["hit"]) == 0.8127572016460904
    assert rows["target_x_trajectory_label_diagnostic"]["closes_gap"] == "1"
    assert rows["target_unlabeled_geometry"]["available"] == "0"


def test_c52_decomposition_reports_key_and_label_content_separately():
    rows = {r["component"]: r for r in _rows("gauge_key_decomposition.csv")}
    assert len(rows) == 10
    assert rows["best_existing_source_score"]["target_label_derived"] == "0"
    assert rows["best_source_observable_geometry"]["key_only"] == "1"
    assert rows["best_source_observable_geometry"]["closes_gap"] == "0"
    assert float(rows["best_source_observable_geometry"]["increment_vs_best_raw_source"]) == -0.018518518518518545
    assert rows["target_id_key_only"]["key_only"] == "1"
    assert rows["trajectory_id_key_only"]["key_only"] == "1"
    assert rows["target_x_trajectory_key_only"]["key_only"] == "1"
    assert rows["trajectory_centered_label_content"]["target_label_derived"] == "1"
    assert rows["trajectory_centered_label_content"]["closes_gap"] == "1"
    assert rows["target_x_trajectory_label_content"]["target_label_derived"] == "1"
    assert rows["target_unlabeled_geometry"]["available"] == "0"


def test_c52_key_nulls_emit_n5_n6_and_n7_quarantine():
    rows = _rows("key_null_calibration_summary.csv")
    assert len(rows) == 6
    assert {r["null_name"] for r in rows} == {
        "N5_key_only_identity_tie_null",
        "N6_source_geometry_score_shuffle",
        "N7_label_derived_diagnostic_quarantine",
    }
    n5 = [r for r in rows if r["null_name"] == "N5_key_only_identity_tie_null"]
    assert {r["key_or_field"] for r in n5} == {
        "target_id", "trajectory_id", "additive_target_plus_trajectory", "target_x_trajectory"
    }
    assert {r["status"] for r in n5} == {"analytical_key_tie"}
    assert {float(r["observed_hit"]) for r in n5} == {0.429723378036041}
    n6 = next(r for r in rows if r["null_name"] == "N6_source_geometry_score_shuffle")
    assert n6["key_or_field"] == "source_distance_q25:high"
    assert float(n6["observed_hit"]) == 0.4876543209876543
    assert float(n6["null_mean"]) == 0.427758487654321
    assert float(n6["percentile"]) == 0.96875
    assert int(n6["n_permutations"]) == 64
    n7 = next(r for r in rows if r["null_name"] == "N7_label_derived_diagnostic_quarantine")
    assert n7["status"] == "unavailable_for_key_only_null"
    assert n7["key_or_field"] == "trajectory_centered_label_diagnostic"
    assert float(n7["observed_hit"]) == 0.8127572016460904


def test_c52_cell_ledger_and_reason_codes_quantify_residual():
    cells = _rows("target_trajectory_cell_ledger.csv")
    assert len(cells) == 162
    assert sum(int(r["key_only_closes_cell"]) for r in cells) == 12
    assert sum(int(r["label_diagnostic_closes_cell"]) for r in cells) == 131
    assert {r["target_labels_diagnostic_only"] for r in cells} == {"1"}
    assert {r["no_selection_artifact"] for r in cells} == {"1"}
    reasons = {r["primary_reason"]: sum(x["primary_reason"] == r["primary_reason"] for x in cells)
               for r in cells}
    assert reasons == {
        "label_content_required_after_key_tie": 97,
        "source_score_underuses_diagnostic_island": 34,
        "trajectory_fragmented_low_local_bayes": 31,
    }
    first = next(r for r in cells if r["trajectory_id"] == "0|1|0|S0_full_support")
    assert float(first["base_hit"]) == 0.05
    assert float(first["local_bayes_hit"]) == 0.0
    assert first["key_only_closes_cell"] == "0"
    assert first["label_diagnostic_closes_cell"] == "0"

    ledger = {r["reason_code"]: r for r in _rows("residual_failure_reason_ledger.csv")}
    assert len(ledger) == 4
    assert int(ledger["KEY_ONLY_HAS_NO_WITHIN_TRAJECTORY_RANK"]["n_trajectories"]) == 162
    assert int(ledger["SOURCE_SCORE_UNDERUSES_DIAGNOSTIC_ISLAND"]["n_trajectories"]) == 34
    assert int(ledger["LOW_TRAJECTORY_LOCAL_BAYES"]["n_trajectories"]) == 31
    assert int(ledger["LABEL_DERIVED_DIAGNOSTIC_CLOSES_RESIDUAL"]["n_trajectories"]) == 131


def test_c52_tables_have_expected_shape_and_gates():
    d = _summary()
    assert d["table_row_counts"] == {
        "conditioning_ladder_summary": 13,
        "gauge_key_decomposition": 10,
        "key_null_calibration_summary": 6,
        "no_selector_artifact_gate": 10,
        "red_team_verification": 8,
        "residual_failure_reason_ledger": 4,
        "target_trajectory_cell_ledger": 162,
    }
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    red = {r["check"]: r["passed"] for r in _rows("red_team_verification.csv")}
    assert red and set(red.values()) == {"1"}
    assert "conditioning_ladder_rows" not in d
    assert os.path.getsize(REPORT_JSON) < 50_000


def test_c52_run_loads_committed_artifacts_without_recompute():
    res = c52.run()
    assert res["decision"]["decision"] == "C52-G_diagnostic_label_content_required"
    assert res["n_candidate_rows"] == 3804
    assert len(res["target_trajectory_cell_ledger_rows"]) == 162
    assert all(g["passed"] for g in c52.no_selector_gate(res))


def test_c52_outputs_do_not_emit_selector_artifacts_or_forbidden_claims():
    for name in (
        "conditioning_ladder_summary.csv",
        "gauge_key_decomposition.csv",
        "key_null_calibration_summary.csv",
        "target_trajectory_cell_ledger.csv",
        "residual_failure_reason_ledger.csv",
    ):
        text = open(os.path.join(TABLE_DIR, name)).read().lower()
        assert "model_hash" not in text
        assert "checkpoint_hash" not in text
        assert "selected_candidate_id" not in text
        assert "checkpoint recommendation" not in text
    forbidden = (
        "deployable selector",
        "oaci rescue",
        "source-only rescue",
        "target-unlabeled method",
        "target-grouped method",
        "trajectory-conditioned method",
        "checkpoint recommendation",
        "production rule",
        "action rule",
        "usable policy",
    )
    for name in ("C52_MINIMAL_GAUGE_KEY_SUFFICIENCY.md",
                 "C52_MINIMAL_GAUGE_KEY_SUFFICIENCY.json",
                 "C52_RED_TEAM_VERIFICATION.md"):
        text = open(os.path.join("oaci/reports", name)).read().lower()
        for phrase in forbidden:
            assert phrase not in text
    md = open("oaci/reports/C52_MINIMAL_GAUGE_KEY_SUFFICIENCY.md").read()
    assert "C52-G_diagnostic_label_content_required" in md
    assert "key availability from label-derived diagnostic content" in md
