"""C42 Source-Rank Actionability / Rank-to-Selector Gap tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.rank_actionability import report, schema


REPORT_JSON = "oaci/reports/C42_SOURCE_RANK_ACTIONABILITY_AUDIT.json"
TABLE_DIR = "oaci/reports/c42_tables"


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c42_config_and_gates_are_frozen():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.SOURCE_RANK_PAIRWISE_SIGNAL_GATE == 0.55
    assert schema.TOP1_RELIABLE_JOINT_GOOD_GATE == 0.70
    assert schema.PLATEAU_EPS == 0.02


def test_c42_summary_taxonomy_closes_escape_hatch():
    d = _summary()
    assert d["diagnostic_only_non_deployable"] is True
    assert d["n_candidate_rows"] == 3804
    assert d["n_trajectories"] == 162
    cases = set(d["taxonomy"]["cases"])
    assert cases == {schema.R1, schema.R2, schema.R3, schema.R6, schema.R7, schema.R8, schema.R9}
    assert schema.R4 not in cases
    assert schema.R5 not in cases
    assert schema.R10 not in cases


def test_c42_pairwise_signal_and_top1_gap_numbers():
    d = _summary()
    gap = {(r["score"], r["selection_rule"], r["label"]): r for r in d["auc_to_topk_gap_summary_rows"]}
    src = gap[("C30_source_rank_score", "top1", "primary_joint_good")]
    leak = gap[("selection_leakage_point", "top1", "primary_joint_good")]
    assert d["c30_source_rank_auc"] == 0.658600107451297
    assert d["c41_selection_leakage_auc"] == 0.49373160501299335
    assert src["mean_pairwise_auc_vs_target_utility"] == 0.5904967774230292
    assert src["mean_hit_rate"] == 0.5061728395061729
    assert src["mean_enrichment_ratio"] == 1.1542183729112576
    assert leak["mean_pairwise_auc_vs_target_utility"] == 0.49373160501299335


def test_c42_top1_vs_oaci_and_random():
    d = _summary()
    top1 = d["diagnostic_top1_summary"]
    src = top1["C30_source_rank_score"]
    oaci = top1["actual_oaci_selector"]
    rand = top1["random_trajectory_conditioned"]
    assert src["top1_joint_good_rate"] == 0.5061728395061729
    assert oaci["top1_joint_good_rate"] == 0.4444444444444444
    assert rand["top1_joint_good_rate"] == 0.4297233780360411
    assert src["top1_joint_good_gain_vs_random"] == 0.07644946147013176
    assert src["top1_regret_vs_target_oracle"] == 0.6568196494026767
    assert src["fraction_top1_target_better_than_actual_oaci"] == 0.5370370370370371


def test_c42_stability_gauge_and_conflict_summaries():
    d = _summary()
    stability = d["source_rank_top_region_stability_summary"]
    assert stability["mean_plateau_size"] == 2.191358024691358
    assert stability["low_margin_fraction"] == 0.5370370370370371
    assert stability["top_region_plateau_or_instability"] is True
    gauge = d["source_rank_gauge_sensitivity_summary"]
    assert gauge["max_centered_top1_joint_good_gain_vs_raw"] == 0.0
    assert gauge["gauge_breaks_source_rank_actionability"] is False
    conflict = d["leakage_vs_rank_conflict_summary"]
    assert conflict["rank_top_target_better_than_oaci_count"] == 87
    assert conflict["leakage_blocks_rank_better_count"] == 87
    assert conflict["leakage_blocks_rank_better_fraction"] == 0.5370370370370371
    assert conflict["target_gauge_delta_available"] is False


def test_c42_tables_have_expected_shape_and_gates():
    assert len(_rows("rank_actionability_score_registry.csv")) == 9
    assert len(_rows("auc_to_topk_gap.csv")) == 72
    assert len(_rows("diagnostic_top1_by_score.csv")) == 7
    assert len(_rows("source_rank_top_region_stability.csv")) == 162
    assert len(_rows("source_rank_gauge_sensitivity.csv")) == 5
    assert len(_rows("leakage_vs_rank_conflict.csv")) == 162
    assert len(_rows("source_rank_regret_vs_oracle.csv")) == 7
    assert len(_rows("trajectory_conditioned_random_baseline.csv")) == 1944
    gates = {r["check"]: r["passed"] for r in _rows("no_selector_artifact_gate.csv")}
    assert gates and set(gates.values()) == {"1"}
    tax = {r["case"]: r["established"] for r in _rows("c42_case_taxonomy.csv")}
    assert tax[schema.R1] == "1"
    assert tax[schema.R9] == "1"
    assert tax[schema.R4] == "0"
    assert tax[schema.R10] == "0"


def test_c42_score_registry_marks_non_source_fields_and_no_proxy():
    rows = {r["score"]: r for r in _rows("rank_actionability_score_registry.csv")}
    assert rows["target_unlabeled_R3"]["candidate_level_available"] == "0"
    assert rows["target_unlabeled_R3"]["proxy_used"] == "0"
    assert rows["target_unlabeled_R3"]["non_source_only"] == "1"
    assert rows["target_grouped_diagnostic_ceiling"]["diagnostic_ceiling"] == "1"
    assert rows["C30_source_rank_score"]["orientation"] == "higher"


def test_c42_conflict_table_omits_checkpoint_artifacts():
    text = open(os.path.join(TABLE_DIR, "leakage_vs_rank_conflict.csv")).read()
    assert "selected_candidate_id" not in text
    assert "rank_top_candidate_id" not in text
    assert "model_hash" not in text
    assert "checkpoint_hash" not in text
    assert "no_candidate_id_emitted" in text


def test_c42_report_run_loads_committed_artifacts_without_recompute():
    res = report.run()
    assert res["taxonomy"]["cases"] == [schema.R1, schema.R2, schema.R3, schema.R6, schema.R7, schema.R8, schema.R9]
    assert res["n_candidate_rows"] == 3804
    assert all(g["passed"] for g in report.no_selector_gate(res))


def test_c42_outputs_do_not_overclaim():
    for name in ("C42_SOURCE_RANK_ACTIONABILITY_AUDIT.md",
                 "C42_SOURCE_RANK_ACTIONABILITY_AUDIT.json",
                 "C42_AUC_TO_SELECTION_GAP.md",
                 "C42_LEAKAGE_VS_RANK_CONFLICT.md"):
        text = open(os.path.join("oaci/reports", name)).read()
        low = text.lower()
        assert "source-rank selector" not in low
        assert "source rank selector" not in low
        assert "oaci rescue" not in low
        assert "target-free detector" not in low
        assert "external validation success" not in low
        assert "target-grouped oracle as method" not in low
    md = open("oaci/reports/C42_SOURCE_RANK_ACTIONABILITY_AUDIT.md").read()
    assert "R9_source_rank_escape_hatch_closed" in md
    assert "R4_source_rank_top1_reliable_diagnostic" not in md.split("**cases: `", 1)[1].split("`**", 1)[0]
