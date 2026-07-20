"""C56 Mechanism Closure / Information-Boundary Review Gauntlet tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.conditioned_ceiling_coverage import c56_mechanism_closure_review_gauntlet as c56
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C56_MECHANISM_CLOSURE_REVIEW_GAUNTLET.json"
TABLE_DIR = "oaci/reports/c56_tables"


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def test_c56_config_and_decision_are_frozen():
    assert c56._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert c56.MILESTONE == "C56"
    assert set(c56.DECISIONS) == {
        "C56-A_mechanism_closed_ready_for_manuscript_scaffold",
        "C56-B_mechanism_closed_but_literature_alignment_incomplete",
        "C56-C_specific_escape_hatch_open_requires_C57",
        "C56-D_artifact_or_claim_inconsistency_requires_repair",
        "C56-E_split_label_extension_required_before_major_claim",
        "C56-F_inconclusive_reopen_exploration",
    }
    d = _summary()
    assert d["milestone"] == "C56"
    assert d["config_hash"] == "664007686afb520f"
    assert d["diagnostic_only_non_deployable"] is True
    dec = d["decision"]
    assert dec["primary"] == "C56-A_mechanism_closed_ready_for_manuscript_scaffold"
    assert dec["mechanism_closed"] is True
    assert dec["c55_null_ambiguity_resolved"] is True
    assert dec["untraceable_key_number_count"] == 0
    assert dec["red_team_failure_count"] == 0
    assert dec["recommended_next_direction"] == "manuscript/theory scaffold"


def test_c56_table_shapes_cover_required_review_package():
    d = _summary()
    assert d["table_row_counts"] == {
        "artifact_manifest": 32,
        "availability_ledger": 10,
        "c55_null_provenance": 8,
        "claim_support_matrix": 6,
        "claim_to_literature_map": 5,
        "closed_escape_hatches": 8,
        "forbidden_claim_scan": 9,
        "forbidden_literature_overclaims": 4,
        "information_class_ladder": 8,
        "key_number_provenance": 35,
        "large_artifact_scan": 32,
        "literature_alignment_matrix": 5,
        "mechanism_edges": 7,
        "mechanism_nodes": 10,
        "milestone_evidence_ledger": 42,
        "non_theorem_guardrails": 4,
        "open_caveats": 4,
        "red_team_failure_ledger": 10,
        "reviewer_answer_evidence_map": 12,
        "reviewer_question_bank": 12,
        "schema_validation_summary": 25,
        "sufficiency_boundary_matrix": 8,
        "target_label_use_ledger": 10,
        "taxonomy_timeline": 144,
        "test_command_manifest": 4,
        "theory_candidate_statements": 3,
        "validation_timeline": 42,
    }
    required = {
        "milestone_evidence_ledger.csv",
        "key_number_provenance.csv",
        "taxonomy_timeline.csv",
        "validation_timeline.csv",
        "mechanism_nodes.csv",
        "mechanism_edges.csv",
        "claim_support_matrix.csv",
        "closed_escape_hatches.csv",
        "open_caveats.csv",
        "information_class_ladder.csv",
        "sufficiency_boundary_matrix.csv",
        "theory_candidate_statements.csv",
        "non_theorem_guardrails.csv",
        "literature_alignment_matrix.csv",
        "claim_to_literature_map.csv",
        "forbidden_literature_overclaims.csv",
        "availability_ledger.csv",
        "target_label_use_ledger.csv",
        "c55_null_provenance.csv",
        "forbidden_claim_scan.csv",
        "red_team_failure_ledger.csv",
        "reviewer_question_bank.csv",
        "reviewer_answer_evidence_map.csv",
        "artifact_manifest.csv",
        "test_command_manifest.csv",
        "schema_validation_summary.csv",
        "large_artifact_scan.csv",
    }
    assert required == {p for p in os.listdir(TABLE_DIR) if p.endswith(".csv")}


def test_c56_key_number_provenance_traces_main_report_numbers():
    rows = {r["provenance_id"]: r for r in _rows("key_number_provenance.csv")}
    assert len(rows) == 35
    expected = {
        "K_C31_joint_good_rate": 0.4242902208201893,
        "K_C42_source_rank_top1_joint": 0.5061728395061729,
        "K_C43_best_source_scalarization_top1": 0.5740740740740741,
        "K_C46_cross_target_q10": 0.9369369369369369,
        "K_C48_local_ceiling_hit": 1.0,
        "K_C50_trajectory_fail_fraction": 0.43209876543209874,
        "K_C52_best_key_only_hit": 0.4876543209876543,
        "K_C52_best_label_derived_hit": 0.8127572016460904,
        "K_C55_template_only_best": 0.7037037037037037,
        "K_C55_endpoint_scalar_transfer": 0.9444444444444444,
        "K_C55_same_minus_template_gap": 0.2407407407407407,
        "K_C55_max_null_p95": 0.7712962962962961,
    }
    for key, value in expected.items():
        assert key in rows
        assert float(rows[key]["value"]) == value
        assert rows[key]["trace_status"] == "verified"
        assert rows[key]["artifact"].startswith("oaci/reports/")
    md = open("oaci/reports/C56_MECHANISM_CLOSURE_REVIEW_GAUNTLET.md").read()
    for key in expected:
        if key in {
            "K_C31_joint_good_rate",
            "K_C42_source_rank_top1_joint",
            "K_C43_best_source_scalarization_top1",
            "K_C46_cross_target_q10",
            "K_C48_local_ceiling_hit",
            "K_C50_trajectory_fail_fraction",
            "K_C52_best_key_only_hit",
            "K_C52_best_label_derived_hit",
            "K_C55_template_only_best",
            "K_C55_endpoint_scalar_transfer",
            "K_C55_same_minus_template_gap",
            "K_C55_max_null_p95",
        }:
            assert key in md


def test_c56_c55_null_provenance_resolves_ambiguity():
    d = _summary()
    assert d["c55_null_disambiguation"] == {
        "endpoint_scalar_transfer_beats_max_null_p95": True,
        "endpoint_scalar_transfer_hit": 0.9444444444444444,
        "max_null_p95": 0.7712962962962961,
        "template_only_beats_max_null_p95": False,
        "template_only_hit": 0.7037037037037037,
    }
    rows = {(r["observed_statistic"], r["null_family"]): r for r in _rows("c55_null_provenance.csv")}
    endpoint = rows[("C55_endpoint_scalar_transfer", "max_over_C55_transfer_nulls")]
    template = rows[("C55_template_only_best", "max_over_C55_transfer_nulls")]
    assert float(endpoint["observed_hit"]) == 0.9444444444444444
    assert float(endpoint["null_p95"]) == 0.7712962962962961
    assert endpoint["comparison_pass"] == "1"
    assert float(template["observed_hit"]) == 0.7037037037037037
    assert float(template["null_p95"]) == 0.7712962962962961
    assert template["comparison_pass"] == "0"
    assert "not claimed" in template["claim_allowed"]
    assert rows[("C55_endpoint_scalar_transfer", "N5_trajectory_block_shuffle")]["comparison_pass"] == "1"


def test_c56_availability_ledger_blocks_source_available_endpoint_closure():
    rows = {r["score_or_claim"]: r for r in _rows("availability_ledger.csv")}
    assert rows["best_strict_source"]["available_under_original_source_only_DG"] == "1"
    assert rows["best_strict_source"]["diagnostic_only"] == "0"
    assert rows["best_key_only"]["uses_key_only_inputs"] == "1"
    assert rows["best_key_only"]["uses_target_grouped_inputs"] == "1"
    assert rows["best_key_only"]["available_under_original_source_only_DG"] == "0"
    assert rows["C55_template_only_best"]["uses_other_cell_target_labels"] == "1"
    assert rows["C55_template_only_best"]["uses_target_endpoint_scalar_on_test_candidate"] == "0"
    assert rows["C55_template_only_best"]["diagnostic_only"] == "1"
    endpoint = rows["C55_endpoint_scalar_transfer"]
    assert endpoint["uses_target_endpoint_scalar_on_test_candidate"] == "1"
    assert endpoint["uses_other_cell_target_labels"] == "1"
    assert endpoint["available_under_original_source_only_DG"] == "0"
    assert endpoint["diagnostic_only"] == "1"
    split = rows["split_label_constructed_endpoint_template"]
    assert split["uses_trial_level_split_labels"] == "1"
    assert split["reported_hit_or_value"] == "unavailable"


def test_c56_information_boundary_and_theory_guardrails_are_non_theorem():
    ladder = {r["information_class"]: r for r in _rows("information_class_ladder.csv")}
    assert set(ladder) == {
        "I0_random_or_tie",
        "I1_strict_source_observables",
        "I2_source_plus_target_or_trajectory_keys",
        "I3_target_unlabeled_transductive_geometry",
        "I4_target_grouped_zero_label_structure",
        "I5_few_label_or_split_label_calibration",
        "I6_target_label_diagnostic_content",
        "I7_same_label_endpoint_oracle",
    }
    assert ladder["I1_strict_source_observables"]["sufficiency_boundary"] == "not reliable"
    assert ladder["I5_few_label_or_split_label_calibration"]["empirical_status"] == "not evaluated"
    assert ladder["I7_same_label_endpoint_oracle"]["sufficiency_boundary"] == "diagnostic endpoint oracle"
    theory = _rows("theory_candidate_statements.csv")
    assert len(theory) == 3
    assert {r["status"] for r in theory} == {"future theorem candidate"}
    guards = _rows("non_theorem_guardrails.csv")
    assert len(guards) == 4
    assert {r["passed"] for r in guards} == {"1"}


def test_c56_mechanism_edges_and_escape_hatches_are_supported():
    edges = _rows("mechanism_edges.csv")
    assert len(edges) == 7
    assert {r["status"] for r in edges} == {"confirmed"}
    assert all(r["supporting_milestones"] for r in edges)
    edge_by_id = {r["edge_id"]: r for r in edges}
    assert edge_by_id["E7"]["from_node"] == "N_endpoint_oracle"
    assert edge_by_id["E7"]["to_node"] == "N_endpoint_availability_gap"
    assert "template=0.7037037037037037" in edge_by_id["E7"]["key_number_summary"]
    hatches = {r["escape_hatch"]: r for r in _rows("closed_escape_hatches.csv")}
    assert hatches["endpoint_template_full_transfer"]["closing_milestone"] == "C55"
    assert hatches["key_only_gauge_recovery"]["closing_milestone"] == "C52"
    caveats = {r["caveat_id"]: r for r in _rows("open_caveats.csv")}
    assert caveats["split_label_or_few_label"]["status"] == "open_future"
    assert caveats["formal_lower_bound"]["status"] == "future_theory"


def test_c56_literature_alignment_is_claim_limited():
    rows = _rows("literature_alignment_matrix.csv")
    assert len(rows) == 5
    ids = {r["closest_literature"] for r in rows}
    assert ids == {
        "IRM_1907_02893",
        "DomainBed_2007_01434",
        "ZhaoInvariantDA_1901_09453",
        "PostSelection_1401_3889",
        "InteractiveLowerBounds_2410_05117",
    }
    forbidden = _rows("forbidden_literature_overclaims.csv")
    assert len(forbidden) == 4
    assert {r["status"] for r in forbidden} == {"blocked"}
    lit_md = open("oaci/reports/C56_LITERATURE_ALIGNMENT.md").read()
    assert "https://arxiv.org/abs/1907.02893" in lit_md
    assert "https://arxiv.org/abs/2007.01434" in lit_md
    assert "https://arxiv.org/abs/1901.09453" in lit_md
    assert "https://arxiv.org/abs/1401.3889" in lit_md
    assert "https://arxiv.org/abs/2410.05117" in lit_md


def test_c56_reviewer_dossier_has_required_questions():
    bank = _rows("reviewer_question_bank.csv")
    amap = _rows("reviewer_answer_evidence_map.csv")
    assert len(bank) == 12
    assert len(amap) == 12
    qtext = " ".join(r["question"] for r in bank).lower()
    for phrase in (
        "negative result",
        "good checkpoints absent",
        "target labels leak",
        "local bayes ceiling",
        "conditioning",
        "target-aware action rule",
        "source-visible",
        "cross-cell endpoint-template",
        "split-label",
        "nulls",
        "literature",
        "next direction",
    ):
        assert phrase in qtext
    assert {r["answer_type"] for r in amap} == {"evidence_bounded"}


def test_c56_red_team_and_artifact_hygiene_pass():
    red = _rows("red_team_failure_ledger.csv")
    assert len(red) == 10
    assert {r["failed"] for r in red} == {"0"}
    scan = _rows("forbidden_claim_scan.csv")
    assert len(scan) == 9
    assert {r["affirmative_hits"] for r in scan} == {"0"}
    assert {r["passed"] for r in scan} == {"1"}
    schema = _rows("schema_validation_summary.csv")
    assert schema and {r["passed"] for r in schema} == {"1"}
    large = _rows("large_artifact_scan.csv")
    assert len(large) == 32
    assert {r["over_50mb"] for r in large} == {"0"}
    assert {r["passed"] for r in large} == {"1"}
    manifest = _rows("artifact_manifest.csv")
    assert len(manifest) == 32
    assert all(r["sha256"] for r in manifest)
    assert os.path.getsize(REPORT_JSON) < 50_000


def test_c56_run_loads_committed_artifacts():
    res = c56.run()
    assert res["decision"]["primary"] == "C56-A_mechanism_closed_ready_for_manuscript_scaffold"
    assert len(res["key_number_provenance_rows"]) == 35
    assert len(res["c55_null_provenance_rows"]) == 8
    assert len(res["availability_ledger_rows"]) == 10


def test_c56_outputs_do_not_emit_forbidden_affirmative_claims():
    forbidden = (
        "deployable selector claim",
        "oaci rescue claim",
        "source-only rescue claim",
        "checkpoint recommendation artifact",
        "few-label sufficiency claim",
        "same-label endpoint oracle available at selection time",
        "target-unlabeled deployable method",
        "target-grouped diagnostic described as source-only",
        "theorem claim without formal proof",
    )
    for name in (
        "C56_MECHANISM_CLOSURE_REVIEW_GAUNTLET.md",
        "C56_MECHANISM_CLOSURE_REVIEW_GAUNTLET.json",
        "C56_INFORMATION_BOUNDARY_FORMALIZATION.md",
        "C56_LITERATURE_ALIGNMENT.md",
        "C56_RED_TEAM_VERIFICATION.md",
        "C56_REVIEWER_QA_DOSSIER.md",
        "C56_REVIEW_DECISION.md",
    ):
        text = open(os.path.join("oaci/reports", name)).read().lower()
        for phrase in forbidden:
            assert phrase not in text
    for name in os.listdir(TABLE_DIR):
        if not name.endswith(".csv"):
            continue
        if name in {"forbidden_claim_scan.csv", "forbidden_literature_overclaims.csv"}:
            continue
        text = open(os.path.join(TABLE_DIR, name)).read().lower()
        assert "checkpoint recommendation artifact" not in text
        assert "selected_candidate_id" not in text
        assert "checkpoint_hash" not in text
