"""C57 Manuscript Scaffold / Claim-Contract tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c57_manuscript_scaffold_claim_contract as c57
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C57_MANUSCRIPT_SCAFFOLD_CLAIM_CONTRACT.json"
TABLE_DIR = "oaci/reports/c57_tables"
SCAFFOLD_DIR = "oaci/reports/c57_manuscript_scaffold"


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_c57_config_decision_and_scope_are_frozen():
    assert c57._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert c57.MILESTONE == "C57"
    assert set(c57.DECISIONS) == {
        "C57-A_manuscript_scaffold_ready",
        "C57-B_manuscript_scaffold_ready_but_literature_gap_remaining",
        "C57-C_claim_contract_inconsistency_requires_repair",
        "C57-D_figure_or_evidence_provenance_gap_requires_repair",
        "C57-E_theory_framing_not_ready_reopen_C56",
        "C57-F_not_ready_for_manuscript_scaffold",
    }
    d = _summary()
    assert d["milestone"] == "C57"
    assert d["config_hash"] == "664007686afb520f"
    assert d["diagnostic_only_non_deployable"] is True
    assert d["c56_decision"] == "C56-A_mechanism_closed_ready_for_manuscript_scaffold"
    dec = d["decision"]
    assert dec["primary"] == "C57-A_manuscript_scaffold_ready"
    assert dec["red_team_failure_count"] == 0
    assert dec["untraceable_key_number_count"] == 0
    assert dec["recommended_next_direction"] == "M1 manuscript drafting"


def test_c57_table_shapes_and_scaffold_files_are_complete():
    d = _summary()
    assert d["claim_contract_count"] == 16
    assert d["main_figure_count"] == 8
    assert d["supplement_figure_count"] == 6
    assert d["key_number_count"] == 43
    assert d["table_row_counts"] == {
        "artifact_manifest": 36,
        "bayes_ceiling_vs_action_rule": 5,
        "claim_contract": 16,
        "claim_strength_ladder": 7,
        "claim_to_literature_map": 6,
        "contribution_map": 4,
        "figure_evidence_map": 14,
        "forbidden_claim_boundary": 9,
        "forbidden_claim_scan": 12,
        "forbidden_literature_overclaims": 5,
        "information_class_ladder": 8,
        "key_number_provenance": 43,
        "key_number_to_figure_map": 29,
        "large_artifact_scan": 36,
        "literature_alignment_matrix": 6,
        "manuscript_section_map": 10,
        "missing_or_ambiguous_provenance": 0,
        "red_team_failure_ledger": 11,
        "remaining_caveats": 4,
        "reviewer_answer_evidence_map": 12,
        "reviewer_question_bank": 12,
        "schema_validation_summary": 24,
        "subagent_audit_manifest": 9,
        "taxonomy_crosswalk": 7,
        "term_usage_guardrails": 5,
        "test_command_manifest": 4,
    }
    required_tables = {
        "artifact_manifest.csv",
        "bayes_ceiling_vs_action_rule.csv",
        "claim_contract.csv",
        "claim_strength_ladder.csv",
        "claim_to_literature_map.csv",
        "contribution_map.csv",
        "figure_evidence_map.csv",
        "forbidden_claim_boundary.csv",
        "forbidden_claim_scan.csv",
        "forbidden_literature_overclaims.csv",
        "information_class_ladder.csv",
        "key_number_provenance.csv",
        "key_number_to_figure_map.csv",
        "large_artifact_scan.csv",
        "literature_alignment_matrix.csv",
        "manuscript_section_map.csv",
        "missing_or_ambiguous_provenance.csv",
        "red_team_failure_ledger.csv",
        "remaining_caveats.csv",
        "reviewer_answer_evidence_map.csv",
        "reviewer_question_bank.csv",
        "schema_validation_summary.csv",
        "subagent_audit_manifest.csv",
        "taxonomy_crosswalk.csv",
        "term_usage_guardrails.csv",
        "test_command_manifest.csv",
    }
    assert required_tables == {p for p in os.listdir(TABLE_DIR) if p.endswith(".csv")}
    required_scaffold = {
        "figure_plan.md",
        "information_boundary_formalism.md",
        "manuscript_outline.md",
        "section_claims.md",
        "terminology_contract.md",
        "title_abstract_candidates.md",
    }
    assert required_scaffold == {p for p in os.listdir(SCAFFOLD_DIR) if p.endswith(".md")}


def test_c57_claim_contract_locks_allowed_strengths_and_boundaries():
    claims = {r["claim_id"]: r for r in _rows("claim_contract.csv")}
    assert len(claims) == 16
    assert {r["allowed_strength"] for r in claims.values()} <= set(c57.ALLOWED_STRENGTHS)
    assert claims["CL02"]["allowed_strength"] == "negative_control_result"
    assert "OACI" in claims["CL02"]["claim_text"]
    assert claims["CL11"]["allowed_information_class"] == "I2"
    assert "do not close" in claims["CL11"]["claim_text"]
    assert claims["CL13"]["allowed_information_class"] == "I7"
    assert "unavailable at selection time" in claims["CL13"]["claim_text"]
    assert claims["CL14"]["required_evidence_milestones"] == "C55,C56"
    assert "does not beat" in claims["CL14"]["claim_text"]
    assert claims["CL15"]["allowed_strength"] == "future_work_only"
    forbidden = {r["forbidden_class"] for r in _rows("forbidden_claim_boundary.csv")}
    assert {
        "source_only_solution",
        "deployable_selector",
        "OACI_rescue",
        "few_label_sufficiency",
        "same_label_oracle_as_available_method",
        "formal_theorem_without_proof",
    } <= forbidden


def test_c57_c55_null_disambiguation_is_preserved():
    d = _summary()
    assert d["c55_null_disambiguation"] == {
        "endpoint_scalar_transfer_beats_max_null_p95": True,
        "endpoint_scalar_transfer_hit": 0.9444444444444444,
        "max_null_p95": 0.7712962962962961,
        "template_only_beats_max_null_p95": False,
        "template_only_hit": 0.7037037037037037,
    }
    c55_rows = {r["provenance_id"]: r for r in _rows("key_number_provenance.csv")}
    assert float(c55_rows["K_C55_template_only_best"]["value"]) == 0.7037037037037037
    assert float(c55_rows["K_C55_endpoint_scalar_transfer"]["value"]) == 0.9444444444444444
    assert float(c55_rows["K_C55_max_null_p95"]["value"]) == 0.7712962962962961
    report = open("oaci/reports/C57_MANUSCRIPT_SCAFFOLD_CLAIM_CONTRACT.md").read()
    assert "template-only" in report
    assert "unavailable" in report


def test_c57_key_number_provenance_includes_c34_c38_and_c56_chain():
    rows = {r["provenance_id"]: r for r in _rows("key_number_provenance.csv")}
    assert len(rows) == 43
    expected = {
        "K_C31_joint_good_rate": 0.4242902208201893,
        "K_C34_real_endpoint_regret_fraction": 0.9411764705882353,
        "K_C34_threshold_only_fraction": 0.0,
        "K_C35_preference_robust_fraction": 0.7450980392156863,
        "K_C35_preference_robust_pairs": 114.0,
        "K_C37_ucl_prefers_selected_fraction": 1.0,
        "K_C37_selection_target_conflict_exact_rate": 1.0,
        "K_C38_point_dominant_fraction": 0.9736842105263158,
        "K_C38_leakage_target_gauge_conflict_fraction": 0.9210526315789473,
        "K_C48_local_ceiling_hit": 1.0,
        "K_C52_best_key_only_hit": 0.4876543209876543,
        "K_C52_best_label_derived_hit": 0.8127572016460904,
        "K_C53_best_scalar_endpoint_hit": 0.9444444444444444,
        "K_C55_endpoint_scalar_transfer": 0.9444444444444444,
    }
    for key, val in expected.items():
        assert key in rows
        assert float(rows[key]["value"]) == val
        assert rows[key]["trace_status"] == "verified"
        assert rows[key]["artifact"].startswith("oaci/reports/")
    assert _rows("missing_or_ambiguous_provenance.csv") == []


def test_c57_information_ladder_and_ceiling_action_split_are_locked():
    ladder = {r["information_class"]: r for r in _rows("information_class_ladder.csv")}
    assert set(ladder) == {
        "I0_random_or_tie",
        "I1_strict_source_observables",
        "I2_source_plus_target_or_trajectory_keys",
        "I3_target_unlabeled_transductive_geometry",
        "I4_target_grouped_zero_label_structure",
        "I5_split_label_or_few_label_calibration",
        "I6_target_label_diagnostic_content",
        "I7_same_label_endpoint_oracle",
    }
    assert ladder["I1_strict_source_observables"]["sufficiency_boundary"] == "not reliable"
    assert ladder["I5_split_label_or_few_label_calibration"]["sufficiency_boundary"] == "open future"
    assert ladder["I7_same_label_endpoint_oracle"]["manuscript_phrase"] == "unavailable at selection time"
    split = {r["object"]: r for r in _rows("bayes_ceiling_vs_action_rule.csv")}
    assert split["conditioned local Bayes ceiling"]["uses_target_labels"] == "1"
    assert split["conditioned local Bayes ceiling"]["available_at_selection_time"] == "0"
    assert split["best strict source score"]["available_at_selection_time"] == "1"
    assert split["same-label endpoint scalar"]["available_at_selection_time"] == "0"
    terms = {r["term_id"]: r for r in _rows("taxonomy_crosswalk.csv")}
    assert terms["measurement_vs_control"]["availability_class"] == "I1"
    assert terms["measurement_vs_control"]["available_at_selection_time"] == "1"
    assert terms["diagnostic_ceiling_vs_action_rule"]["available_at_selection_time"] == "0"
    assert terms["template_transfer_vs_endpoint_scalar_availability"]["uses_target_labels"] == "1"
    assert terms["same_label_oracle_vs_split_label_calibration"]["allowed_claim_strength"] == "future_work_only"


def test_c57_figure_plan_maps_key_numbers_to_claims():
    figures = {r["figure_id"]: r for r in _rows("figure_evidence_map.csv")}
    assert len(figures) == 14
    assert sum(1 for r in figures.values() if r["main_or_supplement"] == "main") == 8
    assert sum(1 for r in figures.values() if r["main_or_supplement"] == "supplement") == 6
    assert "K_C55_template_only_best=0.704" in figures["F7"]["key_numbers"]
    assert "K_C55_max_null_p95=0.771" in figures["F7"]["key_numbers"]
    assert figures["F7"]["unsupported_overclaim"] == "template-only transfer beats max null"
    assert "K_C37_ucl_prefers_selected_fraction=1.000" in figures["F5"]["key_numbers"]
    key_map = _rows("key_number_to_figure_map.csv")
    assert len(key_map) == 29
    assert any(r["provenance_id"] == "K_C55_endpoint_scalar_transfer" and r["figure_id"] == "F7" for r in key_map)


def test_c57_literature_and_reviewer_dossier_are_claim_limited():
    lit = {r["literature_id"]: r for r in _rows("literature_alignment_matrix.csv")}
    assert set(lit) == {
        "IRM_1907_02893",
        "DomainBed_2007_01434",
        "ZhaoInvariantDA_1901_09453",
        "PostSelection_1401_3889",
        "InteractiveLowerBounds_2410_05117",
        "EEG_DG_project_bibliography_pending",
    }
    assert "https://arxiv.org/abs/1907.02893" in lit["IRM_1907_02893"]["url_or_status"]
    assert lit["EEG_DG_project_bibliography_pending"]["url_or_status"] == "local bibliography expansion for M1"
    forbidden = _rows("forbidden_literature_overclaims.csv")
    assert len(forbidden) == 5
    assert {r["status"] for r in forbidden} == {"blocked"}
    bank = _rows("reviewer_question_bank.csv")
    assert len(bank) == 12
    qtext = " ".join(r["question"] for r in bank).lower()
    for phrase in ("negative result", "good checkpoints", "target labels", "local bayes", "conditioning", "c55", "split-label", "literature", "eeg-specific"):
        assert phrase in qtext


def test_c57_red_team_artifact_hygiene_and_forbidden_scan_pass():
    red = _rows("red_team_failure_ledger.csv")
    assert len(red) == 11
    assert {r["failed"] for r in red} == {"0"}
    scan = _rows("forbidden_claim_scan.csv")
    assert len(scan) == len(c57.FORBIDDEN_PATTERNS) == 12
    assert {r["affirmative_hits"] for r in scan} == {"0"}
    assert {r["passed"] for r in scan} == {"1"}
    schema = _rows("schema_validation_summary.csv")
    assert len(schema) == 24
    assert {r["passed"] for r in schema} == {"1"}
    large = _rows("large_artifact_scan.csv")
    assert len(large) == 36
    assert {r["over_50mb"] for r in large} == {"0"}
    manifest = _rows("artifact_manifest.csv")
    assert len(manifest) == 36
    assert all(r["sha256"] for r in manifest)
    provenance_md = open("oaci/reports/C57_PROVENANCE_AUDIT.md").read()
    assert "Tracked generated payload artifacts: 36." in provenance_md
    assert "self-reference instability" in provenance_md
    for row in manifest:
        assert os.path.exists(row["path"])
        assert int(row["size_bytes"]) == os.path.getsize(row["path"])
        assert row["sha256"] == _sha256(row["path"])
    manifest_paths = {r["path"] for r in manifest}
    assert "oaci/reports/c57_tables/artifact_manifest.csv" not in manifest_paths
    assert "oaci/reports/c57_tables/large_artifact_scan.csv" not in manifest_paths
    assert os.path.getsize(REPORT_JSON) < 50_000
    subagents = _rows("subagent_audit_manifest.csv")
    assert len(subagents) == 9
    assert {r["integration_status"] for r in subagents} == {"launched_integrated"}


def test_c57_outputs_do_not_emit_selected_or_checkpoint_artifacts():
    for root in ("oaci/reports", SCAFFOLD_DIR, TABLE_DIR):
        for dirpath, _, filenames in os.walk(root):
            if "c57" not in dirpath.lower() and dirpath != "oaci/reports":
                continue
            for name in filenames:
                if not (name.startswith("C57_") or dirpath.endswith("c57_tables") or dirpath.endswith("c57_manuscript_scaffold")):
                    continue
                path = os.path.join(dirpath, name)
                text = open(path, errors="ignore").read().lower()
                assert "selected_candidate_id" not in text
                assert "checkpoint_hash" not in text
                assert "checkpoint recommendation artifact" not in text


def test_c57_run_recomputes_without_new_experiment():
    res = c57.run()
    assert res["c56_decision"] == "C56-A_mechanism_closed_ready_for_manuscript_scaffold"
    assert len(res["claim_contract_rows"]) == 16
    assert len(res["figure_evidence_map_rows"]) == 14
    assert len(res["key_number_provenance_rows"]) == 43
