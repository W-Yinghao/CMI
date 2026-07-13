from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

from oaci.conditioned_ceiling_coverage import c83_evidence_freeze as freeze


def _table(name: str) -> list[dict[str, str]]:
    with (freeze.TABLE_DIR / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _numbers() -> dict[str, dict[str, str]]:
    return {row["number_id"]: row for row in _table("authoritative_number_registry.csv")}


def test_c83_validation_passes_without_scientific_execution():
    result = freeze.validate()["result"]
    assert result["failed"] == result["blocking"] == 0
    assert result["new_real_data_statistics"] == 0
    assert result["EEG_or_label_view_accesses"] == 0
    assert result["manuscript_prose_created"] is False
    assert result["gate"] == freeze.C83_GATE


def test_c82_protocol_lock_result_and_manifest_hashes_replay():
    assert freeze.sha256_file(freeze.REPORT_DIR / "C82_POST_C81_BASELINE_RECOVERY_PROTOCOL.json") == freeze.C82_PROTOCOL_SHA256
    assert freeze.sha256_file(freeze.REPORT_DIR / "C82_ANALYSIS_EXECUTION_LOCK.json") == freeze.C82_LOCK_SHA256
    assert freeze.sha256_file(freeze.REPORT_DIR / "C82_POST_C81_BASELINE_RECOVERY.json") == freeze.C82_RESULT_SHA256
    assert freeze.sha256_file(freeze.C82_TABLE_DIR / "result_artifact_manifest.json") == freeze.C82_MANIFEST_SHA256


def test_c82_commit_chain_replay_is_complete_and_ordered():
    rows = _table("c82_commit_chain_replay.csv")
    assert len(rows) == 11
    assert all(row["status"] == "PASS" for row in rows)
    assert rows[-2]["expected_commit"] == freeze.C82_BASE_HEAD
    assert rows[-1]["object_id"] == "C82_PM_addendum"


def test_c82_all_23_table_hashes_and_row_counts_replay():
    rows = _table("c82_table_manifest_replay.csv")
    assert len(rows) == 23
    assert all(row["hash_pass"] == row["row_count_pass"] == "1" for row in rows)


def test_c82_regression_replay_has_four_accepted_empty_stderr_suites():
    rows = _table("c82_regression_replay.csv")
    assert len(rows) == 4
    assert {row["passed"] for row in rows} == {"43", "460", "871", "1795"}
    assert all(row["status"] == "PASS" and row["stderr_bytes"] == "0" for row in rows)


def test_claim_contract_contains_supported_C1_through_C10():
    rows = _table("claim_contract.csv")
    supported = {row["claim_id"] for row in rows if row["supported"] == "1"}
    assert supported == {f"C{index}" for index in range(1, 11)}
    assert all((freeze.REPO_ROOT / path).exists() for row in rows for path in row["supporting_artifacts"].split(";"))


def test_claim_contract_contains_all_minimum_forbidden_expansions():
    rows = _table("claim_contract.csv")
    forbidden = {row["claim_text_short"] for row in rows if row["supported"] == "0"}
    assert "Universal zero-label impossibility" in forbidden
    assert "Universal one-label sufficiency" in forbidden
    assert "External validity or new-subject generalization" in forbidden
    assert "Deployability" in forbidden
    assert "COTT failure in general" in forbidden
    assert "Cross-regime selector transport" in forbidden
    assert len(forbidden) == 10


def test_machine_claim_contract_disallows_manuscript_drafting():
    contract = json.loads((freeze.REPORT_DIR / "C83_AAAI_CLAIM_CONTRACT.json").read_text())
    assert contract["manuscript_drafting_authorized"] is False
    assert contract["new_real_data_statistics"] == 0
    assert contract["scientific_gate"] == freeze.C82_GATE
    assert contract["historical_C81_gate"] == freeze.C81_GATE


def test_authoritative_number_registry_is_unique_and_exactly_replayable():
    rows = _table("authoritative_number_registry.csv")
    assert len(rows) == 928
    assert len({row["number_id"] for row in rows}) == 928
    assert all(freeze._validate_number(row) for row in rows)


def test_core_universe_counts_are_bound_to_committed_sources():
    numbers = _numbers()
    assert numbers["universe_primary_candidates"]["value"] == "2592"
    assert numbers["universe_contexts"]["value"] == "32"
    assert numbers["registered_methods"]["value"] == "34"
    assert numbers["selection_methods"]["value"] == "19"
    assert numbers["primary_zero_label_methods"]["value"] == "6"
    assert numbers["canonical_result_rows"]["value"] == "672"
    assert numbers["registered_result_tables"]["value"] == "23"
    assert numbers["c82_scientific_red_team_checks"]["value"] == "59"
    assert numbers["c82_final_report_red_team_checks"]["value"] == "50"


def test_core_COTT_S1_and_Q0_values_replay_at_full_precision():
    numbers = _numbers()
    expected = {
        "c82_seed3_S1_mean_standardized_regret": "0.7794759728707445",
        "c82_seed4_S1_mean_standardized_regret": "0.8048232601796825",
        "c82_seed3_U13_mean_standardized_regret": "0.33864056415217253",
        "c82_seed4_U13_mean_standardized_regret": "0.4653350333230629",
        "c80_seed3_budget1_expected_standardized_regret": "0.35338318633180843",
        "c80_seed4_budget1_expected_standardized_regret": "0.3737050812998979",
        "c82_seed3_U13_q1q2_Q1_maxT_p": "0.01556420233463035",
        "c82_seed4_U13_q1q2_Q1_maxT_p": "0.10116731517509728",
        "c82_seed3_U13_q1q2_Q2_simultaneous_upper": "0.1445282639512282",
        "c82_seed4_U13_q1q2_Q2_simultaneous_upper": "0.2509008381540292",
    }
    assert {key: numbers[key]["value"] for key in expected} == expected


def test_COTT_topk_and_measurement_numbers_remain_distinct_endpoints():
    numbers = _numbers()
    assert numbers["c82_seed3_U13_top1"]["value"] == "0.125"
    assert numbers["c82_seed4_U13_top1"]["value"] == "0.0"
    assert numbers["c82_seed3_U13_mean_spearman"]["value"] == "0.2766049969060458"
    assert numbers["c82_seed4_U13_mean_spearman"]["value"] == "0.18423166857231696"
    assert numbers["c82_seed3_U13_mean_pairwise_order_accuracy"]["endpoint"] != numbers["c82_seed3_U13_top1"]["endpoint"]


def test_C80_full_panel_Bstar_and_all_LOTO_values_are_frozen():
    numbers = _numbers()
    assert numbers["c80_seed3_Bstar"]["value"] == numbers["c80_seed4_Bstar"]["value"] == "1"
    loto = [row["value"] for key, row in numbers.items() if key.startswith("c80_loto_")]
    assert len(loto) == 16
    assert set(loto) == {"2", "4"}


def test_C82_global_LOTO_semantics_are_not_per_panel_COTT_evidence():
    addendum = json.loads((freeze.REPORT_DIR / "C82E_PM_GITHUB_AUDIT_ADDENDUM.json").read_text())
    loto = addendum["loto_semantics"]
    assert loto["implementation_uses_full_panel_cross_seed_common_method_set"] is True
    assert loto["common_B_method_set"] == []
    assert loto["global_method_aware_panels_preserved"] == 7
    assert loto["global_panels_total"] == 16
    assert loto["per_panel_per_method_Q1_ledger_emitted"] is False
    assert "Every seed-3 LOTO" in loto["forbidden_wording"]


def test_Q5_is_descriptive_best_within_fixed_class_not_inferential_winner():
    addendum = json.loads((freeze.REPORT_DIR / "C82E_PM_GITHUB_AUDIT_ADDENDUM.json").read_text())
    q5 = addendum["q5_semantics"]
    assert q5["information_class_membership_fixed"] is True
    assert q5["displayed_method_selected_by_observed_minimum_regret"] is True
    assert q5["authoritative_wording"] == "descriptive best registered method within a fixed class"
    figure = _table("figure_3_data.csv")
    best_i0 = [row for row in figure if row["display_role"] == "descriptive_best_within_fixed_class" and row["information_class"] == "I0"]
    assert {(row["seed"], row["method_id"]) for row in best_i0} == {("3", "B4O"), ("4", "B2")}


def test_figure_contracts_are_complete_and_no_images_are_rendered():
    assert len(_table("figure_1_contract.csv")) == 9
    assert len(_table("figure_2_contract.csv")) == 9
    assert len(_table("figure_3_data.csv")) == 50
    assert len(_table("figure_4_data.csv")) == 52
    assert not list(freeze.REPORT_DIR.glob("C83_AAAI_FIGURE*.png"))
    assert not list(freeze.REPORT_DIR.glob("C83_AAAI_FIGURE*.pdf"))


def test_figure_and_table_data_registries_cover_exactly_four_figures_and_three_main_tables():
    figures = _table("figure_data_registry.csv")
    tables = _table("table_data_registry.csv")
    assert {row["figure_id"] for row in figures} == {"F1", "F2", "F3", "F4"}
    assert {row["table_id"] for row in tables if row["placement"] == "main"} == {"T1", "T2", "T3"}
    for row in figures:
        assert len(_table(Path(row["data_path"]).name)) == int(row["row_count"])
    for row in tables:
        assert (freeze.REPO_ROOT / row["data_path"]).exists()


def test_figure_and_main_table_number_ids_resolve_without_rounding_drift():
    numbers = _numbers()
    for name in ("figure_3_data.csv", "figure_4_data.csv"):
        for row in _table(name):
            if row["number_id"] != "NA":
                assert row["value"] == numbers[row["number_id"]]["value"]
    for row in _table("main_table_2.csv"):
        assert row["standardized_regret"] == numbers[row["regret_number_id"]]["value"]
        assert row["source_relative_gain"] == numbers[row["gain_number_id"]]["value"]


def test_main_table_contracts_have_fixed_scope():
    table1 = _table("main_table_1.csv")
    table2 = _table("main_table_2.csv")
    table3 = _table("main_table_3.csv")
    assert len(table1) == 12
    assert len(table2) == 18
    assert {row["method_id"] for row in table2} == {"S1", *freeze.PRIMARY_ZERO_METHODS, "L1", "L7"}
    assert {row["milestone"] for row in table3} == {"C78S", "C79E", "C80E", "C81", "C82E"}
    assert next(row for row in table3 if row["milestone"] == "C81")["gate"] == freeze.C81_GATE
    assert next(row for row in table3 if row["milestone"] == "C82E")["gate"] == freeze.C82_GATE


def test_baseline_fidelity_appendix_accounts_for_34_methods_and_five_unavailable():
    rows = _table("baseline_reference_fidelity_appendix.csv")
    assert len(rows) == 34
    excluded = [row for row in rows if row["fidelity_status"] == "EXCLUDED_BEFORE_HASH"]
    assert {row["method_id"] for row in excluded} == {"S3", "S4", "U8", "U9", "U10"}
    assert all(row["outcome_tuned"] == "0" and row["oracle_reachable"] == "0" for row in rows)


def test_reviewer_risks_are_closed_by_disclosure_not_new_experiments():
    rows = _table("reviewer_risk_ledger.csv")
    assert len(rows) == 17
    assert all(row["new_experiment_required"] == "no_for_C83P" for row in rows)
    assert all(row["status"] == "CLOSED_BY_DISCLOSURE_AND_CLAIM_NARROWING" for row in rows)


def test_reproducibility_index_copies_no_raw_payloads():
    index = json.loads((freeze.REPORT_DIR / "C83_AAAI_REPRODUCIBILITY_INDEX.json").read_text())
    assert index["no_new_scientific_computation"] is True
    assert index["raw_EEG_or_label_arrays_in_git"] is False
    assert index["model_weights_or_caches_in_Git"] is False
    assert len(index["items"]) == 16
    assert all(item["raw_payload_copied"] is False for item in index["items"])


def test_limitations_contract_blocks_external_and_universal_expansions():
    contract = json.loads((freeze.REPORT_DIR / "C83_AAAI_LIMITATIONS_AND_EXTERNAL_VALIDITY_CONTRACT.json").read_text())
    assert contract["independent_datasets"] == 1
    assert contract["training_seeds"] == 2
    assert contract["primary_target_clusters"] == 8
    assert contract["external_validation"] is False
    assert contract["new_subject_generalization"] is False
    assert contract["deployability"] is False
    assert contract["universal_zero_label_impossibility"] is False
    assert contract["universal_one_label_sufficiency"] is False
    assert contract["new_experiment_required_for_C83P"] is False


def test_target4_and_same_label_oracle_remain_zero_in_all_672_rows():
    rows = freeze.read_csv(freeze.C82_TABLE_DIR / "method_context_results.csv")
    assert len(rows) == 672
    assert all(row["target"] != "4" for row in rows)
    assert all(row["target4_primary"] == "False" for row in rows)
    assert all(row["same_label_oracle_accessed"] == "False" for row in rows)


def test_evidence_freeze_implementation_has_no_scientific_stack_imports():
    tree = ast.parse(Path(freeze.__file__).read_text())
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imports.isdisjoint({"numpy", "scipy", "pandas", "torch", "mne", "moabb"})


def test_C83P_creates_contracts_not_manuscript_sections():
    required = {
        "C83_AAAI_CLAIM_CONTRACT.json",
        "C83_AAAI_FIGURE_CONTRACT.md",
        "C83_AAAI_REPRODUCIBILITY_INDEX.md",
        "C83_AAAI_REPRODUCIBILITY_INDEX.json",
    }
    assert required <= {path.name for path in freeze.REPORT_DIR.iterdir()}
    forbidden = ("ABSTRACT", "INTRODUCTION", "RELATED_WORK", "DISCUSSION", "COVER_LETTER")
    assert not [path for path in freeze.REPORT_DIR.iterdir() if path.name.startswith("C83") and any(token in path.name.upper() for token in forbidden)]
