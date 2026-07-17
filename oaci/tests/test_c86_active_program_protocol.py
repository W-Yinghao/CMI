"""C86P protocol, registry, and no-execution boundary tests."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from oaci.theory.c86_active_program import ELIGIBLE_DATASETS, write_readiness_tables


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"
TABLES = REPORTS / "c86p_tables"


def _rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_c86_protocol_hash_status_and_parent_state() -> None:
    path = REPORTS / "C86_ACTIVE_TESTING_PROGRAM_PROTOCOL.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = (REPORTS / "C86_ACTIVE_TESTING_PROGRAM_PROTOCOL.sha256").read_text().split()[0]
    assert digest == sidecar == "d4feac535f8c1144a55d77cd7f322ae961d5c7d5a899dfd15c371484d88fbb7a"
    assert protocol["status"].endswith("NO_REAL_EXECUTION_LOCK")
    assert protocol["frozen_parent_state"]["C84_label_frontier"] == "C84-L4"
    assert protocol["frozen_parent_state"]["theorem_statuses"]["T5"] == "OPEN"
    assert protocol["chronology"]["new_EEG_downloaded_or_opened"] is False
    assert protocol["chronology"]["active_acquisition_executed"] is False
    operationalization = REPORTS / "C86P_ACTIVE_ESTIMATOR_OPERATIONALIZATION_PROTOCOL.json"
    op_digest = hashlib.sha256(operationalization.read_bytes()).hexdigest()
    op_sidecar = (REPORTS / "C86P_ACTIVE_ESTIMATOR_OPERATIONALIZATION_PROTOCOL.sha256").read_text().split()[0]
    assert op_digest == op_sidecar == "0cdb05c113a1c681584dec907a002af809aa019c933e042f15065a4f30c1f1dd"
    op = json.loads(operationalization.read_text())
    assert op["balanced_accuracy_plugin"]["Jeffreys_pseudocount_per_binary_outcome"] == 0.5
    assert op["chronology"]["C86_scientific_results_before_operationalization"] == 0
    correction = REPORTS / "C86P_UNTOUCHED_COHORT_VARIANT_ELIGIBILITY_CORRECTION_PROTOCOL.json"
    correction_digest = hashlib.sha256(correction.read_bytes()).hexdigest()
    correction_sidecar = correction.with_suffix(".sha256").read_text().split()[0]
    assert correction_digest == correction_sidecar == "5948b76a2d08c45c88e157aace1cc421a8c551b1c763a265376ad25921103c0d"
    correction_value = json.loads(correction.read_text())
    assert correction_value["prospective_correction"]["interface_variant"].startswith("paradigm_type=2C")
    assert correction_value["chronology"]["performance_outcomes_used"] is False
    access_correction = REPORTS / "C86P_HISTORICAL_ACCESS_ELIGIBILITY_CORRECTION_PROTOCOL.json"
    access_digest = hashlib.sha256(access_correction.read_bytes()).hexdigest()
    access_sidecar = access_correction.with_suffix(".sha256").read_text().split()[0]
    assert access_digest == access_sidecar == "fd7e214c6c6675b2a9b071b2bf278e2c4393495b2f4cbe03f82528c3098e7064"
    access_value = json.loads(access_correction.read_text())
    assert access_value["prospective_correction"]["Dreyer2023_confirmation_role"].startswith("INELIGIBLE")
    assert access_value["chronology"]["performance_outcomes_used"] is False


def test_c86_protocol_fixes_query_inference_and_taxonomy() -> None:
    protocol = json.loads((REPORTS / "C86_ACTIVE_TESTING_PROGRAM_PROTOCOL.json").read_text())
    assert protocol["label_budget"]["complete_grid"] == [4, 8, 16, 32, "FULL"]
    assert protocol["label_budget"]["unit"] == "total_label_queries_per_target"
    assert protocol["common_active_sampling_contract"]["active_chains"] == 2048
    assert protocol["aggregation_and_inference"]["principal_cluster"] == "target_subject"
    assert protocol["aggregation_and_inference"]["maxT_draws"] == 65536
    assert protocol["aggregation_and_inference"]["no_pooled_three_dataset_pvalue"] is True
    assert protocol["taxonomy"]["precedence"][0].startswith("C86-E_")
    assert protocol["label_complexity"]["frontier_domain"] == [4, 8, 16, 32]
    assert protocol["label_budget"]["FULL_superiority_test"] is False


def test_all_required_c86p_tables_are_present_and_nonempty() -> None:
    required = {
        "c85e_identity_and_claim_replay.csv", "active_testing_literature_registry.csv",
        "active_method_registry.csv", "reference_fidelity_registry.csv",
        "observable_information_contract.csv", "untouched_dataset_eligibility_registry.csv",
        "historical_access_ledger.csv", "interface_compatibility_matrix.csv",
        "dataset_selection_rule_truth_table.csv", "development_confirmation_separation.csv",
        "candidate_zoo_contract.csv", "physical_label_view_contract.csv",
        "total_query_budget_contract.csv", "passive_comparator_fairness_audit.csv",
        "trial_loss_vector_schema.csv", "estimator_and_importance_weight_contract.csv",
        "stopping_rule_contract.csv", "endpoint_and_robust_risk_contract.csv",
        "inference_and_multiplicity_contract.csv", "future_C86L_artifact_contract.csv",
        "future_C86H_confirmation_contract.csv", "synthetic_scenario_registry.csv",
        "risk_register.csv", "failure_reason_ledger.csv",
    }
    observed = {path.name for path in TABLES.glob("*.csv")}
    assert observed == required
    assert all(_rows(name) for name in required)


def test_readiness_table_generation_is_metadata_only_and_deterministic(tmp_path: Path) -> None:
    first = write_readiness_tables(tmp_path / "first")
    second = write_readiness_tables(tmp_path / "second")
    assert first["hashes"] == second["hashes"]
    assert first["table_count"] == 24
    assert first["catalog_rows"] == 53
    assert tuple(first["eligible_datasets"]) == ELIGIBLE_DATASETS
    assert tuple(first["eligible_datasets"]) == (
        "Brandl2020", "Kumar2024", "Yang2025_2C",
    )
    assert first["new_EEG_downloads"] == first["new_label_reads"] == 0
    assert first["active_acquisition_runs"] == 0


def test_method_registry_and_literature_fidelity_are_complete() -> None:
    methods = {row["method_id"]: row for row in _rows("active_method_registry.csv")}
    assert set(methods) == {"P0", "P1", "A1", "A2", "A3", "A4", "C0", "C1", "C2_U11", "C2_U13", "O1"}
    assert methods["P0"]["role"] == "PRIMARY_COMPARATOR"
    assert methods["P1"]["role"] == "SECONDARY_UNFAIR_INFORMATION_REFERENCE"
    assert all(methods[key]["confirmation_tuning"] == "FORBIDDEN" for key in methods)
    assert methods["A2"]["complexity"] == "O(NM^2)"
    assert methods["A4"]["confidence_set"] == "Bonferroni_normal_NLL_plausible_best_set_alpha_0.05"
    sources = {row["source_id"] for row in _rows("active_testing_literature_registry.csv")}
    assert {"Kossen2021", "Farquhar2021", "Hara2024", "Karimi2021"} <= sources
    fidelity = _rows("reference_fidelity_registry.csv")
    assert all(row["exact_reference_replication_claimed"] == "0" for row in fidelity if row["method_id"].startswith("A"))


def test_c86p_creates_no_real_execution_lock_or_authorization() -> None:
    assert not (REPORTS / "C86_EXECUTION_LOCK.json").exists()
    assert not (REPORTS / "C86H_EXECUTION_LOCK.json").exists()
    assert not (REPORTS / "C86L_EXECUTION_LOCK.json").exists()
    assert not (REPORTS / "C86_PI_AUTHORIZATION_RECORD.json").exists()
