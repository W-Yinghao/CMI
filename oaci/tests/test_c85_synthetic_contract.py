"""C85P locked S0-S10 generator schema tests without scenario execution."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from oaci.theory import c85_synthetic_contract as contract


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci" / "reports"
TABLES = REPORTS / "c85p_tables"


def _locked() -> dict:
    return contract.validate_locked_inputs()


def _csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_exact_s0_s10_registry_is_locked_and_unexecuted() -> None:
    generator = _locked()["generator"]
    assert tuple(row["id"] for row in generator["scenarios"]) == tuple(f"S{i}" for i in range(11))
    assert generator["status"] == "LOCKED_IN_C85P_NOT_EXECUTED_UNTIL_C85T"
    assert generator["outcome_informed_design"] is False


def test_seed_rule_is_stable_and_scenario_specific() -> None:
    first = contract.deterministic_seed("S0", 0)
    assert first == contract.deterministic_seed("S0", 0)
    assert first != contract.deterministic_seed("S0", 1)
    assert first != contract.deterministic_seed("S1", 0)
    assert 0 <= first < 2**64


def test_every_scenario_binds_state_or_group_model_and_success_criterion() -> None:
    for row in _locked()["generator"]["scenarios"]:
        assert "states" in row or "groups" in row or "actions" in row
        assert row["risk_functionals"]
        assert row["success_criterion"]
        assert isinstance(row["sample_size"], int) and row["sample_size"] >= 0


def test_restricted_information_counterexamples_are_parameters_not_results() -> None:
    scenarios = {row["id"]: row for row in _locked()["generator"]["scenarios"]}
    assert scenarios["S1"]["experiments"]["E2"] == [[1.0, 0.0], [0.0, 1.0]]
    assert scenarios["S1"]["registered_policies"]["E2"] == "choose_action_1_minus_observed_state"
    assert scenarios["S10"]["garbling_rich_to_coarse"] == scenarios["S10"]["experiments"]["coarse"]
    assert all(row["status"] == "OPEN" for row in _csv("counterexample_registry.csv"))
    assert all(row["exhaustive_execution"] == "0" for row in _csv("counterexample_exhaustive_check.csv"))


def test_tail_scenario_has_symbolic_interval_audit_not_empirical_alpha() -> None:
    scenario = next(row for row in _locked()["generator"]["scenarios"] if row["id"] == "S5")
    assert scenario["alpha_intervals"] == [[0.0, 0.8], [0.8, 0.9], [0.9, 1.0]]
    assert "C84" not in json.dumps(scenario)


def test_dense_and_sparse_geometry_parameters_are_fixed_preexecution() -> None:
    scenarios = {row["id"]: row for row in _locked()["generator"]["scenarios"]}
    assert scenarios["S6"]["epsilon"] == 0.005
    assert scenarios["S6"]["tau"] == 0.01
    assert scenarios["S7"]["epsilon"] == 0.05
    assert scenarios["S7"]["tau"] == 0.01
    assert scenarios["S6"]["status"] if "status" in scenarios["S6"] else True


def test_partial_identification_contract_allows_randomization() -> None:
    scenario = next(row for row in _locked()["generator"]["scenarios"] if row["id"] == "S8")
    assert scenario["randomization_allowed"] is True
    rows = _csv("minimax_regret_problem_registry.csv")
    randomized = next(row for row in rows if row["action_class"] == "randomized")
    assert "LP:" in randomized["finite_form"]


def test_costly_label_query_is_not_one_arm_per_pull() -> None:
    scenario = next(row for row in _locked()["generator"]["scenarios"] if row["id"] == "S9")
    assert "all four actions" in scenario["query_observation"]
    rows = _csv("costly_label_experiment_contract.csv")
    full = next(row for row in rows if row["contract_id"] == "CL1")
    bandit = next(row for row in rows if row["contract_id"] == "CL2")
    assert full["standard_bandit_equivalent"] == "0"
    assert bandit["standard_bandit_equivalent"] == "1"


def test_validation_matrix_never_reports_a_synthetic_result() -> None:
    rows = _csv("synthetic_validation_matrix.csv")
    assert len(rows) == 11
    assert all(row["scientific_execution"] == "0" for row in rows)
    assert {row["result_status"] for row in rows} == {"CONTRACT_VALIDATED_NOT_EXECUTED"}


def test_theorem_and_proof_audits_have_no_completed_project_proof() -> None:
    theorem = _csv("theorem_registry.csv")
    audit = _csv("proof_audit.csv")
    assert len(theorem) == len(audit) == 7
    assert {row["status"] for row in theorem + audit} == {"OPEN"}
    assert all(row["project_proof_complete"] == "0" for row in audit)


def test_outcome_informed_design_guard_blocks_all_empirical_parameter_choices() -> None:
    rows = _csv("outcome_informed_design_guard.csv")
    assert len(rows) == 9
    assert all(row["C84_outcome_may_determine"] == "0" for row in rows)
    assert all(row["status"] == "PASS" for row in rows)


def test_active_testing_registry_is_advisory_only() -> None:
    methods = _csv("active_testing_method_registry.csv")
    requirements = _csv("future_active_protocol_requirements.csv")
    assert all(row["authorized"] == "0" for row in methods)
    assert all(row["blocks_execution_now"] == "1" for row in requirements)


def test_materialized_registry_bytes_replay_builder_output() -> None:
    observed = contract.validate_materialized_tables()
    assert sum(observed.values()) == 193
    assert observed["risk_register.csv"] == 15
    assert observed["failure_reason_ledger.csv"] == 1


def test_contract_cli_reports_validation_only() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "oaci.theory.c85_synthetic_contract", "validate-contract"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    value = json.loads(result.stdout)
    assert value["scenarios"] == 11
    assert value["status"] == "LOCKED_PROTOCOL_ONLY_NOT_EXECUTED"


def test_no_c85_execution_or_real_data_lock_exists() -> None:
    forbidden = [
        "C85T_EXECUTION_LOCK.json",
        "C85E_EXECUTION_LOCK.json",
        "C85_REAL_DATA_AUTHORIZATION.json",
        "C85_ACTIVE_ACQUISITION_LOCK.json",
    ]
    assert not any((REPORTS / name).exists() for name in forbidden)

