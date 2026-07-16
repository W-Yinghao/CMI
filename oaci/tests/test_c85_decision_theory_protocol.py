"""C85P statistical-decision protocol and finite contract tests."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from oaci.theory.c85_decision_experiments import (
    DecisionContractError,
    FiniteDecisionProblem,
    FiniteExperiment,
    candidate_two_state_lecam_regret_bound,
    channels_equal,
    garble_experiment,
    policy_approximation_gap,
    registered_policy_risk,
    unrestricted_optimal_risk,
)
from oaci.theory.c85_lower_bound_contracts import (
    ProofObligation,
    TheoremStatus,
    require_symbolic_robust_parameters,
    validate_c85p_statuses,
)
from oaci.theory.c85_policy_collapse import (
    assert_policy_collapse_loss_identity,
    summarize_realized_policy_use,
)
from oaci.theory.c85_robust_risk import (
    compare_mean_and_tail,
    entropy_effective_size,
    hill2_effective_size,
    near_optimal_set,
    soft_gap_weights,
    upper_tail_cvar,
)
from oaci.theory import c85_synthetic_contract as contract


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci" / "reports"
TABLES = REPORTS / "c85p_tables"
THEORY = ROOT / "oaci" / "theory"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_protocol_and_addendum_sidecars_replay() -> None:
    locked = contract.validate_locked_inputs()
    assert locked["protocol_sha256"] == contract.EXPECTED_PROTOCOL_SHA256
    assert locked["generator_sha256"] == contract.EXPECTED_GENERATOR_SHA256
    assert locked["addendum_sha256"] == contract.EXPECTED_ADDENDUM_JSON_SHA256
    assert _sha(REPORTS / "C85_TPAMI_DECISION_THEORY_PROTOCOL.json") == contract.EXPECTED_PROTOCOL_SHA256


def test_protocol_commit_precedes_theory_implementation_commit() -> None:
    protocol_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "oaci/reports/C85_TPAMI_DECISION_THEORY_PROTOCOL.json"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert protocol_commit == "2449be1c24e313922688b5e957ce6d19cb75d9d6"
    implementation_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "oaci/theory/c85_decision_experiments.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if implementation_commit:
        assert subprocess.run(
            ["git", "merge-base", "--is-ancestor", protocol_commit, implementation_commit],
            cwd=ROOT, check=False,
        ).returncode == 0


def test_c84a_pm_addendum_records_exact_realized_action_equivalence() -> None:
    value = json.loads((REPORTS / "C84A_PM_REALIZED_POLICY_USE_ADDENDUM.json").read_text())
    rows = {row["dataset"]: row for row in value["realized_action_equivalence"]}
    assert rows["Cho2017"]["equivalent_contexts"] == rows["Cho2017"]["total_contexts"] == 160
    assert rows["Lee2019_MI"]["equivalent_contexts"] == 175
    assert rows["PhysionetMI"]["equivalent_contexts"] == 607
    assert value["frozen_science"]["taxonomy_changed"] is False
    assert set(value["protected_counters"].values()) == {0}


def test_all_theorem_targets_remain_open_in_c85p() -> None:
    protocol = json.loads((REPORTS / "C85_TPAMI_DECISION_THEORY_PROTOCOL.json").read_text())
    assert [row["id"] for row in protocol["theorem_targets"]] == [f"T{i}" for i in range(1, 8)]
    assert {row["status_at_C85P"] for row in protocol["theorem_targets"]} == {"OPEN"}
    assert all("c85t_obligation" in row for row in protocol["theorem_targets"])


def test_finite_experiment_garbling_and_unrestricted_risk_contract() -> None:
    rich = FiniteExperiment.from_rows((0, 1), ("z0", "z1"), ((1, 0), (0, 1)))
    coarse = garble_experiment(rich, ("constant",), ((1,), (1,)))
    expected_coarse = FiniteExperiment.from_rows((0, 1), ("constant",), ((1,), (1,)))
    assert channels_equal(coarse, expected_coarse)
    problem = FiniteDecisionProblem(
        states=(0, 1), actions=(0, 1), prior=(0.5, 0.5),
        utilities=((1.0, 0.0), (0.0, 1.0)),
    )
    assert unrestricted_optimal_risk(rich, problem) == pytest.approx(0.0)
    assert unrestricted_optimal_risk(coarse, problem) == pytest.approx(0.5)


def test_registered_policy_gap_is_distinct_from_information_value() -> None:
    experiment = FiniteExperiment.from_rows((0, 1), ("z0", "z1"), ((1, 0), (0, 1)))
    problem = FiniteDecisionProblem(
        states=(0, 1), actions=(0, 1), prior=(0.5, 0.5),
        utilities=((1.0, 0.0), (0.0, 1.0)),
    )
    assert registered_policy_risk(experiment, problem, [(0, 0)]) == pytest.approx(0.5)
    assert policy_approximation_gap(experiment, problem, [(0, 0)]) == pytest.approx(0.5)
    assert unrestricted_optimal_risk(experiment, problem) == pytest.approx(0.0)


def test_probability_and_dimension_contracts_fail_closed() -> None:
    with pytest.raises(DecisionContractError):
        FiniteExperiment.from_rows((0, 1), (0,), ((1,), (0.8,)))
    with pytest.raises(DecisionContractError):
        FiniteDecisionProblem((0,), (0,), (1,), ((1.2,),))
    with pytest.raises(DecisionContractError):
        candidate_two_state_lecam_regret_bound(0.1, 1.1)


def test_lecam_expression_is_explicitly_candidate_only() -> None:
    assert candidate_two_state_lecam_regret_bound(0.4, 0.25) == pytest.approx(0.15)
    source = (THEORY / "c85_decision_experiments.py").read_text()
    assert "not a C85P theorem" in source


def test_upper_tail_cvar_uses_locked_variational_convention() -> None:
    assert upper_tail_cvar([0.0, 1.0], 0.5) == pytest.approx(1.0)
    assert upper_tail_cvar([0.0, 1.0], 0.25) == pytest.approx(2.0 / 3.0)
    with pytest.raises(DecisionContractError):
        upper_tail_cvar([0.0, 1.0], 1.0)


def test_mean_and_tail_are_separate_functionals() -> None:
    summary = compare_mean_and_tail([0.5] * 10, [0.3] * 9 + [1.0], alpha=0.9)
    assert summary["mean_improvement"] > 0
    assert summary["worst_improvement"] < 0
    assert summary["cvar_improvement"] < 0


def test_near_optimal_geometry_is_gap_weighted() -> None:
    utilities = [1.0, 0.999, 0.998, 0.5]
    assert near_optimal_set(utilities, 0.002) == (0, 1, 2)
    weights = soft_gap_weights(utilities, 0.01)
    assert sum(weights) == pytest.approx(1.0)
    assert 2.0 < hill2_effective_size(weights) < 4.0
    assert 2.0 < entropy_effective_size(weights) < 4.0


def test_exact_policy_collapse_has_zero_incremental_realized_value() -> None:
    summary = summarize_realized_policy_use(
        [0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3],
    )
    assert summary.exact_collapse
    assert summary.action_divergence == 0.0
    assert summary.incremental_fixed_policy_risk_value == pytest.approx(0.0)
    assert_policy_collapse_loss_identity(
        [0, 0], [0, 0], [0.1, 0.2], [0.1, 0.2]
    )


def test_policy_collapse_does_not_accept_loss_drift() -> None:
    with pytest.raises(DecisionContractError):
        assert_policy_collapse_loss_identity([0], [0], [0.1], [0.2])


def test_proof_status_validator_rejects_premature_proof_claim() -> None:
    open_obligation = ProofObligation("T4", TheoremStatus.OPEN, ("A3",), "candidate", ("derive constant",))
    validate_c85p_statuses([open_obligation])
    proved = ProofObligation("T4", TheoremStatus.PROVED, ("A3",), "candidate", ("derive constant",))
    with pytest.raises(DecisionContractError):
        validate_c85p_statuses([proved])


def test_robust_theory_parameters_remain_symbolic() -> None:
    require_symbolic_robust_parameters(
        "SYMBOLIC_(0,1)", "SYMBOLIC_POSITIVE", "SYMBOLIC_POSITIVE"
    )
    with pytest.raises(DecisionContractError):
        require_symbolic_robust_parameters("0.9", "0.05", "0.01")


def test_all_required_registry_tables_materialize_exactly() -> None:
    observed = contract.validate_materialized_tables()
    assert len(observed) == 32
    assert observed["terminology_registry.csv"] == 18
    assert observed["theorem_registry.csv"] == 7
    assert observed["synthetic_scenario_registry.csv"] == 11
    assert observed["literature_source_registry.csv"] >= 14


def test_leading_numeric_cumulative_and_focused_suites_include_c85() -> None:
    from oaci.multidataset import c84r_regression_suite as suites

    c85_names = {
        "test_c85_decision_theory_protocol.py",
        "test_c85_synthetic_contract.py",
    }
    assert suites.milestone_number("test_c85_synthetic_contract.py") == 85
    for suite in ("focused", "c65", "c23", "full"):
        paths = suites.suite_files(suite)
        if suite == "full":
            assert paths == [suites.TEST_DIR]
        else:
            assert c85_names <= {path.name for path in paths}


def test_literature_registry_uses_primary_or_canonical_sources() -> None:
    rows = _csv("literature_source_registry.csv")
    assert all(row["verification_status"] == "VERIFIED_PRIMARY_OR_CANONICAL" for row in rows)
    assert all(row["conditions_imported_without_audit"] == "0" for row in rows)
    assert {"10.1214/aoms/1177729032", "10.21314/JOR.2000.038", "10.1016/S0304-4076(99)00045-7"} <= {row["doi"] for row in rows}


def test_theory_package_has_no_empirical_or_training_import() -> None:
    forbidden_roots = {"torch", "mne", "moabb"}
    forbidden_oaci = {"multidataset", "train", "methods", "models"}
    for path in THEORY.glob("c85_*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] in forbidden_roots for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module.split(".")[0] not in forbidden_roots
                if module.startswith("oaci."):
                    assert module.split(".")[1] not in forbidden_oaci


def test_theory_sources_contain_no_external_project_data_root() -> None:
    text = "\n".join(path.read_text() for path in THEORY.glob("c85_*.py"))
    assert "/projects/EEG-foundation-model" not in text
    assert "C84S_RESULT_DIR" not in text
    assert "target_construction_label_view" not in text
