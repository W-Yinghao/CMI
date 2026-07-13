from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from oaci.conditioned_ceiling_coverage import c80_label_budget_frontier as frontier
from oaci.conditioned_ceiling_coverage import c80_synthetic_label_budget as synthetic


def _rows(name: str):
    with (frontier.TABLE_DIR / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_protocol_hash_replays_exactly():
    assert frontier.sha256_file(frontier.PROTOCOL_PATH) == frontier.PROTOCOL_SHA_PATH.read_text().strip()
    assert frontier.PROTOCOL_SHA_PATH.read_text().strip() == "c6292597f5610cb96e8eaf0313eaa741f8fa9b11dd89ff9e4d84db1fa33add85"


def test_protocol_epistemic_status_is_existing_field_and_not_confirmation():
    protocol, _ = frontier.load_protocol()
    status = protocol["epistemic_status"]
    assert status["designed_after_C79E_outcomes"] is True
    assert status["prospective_to_new_C80_budget_computations"] is True
    assert status["retrospective_existing_field_analysis"] is True
    assert status["independent_confirmation"] is False
    assert status["external_validation"] is False


def test_locked_grid_removes_infeasible_64_before_hash():
    protocol, _ = frontier.load_protocol()
    assert tuple(protocol["budget_design"]["requested_grid"]) == frontier.REQUESTED_BUDGETS
    assert tuple(protocol["budget_design"]["locked_primary_grid"]) == frontier.BUDGETS
    assert 64 not in frontier.BUDGETS
    removed = protocol["budget_design"]["removed_before_hash"]
    assert removed == [{
        "budget": 64,
        "reason": "availability_only_infeasible_minimum_class_count_61",
        "affected_targets": [2, 5, 6, 7, 9],
        "scientific_outcomes_inspected": 0,
    }]


def test_construction_availability_has_minimum_61_and_no_outcome():
    rows = _rows("construction_label_availability.csv")
    assert len(rows) == 8
    assert min(int(row["min_class_count"]) for row in rows) == 61
    assert all(row["scientific_outcome_computed"] == "0" for row in rows)
    assert all(row["seed3_seed4_split_hash_equal"] == "1" for row in rows)


def test_registry_is_exactly_80_of_80():
    assert frontier.registry_audit() == {"paths": 5, "categories": 16, "bound_cells": 80, "blank_cells": 0}


def test_protocol_audit_replays_all_compact_inputs_without_budget_outcome():
    audit = frontier.protocol_audit()
    assert audit["accepted_input_hashes"] == 10
    assert audit["real_budget_statistics"] == 0
    assert audit["same_label_oracle_accessed"] is False
    assert audit["C80E_authorized"] is False


def test_deterministic_stream_seed_is_stable_and_cell_specific():
    first = frontier.deterministic_stream_seed(3, 1, 0, 7)
    assert first == frontier.deterministic_stream_seed(3, 1, 0, 7)
    assert first != frontier.deterministic_stream_seed(4, 1, 0, 7)
    assert first != frontier.deterministic_stream_seed(3, 2, 0, 7)


def test_Q0_nested_sampling_is_class_balanced_and_nested():
    labels = np.repeat(np.arange(4), 66)
    samples = frontier.nested_class_samples(labels, rng=np.random.default_rng(9))
    for budget in frontier.BUDGETS[:-1]:
        selected = samples[budget]
        assert len(selected) == 4 * int(budget)
        assert [int(np.sum(labels[selected] == class_id)) for class_id in range(4)] == [budget] * 4
    for left, right in zip(frontier.BUDGETS[:-2], frontier.BUDGETS[1:-1]):
        assert set(samples[left]).issubset(set(samples[right]))
    assert set(samples[32]).issubset(set(samples["FULL"]))


def test_Q0_infeasible_budget_fails_loudly():
    labels = np.concatenate((np.repeat(0, 31), np.repeat(1, 40), np.repeat(2, 40), np.repeat(3, 40)))
    with pytest.raises(RuntimeError, match="infeasible finite budget 32"):
        frontier.nested_class_samples(labels, rng=np.random.default_rng(1))


def test_exact_selector_score_orients_all_three_endpoints():
    x = np.linspace(0.0, 1.0, frontier.CANDIDATES_PER_CELL)
    metrics = np.column_stack((x, 1.0 - x, 1.0 - x))
    score = frontier.score_from_endpoint_metrics(metrics)
    assert np.all(np.diff(score) > 0)
    assert frontier.descending_candidate_order(score)[0] == 80


def test_exact_tie_rule_uses_reverse_default_argsort_order():
    scores = np.zeros(frontier.CANDIDATES_PER_CELL)
    assert frontier.descending_candidate_order(scores)[0] == 80


def test_standardized_regret_is_zero_best_one_worst():
    utility = np.linspace(0.0, 1.0, frontier.CANDIDATES_PER_CELL)
    assert frontier.standardized_regret(utility, 80) == 0.0
    assert frontier.standardized_regret(utility, 0) == 1.0


def test_standardized_regret_handles_flat_field():
    assert frontier.standardized_regret(np.ones(frontier.CANDIDATES_PER_CELL), 17) == 0.0


def test_exact_maxT_controls_all_registered_budgets_together():
    effects = np.full((frontier.TARGETS, len(frontier.BUDGETS)), 0.15)
    pvalues = frontier.exact_maxT_pvalues(effects)
    assert pvalues.shape == (len(frontier.BUDGETS),)
    assert np.all(pvalues <= 0.05)


def test_bstar_requires_all_larger_budgets_to_qualify():
    base = np.asarray([0.15, 0.00, 0.15, 0.00, 0.15, 0.15, 0.15])
    effects = np.repeat(base[None, :], frontier.TARGETS, axis=0)
    result = frontier.budget_qualification(effects)
    assert result["direct_qualification"].tolist() == [True, False, True, False, True, True, True]
    assert result["closure_qualification"].tolist() == [False, False, False, False, True, True, True]
    assert result["Bstar"] == 16


def test_catastrophic_targets_block_frontier():
    effects = np.full((frontier.TARGETS, len(frontier.BUDGETS)), 0.15)
    effects[0] = -0.20
    result = frontier.budget_qualification(effects)
    assert result["Bstar"] is None
    assert np.all(result["catastrophic"])


def test_bstar_distance_is_grid_steps_and_absence_is_not_stable():
    assert frontier.bstar_grid_distance(8, 32) == 2
    assert frontier.bstar_grid_distance(32, "FULL") == 1
    assert frontier.bstar_grid_distance(None, 8) is None


def test_synthetic_frontier_scenarios_all_pass():
    rows = _rows("synthetic_frontier_calibration.csv")
    assert len(rows) == 9
    assert all(row["passed"] == "1" for row in rows)


def test_synthetic_Bstar_recovery_meets_locked_threshold():
    rows = _rows("synthetic_bstar_recovery.csv")
    assert len(rows) == 18
    assert min(float(row["recovery_rate"]) for row in rows) >= 0.90


def test_synthetic_familywise_error_and_pseudoreplication_controls_pass():
    fwer = _rows("synthetic_familywise_error.csv")
    registered = next(row for row in fwer if row["method"] == "exact_target_signflip_maxT")
    assert float(registered["familywise_error"]) <= 0.06
    dependence = _rows("synthetic_dependence_calibration.csv")
    assert all(row["passed"] == "1" for row in dependence)


def test_monte_carlo_precision_selects_only_2048():
    rows = _rows("monte_carlo_precision_selection.csv")
    passing = [int(row["candidate_chains"]) for row in rows if row["bound_pass"] == row["empirical_pass"] == "1"]
    selected = [int(row["candidate_chains"]) for row in rows if row["selected"] == "1"]
    assert passing == [2048]
    assert selected == [2048]


def test_failed_synthetic_attempt_is_retained():
    rows = _rows("failure_reason_ledger.csv")
    failure = next(row for row in rows if row["stage"] == "synthetic_attempt_1")
    assert failure["blocking"] == "0"
    assert failure["status"] == "CLOSED_IMPLEMENTATION_ONLY_REPAIR"


def test_C80E_real_analysis_fails_closed_without_lock_and_authorization():
    with pytest.raises(RuntimeError):
        frontier.assert_c80e_authorized()
    with pytest.raises(RuntimeError):
        frontier.main(["run-real"])


def test_binding_contract_excludes_target4_oracle_and_real_analysis():
    binding = frontier.binding_contract()
    assert binding["primary_targets"] == [1, 2, 3, 5, 6, 7, 8, 9]
    assert binding["target4_primary"] is False
    assert binding["same_label_oracle"] is False
    assert binding["real_budget_analysis_started"] is False
    assert binding["C80E_authorized"] is False


@pytest.mark.parametrize("module", [frontier, synthetic])
def test_C80P_modules_import_no_EEG_GPU_or_training_packages(module):
    tree = ast.parse(Path(module.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not imported.intersection({"torch", "mne", "moabb"})


def test_C80P_source_has_no_external_payload_or_route_loader():
    source = Path(frontier.__file__).read_text()
    assert "/projects/" not in source
    assert "np.load" not in source
    assert "target_evaluation_view" not in source


def test_risk_register_has_no_open_blocker():
    rows = _rows("risk_register.csv")
    assert len(rows) >= 29
    assert all(row["blocking"] == "0" for row in rows)


def test_no_real_budget_result_artifact_exists():
    forbidden = [
        "C80_LABEL_BUDGET_FRONTIER_RESULT.json",
        "C80_REAL_DATA_BUDGET_CURVE.csv",
        "C80_BSTAR_RESULT.csv",
    ]
    assert all(not (frontier.REPORT_DIR / name).exists() for name in forbidden)
    historical = json.loads((frontier.REPORT_DIR / "C80E_PI_AUTHORIZATION_RECORD.json").read_text())
    assert historical["execution_blocked_by_preflight"] is True
    assert frontier.historical_authorization_usable() is False
