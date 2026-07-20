"""Pure shadow tests for C86 active-estimation and robust-risk semantics."""
from __future__ import annotations

import ast
import itertools
from pathlib import Path

import numpy as np
import pytest

from oaci.theory.c86_active_program import (
    C86PContractError,
    empirical_upper_cvar,
    estimate_historical_composite,
    lure_mean,
    lure_weights,
    midrank_percentile,
    mixed_query_probabilities,
)


ROOT = Path(__file__).resolve().parents[2]


def test_uniform_sampling_reduces_lure_weights_to_one() -> None:
    n, budget = 40, 8
    probabilities = [1.0 / (n - step) for step in range(budget)]
    assert np.array_equal(lure_weights(n, budget, probabilities), np.ones(budget))
    values = np.arange(budget, dtype=float)
    assert lure_mean(values, population_size=n, query_probabilities=probabilities) == pytest.approx(3.5)


def test_lure_is_exactly_unbiased_under_adaptive_without_replacement_enumeration() -> None:
    population = np.array([0.0, 1.0, 4.0])
    first = np.array([0.6, 0.3, 0.1])
    expected = 0.0
    for i, j in itertools.permutations(range(3), 2):
        remaining = [index for index in range(3) if index != i]
        second_weights = np.array([first[index] for index in remaining])
        second_weights /= second_weights.sum()
        q2 = float(second_weights[remaining.index(j)])
        path_probability = float(first[i] * q2)
        estimate = float(lure_mean(
            population[[i, j]], population_size=3,
            query_probabilities=[float(first[i]), q2],
        ))
        expected += path_probability * estimate
    assert expected == pytest.approx(float(np.mean(population)), abs=1e-14)


def test_query_probability_floor_and_all_zero_fallback() -> None:
    probabilities = mixed_query_probabilities([0.0, 1.0, 3.0])
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.min(probabilities) >= 0.05 / 3
    assert np.array_equal(mixed_query_probabilities([0.0, 0.0]), np.array([0.5, 0.5]))
    with pytest.raises(C86PContractError, match="invalid query score"):
        mixed_query_probabilities([0.0, -1.0])


def test_historical_composite_shadow_estimator_preserves_shape_and_tie_rule() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)
    probabilities = np.empty((4, 81, 2), dtype=np.float64)
    for candidate in range(81):
        strength = 0.51 + 0.48 * candidate / 80.0
        for row, label in enumerate(labels):
            probabilities[row, candidate, label] = strength
            probabilities[row, candidate, 1 - label] = 1.0 - strength
    result = estimate_historical_composite(
        probabilities, labels, population_size=40,
        query_probabilities=[1 / 40, 1 / 39, 1 / 38, 1 / 37],
    )
    assert result["balanced_accuracy"].shape == (81,)
    assert result["NLL"].shape == result["ECE"].shape == (81,)
    assert result["composite_utility"].shape == (81,)
    assert result["selected_action"] == 80
    tied = midrank_percentile(np.ones(81))
    assert np.all(tied == 0.5)


def test_missing_queried_class_uses_locked_symmetric_jeffreys_plugin() -> None:
    probabilities = np.full((4, 81, 2), 0.5, dtype=np.float64)
    result = estimate_historical_composite(
        probabilities, [0, 0, 0, 0], population_size=40,
        query_probabilities=[1 / 40, 1 / 39, 1 / 38, 1 / 37],
    )
    assert np.all(np.isfinite(result["balanced_accuracy"]))
    assert np.all(result["balanced_accuracy"] == pytest.approx(0.7439024390243902))
    assert result["selected_action"] == 0


def test_empirical_cvar_uses_fractional_boundary_mass() -> None:
    losses = [0.0, 1.0, 2.0]
    assert empirical_upper_cvar(losses, 0.50) == pytest.approx(5.0 / 3.0)
    assert empirical_upper_cvar(losses, 0.75) == pytest.approx(2.0)
    assert empirical_upper_cvar(losses, 0.90) == pytest.approx(2.0)
    with pytest.raises(C86PContractError, match=r"in \(0,1\)"):
        empirical_upper_cvar(losses, 1.0)


def test_c86_shadow_module_has_no_real_data_or_execution_imports() -> None:
    path = ROOT / "oaci/theory/c86_active_program.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    forbidden = ("mne", "moabb", "torch", "oaci.data", "oaci.multidataset", "c85u", "c85e_execute")
    assert not [name for name in imports if name.startswith(forbidden)]
    source = path.read_text(encoding="utf-8")
    assert "get_data(" not in source
    assert "urlretrieve(" not in source
    assert "subprocess" not in imports


def test_synthetic_registry_is_schema_only_not_registered_execution() -> None:
    path = ROOT / "oaci/reports/c86p_tables/synthetic_scenario_registry.csv"
    text = path.read_text(encoding="utf-8")
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["scenario_id"] for row in rows] == [f"C86S{index:02d}" for index in range(11)]
    assert all(row["C86P_mode"] == "LOCKED_SCHEMA_ONLY_NOT_EXECUTED" for row in rows)
    assert all(row["target_groups"] == "12" and row["contexts_per_target"] == "8" for row in rows)
    assert all(row["candidate_count"] == "81" and row["active_chains"] == "2048" for row in rows)
    assert all(row["registered_draws"] == row["real_data"] == "0" for row in rows)
    assert all("C86_SYNTHETIC_V1" in row["seed_rule"] for row in rows)
    assert rows[8]["failure_injection"] == "set_one_post_mixture_remaining_trial_probability_to_zero"
