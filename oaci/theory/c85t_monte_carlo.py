"""Deterministic Monte Carlo engines for C85T and shadow readiness fixtures."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .c85_decision_experiments import DecisionContractError
from .c85t_exact_scenarios import as_fraction, near_optimal_geometry
from .c85t_rng import (
    REGISTERED_EXECUTION_TOKEN,
    draw_s9_rademacher_prefixes,
    draw_standard_normal,
)


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise DecisionContractError("Monte Carlo summary requires finite 1D values")
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(array.size))


def _interval(mean: float, standard_error: float) -> tuple[float, float]:
    return mean - 1.96 * standard_error, mean + 1.96 * standard_error


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def simulate_near_optimal_selection(
    *,
    scenario_id: str,
    utilities: Sequence[float],
    epsilon: float,
    tau: float,
    pairwise_sigma: float,
    replicates: int,
    execution_token: str | None = None,
) -> dict[str, Any]:
    """Run the locked independent-action-error model in replicate order."""

    values = np.asarray(utilities, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise DecisionContractError("near-optimal utilities must be finite 1D values")
    if replicates != 4096:
        raise DecisionContractError("C85T Monte Carlo replicate count must equal 4096")
    geometry = near_optimal_geometry(
        tuple(float(value) for value in values), epsilon, tau, pairwise_sigma
    )
    optimum_action = int(np.argmax(values))
    near = set(geometry["epsilon_near_optimal_set"])
    selected = np.empty(replicates, dtype="<u2")
    top1 = np.empty(replicates, dtype=np.uint8)
    outside = np.empty(replicates, dtype=np.uint8)
    regrets = np.empty(replicates, dtype="<f8")
    scale = pairwise_sigma / math.sqrt(2.0)
    for replicate in range(replicates):
        standard = draw_standard_normal(
            scenario_id,
            replicate,
            values.size,
            execution_token=execution_token,
        )
        estimated = values + standard * scale
        action = int(np.argmax(estimated))
        selected[replicate] = action
        top1[replicate] = int(action == optimum_action)
        outside[replicate] = int(action not in near)
        regrets[replicate] = float(values[optimum_action] - values[action])

    top1_mean, top1_se = _mean_se(top1)
    outside_mean, outside_se = _mean_se(outside)
    regret_mean, regret_se = _mean_se(regrets)
    return {
        "scenario_id": scenario_id,
        "replicates": replicates,
        "selected_action_counts": {
            str(action): int(np.count_nonzero(selected == action))
            for action in range(values.size)
        },
        "top_1_probability": top1_mean,
        "top_1_monte_carlo_se": top1_se,
        "top_1_95pct_mc_interval": _interval(top1_mean, top1_se),
        "outside_A_epsilon_probability": outside_mean,
        "outside_A_epsilon_monte_carlo_se": outside_se,
        "outside_A_epsilon_95pct_mc_interval": _interval(outside_mean, outside_se),
        "mean_regret": regret_mean,
        "mean_regret_monte_carlo_se": regret_se,
        "mean_regret_95pct_mc_interval": _interval(regret_mean, regret_se),
        "geometry": geometry,
        "raw_output_sha256": _array_digest(selected, top1, outside, regrets),
    }


def _design_estimate(
    low: np.ndarray,
    high: np.ndarray,
    *,
    low_count: int,
    high_count: int,
    stratum_masses: tuple[float, float],
    base_losses: np.ndarray,
    action1_offset: float,
    sigmas: tuple[float, float],
) -> np.ndarray:
    if low_count <= 0 or high_count <= 0:
        raise DecisionContractError("each S9 design stratum count must be positive")
    result = np.asarray(base_losses, dtype=np.float64).copy()
    result[1] = (
        result[0]
        + action1_offset
        + stratum_masses[0] * sigmas[0] * float(low[:low_count].mean())
        + stratum_masses[1] * sigmas[1] * float(high[:high_count].mean())
    )
    return result


def simulate_full_information_designs(
    *,
    scenario_id: str,
    replicates: int,
    stratum_masses: tuple[float, float],
    sigmas: tuple[float, float],
    passive_allocation: tuple[int, int],
    neyman_allocation: tuple[int, int],
    population_mean_losses: Sequence[float],
    action1_offset: float,
    execution_token: str | None = None,
) -> dict[str, Any]:
    """Compare paired passive and Neyman full-information designs."""

    if replicates != 4096:
        raise DecisionContractError("C85T Monte Carlo replicate count must equal 4096")
    if passive_allocation != (51, 13) or neyman_allocation != (18, 46):
        raise DecisionContractError("S9 prefix coupling requires allocations 51/13 and 18/46")
    if len(stratum_masses) != 2 or not math.isclose(sum(stratum_masses), 1.0):
        raise DecisionContractError("stratum masses must sum to one")
    population = np.asarray(population_mean_losses, dtype=np.float64)
    if population.shape != (4,) or not np.isfinite(population).all():
        raise DecisionContractError("full-information population must contain four losses")
    true_best = int(np.argmin(population))
    base = population.copy()
    base[1] = base[0]

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for design in ("passive", "neyman"):
        arrays[design] = {
            "selected": np.empty(replicates, dtype=np.uint8),
            "correct": np.empty(replicates, dtype=np.uint8),
            "top2": np.empty(replicates, dtype=np.uint8),
            "regret": np.empty(replicates, dtype="<f8"),
            "d_hat": np.empty(replicates, dtype="<f8"),
        }

    for replicate in range(replicates):
        low, high = draw_s9_rademacher_prefixes(
            scenario_id, replicate, execution_token=execution_token
        )
        estimates = {
            "passive": _design_estimate(
                low,
                high,
                low_count=passive_allocation[0],
                high_count=passive_allocation[1],
                stratum_masses=stratum_masses,
                base_losses=base,
                action1_offset=action1_offset,
                sigmas=sigmas,
            ),
            "neyman": _design_estimate(
                low,
                high,
                low_count=neyman_allocation[0],
                high_count=neyman_allocation[1],
                stratum_masses=stratum_masses,
                base_losses=base,
                action1_offset=action1_offset,
                sigmas=sigmas,
            ),
        }
        for design, estimated in estimates.items():
            order = np.argsort(estimated, kind="stable")
            action = int(order[0])
            arrays[design]["selected"][replicate] = action
            arrays[design]["correct"][replicate] = int(action == true_best)
            arrays[design]["top2"][replicate] = int(true_best in order[:2])
            arrays[design]["regret"][replicate] = population[action] - population[true_best]
            arrays[design]["d_hat"][replicate] = estimated[1] - estimated[0]

    summaries: dict[str, Any] = {}
    for design, values in arrays.items():
        selected = values["selected"]
        regret_mean, regret_se = _mean_se(values["regret"])
        correct_mean, correct_se = _mean_se(values["correct"])
        top2_mean, top2_se = _mean_se(values["top2"])
        d_mean, d_se = _mean_se(values["d_hat"])
        summaries[design] = {
            "selected_action_counts": {
                str(action): int(np.count_nonzero(selected == action))
                for action in range(population.size)
            },
            "mean_selection_regret": regret_mean,
            "mean_selection_regret_mc_se": regret_se,
            "correct_best_probability": correct_mean,
            "correct_best_mc_se": correct_se,
            "top_2_coverage": top2_mean,
            "top_2_mc_se": top2_se,
            "d_hat_mean": d_mean,
            "d_hat_sample_variance": float(values["d_hat"].var(ddof=1)),
            "d_hat_mean_mc_se": d_se,
            "raw_output_sha256": _array_digest(*values.values()),
        }

    paired: dict[str, Any] = {}
    for endpoint in ("regret", "correct", "top2", "d_hat"):
        difference = arrays["passive"][endpoint].astype(np.float64) - arrays["neyman"][endpoint].astype(np.float64)
        mean, se = _mean_se(difference)
        paired[f"passive_minus_neyman_{endpoint}"] = {
            "mean": mean,
            "monte_carlo_se": se,
            "95pct_mc_interval": _interval(mean, se),
        }

    analytic = {
        "passive_d_hat_variance": sum(
            mass * mass * sigma * sigma / count
            for mass, sigma, count in zip(stratum_masses, sigmas, passive_allocation)
        ),
        "neyman_d_hat_variance": sum(
            mass * mass * sigma * sigma / count
            for mass, sigma, count in zip(stratum_masses, sigmas, neyman_allocation)
        ),
    }
    return {
        "scenario_id": scenario_id,
        "replicates": replicates,
        "designs": summaries,
        "paired_differences": paired,
        "analytic_variance": analytic,
        "universal_active_superiority_claim": False,
    }


def execute_registered_monte_carlo(
    contract: Mapping[str, Any], *, execution_token: str
) -> dict[str, Any]:
    if execution_token != REGISTERED_EXECUTION_TOKEN:
        raise DecisionContractError("registered Monte Carlo requires C85T authorization")
    scenarios = {row["id"]: row for row in contract["scenarios"]}
    result: dict[str, Any] = {}
    for scenario_id in ("S6", "S7"):
        row = scenarios[scenario_id]
        result[scenario_id] = simulate_near_optimal_selection(
            scenario_id=scenario_id,
            utilities=row["utilities"][0],
            epsilon=float(row["epsilon"]),
            tau=float(row["tau"]),
            pairwise_sigma=float(row["pairwise_sigma"]),
            replicates=int(row["sample_size"]),
            execution_token=execution_token,
        )
    s9 = scenarios["S9"]
    result["S9"] = simulate_full_information_designs(
        scenario_id="S9",
        replicates=int(s9["sample_size"]),
        stratum_masses=tuple(float(as_fraction(value)) for value in s9["stratum_probabilities"]),
        sigmas=(1 / 50, 1 / 5),
        passive_allocation=(s9["passive_allocation"]["L"], s9["passive_allocation"]["H"]),
        neyman_allocation=(s9["neyman_allocation"]["L"], s9["neyman_allocation"]["H"]),
        population_mean_losses=tuple(float(as_fraction(value)) for value in s9["population_mean_losses"]),
        action1_offset=1 / 20,
        execution_token=execution_token,
    )
    return result
