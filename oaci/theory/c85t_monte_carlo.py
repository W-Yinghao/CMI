"""Deterministic Monte Carlo engines for C85T and shadow readiness fixtures."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .c85_decision_experiments import DecisionContractError
from .c85t_exact_scenarios import as_fraction, near_optimal_geometry
from .c85t_execution_guard import require_registered_capability
from .c85t_rng import (
    canonical_int64_sha256,
    draw_s9_rademacher_int64,
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


def probability_interval_v2(mean: float, standard_error: float) -> dict[str, Any]:
    raw = _interval(mean, standard_error)
    reported = (max(0.0, raw[0]), min(1.0, raw[1]))
    return {
        "raw_95pct_mc_interval": list(raw),
        "reported_95pct_mc_interval": list(reported),
        "interval_clipped": reported != raw,
    }


def _validate_replicate_ids(values: np.ndarray, replicates: int = 4096) -> None:
    observed = np.asarray(values)
    expected = np.arange(replicates, dtype="<u2")
    if observed.dtype != expected.dtype or observed.shape != expected.shape:
        raise DecisionContractError("replicate ID dtype or shape drifted")
    if not np.array_equal(observed, expected):
        raise DecisionContractError("replicate IDs are missing, duplicated, or reordered")


def summarize_near_replicates_v2(
    scenario_id: str, arrays: Mapping[str, np.ndarray], geometry: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "replicate_id": ("<u2", (4096,)),
        "selected_action": ("<u2", (4096,)),
        "top1": ("uint8", (4096,)),
        "outside_A_epsilon": ("uint8", (4096,)),
        "selection_regret": ("<f8", (4096,)),
    }
    if set(arrays) != set(required):
        raise DecisionContractError("S6/S7 replicate field coverage drifted")
    for name, (dtype, shape) in required.items():
        value = np.asarray(arrays[name])
        if value.dtype != np.dtype(dtype) or value.shape != shape:
            raise DecisionContractError(f"S6/S7 replicate field drifted: {name}")
        if not np.isfinite(value).all():
            raise DecisionContractError(f"S6/S7 replicate field is nonfinite: {name}")
    _validate_replicate_ids(arrays["replicate_id"])
    if not np.isin(arrays["top1"], (0, 1)).all() or not np.isin(
        arrays["outside_A_epsilon"], (0, 1)
    ).all():
        raise DecisionContractError("S6/S7 indicator array is not binary")
    top1_mean, top1_se = _mean_se(arrays["top1"])
    outside_mean, outside_se = _mean_se(arrays["outside_A_epsilon"])
    regret_mean, regret_se = _mean_se(arrays["selection_regret"])
    selected = arrays["selected_action"]
    return {
        "scenario_id": scenario_id,
        "replicates": 4096,
        "selected_action_counts": {
            str(action): int(np.count_nonzero(selected == action))
            for action in sorted(set(int(value) for value in selected))
        },
        "top_1_probability": top1_mean,
        "top_1_monte_carlo_se": top1_se,
        "top_1_interval": probability_interval_v2(top1_mean, top1_se),
        "outside_A_epsilon_probability": outside_mean,
        "outside_A_epsilon_monte_carlo_se": outside_se,
        "outside_A_epsilon_interval": probability_interval_v2(
            outside_mean, outside_se
        ),
        "mean_regret": regret_mean,
        "mean_regret_monte_carlo_se": regret_se,
        "mean_regret_raw_95pct_mc_interval": list(_interval(regret_mean, regret_se)),
        "geometry": dict(geometry),
        "raw_output_sha256": _array_digest(
            arrays["replicate_id"],
            arrays["selected_action"],
            arrays["top1"],
            arrays["outside_A_epsilon"],
            arrays["selection_regret"],
        ),
        "aggregate_source": "RELOADED_SAVED_REPLICATE_ARRAYS",
    }


def simulate_near_optimal_selection_v2(
    *,
    scenario_id: str,
    utilities: Sequence[float],
    epsilon: float,
    tau: float,
    pairwise_sigma: float,
    capability: object | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    values = np.asarray(utilities, dtype="<f8")
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise DecisionContractError("near-optimal utilities must be finite 1D values")
    geometry = near_optimal_geometry(
        tuple(float(value) for value in values), epsilon, tau, pairwise_sigma
    )
    optimum_action = int(np.argmax(values))
    near = set(geometry["epsilon_near_optimal_set"])
    arrays = {
        "replicate_id": np.arange(4096, dtype="<u2"),
        "selected_action": np.empty(4096, dtype="<u2"),
        "top1": np.empty(4096, dtype=np.uint8),
        "outside_A_epsilon": np.empty(4096, dtype=np.uint8),
        "selection_regret": np.empty(4096, dtype="<f8"),
    }
    scale = pairwise_sigma / math.sqrt(2.0)
    for replicate in range(4096):
        standard = draw_standard_normal(
            scenario_id,
            replicate,
            values.size,
            capability=capability,
        )
        action = int(np.argmax(values + standard * scale))
        arrays["selected_action"][replicate] = action
        arrays["top1"][replicate] = int(action == optimum_action)
        arrays["outside_A_epsilon"][replicate] = int(action not in near)
        arrays["selection_regret"][replicate] = values[optimum_action] - values[action]
    return summarize_near_replicates_v2(scenario_id, arrays, geometry), arrays


def _summarize_s9_arrays_v2(
    arrays: Mapping[str, np.ndarray], population: np.ndarray
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for design in ("passive", "neyman"):
        names = {
            "replicate_id": f"{design}_replicate_id",
            "selected_action": f"{design}_selected_action",
            "correct_best": f"{design}_correct_best",
            "top2_coverage": f"{design}_top2_coverage",
            "selection_regret": f"{design}_selection_regret",
            "D_hat": f"{design}_D_hat",
        }
        expected = {
            "replicate_id": "<u2",
            "selected_action": "uint8",
            "correct_best": "uint8",
            "top2_coverage": "uint8",
            "selection_regret": "<f8",
            "D_hat": "<f8",
        }
        values: dict[str, np.ndarray] = {}
        for logical, physical in names.items():
            if physical not in arrays:
                raise DecisionContractError(f"S9 replicate field is absent: {physical}")
            value = np.asarray(arrays[physical])
            if value.dtype != np.dtype(expected[logical]) or value.shape != (4096,):
                raise DecisionContractError(f"S9 replicate field drifted: {physical}")
            if not np.isfinite(value).all():
                raise DecisionContractError(f"S9 replicate field is nonfinite: {physical}")
            values[logical] = value
        _validate_replicate_ids(values["replicate_id"])
        for binary in ("correct_best", "top2_coverage"):
            if not np.isin(values[binary], (0, 1)).all():
                raise DecisionContractError(f"S9 indicator is not binary: {binary}")
        regret_mean, regret_se = _mean_se(values["selection_regret"])
        correct_mean, correct_se = _mean_se(values["correct_best"])
        top2_mean, top2_se = _mean_se(values["top2_coverage"])
        d_mean, d_se = _mean_se(values["D_hat"])
        summaries[design] = {
            "selected_action_counts": {
                str(action): int(np.count_nonzero(values["selected_action"] == action))
                for action in range(population.size)
            },
            "mean_selection_regret": regret_mean,
            "mean_selection_regret_mc_se": regret_se,
            "mean_selection_regret_raw_95pct_mc_interval": list(
                _interval(regret_mean, regret_se)
            ),
            "correct_best_probability": correct_mean,
            "correct_best_mc_se": correct_se,
            "correct_best_interval": probability_interval_v2(
                correct_mean, correct_se
            ),
            "top_2_coverage": top2_mean,
            "top_2_mc_se": top2_se,
            "top_2_interval": probability_interval_v2(top2_mean, top2_se),
            "d_hat_mean": d_mean,
            "d_hat_sample_variance": float(values["D_hat"].var(ddof=1)),
            "d_hat_mean_mc_se": d_se,
            "raw_output_sha256": _array_digest(*values.values()),
        }
    paired: dict[str, Any] = {}
    for endpoint in (
        "selection_regret",
        "correct_best",
        "top2_coverage",
        "D_hat",
    ):
        name = f"paired_passive_minus_neyman_{endpoint}"
        if name not in arrays:
            raise DecisionContractError(f"S9 paired endpoint is absent: {name}")
        difference = np.asarray(arrays[name])
        if difference.dtype != np.dtype("<f8") or difference.shape != (4096,):
            raise DecisionContractError(f"S9 paired endpoint drifted: {name}")
        mean, se = _mean_se(difference)
        paired[name] = {
            "mean": mean,
            "monte_carlo_se": se,
            "raw_95pct_mc_interval": list(_interval(mean, se)),
        }
    return {
        "scenario_id": "S9",
        "replicates": 4096,
        "designs": summaries,
        "paired_differences": paired,
        "aggregate_source": "RELOADED_SAVED_REPLICATE_ARRAYS",
    }


def simulate_full_information_designs_v2(
    *,
    scenario_id: str,
    stratum_masses: tuple[float, float],
    sigmas: tuple[float, float],
    passive_allocation: tuple[int, int],
    neyman_allocation: tuple[int, int],
    population_mean_losses: Sequence[float],
    action1_offset: float,
    capability: object | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    if passive_allocation != (51, 13) or neyman_allocation != (18, 46):
        raise DecisionContractError("S9 prefix coupling requires allocations 51/13 and 18/46")
    population = np.asarray(population_mean_losses, dtype="<f8")
    if population.shape != (4,) or not np.isfinite(population).all():
        raise DecisionContractError("S9 population must contain four finite losses")
    true_best = int(np.argmin(population))
    base = population.copy()
    base[1] = base[0]
    arrays: dict[str, np.ndarray] = {}
    for design in ("passive", "neyman"):
        arrays[f"{design}_replicate_id"] = np.arange(4096, dtype="<u2")
        arrays[f"{design}_selected_action"] = np.empty(4096, dtype=np.uint8)
        arrays[f"{design}_correct_best"] = np.empty(4096, dtype=np.uint8)
        arrays[f"{design}_top2_coverage"] = np.empty(4096, dtype=np.uint8)
        arrays[f"{design}_selection_regret"] = np.empty(4096, dtype="<f8")
        arrays[f"{design}_D_hat"] = np.empty(4096, dtype="<f8")
    digest_rows: list[dict[str, Any]] = []
    for replicate in range(4096):
        low_raw, high_raw = draw_s9_rademacher_int64(
            scenario_id, replicate, capability=capability
        )
        if low_raw.dtype != np.dtype("<i8") or high_raw.dtype != np.dtype("<i8"):
            raise DecisionContractError("S9 raw draw dtype must be little-endian int64")
        digest = hashlib.sha256()
        digest.update(low_raw.tobytes(order="C"))
        digest.update(high_raw.tobytes(order="C"))
        digest_rows.append(
            {
                "replicate_id": replicate,
                "L_sha256": canonical_int64_sha256(low_raw),
                "H_sha256": canonical_int64_sha256(high_raw),
                "combined_sha256": digest.hexdigest(),
                "dtype": "<i8",
                "L_count": 51,
                "H_count": 46,
            }
        )
        low = low_raw.astype(np.int8)
        high = high_raw.astype(np.int8)
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
            arrays[f"{design}_selected_action"][replicate] = action
            arrays[f"{design}_correct_best"][replicate] = int(action == true_best)
            arrays[f"{design}_top2_coverage"][replicate] = int(
                true_best in order[:2]
            )
            arrays[f"{design}_selection_regret"][replicate] = (
                population[action] - population[true_best]
            )
            arrays[f"{design}_D_hat"][replicate] = estimated[1] - estimated[0]
    for endpoint in (
        "selection_regret",
        "correct_best",
        "top2_coverage",
        "D_hat",
    ):
        arrays[f"paired_passive_minus_neyman_{endpoint}"] = np.asarray(
            arrays[f"passive_{endpoint}"].astype("<f8")
            - arrays[f"neyman_{endpoint}"].astype("<f8"),
            dtype="<f8",
        )
    summary = _summarize_s9_arrays_v2(arrays, population)
    summary["analytic_variance"] = {
        "passive_d_hat_variance": sum(
            mass * mass * sigma * sigma / count
            for mass, sigma, count in zip(
                stratum_masses, sigmas, passive_allocation
            )
        ),
        "neyman_d_hat_variance": sum(
            mass * mass * sigma * sigma / count
            for mass, sigma, count in zip(
                stratum_masses, sigmas, neyman_allocation
            )
        ),
    }
    summary["universal_active_superiority_claim"] = False
    return summary, arrays, digest_rows


def simulate_near_optimal_selection(
    *,
    scenario_id: str,
    utilities: Sequence[float],
    epsilon: float,
    tau: float,
    pairwise_sigma: float,
    replicates: int,
    capability: object | None = None,
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
            capability=capability,
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
            scenario_id, replicate
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
    contract: Mapping[str, Any], *, capability: object
) -> dict[str, Any]:
    require_registered_capability(capability)
    scenarios = {row["id"]: row for row in contract["scenarios"]}
    result: dict[str, Any] = {}
    for scenario_id in ("S6", "S7"):
        row = scenarios[scenario_id]
        summary, arrays = simulate_near_optimal_selection_v2(
            scenario_id=scenario_id,
            utilities=row["utilities"][0],
            epsilon=float(row["epsilon"]),
            tau=float(row["tau"]),
            pairwise_sigma=float(row["pairwise_sigma"]),
            capability=capability,
        )
        result[scenario_id] = {"summary": summary, "arrays": arrays}
    s9 = scenarios["S9"]
    summary, arrays, digest_rows = simulate_full_information_designs_v2(
        scenario_id="S9",
        stratum_masses=tuple(float(as_fraction(value)) for value in s9["stratum_probabilities"]),
        sigmas=(1 / 50, 1 / 5),
        passive_allocation=(s9["passive_allocation"]["L"], s9["passive_allocation"]["H"]),
        neyman_allocation=(s9["neyman_allocation"]["L"], s9["neyman_allocation"]["H"]),
        population_mean_losses=tuple(float(as_fraction(value)) for value in s9["population_mean_losses"]),
        action1_offset=1 / 20,
        capability=capability,
    )
    result["S9"] = {
        "summary": summary,
        "arrays": arrays,
        "raw_draw_digest_rows": digest_rows,
    }
    return result
