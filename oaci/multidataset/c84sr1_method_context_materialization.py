"""Immutable Stage-B selections to canonical C84SR1 method-context rows."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import stats

from . import c84s_evaluation as evaluation
from .c84s_common import require
from .c84sr1_common import (
    CANDIDATES, FIXED_METHODS, PRIMARY_ZERO_METHODS, Q0_BUDGET_CODES,
    SCORE_METHODS, expected_methods, finite_budgets,
)


METHOD_CONTEXT_FIELDS_V2 = (
    "dataset", "target_subject_id", "panel", "training_seed", "level",
    "method_id", "standardized_regret", "selected_utility",
    "source_relative_regret_gain", "top1", "top5", "top10", "coverage",
    "selected_regime", "rank_measurement_applicable",
    "performance_estimate_applicable", "Spearman", "Kendall",
    "pairwise_ordering_accuracy", "accuracy_estimation_MAE",
)
Q0_REGIME_FIELDS = (
    "dataset", "target_subject_id", "panel", "training_seed", "level",
    "budget", "regime", "chain_count", "fraction",
)
Q0_MC_FIELDS = (
    "dataset", "target_subject_id", "panel", "training_seed", "level",
    "budget", "chain_count", "mean_regret", "regret_MCSE",
    "mean_selected_utility", "selected_utility_min", "selected_utility_max",
    "top1_MCSE", "top5_MCSE", "top10_MCSE",
)
PERFORMANCE_ESTIMATE_METHODS = {"S1", "U7", "U13", "U15"}


def _positions(orders: np.ndarray) -> np.ndarray:
    orders = np.asarray(orders, dtype=np.int16)
    require(orders.ndim == 2 and orders.shape[1] == CANDIDATES,
            "candidate-order matrix shape drift")
    positions = np.empty_like(orders)
    rows = np.arange(len(orders))[:, None]
    positions[rows, orders] = np.arange(CANDIDATES, dtype=np.int16)[None, :]
    return positions


def order_measurement_batch(orders: np.ndarray, utility: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tie-resolved ordinal rank metrics for every Q0 chain."""
    orders = np.asarray(orders, dtype=np.int16)
    utility = np.asarray(utility, dtype=float)
    require(orders.ndim == 2 and orders.shape[1] == CANDIDATES, "Q0 rank order shape drift")
    require(utility.shape == (CANDIDATES,) and np.all(np.isfinite(utility)), "utility vector drift")
    position = _positions(orders).astype(float)
    x = -position
    y = stats.rankdata(utility, method="average")
    x_center = x - np.mean(x, axis=1, keepdims=True)
    y_center = y - np.mean(y)
    denominator = np.sqrt(np.sum(x_center ** 2, axis=1) * np.sum(y_center ** 2))
    spearman = np.divide(
        np.sum(x_center * y_center[None, :], axis=1), denominator,
        out=np.zeros(len(orders), dtype=float), where=denominator > 0,
    )
    left, right = np.triu_indices(CANDIDATES, 1)
    utility_delta = utility[left] - utility[right]
    informative = np.abs(utility_delta) > 1e-15
    predicted = np.sign(position[:, right] - position[:, left])
    truth = np.sign(utility_delta)[None, :]
    concordance = np.sum((predicted[:, informative] * truth[:, informative]) > 0, axis=1)
    discordance = np.sum((predicted[:, informative] * truth[:, informative]) < 0, axis=1)
    informative_count = int(np.sum(informative))
    pairwise = (
        np.full(len(orders), 0.5, dtype=float)
        if informative_count == 0
        else concordance.astype(float) / informative_count
    )
    total_pairs = len(left)
    kendall_denominator = np.sqrt(total_pairs * informative_count)
    kendall = (
        np.zeros(len(orders), dtype=float)
        if kendall_denominator <= 0
        else (concordance - discordance).astype(float) / kendall_denominator
    )
    return spearman, kendall, pairwise


def _batch_endpoints(
    orders: np.ndarray, utility: np.ndarray, regimes: Sequence[str],
) -> dict[str, np.ndarray]:
    orders = np.asarray(orders, dtype=np.int16)
    utility = np.asarray(utility, dtype=float)
    best = int(np.lexsort((np.arange(CANDIDATES), -utility))[0])
    spread = float(np.max(utility) - np.min(utility))
    selected = orders[:, 0]
    selected_utility = utility[selected]
    regret = (
        np.zeros(len(orders), dtype=float)
        if spread <= 1e-15
        else (np.max(utility) - selected_utility) / spread
    )
    regime_array = np.asarray(regimes, dtype=str)
    return {
        "standardized_regret": regret,
        "selected_utility": selected_utility,
        "top1": np.any(orders[:, :1] == best, axis=1).astype(float),
        "top5": np.any(orders[:, :5] == best, axis=1).astype(float),
        "top10": np.any(orders[:, :10] == best, axis=1).astype(float),
        "selected_regime": regime_array[selected],
    }


def _row(
    identity: Mapping[str, Any], method: str, endpoint: Mapping[str, Any],
    *, rank_applicable: bool, performance_applicable: bool,
    measurement: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output = {
        "dataset": str(identity["dataset"]),
        "target_subject_id": str(identity["target_subject_id"]),
        "panel": str(identity["panel"]),
        "training_seed": int(identity["training_seed"]),
        "level": int(identity["level"]),
        "method_id": method,
        "standardized_regret": float(endpoint["standardized_regret"]),
        "selected_utility": float(endpoint["selected_utility"]),
        "source_relative_regret_gain": 0.0,
        "top1": float(endpoint["top1"]),
        "top5": float(endpoint["top5"]),
        "top10": float(endpoint["top10"]),
        "coverage": float(endpoint.get("coverage", 1.0)),
        "selected_regime": str(endpoint["selected_regime"]),
        "rank_measurement_applicable": int(rank_applicable),
        "performance_estimate_applicable": int(performance_applicable),
        "Spearman": None,
        "Kendall": None,
        "pairwise_ordering_accuracy": None,
        "accuracy_estimation_MAE": None,
    }
    if rank_applicable:
        require(measurement is not None, f"rank measurement absent: {method}")
        for field in ("Spearman", "Kendall", "pairwise_ordering_accuracy"):
            output[field] = float(measurement[field])
    if performance_applicable:
        require(measurement is not None and measurement.get("accuracy_estimation_MAE") is not None,
                f"performance-estimate MAE absent: {method}")
        output["accuracy_estimation_MAE"] = float(measurement["accuracy_estimation_MAE"])
    return output


def materialize_context(
    *,
    identity: Mapping[str, Any],
    candidate_ids: Sequence[str],
    regimes: Sequence[str],
    utility: np.ndarray,
    evaluation_metrics: np.ndarray,
    score_vectors: Mapping[str, np.ndarray],
    fixed_selected_indices: Mapping[str, int],
    q0_payload: Mapping[str, np.ndarray],
    q0_chains: int = 2048,
    budget_provider: Callable[[str], tuple[int, ...]] = finite_budgets,
    method_provider: Callable[[str], tuple[str, ...]] = expected_methods,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize all deterministic and stochastic methods for one context."""
    dataset = str(identity["dataset"])
    candidate_ids = tuple(map(str, candidate_ids))
    regimes = tuple(map(str, regimes))
    utility = np.asarray(utility, dtype=float)
    evaluation_metrics = np.asarray(evaluation_metrics, dtype=float)
    require(len(candidate_ids) == len(set(candidate_ids)) == CANDIDATES, "candidate IDs drift")
    require(len(regimes) == CANDIDATES and utility.shape == (CANDIDATES,), "context vector shape drift")
    require(evaluation_metrics.shape == (CANDIDATES, 3), "evaluation metric matrix drift")
    require(set(score_vectors) == set(SCORE_METHODS), "score-method set drift")
    require(set(fixed_selected_indices) == set(FIXED_METHODS), "fixed-method set drift")

    endpoints: dict[str, dict[str, Any]] = {}
    measurements: dict[str, dict[str, Any] | None] = {}
    rank_applicable: dict[str, bool] = {}
    performance_applicable: dict[str, bool] = {}

    endpoints["B0"] = evaluation.evaluate_uniform_random(utility)
    endpoints["B5"] = evaluation.evaluate_oracle(utility, regimes)
    for method in ("B0", "B5"):
        measurements[method] = None
        rank_applicable[method] = performance_applicable[method] = False

    best = int(np.lexsort((np.arange(CANDIDATES), -utility))[0])
    for method in FIXED_METHODS:
        selected = int(fixed_selected_indices[method])
        hit = float(selected == best)
        endpoints[method] = {
            "standardized_regret": evaluation.standardized_regret(utility, selected),
            "selected_utility": float(utility[selected]),
            "top1": hit, "top5": hit, "top10": hit, "coverage": 1.0,
            "selected_regime": regimes[selected],
        }
        measurements[method] = None
        rank_applicable[method] = performance_applicable[method] = False

    for method in SCORE_METHODS:
        score = np.asarray(score_vectors[method], dtype=float)
        require(score.shape == (CANDIDATES,) and np.all(np.isfinite(score)), f"score vector drift: {method}")
        order = np.lexsort((np.arange(CANDIDATES), -score))
        endpoints[method] = evaluation.evaluate_order(order, utility, regimes)
        estimate = method in PERFORMANCE_ESTIMATE_METHODS
        measurements[method] = evaluation.measurement_metrics(
            score, utility, estimate_semantics=estimate,
            estimated_performance_target=evaluation_metrics[:, 0] if estimate else None,
        )
        rank_applicable[method] = True
        performance_applicable[method] = estimate

    q0_regime_rows: list[dict[str, Any]] = []
    q0_mc_rows: list[dict[str, Any]] = []
    finite_codes = np.asarray(q0_payload["finite_budget_code"], dtype=np.uint8)
    finite_orders = np.asarray(q0_payload["finite_candidate_order"], dtype=np.uint8)
    for budget in budget_provider(dataset):
        mask = finite_codes == Q0_BUDGET_CODES[budget]
        orders = finite_orders[mask]
        require(len(orders) == q0_chains, f"Q0 integrated chain count drift: {dataset}/B{budget}")
        batch = _batch_endpoints(orders, utility, regimes)
        spearman, kendall, pairwise = order_measurement_batch(orders, utility)
        method = f"Q0_B{budget}"
        endpoints[method] = {
            "standardized_regret": float(np.mean(batch["standardized_regret"])),
            "selected_utility": float(np.mean(batch["selected_utility"])),
            "top1": float(np.mean(batch["top1"])),
            "top5": float(np.mean(batch["top5"])),
            "top10": float(np.mean(batch["top10"])),
            "coverage": 1.0, "selected_regime": "STOCHASTIC_Q0",
        }
        measurements[method] = {
            "Spearman": float(np.mean(spearman)),
            "Kendall": float(np.mean(kendall)),
            "pairwise_ordering_accuracy": float(np.mean(pairwise)),
            "accuracy_estimation_MAE": None,
        }
        rank_applicable[method] = True
        performance_applicable[method] = False
        selected_regime = np.asarray(batch["selected_regime"], dtype=str)
        for regime in ("ERM", "OACI", "SRC"):
            count = int(np.sum(selected_regime == regime))
            q0_regime_rows.append({
                **{key: identity[key] for key in ("dataset", "target_subject_id", "panel", "training_seed", "level")},
                "budget": str(budget), "regime": regime,
                "chain_count": count, "fraction": count / float(q0_chains),
            })
        def mcse(values: np.ndarray) -> float:
            return float(np.std(values, ddof=1) / np.sqrt(len(values)))
        q0_mc_rows.append({
            **{key: identity[key] for key in ("dataset", "target_subject_id", "panel", "training_seed", "level")},
            "budget": str(budget), "chain_count": q0_chains,
            "mean_regret": float(np.mean(batch["standardized_regret"])),
            "regret_MCSE": mcse(batch["standardized_regret"]),
            "mean_selected_utility": float(np.mean(batch["selected_utility"])),
            "selected_utility_min": float(np.min(batch["selected_utility"])),
            "selected_utility_max": float(np.max(batch["selected_utility"])),
            "top1_MCSE": mcse(batch["top1"]),
            "top5_MCSE": mcse(batch["top5"]),
            "top10_MCSE": mcse(batch["top10"]),
        })

    full_order = np.asarray(q0_payload["FULL_candidate_order"], dtype=np.uint8)[0]
    full_batch = _batch_endpoints(full_order[None, :], utility, regimes)
    full_s, full_k, full_p = order_measurement_batch(full_order[None, :], utility)
    endpoints["Q0_FULL"] = {
        "standardized_regret": float(full_batch["standardized_regret"][0]),
        "selected_utility": float(full_batch["selected_utility"][0]),
        "top1": float(full_batch["top1"][0]), "top5": float(full_batch["top5"][0]),
        "top10": float(full_batch["top10"][0]), "coverage": 1.0,
        "selected_regime": str(full_batch["selected_regime"][0]),
    }
    measurements["Q0_FULL"] = {
        "Spearman": float(full_s[0]), "Kendall": float(full_k[0]),
        "pairwise_ordering_accuracy": float(full_p[0]), "accuracy_estimation_MAE": None,
    }
    rank_applicable["Q0_FULL"] = True
    performance_applicable["Q0_FULL"] = False

    expected = set(method_provider(dataset))
    require(set(endpoints) == expected, f"materialized method set drift: {dataset}")
    source_regret = float(endpoints["S1"]["standardized_regret"])
    rows = []
    for method in method_provider(dataset):
        row = _row(
            identity, method, endpoints[method],
            rank_applicable=rank_applicable[method],
            performance_applicable=performance_applicable[method],
            measurement=measurements[method],
        )
        row["source_relative_regret_gain"] = evaluation.source_relative_regret_gain(
            source_regret, row["standardized_regret"],
        )
        rows.append(row)
    return rows, q0_regime_rows, q0_mc_rows
