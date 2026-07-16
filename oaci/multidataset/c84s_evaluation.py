"""Held-evaluation utility and decision endpoints for immutable selections."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from .c84s_common import require
from .c84s_q0_budget import endpoint_metrics, midrank_percentile


CANDIDATES = 81


def context_candidate_utility(
    candidate_logits: np.ndarray, evaluation_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(candidate_logits, dtype=float)
    labels = np.asarray(evaluation_labels, dtype=int)
    require(logits.ndim == 3 and logits.shape[0] == CANDIDATES and logits.shape[2] == 2,
            "evaluation candidate logits shape drift")
    require(logits.shape[1] == len(labels), "evaluation trial count drift")
    metrics = np.asarray([
        list(endpoint_metrics(logits[index], labels).values())
        for index in range(CANDIDATES)
    ])
    oriented = np.column_stack((
        midrank_percentile(metrics[:, 0]),
        midrank_percentile(-metrics[:, 1]),
        midrank_percentile(-metrics[:, 2]),
    ))
    return np.mean(oriented, axis=1), metrics


def standardized_regret(utility: np.ndarray, selected_index: int) -> float:
    utility = np.asarray(utility, dtype=float)
    require(utility.shape == (CANDIDATES,) and np.all(np.isfinite(utility)), "utility vector drift")
    require(0 <= int(selected_index) < CANDIDATES, "selected candidate index drift")
    spread = float(np.max(utility) - np.min(utility))
    if spread <= 1e-15:
        return 0.0
    return float((np.max(utility) - utility[int(selected_index)]) / spread)


def evaluate_order(order: Sequence[int], utility: np.ndarray, regimes: Sequence[str]) -> dict[str, Any]:
    order_array = np.asarray(order, dtype=int)
    utility = np.asarray(utility, dtype=float)
    require(order_array.ndim == 1 and len(order_array) == CANDIDATES, "selection order shape drift")
    require(set(order_array.tolist()) == set(range(CANDIDATES)), "selection order is not a permutation")
    require(len(regimes) == CANDIDATES, "regime vector shape drift")
    best = int(np.lexsort((np.arange(CANDIDATES), -utility))[0])
    selected = int(order_array[0])
    return {
        "standardized_regret": standardized_regret(utility, selected),
        "selected_utility": float(utility[selected]),
        "top1": int(best in set(order_array[:1])),
        "top5": int(best in set(order_array[:5])),
        "top10": int(best in set(order_array[:10])),
        "coverage": 1.0,
        "selected_regime": str(regimes[selected]),
    }


def evaluate_uniform_random(utility: np.ndarray) -> dict[str, Any]:
    utility = np.asarray(utility, dtype=float)
    regrets = np.asarray([standardized_regret(utility, index) for index in range(CANDIDATES)])
    return {
        "standardized_regret": float(np.mean(regrets)),
        "selected_utility": float(np.mean(utility)),
        "top1": 1.0 / CANDIDATES,
        "top5": 5.0 / CANDIDATES,
        "top10": 10.0 / CANDIDATES,
        "coverage": 1.0,
        "selected_regime": "ANALYTIC_UNIFORM_RANDOM",
    }


def evaluate_oracle(utility: np.ndarray, regimes: Sequence[str]) -> dict[str, Any]:
    utility = np.asarray(utility, dtype=float)
    best = int(np.lexsort((np.arange(CANDIDATES), -utility))[0])
    return {
        "standardized_regret": 0.0,
        "selected_utility": float(utility[best]),
        "top1": 1.0, "top5": 1.0, "top10": 1.0, "coverage": 1.0,
        "selected_regime": str(regimes[best]),
    }


def source_relative_regret_gain(source_regret: float, method_regret: float) -> float:
    source = float(source_regret)
    method = float(method_regret)
    require(np.isfinite(source) and np.isfinite(method), "regret is nonfinite")
    if source <= 1e-15:
        return 0.0
    return float((source - method) / source)


def measurement_metrics(
    score: np.ndarray,
    utility: np.ndarray,
    *,
    estimate_semantics: bool,
    estimated_performance_target: np.ndarray | None = None,
) -> dict[str, Any]:
    score = np.asarray(score, dtype=float)
    utility = np.asarray(utility, dtype=float)
    require(score.shape == utility.shape == (CANDIDATES,), "measurement vector shape drift")
    spearman = float(stats.spearmanr(score, utility)[0])
    kendall = float(stats.kendalltau(score, utility)[0])
    if not np.isfinite(spearman):
        spearman = 0.0
    if not np.isfinite(kendall):
        kendall = 0.0
    left, right = np.triu_indices(CANDIDATES, 1)
    score_delta, utility_delta = score[left] - score[right], utility[left] - utility[right]
    informative = (np.abs(score_delta) > 1e-15) & (np.abs(utility_delta) > 1e-15)
    pairwise = float(np.mean(np.sign(score_delta[informative]) == np.sign(utility_delta[informative]))) if np.any(informative) else 0.5
    output: dict[str, Any] = {
        "Spearman": spearman, "Kendall": kendall,
        "pairwise_ordering_accuracy": pairwise,
        "performance_estimate_semantics": bool(estimate_semantics),
        "accuracy_estimation_MAE": None,
        "incremental_R2": None,
    }
    if estimate_semantics:
        require(estimated_performance_target is not None,
                "performance-estimate metric requires its registered measured target")
        target = np.asarray(estimated_performance_target, dtype=float)
        require(target.shape == (CANDIDATES,) and np.all(np.isfinite(target)),
                "estimated-performance target vector drift")
        residual = target - score
        output["accuracy_estimation_MAE"] = float(np.mean(np.abs(residual)))
        denominator = float(np.sum((target - np.mean(target)) ** 2))
        output["incremental_R2"] = float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 1e-15 else 0.0
    return output


def reliability_between_construction_and_evaluation(
    construction_score: np.ndarray, evaluation_utility: np.ndarray,
) -> dict[str, float]:
    metrics = measurement_metrics(construction_score, evaluation_utility, estimate_semantics=False)
    return {
        "Spearman": float(metrics["Spearman"]),
        "Kendall": float(metrics["Kendall"]),
        "pairwise_ordering_accuracy": float(metrics["pairwise_ordering_accuracy"]),
    }
