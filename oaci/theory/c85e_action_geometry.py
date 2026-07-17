"""Candidate geometry on the frozen composite-utility scale."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


EPSILON_GRID = (0.005, 0.01, 0.02, 0.05)
TAU_GRID = (0.005, 0.01, 0.02, 0.05, 0.10)
CANDIDATES = 81


class C85EGeometryError(ValueError):
    """Raised when a utility vector or geometry grid violates the contract."""


def _utility_vector(utility: Sequence[float]) -> np.ndarray:
    value = np.asarray(utility, dtype=np.float64)
    if value.shape != (CANDIDATES,) or not np.all(np.isfinite(value)):
        raise C85EGeometryError("C85E requires one finite 81-candidate utility vector")
    return value


def raw_utility_gaps(utility: Sequence[float]) -> np.ndarray:
    """Return max utility minus each candidate utility, without normalization."""
    value = _utility_vector(utility)
    return np.max(value) - value


def canonical_utility_order(utility: Sequence[float]) -> np.ndarray:
    value = _utility_vector(utility)
    return np.lexsort((np.arange(CANDIDATES), -value)).astype(np.int16)


def soft_action_weights(utility: Sequence[float], tau: float) -> np.ndarray:
    """Stable soft weights over raw utility gaps."""
    if float(tau) not in TAU_GRID:
        raise C85EGeometryError("C85E tau is outside the locked grid")
    gaps = raw_utility_gaps(utility)
    log_weight = -gaps / float(tau)
    log_weight -= np.max(log_weight)
    weight = np.exp(log_weight)
    weight /= np.sum(weight)
    if not np.all(np.isfinite(weight)) or not np.isclose(np.sum(weight), 1.0):
        raise C85EGeometryError("C85E soft weights are numerically invalid")
    return weight


def context_geometry(
    utility: Sequence[float], *, identity: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]] | dict[str, Any]]:
    """Materialize all locked, descriptive geometry rows for one context."""
    value = _utility_vector(utility)
    prefix = dict(identity or {})
    order = canonical_utility_order(value)
    gaps = raw_utility_gaps(value)
    best = float(value[order[0]])
    summary = {
        **prefix,
        "best_candidate_index": int(order[0]),
        "best_second_raw_utility_gap": float(best - value[order[1]]),
        "best_fifth_raw_utility_gap": float(best - value[order[4]]),
        "best_tenth_raw_utility_gap": float(best - value[order[9]]),
        "utility_range": float(np.max(value) - np.min(value)),
        "exact_comaximizer_count": int(np.sum(value == np.max(value))),
        "geometry_scale": "RAW_COMPOSITE_UTILITY_GAP",
        "result_tag": "POST_C84S_EXPLORATORY",
    }
    near_optimal = [
        {
            **prefix,
            "epsilon": epsilon,
            "near_optimal_set_size": int(np.sum(gaps <= epsilon)),
            "geometry_scale": "RAW_COMPOSITE_UTILITY_GAP",
            "result_tag": "POST_C84S_EXPLORATORY",
        }
        for epsilon in EPSILON_GRID
    ]
    multiplicity: list[dict[str, Any]] = []
    for tau in TAU_GRID:
        weight = soft_action_weights(value, tau)
        positive = weight > 0.0
        hill2 = 1.0 / float(np.sum(weight * weight))
        entropy = float(np.exp(-np.sum(weight[positive] * np.log(weight[positive]))))
        multiplicity.append({
            **prefix,
            "tau": tau,
            "hill2_effective_size": hill2,
            "entropy_effective_size": entropy,
            "geometry_scale": "RAW_COMPOSITE_UTILITY_GAP",
            "result_tag": "POST_C84S_EXPLORATORY",
        })
    return {"summary": summary, "near_optimal": near_optimal, "multiplicity": multiplicity}


__all__ = [
    "CANDIDATES", "C85EGeometryError", "EPSILON_GRID", "TAU_GRID",
    "canonical_utility_order", "context_geometry", "raw_utility_gaps",
    "soft_action_weights",
]
