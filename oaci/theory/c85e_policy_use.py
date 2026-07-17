"""Realized deterministic and stochastic policy-use summaries for C85E."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CONTEXT_FIELDS = (
    "dataset", "target_subject_id", "panel", "training_seed", "level",
)
DETERMINISTIC_METHODS = (
    "B1", "B2", "B3", "B4O", "B4S", "S1", "U5", "U7", "U11",
    "U13", "U14", "U15", "Q0_FULL",
)
REFERENCE_METHODS = ("B1", "S1")


class C85EPolicyUseError(ValueError):
    """Raised when action records violate identity or weighting contracts."""


def _context_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in CONTEXT_FIELDS)


def _entropy(counts: Iterable[float]) -> float:
    value = np.asarray(tuple(counts), dtype=np.float64)
    if value.ndim != 1 or np.any(value < 0.0) or np.sum(value) <= 0.0:
        raise C85EPolicyUseError("invalid action-distribution masses")
    probability = value / np.sum(value)
    positive = probability > 0.0
    return float(-np.sum(probability[positive] * np.log(probability[positive])))


def _validate_action(value: Any) -> int:
    action = int(value)
    if not 0 <= action <= 80:
        raise C85EPolicyUseError("canonical action must be in 0..80")
    return action


def summarize_deterministic_policy_use(
    method_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    method_id: str,
    reference_id: str,
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two fixed action maps over an explicitly declared scope."""
    method = {_context_key(row): row for row in method_rows}
    reference = {_context_key(row): row for row in reference_rows}
    if len(method) != len(method_rows) or len(reference) != len(reference_rows):
        raise C85EPolicyUseError("duplicate deterministic context identity")
    if not method or set(method) != set(reference):
        raise C85EPolicyUseError("deterministic action-map coverage mismatch")
    divergence: list[int] = []
    regret_differences: list[float] = []
    actions: list[int] = []
    regimes: Counter[str] = Counter()
    target_flags: dict[str, list[int]] = defaultdict(list)
    for key in sorted(method):
        observed = method[key]
        baseline = reference[key]
        action = _validate_action(observed["selected_candidate_index"])
        reference_action = _validate_action(baseline["selected_candidate_index"])
        differs = int(action != reference_action)
        divergence.append(differs)
        actions.append(action)
        regimes[str(observed["selected_regime"])] += 1
        target_flags[str(observed["target_subject_id"])].append(differs)
        difference = float(observed["standardized_regret"]) - float(
            baseline["standardized_regret"]
        )
        if not math.isfinite(difference):
            raise C85EPolicyUseError("nonfinite standardized-regret difference")
        regret_differences.append(difference)
    divergent = np.asarray(divergence, dtype=bool)
    risk = np.asarray(regret_differences, dtype=np.float64)
    total_difference = float(np.sum(risk))
    divergent_sum = float(np.sum(risk[divergent]))
    conditional = None if not np.any(divergent) else float(np.mean(risk[divergent]))
    contribution = None if not np.any(divergent) or abs(total_difference) <= 1e-15 else (
        divergent_sum / total_difference
    )
    action_counts = Counter(actions)
    return {
        **dict(scope),
        "method_id": method_id,
        "reference_id": reference_id,
        "contexts": len(method),
        "action_divergence_rate": float(np.mean(divergent)),
        "exact_equivalence_contexts": int(len(method) - np.sum(divergent)),
        "target_action_divergence_rate": float(np.mean([
            np.mean(flags) for flags in target_flags.values()
        ])),
        "targets_with_any_action_divergence_rate": float(np.mean([
            int(any(flags)) for flags in target_flags.values()
        ])),
        "canonical_action_entropy": _entropy(action_counts.values()),
        "selected_regime_distribution": {
            key: value / len(method) for key, value in sorted(regimes.items())
        },
        "mean_standardized_regret_difference": float(np.mean(risk)),
        "divergent_context_regret_difference": conditional,
        "divergent_context_risk_contribution_fraction": contribution,
        "exact_collapse": bool(not np.any(divergent)),
        "T3_exactly_applicable": bool(not np.any(divergent)),
        "risk_scale": "HISTORICAL_C84_STANDARDIZED_REGRET",
        "result_tag": "POST_C84S_EXPLORATORY",
    }


def exact_equivalence_scopes(
    method_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *, method_id: str, reference_id: str,
) -> list[dict[str, Any]]:
    """Evaluate exact collapse only at the three locked scope families."""
    rows: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in method_rows})
    for dataset in datasets:
        for level in (0, 1):
            method_scope = [
                row for row in method_rows
                if str(row["dataset"]) == dataset and int(row["level"]) == level
            ]
            reference_scope = [
                row for row in reference_rows
                if str(row["dataset"]) == dataset and int(row["level"]) == level
            ]
            rows.append(summarize_deterministic_policy_use(
                method_scope, reference_scope, method_id=method_id,
                reference_id=reference_id,
                scope={"scope": "dataset_x_level", "dataset": dataset, "level": level},
            ))
        method_scope = [row for row in method_rows if str(row["dataset"]) == dataset]
        reference_scope = [row for row in reference_rows if str(row["dataset"]) == dataset]
        rows.append(summarize_deterministic_policy_use(
            method_scope, reference_scope, method_id=method_id,
            reference_id=reference_id,
            scope={"scope": "dataset_full_panel", "dataset": dataset, "level": "ALL"},
        ))
    rows.append(summarize_deterministic_policy_use(
        method_rows, reference_rows, method_id=method_id, reference_id=reference_id,
        scope={"scope": "global_three_dataset_field", "dataset": "ALL", "level": "ALL"},
    ))
    return rows


def summarize_stochastic_q0_context(
    *,
    selected_actions: Sequence[int],
    selected_regimes: Sequence[str],
    reference_action: int,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe a frozen finite-budget Q0 action distribution without collapsing it."""
    actions = np.asarray(selected_actions, dtype=np.int64)
    if actions.shape != (2048,) or np.any((actions < 0) | (actions > 80)):
        raise C85EPolicyUseError("finite Q0 requires exactly 2,048 valid frozen actions")
    if len(selected_regimes) != len(actions):
        raise C85EPolicyUseError("finite Q0 regime/action length mismatch")
    reference = _validate_action(reference_action)
    action_counts = Counter(map(int, actions.tolist()))
    regime_counts = Counter(map(str, selected_regimes))
    return {
        **dict(identity),
        "chains": 2048,
        "probability_action_differs_from_reference": float(np.mean(actions != reference)),
        "action_entropy": _entropy(action_counts.values()),
        "action_distribution": {
            str(key): value / 2048.0 for key, value in sorted(action_counts.items())
        },
        "regime_distribution": {
            key: value / 2048.0 for key, value in sorted(regime_counts.items())
        },
        "stochastic_policy_preserved": True,
        "chains_are_scientific_sample": False,
        "result_tag": "POST_C84S_EXPLORATORY",
    }


__all__ = [
    "CONTEXT_FIELDS", "DETERMINISTIC_METHODS", "REFERENCE_METHODS",
    "C85EPolicyUseError", "exact_equivalence_scopes",
    "summarize_deterministic_policy_use", "summarize_stochastic_q0_context",
]
