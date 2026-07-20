"""Descriptive rank, top-k, regret, and geometry separation for C85E."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CONTEXT_FIELDS = (
    "dataset", "target_subject_id", "panel", "training_seed", "level",
)


class C85ERankGeometryError(ValueError):
    """Raised when frozen measurement and geometry identities do not align."""


def _context_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in CONTEXT_FIELDS)


def join_measurement_and_geometry(
    method_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    *, divergence_by_context: Mapping[tuple[Any, ...], float] | None = None,
) -> list[dict[str, Any]]:
    """Join by exact context identity while preserving applicability nulls."""
    geometry = {_context_key(row): row for row in geometry_rows}
    if len(geometry) != len(geometry_rows):
        raise C85ERankGeometryError("duplicate geometry context identity")
    output: list[dict[str, Any]] = []
    for method in method_rows:
        key = _context_key(method)
        if key not in geometry:
            raise C85ERankGeometryError("method row lacks exact geometry context")
        row = geometry[key]
        rank_applicable = bool(int(method["rank_measurement_applicable"]))
        joined = {
            **{field: method[field] for field in CONTEXT_FIELDS},
            "method_id": method["method_id"],
            "rank_measurement_applicable": rank_applicable,
            "Spearman": None if not rank_applicable else float(method["Spearman"]),
            "Kendall": None if not rank_applicable else float(method["Kendall"]),
            "pairwise_ordering_accuracy": None if not rank_applicable else float(
                method["pairwise_ordering_accuracy"]
            ),
            "top1": float(method["top1"]),
            "top5": float(method["top5"]),
            "top10": float(method["top10"]),
            "selected_standardized_regret": float(method["standardized_regret"]),
            "selected_utility": float(method["selected_utility"]),
            "action_divergence": None if divergence_by_context is None else float(
                divergence_by_context[key]
            ),
            "best_second_raw_utility_gap": float(row["best_second_raw_utility_gap"]),
            "best_fifth_raw_utility_gap": float(row["best_fifth_raw_utility_gap"]),
            "best_tenth_raw_utility_gap": float(row["best_tenth_raw_utility_gap"]),
            "utility_range": float(row["utility_range"]),
            "result_tag": "POST_C84S_EXPLORATORY",
        }
        numeric = [
            value for key_name, value in joined.items()
            if key_name not in CONTEXT_FIELDS and key_name not in {
                "method_id", "result_tag", "rank_measurement_applicable"
            } and value is not None
        ]
        if not all(np.isfinite(float(value)) for value in numeric):
            raise C85ERankGeometryError("nonfinite joined measurement")
        output.append(joined)
    return output


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def descriptive_rank_association(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman coefficient only; no p-value is computed or returned."""
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise C85ERankGeometryError("descriptive association shape drift")
    if len(left) < 2:
        return None
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise C85ERankGeometryError("descriptive association is nonfinite")
    left_rank, right_rank = _midranks(left), _midranks(right)
    if np.std(left_rank) <= 1e-15 or np.std(right_rank) <= 1e-15:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def target_equal_rows(
    rows: Iterable[Mapping[str, Any]], *, value_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Average repeated contexts within target before any dataset summary."""
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["dataset"]), str(row["target_subject_id"]),
            str(row["method_id"]), int(row["level"]),
        )
        grouped[key].append(row)
    result: list[dict[str, Any]] = []
    for (dataset, target, method, level), group in sorted(grouped.items()):
        if len(group) != 4:
            raise C85ERankGeometryError("level-specific target repeat count must be four")
        item: dict[str, Any] = {
            "dataset": dataset, "target_subject_id": target,
            "method_id": method, "level": level, "context_repeats": 4,
            "result_tag": "POST_C84S_EXPLORATORY",
        }
        for field in value_fields:
            values = [row[field] for row in group]
            item[field] = None if any(value is None for value in values) else float(np.mean(values))
        result.append(item)
    return result


def geometry_regret_associations(
    target_rows: Sequence[Mapping[str, Any]], *, geometry_fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        grouped[(str(row["dataset"]), str(row["method_id"]), int(row["level"]))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, method, level), group in sorted(grouped.items()):
        regret = [float(row["selected_standardized_regret"]) for row in group]
        for field in geometry_fields:
            coefficient = descriptive_rank_association(
                [float(row[field]) for row in group], regret,
            )
            output.append({
                "dataset": dataset, "method_id": method, "level": level,
                "geometry_field": field,
                "descriptive_spearman": coefficient,
                "target_count": len(group),
                "p_value": None,
                "result_tag": "POST_C84S_EXPLORATORY",
            })
    return output


def leave_one_target_sign_stability(
    target_rows: Sequence[Mapping[str, Any]], *, effect_field: str,
) -> list[dict[str, Any]]:
    """Describe leave-target sign stability without inference or p-values."""
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        grouped[(str(row["dataset"]), str(row["method_id"]), int(row["level"]))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, method, level), group in sorted(grouped.items()):
        effects = np.asarray([float(row[effect_field]) for row in group], dtype=np.float64)
        full_sign = int(np.sign(np.mean(effects)))
        omitted_signs = (
            [int(np.sign(np.mean(np.delete(effects, index)))) for index in range(len(effects))]
            if len(effects) > 1 else []
        )
        output.append({
            "dataset": dataset, "method_id": method, "level": level,
            "target_count": len(effects), "full_mean_sign": full_sign,
            "leave_target_defined": bool(omitted_signs),
            "leave_target_same_sign_count": None if not omitted_signs else int(
                np.sum(np.asarray(omitted_signs) == full_sign)
            ),
            "leave_target_sign_flip_count": None if not omitted_signs else int(
                np.sum(np.asarray(omitted_signs) != full_sign)
            ),
            "p_value": None, "result_tag": "POST_C84S_EXPLORATORY",
        })
    return output


__all__ = [
    "C85ERankGeometryError", "descriptive_rank_association",
    "geometry_regret_associations", "join_measurement_and_geometry",
    "leave_one_target_sign_stability", "target_equal_rows",
]
