"""Equal-target robust-risk summaries on historical standardized regret."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CVAR_ALPHA_GRID = (0.50, 0.75, 0.90)


class C85ERiskError(ValueError):
    """Raised when target-risk inputs violate the registered aggregation."""


def empirical_upper_cvar(losses: Sequence[float], alpha: float) -> float:
    """Exact equal-mass empirical upper-loss CVaR with fractional boundary mass."""
    value = np.sort(np.asarray(losses, dtype=np.float64))
    if value.ndim != 1 or len(value) == 0 or not np.all(np.isfinite(value)):
        raise C85ERiskError("empirical CVaR requires a nonempty finite loss vector")
    if float(alpha) not in CVAR_ALPHA_GRID:
        raise C85ERiskError("empirical CVaR alpha is outside the locked grid")
    n = len(value)
    mass = 0.0
    integral = 0.0
    for index, loss in enumerate(value):
        lower = index / n
        upper = (index + 1) / n
        segment = max(0.0, upper - max(float(alpha), lower))
        integral += segment * float(loss)
        mass += segment
    expected_mass = 1.0 - float(alpha)
    if not np.isclose(mass, expected_mass, atol=1e-15, rtol=0.0):
        raise C85ERiskError("empirical CVaR boundary mass drift")
    return integral / expected_mass


def _target_rows(
    context_rows: Iterable[Mapping[str, Any]], *, level: int | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in context_rows:
        if level is not None and int(row["level"]) != level:
            continue
        key = (str(row["dataset"]), str(row["target_subject_id"]), str(row["method_id"]))
        value = float(row["standardized_regret"])
        if not np.isfinite(value):
            raise C85ERiskError("nonfinite standardized target regret")
        grouped[key].append(value)
    expected_repeats = 8 if level is None else 4
    result = []
    for (dataset, target, method), values in sorted(grouped.items()):
        if len(values) != expected_repeats:
            raise C85ERiskError("target context-repeat coverage drift")
        result.append({
            "dataset": dataset,
            "target_subject_id": target,
            "method_id": method,
            "level": "ALL" if level is None else level,
            "standardized_target_regret": float(np.mean(values)),
            "context_repeats": len(values),
        })
    return result


def target_equal_regrets(
    context_rows: Iterable[Mapping[str, Any]], *, level: int | None = None,
) -> list[dict[str, Any]]:
    """Aggregate 8 full-panel or 4 level-specific contexts within each target."""
    if level not in (None, 0, 1):
        raise C85ERiskError("level must be None, 0, or 1")
    return _target_rows(context_rows, level=level)


def robust_risk_profile(target_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize one target-equal risk vector per dataset/method/level."""
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in target_rows:
        grouped[(str(row["dataset"]), str(row["method_id"]), str(row["level"]))].append(
            float(row["standardized_target_regret"])
        )
    output: list[dict[str, Any]] = []
    for (dataset, method, level), values in sorted(grouped.items()):
        losses = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(losses)):
            raise C85ERiskError("nonfinite target-risk profile")
        row: dict[str, Any] = {
            "dataset": dataset,
            "method_id": method,
            "level": level,
            "target_count": len(losses),
            "mean_standardized_regret": float(np.mean(losses)),
            "median_standardized_regret": float(np.median(losses)),
            "worst_target_standardized_regret": float(np.max(losses)),
            "upper_quantile_0_50": float(np.quantile(losses, 0.50, method="inverted_cdf")),
            "upper_quantile_0_75": float(np.quantile(losses, 0.75, method="inverted_cdf")),
            "upper_quantile_0_90": float(np.quantile(losses, 0.90, method="inverted_cdf")),
            "risk_scale": "HISTORICAL_C84_STANDARDIZED_REGRET",
            "principal_group": "TARGET_SUBJECT",
            "result_tag": "POST_C84S_EXPLORATORY",
        }
        for alpha in CVAR_ALPHA_GRID:
            row[f"CVaR_{alpha:.2f}"] = empirical_upper_cvar(losses, alpha)
        output.append(row)
    return output


def target_reference_improvement(
    method_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    method = {
        (str(row["dataset"]), str(row["target_subject_id"]), str(row["level"])): row
        for row in method_rows
    }
    reference = {
        (str(row["dataset"]), str(row["target_subject_id"]), str(row["level"])): row
        for row in reference_rows
    }
    if len(method) != len(method_rows) or len(reference) != len(reference_rows) or set(method) != set(reference):
        raise C85ERiskError("target reference-risk identity mismatch")
    return [{
        "dataset": key[0], "target_subject_id": key[1], "level": key[2],
        "standardized_regret_improvement": float(
            reference[key]["standardized_target_regret"]
        ) - float(method[key]["standardized_target_regret"]),
        "result_tag": "POST_C84S_EXPLORATORY",
    } for key in sorted(method)]


__all__ = [
    "CVAR_ALPHA_GRID", "C85ERiskError", "empirical_upper_cvar",
    "robust_risk_profile", "target_equal_regrets", "target_reference_improvement",
]
