"""Collapse guards for TALOS target-unlabeled predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CollapseGuardResult:
    passed: bool
    entropy_mean: float
    normalized_entropy_mean: float
    max_label_fraction: float
    predicted_label_counts: tuple[int, ...]
    checks: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_collapse_guards(
    proba: np.ndarray,
    *,
    min_normalized_entropy: float = 0.05,
    max_single_label_fraction: float = 0.98,
) -> CollapseGuardResult:
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[0] == 0 or proba.shape[1] < 2:
        raise ValueError("proba must be non-empty with at least two classes")
    entropy = -np.sum(proba * np.log(np.maximum(proba, 1e-8)), axis=1)
    entropy_mean = float(entropy.mean())
    normalized = float(entropy_mean / max(np.log(float(proba.shape[1])), 1e-8))
    pred = proba.argmax(axis=1)
    counts = np.bincount(pred, minlength=proba.shape[1]).astype(np.int64)
    max_frac = float(counts.max() / max(1, counts.sum()))
    checks = []
    warnings = []
    if normalized >= min_normalized_entropy:
        checks.append("entropy_non_degenerate")
    else:
        warnings.append("entropy_collapse")
    if max_frac <= max_single_label_fraction:
        checks.append("not_single_class")
    else:
        warnings.append("single_class_prediction")
    return CollapseGuardResult(
        passed=not warnings,
        entropy_mean=entropy_mean,
        normalized_entropy_mean=normalized,
        max_label_fraction=max_frac,
        predicted_label_counts=tuple(int(x) for x in counts),
        checks=tuple(checks),
        warnings=tuple(warnings),
    )
