"""Source-risk metrics for the risk-feasible trainer.

The **primal** constraint metric must be differentiable: only ``ce`` or ``balanced_ce``.
``balanced_err`` is a 0/1 metric for **guard / report only** and is rejected as a primal
metric. ``balanced_ce`` is defined on the FULL set (the class-balanced mean of per-class mean
CE), so the objective is invariant to how minibatches are formed — see
``per_class_ce_sums`` for the correct cross-batch accumulation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

DIFFERENTIABLE_PRIMAL = ("ce", "balanced_ce")


def assert_differentiable_primal(metric: str) -> None:
    if metric not in DIFFERENTIABLE_PRIMAL:
        raise ValueError(
            f"primal risk metric must be differentiable, one of {DIFFERENTIABLE_PRIMAL}; got "
            f"{metric!r}. 'balanced_err' is a guard/report metric only, never the primal."
        )


def _n_classes(y: torch.Tensor, n_classes: int | None) -> int:
    return int(n_classes) if n_classes is not None else int(y.max().item()) + 1


def task_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy (nats)."""
    return F.cross_entropy(logits, y, reduction="mean")


def per_class_ce_sums(logits: torch.Tensor, y: torch.Tensor, n_classes: int):
    """Per-class (sum of CE, count) — the additive sufficient statistics for ``balanced_ce``.
    Accumulating these across an arbitrary partition and then taking ``mean_c(sum/count)``
    reproduces the full-set ``balanced_ce`` exactly (batch-partition invariant)."""
    per = F.cross_entropy(logits, y, reduction="none")
    sums = torch.zeros(n_classes, dtype=per.dtype)
    counts = torch.zeros(n_classes, dtype=per.dtype)
    for c in range(n_classes):
        m = y == c
        if m.any():
            sums[c] = per[m].sum()
            counts[c] = m.sum()
    return sums, counts


def balanced_ce(logits: torch.Tensor, y: torch.Tensor, n_classes: int | None = None) -> torch.Tensor:
    """Class-balanced CE on the FULL set: mean over present classes of per-class mean CE."""
    nc = _n_classes(y, n_classes)
    per = F.cross_entropy(logits, y, reduction="none")
    losses = []
    for c in range(nc):
        m = y == c
        if m.any():
            losses.append(per[m].mean())
    return torch.stack(losses).mean()


def source_risk(logits: torch.Tensor, y: torch.Tensor, metric: str, n_classes: int | None = None) -> torch.Tensor:
    """Differentiable primal source risk (``ce`` or ``balanced_ce``)."""
    assert_differentiable_primal(metric)
    return task_ce(logits, y) if metric == "ce" else balanced_ce(logits, y, n_classes)


def balanced_error(logits: torch.Tensor, y: torch.Tensor, n_classes: int | None = None) -> float:
    """Class-balanced 0/1 error — GUARD/REPORT ONLY (non-differentiable; never the primal)."""
    nc = _n_classes(y, n_classes)
    pred = logits.argmax(dim=1)
    errs = []
    for c in range(nc):
        m = y == c
        if m.any():
            errs.append((pred[m] != c).float().mean())
    return float(torch.stack(errs).mean().item())
