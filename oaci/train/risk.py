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


def per_class_ce_sums(logits: torch.Tensor, y: torch.Tensor, n_classes: int, weight=None):
    """Per-class (Σ wᵢℓᵢ, Σ wᵢ) — additive sufficient statistics for (mass-)weighted ``balanced_ce``
    (default weight = 1). Accumulating across an arbitrary partition then ``mean_c(sum/mass)``
    reproduces the full-set value exactly (partition- and window-duplication-invariant)."""
    per = F.cross_entropy(logits, y, reduction="none")
    w = torch.ones_like(per) if weight is None else torch.as_tensor(weight, dtype=per.dtype, device=per.device)
    sums = torch.zeros(n_classes, dtype=per.dtype)
    mass = torch.zeros(n_classes, dtype=per.dtype)
    for c in range(n_classes):
        m = y == c
        if m.any():
            sums[c] = (w[m] * per[m]).sum()
            mass[c] = w[m].sum()
    return sums, mass


def balanced_ce(logits: torch.Tensor, y: torch.Tensor, n_classes: int | None = None, weight=None) -> torch.Tensor:
    """Class-balanced CE: mean over present classes of per-class (optionally weighted) mean CE.
    A per-example ``weight`` reweights WITHIN each class only (the class equal-weighting is
    inherent to balanced_ce), so a class-stratified task sampler's weights leave it invariant."""
    nc = _n_classes(y, n_classes)
    per = F.cross_entropy(logits, y, reduction="none")
    w = None if weight is None else torch.as_tensor(weight, dtype=per.dtype, device=per.device)
    losses = []
    for c in range(nc):
        m = y == c
        if m.any():
            losses.append(per[m].mean() if w is None else (w[m] * per[m]).sum() / w[m].sum())
    return torch.stack(losses).mean()


def source_risk(logits: torch.Tensor, y: torch.Tensor, metric: str, n_classes: int | None = None, weight=None) -> torch.Tensor:
    """Differentiable primal source risk (``ce`` or ``balanced_ce``).

    ``weight`` is the per-example TASK weight (distinct from the adversary's importance weight;
    they do not share a target distribution). ``ce`` uses a global weighted mean — the weight
    ``w^task_i = n_y/m_y^(b)`` restores the fixed class prior ``p(y)``; ``balanced_ce`` uses a
    per-class weighted mean (the constant within-class weight cancels)."""
    assert_differentiable_primal(metric)
    if metric == "ce":
        if weight is None:
            return task_ce(logits, y)
        w = torch.as_tensor(weight, dtype=logits.dtype, device=logits.device)
        ce = F.cross_entropy(logits, y, reduction="none")
        return (w * ce).sum() / w.sum()
    return balanced_ce(logits, y, n_classes, weight=weight)


def balanced_error(logits: torch.Tensor, y: torch.Tensor, n_classes: int | None = None, weight=None) -> float:
    """Class-balanced 0/1 error — GUARD/REPORT ONLY (non-differentiable; never the primal). The
    per-class error is MASS-weighted (default weight = 1)."""
    nc = _n_classes(y, n_classes)
    pred = logits.argmax(dim=1)
    wrong = (pred != y).float()
    w = torch.ones_like(wrong) if weight is None else torch.as_tensor(weight, dtype=wrong.dtype, device=wrong.device)
    errs = []
    for c in range(nc):
        m = y == c
        if m.any():
            errs.append((w[m] * wrong[m]).sum() / w[m].sum())
    return float(torch.stack(errs).mean().item())
