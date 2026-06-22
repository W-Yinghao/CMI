"""Checkpoint selection over the single-``model_state`` ABI: feasibility first, then an outer
score; ERM kept on exact-or-within-tolerance ties; byte-exact ERM fallback.

* A Stage-2 checkpoint is **feasible** iff ``R_src ≤ τ + numerical_tol``.
* ERM is ALWAYS a candidate (feasible: ``R̂_ERM ≤ τ``), scored by the same ``score_fn``. ERM is
  selected UNLESS a feasible Stage-2 checkpoint BEATS it by MORE than ``selection_score_tolerance``.
* The default score is the training ``train_surrogate`` (method-generic). An injected ``score_fn``
  (e.g. the leakage module's UCL on a frozen representation) is reported as ``selection_*`` and is
  adaptive; never backprop through it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .checkpoint import state_hash  # re-export so existing imports keep working

__all__ = ["Selection", "select_checkpoint", "state_hash"]


@dataclass
class Selection:
    used_erm_fallback: bool
    selected_erm: bool
    selection_reason: str              # 'erm_best' | 'stage2_best' | 'no_stage2_feasible'
    selected_epoch: int                # -1 == ERM checkpoint
    model_state: dict
    model_hash: str
    R_src: float
    selection_score: float | None
    n_feasible: int
    score_name: str


def select_checkpoint(result, numerical_tol=None, score_fn=None, score_name="train_surrogate",
                      selection_score_tolerance=0.0) -> Selection:
    tol = numerical_tol if numerical_tol is not None else 1e-4
    erm = result.erm_record
    tau = result.erm_stage.tau
    feasible = [c for c in result.trajectory if c.R_src <= tau + tol]
    key = score_fn if score_fn is not None else (lambda c: c.train_surrogate)
    erm_score = key(erm)

    if feasible:
        best_s2 = min(feasible, key=key)
        if key(best_s2) < erm_score - selection_score_tolerance:
            best, selected_erm, reason = best_s2, False, "stage2_best"
        else:
            best, selected_erm, reason = erm, True, "erm_best"
    else:
        best, selected_erm, reason = erm, True, "no_stage2_feasible"
    return Selection(
        used_erm_fallback=(not feasible), selected_erm=selected_erm, selection_reason=reason,
        selected_epoch=best.epoch, model_state=best.model_state, model_hash=best.model_hash,
        R_src=best.R_src, selection_score=float(key(best)), n_feasible=len(feasible),
        score_name=score_name,
    )
