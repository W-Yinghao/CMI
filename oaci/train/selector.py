"""Checkpoint selection: feasibility first, then an outer leakage score; ERM fallback.

* A Stage-2 checkpoint is **feasible** iff ``R_src ≤ τ + numerical_tol``.
* Among feasible checkpoints, pick the one minimising the outer ``score_fn`` (lower = less
  leakage). Default score = the training ``leakage_surrogate``. An injected ``score_fn`` (e.g.
  the leakage module's ``bootstrap_ucl`` on a frozen representation) may be used — but because
  the choice is then made adaptively over many checkpoints, the resulting value is reported as
  ``selection_bootstrap_ucl`` (optimistic; the paper CI must be recomputed on an independent
  audit split or simultaneously corrected). Never backprop through ``score_fn``.
* If **no** feasible Stage-2 checkpoint exists, restore the ERM checkpoint **byte-exactly**
  (``used_erm_fallback = True``).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


def state_hash(state: dict) -> str:
    """Stable byte hash of a state dict (for immutability / byte-exact restore checks)."""
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k]
        h.update(k.encode())
        t = v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        h.update(t.numpy().tobytes())
    return h.hexdigest()


@dataclass
class Selection:
    used_erm_fallback: bool            # True iff NO feasible Stage-2 checkpoint existed
    selected_erm: bool                 # True iff the ERM checkpoint won the lexicographic score
    selection_reason: str              # 'erm_best' | 'stage2_best' | 'no_stage2_feasible'
    selected_epoch: int                # -1 == ERM checkpoint
    enc_state: dict
    head_state: dict
    R_src: float
    selection_score: float | None      # outer score of the chosen checkpoint (adaptive)
    n_feasible: int
    score_name: str


def select_checkpoint(result, numerical_tol=None, score_fn=None, score_name="leakage_surrogate",
                      selection_score_tolerance=0.0) -> Selection:
    """Risk-feasible lexicographic selection. ERM is ALWAYS a candidate (feasible: ``R̂_ERM ≤ τ``),
    scored by the SAME ``score_fn`` as the feasible Stage-2 checkpoints. ERM is selected UNLESS a
    feasible Stage-2 checkpoint is BETTER (lower score) by MORE than ``selection_score_tolerance``
    — so an exact OR within-tolerance tie conservatively keeps the (byte-exact) ERM checkpoint."""
    tol = result.cfg.numerical_tol if numerical_tol is None else numerical_tol
    erm = result.erm_record
    feasible = [c for c in result.trajectory if c.R_src <= result.tau + tol]
    key = score_fn if score_fn is not None else (lambda c: c.leakage_surrogate)
    erm_score = key(erm)

    if feasible:
        best_s2 = min(feasible, key=key)
        if key(best_s2) < erm_score - selection_score_tolerance:   # Stage-2 must BEAT ERM by > tol
            best, selected_erm, reason = best_s2, False, "stage2_best"
        else:
            best, selected_erm, reason = erm, True, "erm_best"
    else:
        best, selected_erm, reason = erm, True, "no_stage2_feasible"
    return Selection(
        used_erm_fallback=(not feasible), selected_erm=selected_erm, selection_reason=reason,
        selected_epoch=best.epoch, enc_state=best.enc_state, head_state=best.head_state,
        R_src=best.R_src, selection_score=float(key(best)), n_feasible=len(feasible),
        score_name=score_name,
    )
