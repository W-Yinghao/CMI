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


def select_checkpoint(result, numerical_tol=None, score_fn=None, score_name="leakage_surrogate") -> Selection:
    """Risk-feasible lexicographic selection. ERM is ALWAYS in the candidate set (it is feasible:
    ``R̂_ERM ≤ τ``) and is scored by the SAME ``score_fn`` as the feasible Stage-2 checkpoints; the
    minimum-score candidate wins. So a Stage-2 checkpoint that is risk-feasible but has WORSE
    leakage than ERM never displaces ERM. If no Stage-2 checkpoint is feasible, ERM is restored
    byte-exactly (the ``erm_record`` carries the ERM checkpoint's states)."""
    tol = result.cfg.numerical_tol if numerical_tol is None else numerical_tol
    erm = result.erm_record
    feasible = [c for c in result.trajectory if c.R_src <= result.tau + tol]
    key = score_fn if score_fn is not None else (lambda c: c.leakage_surrogate)

    best = min(feasible + [erm], key=key)            # ERM always competes
    selected_erm = best is erm
    reason = "no_stage2_feasible" if not feasible else ("erm_best" if selected_erm else "stage2_best")
    return Selection(
        used_erm_fallback=(not feasible), selected_erm=selected_erm, selection_reason=reason,
        selected_epoch=best.epoch, enc_state=best.enc_state, head_state=best.head_state,
        R_src=best.R_src, selection_score=float(key(best)), n_feasible=len(feasible),
        score_name=score_name,
    )
