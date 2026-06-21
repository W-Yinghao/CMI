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
    used_erm_fallback: bool
    selected_epoch: int                # -1 == ERM checkpoint
    enc_state: dict
    head_state: dict
    R_src: float
    selection_score: float | None      # outer score of the chosen checkpoint (adaptive)
    n_feasible: int
    score_name: str


def select_checkpoint(result, numerical_tol=None, score_fn=None, score_name="leakage_surrogate") -> Selection:
    """Select per the risk-feasibility rule. ``result`` is a ``TrainResult``."""
    tol = result.cfg.numerical_tol if numerical_tol is None else numerical_tol
    feasible = [c for c in result.trajectory if c.R_src <= result.tau + tol]

    if not feasible:
        # byte-exact ERM restore
        return Selection(
            used_erm_fallback=True, selected_epoch=-1,
            enc_state=result.erm_ckpt["enc"], head_state=result.erm_ckpt["head"],
            R_src=result.R_ERM_hat, selection_score=None, n_feasible=0, score_name=score_name,
        )

    key = score_fn if score_fn is not None else (lambda c: c.leakage_surrogate)
    best = min(feasible, key=key)
    return Selection(
        used_erm_fallback=False, selected_epoch=best.epoch,
        enc_state=best.enc_state, head_state=best.head_state,
        R_src=best.R_src, selection_score=float(key(best)), n_feasible=len(feasible),
        score_name=score_name,
    )
