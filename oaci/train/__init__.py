"""OACI risk-feasible trainer: min_θ L̃_train(θ) s.t. R̂_src(θ) ≤ τ = R̂_ERM + ε.

The inner game uses a PyTorch ``ConditionalDomainAdversary`` (``C_D``), separate from the
non-differentiable ``oaci/leakage/`` outer score (``L_Q^ov`` / ``bootstrap_ucl``), which is
injected only for checkpoint scoring on frozen representations.
"""
from __future__ import annotations

from .adversary import ConditionalDomainAdversary, reference_entropy_bar
from .primal_dual import (
    CheckpointRecord,
    Encoder,
    TaskHead,
    TrainConfig,
    TrainResult,
    dual_update,
    train_risk_feasible,
)
from .risk import assert_differentiable_primal, balanced_ce, balanced_error, source_risk, task_ce
from .selector import Selection, select_checkpoint, state_hash

__all__ = [
    "ConditionalDomainAdversary",
    "reference_entropy_bar",
    "Encoder",
    "TaskHead",
    "TrainConfig",
    "TrainResult",
    "CheckpointRecord",
    "dual_update",
    "train_risk_feasible",
    "assert_differentiable_primal",
    "balanced_ce",
    "balanced_error",
    "source_risk",
    "task_ce",
    "Selection",
    "select_checkpoint",
    "state_hash",
]
