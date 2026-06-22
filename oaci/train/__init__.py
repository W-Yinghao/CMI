"""OACI risk-feasible trainer: min_θ L̃_train(θ) s.t. R̂_src(θ) ≤ τ = R̂_ERM + ε.

Model-agnostic engine (``engine.py``) over a fixed ``ModelOutput(logits, z)`` backbone + a single
``model_state`` checkpoint (``checkpoint.py``). The inner game uses a PyTorch
``ConditionalDomainAdversary`` (``C_D``), separate from the non-differentiable ``oaci/leakage/``
outer score (``L_Q^ov`` / ``bootstrap_ucl``), which is injected only for checkpoint scoring.
"""
from __future__ import annotations

from .adversary import ConditionalDomainAdversary, reference_entropy_bar
from .checkpoint import CheckpointRecord, ERMStage, TrainResult, model_state_hash, state_hash
from .engine import EngineConfig, InvocationRegistry, dual_update, effective_risk_weight, train_stage1, train_stage2
from .primal_dual import OaciObjective, TrainConfig, make_training_data, train_risk_feasible
from .risk import assert_differentiable_primal, balanced_ce, balanced_error, source_risk, task_ce
from .selector import Selection, select_checkpoint

__all__ = [
    "ConditionalDomainAdversary",
    "reference_entropy_bar",
    "TrainConfig",
    "TrainResult",
    "ERMStage",
    "CheckpointRecord",
    "EngineConfig",
    "InvocationRegistry",
    "OaciObjective",
    "make_training_data",
    "train_stage1",
    "train_stage2",
    "dual_update",
    "effective_risk_weight",
    "train_risk_feasible",
    "model_state_hash",
    "assert_differentiable_primal",
    "balanced_ce",
    "balanced_error",
    "source_risk",
    "task_ce",
    "Selection",
    "select_checkpoint",
    "state_hash",
]
