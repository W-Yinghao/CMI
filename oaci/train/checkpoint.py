"""Single model-state checkpoint ABI for the model-agnostic engine.

One ``model_state`` (parameters AND buffers, incl. BatchNorm ``running_mean``/``running_var``/
``num_batches_tracked``) replaces the old split ``enc_state``/``head_state``. ``state_hash`` binds
key length+key, dtype, ndim+shape and raw contiguous bytes — device-independent, but it
distinguishes identical bytes that differ in dtype or shape.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn


def state_hash(state: dict) -> str:
    """Stable, device-independent byte hash of a state dict. Binds key length+key | dtype | ndim+
    shape | raw contiguous bytes, so it separates equal bytes with different dtype/shape."""
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k]
        t = v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        kb = str(k).encode()
        h.update(len(kb).to_bytes(8, "little")); h.update(kb)
        h.update(str(t.dtype).encode())
        h.update(t.dim().to_bytes(4, "little"))
        for s in t.shape:
            h.update(int(s).to_bytes(8, "little"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def clone_state_cpu(model: nn.Module) -> dict:
    """All parameters + buffers, detached and CPU-cloned so NOTHING shares storage with the live
    model (a later optimizer step cannot mutate a saved checkpoint)."""
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


def model_state_hash(model: nn.Module) -> str:
    return state_hash(clone_state_cpu(model))


@dataclass(frozen=True)
class CheckpointRecord:
    epoch: int
    optimizer_step: int
    model_state: dict          # parameters + ALL buffers
    model_hash: str
    R_src: float               # realised guard-set source risk (primal metric)
    balanced_err: float        # guard/report metric
    train_surrogate: float     # method-generic training surrogate (e.g. H_ref - C_D for OACI)
    lam: float


@dataclass(frozen=True)
class ERMStage:
    checkpoint: CheckpointRecord
    R_ERM_hat: float
    tau: float
    task_plan_hash: str
    stage1_invocation_id: str


@dataclass
class TrainResult:
    method_name: str
    active: bool
    inactive_reason: str | None
    erm_stage: ERMStage
    erm_record: CheckpointRecord
    trajectory: list = field(default_factory=list)
    initial_model_hash: str = ""
    task_plan_hash: str = ""
    alignment_plan_hash: str | None = None

    # ---- convenience for selector/compat (NO second checkpoint semantics) ----
    @property
    def tau(self) -> float:
        return self.erm_stage.tau

    @property
    def R_ERM_hat(self) -> float:
        return self.erm_stage.R_ERM_hat
