"""MethodObjective protocol + test objectives for the A1 engine.

A1 ships only the protocol and minimal objectives used to test the engine; the three real methods
(OACI / global-LPC / uniform) arrive in A2. ``full_surrogate`` is a method-generic name — the
engine never assumes the OACI ``H_ref − C_D`` form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ActiveStatus:
    active: bool
    reason: str | None = None


@dataclass(frozen=True)
class BatchView:
    """Per-microbatch labels/domain/weights for the rows the engine has already gathered into z."""
    y: torch.Tensor
    d: torch.Tensor | None
    w: torch.Tensor


@runtime_checkable
class MethodObjective(Protocol):
    name: str
    def active_status(self) -> ActiveStatus: ...
    def build_critic(self, feat_dim: int, device) -> nn.Module | None: ...
    def critic_loss(self, critic, z_detached, batch: BatchView) -> torch.Tensor: ...
    def encoder_penalty(self, critic, z, batch: BatchView) -> torch.Tensor: ...
    def full_surrogate(self, model, data, device, chunk_size) -> float: ...
    def diagnostics(self) -> dict: ...


class InactiveObjective:
    """Always inactive — the engine must return byte-exact ERM without building a critic/optimizer."""
    name = "inactive"

    def __init__(self, reason="inactive_for_test"):
        self._reason = reason

    def active_status(self) -> ActiveStatus:
        return ActiveStatus(False, self._reason)

    def build_critic(self, feat_dim, device):           # pragma: no cover - must never be called
        raise AssertionError("build_critic called on an inactive objective")

    def critic_loss(self, critic, z_detached, batch):   # pragma: no cover
        raise AssertionError("critic_loss called on an inactive objective")

    def encoder_penalty(self, critic, z, batch):        # pragma: no cover
        raise AssertionError("encoder_penalty called on an inactive objective")

    def full_surrogate(self, model, data, device, chunk_size):
        return 0.0

    def diagnostics(self):
        return {"active": False, "reason": self._reason}


class QuadraticPenaltyObjective:
    """No-critic differentiable penalty ``coeff·mean(z²)`` — drives the representation toward 0
    (a feasible, critic-free objective to exercise the encoder path)."""
    name = "quadratic"

    def __init__(self, coeff=1.0):
        self.coeff = float(coeff)

    def active_status(self):
        return ActiveStatus(True, None)

    def build_critic(self, feat_dim, device):
        return None

    def critic_loss(self, critic, z_detached, batch):   # pragma: no cover - no critic
        raise AssertionError("quadratic objective has no critic")

    def encoder_penalty(self, critic, z, batch):
        w = batch.w.to(z.dtype)
        return self.coeff * (w.unsqueeze(1) * z.pow(2)).sum() / w.sum().clamp_min(1e-12)

    def full_surrogate(self, model, data, device, chunk_size):
        with torch.inference_mode():
            from .bn import all_eval
            with all_eval(model):
                z = model(data.X.to(device)).z
            w = data.sample_mass.to(z.dtype)
            return float((w.unsqueeze(1) * z.pow(2)).sum() / w.sum().clamp_min(1e-12))

    def diagnostics(self):
        return {"active": True, "coeff": self.coeff}
