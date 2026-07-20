"""Fixed backbone interface: every model returns ``ModelOutput(logits, z)`` where ``z`` is the
representation the conditional-domain critic / leakage probe operate on. The trainer is
model-agnostic against this contract."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    logits: torch.Tensor   # [B, C]
    z: torch.Tensor        # [B, P]


class RepresentationClassifier(nn.Module):
    """Base class: ``forward(x) -> ModelOutput``."""

    def forward(self, x: torch.Tensor) -> ModelOutput:  # pragma: no cover - interface
        raise NotImplementedError
