"""Backbone factory + a synthetic MLP adapter (for the CPU tests / synthetic demos). Both return
``ModelOutput(logits, z)`` so the (model-agnostic) trainer treats them identically."""
from __future__ import annotations

import torch
import torch.nn as nn

from .output import ModelOutput, RepresentationClassifier
from .shallow import ShallowConvNet


class MLPClassifier(RepresentationClassifier):
    """2-D synthetic adapter: ``x[B,F] -> z[B,z_dim] -> logits``."""

    def __init__(self, in_dim: int, z_dim: int, n_classes: int, hidden: int = 0):
        super().__init__()
        self.enc = (nn.Linear(in_dim, z_dim) if hidden <= 0
                    else nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, z_dim)))
        self.head = nn.Linear(z_dim, n_classes)
        self.feat_dim = z_dim

    def forward(self, x: torch.Tensor) -> ModelOutput:
        z = self.enc(x)
        return ModelOutput(logits=self.head(z), z=z)


def build_model(name: str, *, in_chans=None, in_times=None, in_dim=None, n_classes=2, **arch):
    """Build a backbone by registry name. ``shallow_convnet`` needs ``in_chans``/``in_times``;
    ``mlp`` needs ``in_dim``."""
    if name == "shallow_convnet":
        if in_chans is None or in_times is None:
            raise ValueError("shallow_convnet requires in_chans and in_times")
        return ShallowConvNet(in_chans=in_chans, in_times=in_times, n_classes=n_classes, **arch)
    if name == "mlp":
        if in_dim is None:
            raise ValueError("mlp requires in_dim")
        return MLPClassifier(in_dim=in_dim, z_dim=arch.get("z_dim", 8), n_classes=n_classes,
                             hidden=arch.get("hidden", 0))
    raise ValueError(f"unknown backbone {name!r} (known: shallow_convnet, mlp)")
