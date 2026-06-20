"""Training: the two-step alternating H2-CMI trainer wiring every component together."""
from __future__ import annotations

from h2cmi.train.trainer import H2Model, train_h2, set_seed

__all__ = ["H2Model", "train_h2", "set_seed"]
