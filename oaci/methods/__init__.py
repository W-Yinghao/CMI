"""OACI methods: per-method activity + (next commit) the four risk-feasible Stage-2 methods
(ERM frozen / OACI support-aware adversary / global_lpc full-domain posterior-KL / uniform target),
all sharing one ERM checkpoint, τ, task-batch plan and overlap-aware outer scorer."""
from __future__ import annotations

from .base import METHODS, method_activity

__all__ = ["METHODS", "method_activity"]
