"""OACI formal methods: the four risk-feasible Stage-2 methods sharing one ERM checkpoint, τ and
task plan. ERM (frozen) / OACI (support-aware adversary, eligible rows only) / global_lpc & uniform
(full-domain posterior-KL to π_y^α / uniform, sharing the full-domain alignment plan)."""
from __future__ import annotations

from .activity import MethodStatus, all_method_status, method_status
from .base import METHODS, method_activity
from .erm import erm_result
from .global_lpc import GlobalLPCObjective
from .oaci import OACIObjective, support_hash
from .posterior import PosteriorDomainCritic, PosteriorObjective
from .uniform import UniformObjective

__all__ = [
    "METHODS", "MethodStatus", "method_status", "all_method_status", "method_activity",
    "erm_result", "OACIObjective", "support_hash", "PosteriorObjective", "PosteriorDomainCritic",
    "GlobalLPCObjective", "UniformObjective",
]
