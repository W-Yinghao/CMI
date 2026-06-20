"""Source-only learned safety gate for selective test-time adaptation.

See :mod:`h2cmi.gate.safety_gate` (review section 7).
"""
from __future__ import annotations

from h2cmi.gate.safety_gate import (
    GATE_FEATURE_KEYS,
    SafetyGate,
    gate_features,
)

__all__ = ["SafetyGate", "gate_features", "GATE_FEATURE_KEYS"]
