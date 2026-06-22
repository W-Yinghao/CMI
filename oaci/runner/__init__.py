"""OACI in-memory four-method runner (A2b). This commit ships the selection scoring session and the
interface fixes; the level/fold orchestration follows. No disk artifacts here."""
from __future__ import annotations

from .scoring import LeakageNonEstimableError, SelectionScoringSession, compute_leakage_score

__all__ = ["LeakageNonEstimableError", "SelectionScoringSession", "compute_leakage_score"]
