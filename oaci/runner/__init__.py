"""OACI in-memory four-method runner (A2b). This commit ships the data contracts (keys / maps /
provenance / canonical features) + the selection scoring session; the support/scope/plans contracts
and the level/fold orchestration follow. No disk artifacts here."""
from __future__ import annotations

from ..leakage.errors import (BootstrapPlanNonEstimable, FoldPlanNonEstimable, LeakageNonEstimableError,
                              NoComparableSupport)
from .data import FoldData
from .features import FeatureArtifact, extract_frozen_features
from .keys import FoldKey, RunKey, canonical_json_hash, feed_float64, feed_int64, feed_string
from .maps import FrozenMaps, build_frozen_maps
from .provenance import IllegalPhaseTransition, ProvenanceEvent, RunnerPhase, RunProvenance
from .scoring import SelectionScoringSession, compute_leakage_score
from .support import (DeletionCell, DeletionSchedule, LevelSupportState, build_level_support,
                      level0_reference_prior, make_deletion_schedule)

__all__ = [
    "FoldKey", "RunKey", "canonical_json_hash", "feed_string", "feed_int64", "feed_float64",
    "FrozenMaps", "build_frozen_maps",
    "RunnerPhase", "RunProvenance", "ProvenanceEvent", "IllegalPhaseTransition",
    "FeatureArtifact", "extract_frozen_features",
    "FoldData", "DeletionCell", "DeletionSchedule", "LevelSupportState", "make_deletion_schedule",
    "build_level_support", "level0_reference_prior",
    "LeakageNonEstimableError", "NoComparableSupport", "FoldPlanNonEstimable", "BootstrapPlanNonEstimable",
    "SelectionScoringSession", "compute_leakage_score",
]
