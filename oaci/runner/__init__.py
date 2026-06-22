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
from .config import ModelSpec, RunnerExecutionConfig
from .fold import DEFAULT_METHOD_ORDER, run_level_training_selection
from .objectives import make_objective
from .plans import LevelPlans, build_level_plans
from .results import (LevelTrainingSelectionResult, ObjectiveSpec, SelectedMethod, Stage1Run, TrainedMethod)
from .selection import (FeatureArtifactCache, FeatureArtifactKey, make_leakage_score_key,
                        select_methods, unique_feasible_records)
from .stage1 import run_stage1_once, stage1_invocation_id
from .stage2 import train_four_methods
from .provenance import IllegalPhaseTransition, ProvenanceEvent, RunnerPhase, RunProvenance
from .scope import (AuditScope, FoldScope, LevelPopulation, ScopePlanConfig, build_audit_scope,
                    build_fold_scope, build_level_population)
from .scoring import SelectionScoringSession, compute_leakage_score, overlap_probe_sample_ids
from .scientific_hash import scientific_value_hash
from .audit_results import (AuditCacheStats, AuditMethodResult, LevelAuditIntermediate,
                            MethodSelectionSnapshot, SelectionSnapshot)
from .audit import build_training_data_for_design, make_selection_snapshot, run_post_selection_audit
from .fold import run_level_through_audit
from .support import (DeletionCell, DeletionSchedule, LevelSupportState, build_level_support,
                      level0_reference_prior, make_deletion_schedule)

__all__ = [
    "FoldKey", "RunKey", "canonical_json_hash", "feed_string", "feed_int64", "feed_float64",
    "FrozenMaps", "build_frozen_maps",
    "RunnerPhase", "RunProvenance", "ProvenanceEvent", "IllegalPhaseTransition",
    "FeatureArtifact", "extract_frozen_features",
    "FoldData", "DeletionCell", "DeletionSchedule", "LevelSupportState", "make_deletion_schedule",
    "build_level_support", "level0_reference_prior",
    "LevelPopulation", "AuditScope", "FoldScope", "ScopePlanConfig",
    "build_level_population", "build_audit_scope", "build_fold_scope",
    "LevelPlans", "build_level_plans",
    "RunnerExecutionConfig", "ModelSpec", "run_stage1_once", "stage1_invocation_id", "make_objective",
    "train_four_methods", "unique_feasible_records", "make_leakage_score_key", "select_methods",
    "FeatureArtifactCache", "FeatureArtifactKey", "run_level_training_selection", "DEFAULT_METHOD_ORDER",
    "Stage1Run", "ObjectiveSpec", "TrainedMethod", "SelectedMethod", "LevelTrainingSelectionResult",
    "LeakageNonEstimableError", "NoComparableSupport", "FoldPlanNonEstimable", "BootstrapPlanNonEstimable",
    "SelectionScoringSession", "compute_leakage_score", "overlap_probe_sample_ids",
    "scientific_value_hash", "make_selection_snapshot", "build_training_data_for_design",
    "run_post_selection_audit", "run_level_through_audit", "MethodSelectionSnapshot", "SelectionSnapshot",
    "AuditMethodResult", "AuditCacheStats", "LevelAuditIntermediate",
]
