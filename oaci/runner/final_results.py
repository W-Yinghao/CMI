"""Final in-memory ABI for one completed level / fold (A2b-1b-ii-b). Immutable; tuple items with
read-only mapping views; every result carries a full SHA-256 identity. Phase reaches COMPLETE."""
from __future__ import annotations

from dataclasses import dataclass

from .metrics import EvaluationMetrics  # noqa: F401  (re-exported for callers)


@dataclass(frozen=True)
class PredictionCacheStats:
    source_guard_requests: int
    source_guard_computes: int
    source_guard_hits: int
    source_audit_requests: int
    source_audit_computes: int
    source_audit_hits: int
    target_requests: int
    target_computes: int
    target_hits: int
    stats_hash: str


@dataclass(frozen=True)
class MethodRunResult:
    method_name: str
    active: bool
    inactive_reason: str | None
    shared_erm_hash: str
    shared_tau: float
    shared_stage2_task_plan_hash: str
    initial_erm_hash: str
    train_result: object
    training_diagnostics_items: tuple
    selection: object
    selection_status: str
    selection_leakage: object | None
    audit_status: str
    audit_leakage: object | None
    source_guard_predictions: object
    source_audit_predictions: object
    target_predictions: object
    source_guard_metrics: EvaluationMetrics
    source_audit_metrics: EvaluationMetrics
    target_metrics: EvaluationMetrics
    method_result_hash: str

    @property
    def training_diagnostics(self) -> dict:
        return dict(self.training_diagnostics_items)


@dataclass(frozen=True)
class LevelRunResult:
    run_key: object
    support_state: object
    plans: object
    erm_stage: object
    method_items: tuple
    provenance: object               # ProvenanceSnapshot
    phase: object                    # RunnerPhase, must be COMPLETE
    selection_snapshot_hash: str
    selection_cache_stats: object
    audit_cache_stats: object
    prediction_cache_stats: PredictionCacheStats
    invariant_items: tuple
    level_result_hash: str

    @property
    def methods(self) -> dict:
        return dict(self.method_items)

    @property
    def invariants(self) -> dict:
        return dict(self.invariant_items)


@dataclass(frozen=True)
class FoldRunResult:
    fold_scope: object
    level_items: tuple
    fold_result_hash: str

    @property
    def levels(self) -> dict:
        return dict(self.level_items)
