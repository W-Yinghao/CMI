"""Selection snapshot + post-lock audit result ABI (A2b-1b-ii-a). Phase stops at AUDIT."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSelectionSnapshot:
    method_name: str
    model_hash: str
    recomputed_state_hash: str
    selected_epoch: int
    R_src: float
    selection_score: float | None
    selected_erm: bool
    used_erm_fallback: bool
    selection_reason: str
    score_name: str
    n_feasible: int
    selection_status: str
    selection_leakage_hash: str | None
    feasible_unique_hashes: tuple


@dataclass(frozen=True)
class SelectionSnapshot:
    methods: tuple
    snapshot_hash: str


@dataclass(frozen=True)
class AuditMethodResult:
    method_name: str
    status: str
    model_hash: str
    leakage: object | None
    leakage_hash: str | None


@dataclass(frozen=True)
class AuditCacheStats:
    n_methods: int
    n_unique_selected_models: int
    feature_requests: int
    feature_computes: int
    feature_hits: int
    score_requests: int
    score_computes: int
    score_hits: int
    stats_hash: str


@dataclass(frozen=True)
class LevelAuditIntermediate:
    training_selection: object
    selection_snapshot_before: SelectionSnapshot
    selection_snapshot_after: SelectionSnapshot
    audit_method_items: tuple                  # ((name, AuditMethodResult), ...) canonical order
    audit_cache_stats: AuditCacheStats
    provenance: object
    phase: object                              # RunnerPhase, must be AUDIT
    invariants: dict

    @property
    def audit_methods(self) -> dict:
        return dict(self.audit_method_items)
