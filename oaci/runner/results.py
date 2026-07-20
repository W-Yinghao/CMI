"""In-memory result ABI for A2b-1b-i (training + source-train selection). Consumed by A2b-1b-ii to
build the final MethodRunResult / LevelRunResult."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage1Run:
    erm_stage: object
    invocation_id: str
    invocation_count: int
    registry_total_claims: int
    model_spec_hash: str
    engine_base_seed: int


@dataclass(frozen=True)
class ObjectiveSpec:
    method_name: str
    active: bool
    inactive_reason: str | None
    support_hash: str
    prior_matrix: object | None      # np.ndarray [n_classes, |D0|] or None
    objective_spec_hash: str


@dataclass(frozen=True)
class TrainedMethod:
    method_name: str
    active: bool
    inactive_reason: str | None
    shared_erm_hash: str
    shared_tau: float
    shared_stage2_task_plan_hash: str
    objective_spec: ObjectiveSpec
    train_result: object             # TrainResult
    training_diagnostics: dict


@dataclass(frozen=True)
class SelectedMethod:
    trained: TrainedMethod
    selection: object                # Selection
    selection_status: str
    selection_leakage: object | None
    feasible_unique_hashes: tuple


@dataclass(frozen=True)
class LevelTrainingSelectionResult:
    run_key: object
    stage1: Stage1Run
    trained_methods: dict
    selected_methods: dict
    leakage_cache_stats: dict
    feature_cache_stats: dict
    provenance: object
    phase: object                    # RunnerPhase, must be SELECTION
    invariants: dict
