"""Staged (GPU-record / CPU-replay) level executor (C4b-2).

The only device-dependent work in a level is the forward pass behind a deterministic key. The staged
executor splits the level so the V100 is not held during the CPU-bound leakage scoring:

* **Stage A (GPU record)** -- train the full-budget models, then GPU-extract EVERY risk-feasible unique
  candidate checkpoint's source-train features (selection), source-audit features (audit) and the three
  prediction-logit roles, recording them into a role-segregated :class:`ReplayStore`. No leakage scoring.
* **Stage B (CPU replay)** -- run selection -> lock -> audit -> finalize entirely from the store (replay
  mode, a missing key is a hard error, never a CPU forward). No GPU.

Pure execution acceleration: the trained models, the risk-feasible candidate set, the probe / bootstrap /
fold plans, the selector tie rule, the audit metric and the prediction aggregation are untouched, so the
staged LevelRunResult is BIT-IDENTICAL to the monolithic ``run_level_complete`` (same checkpoint / feature
/ selection / audit / prediction / metrics / level_result hashes). Target logits are a sealed forward-only
cache: selection only ever requests ``feat:source_train`` and so structurally cannot open the audit or
target stores; ``target_fit_ids`` stays empty.
"""
from __future__ import annotations

import torch

from ..train.engine import InvocationRegistry
from ..train.rng import derive_seed
from .audit import build_training_data_for_design, run_post_selection_audit
from .features import extract_frozen_features
from .finalize import finalize_level_run
from .fold import DEFAULT_METHOD_ORDER, _level_invariants
from .predict import PredictionCacheKey, predict_checkpoint
from .provenance import RunnerPhase, RunProvenance
from .replay_store import ReplayStore, set_replay_store
from .replay_store import resolve_artifact as _resolve
from .results import LevelTrainingSelectionResult
from .selection import FeatureArtifactKey, select_methods, unique_feasible_records
from .stage1 import run_stage1_once
from .stage2 import train_four_methods

_PRED_ROLES = ("source_guard", "source_audit", "target_audit")
_STAGE2 = ("OACI", "global_lpc", "uniform")


def feasible_candidate_states(stage1, trained, tol) -> dict:
    """ERM plus every risk-feasible unique Stage-2 checkpoint, deduplicated by model hash."""
    out = {stage1.erm_stage.checkpoint.model_hash: stage1.erm_stage.checkpoint.model_state}
    for name in _STAGE2:
        for rec in unique_feasible_records(trained[name].train_result, numerical_tol=tol):
            out.setdefault(rec.model_hash, rec.model_state)
    return out


def train_level(run_key, level_population, level_plans, support_state, fold_scope, execution_cfg,
                model_spec, model_factory, device, method_order=DEFAULT_METHOD_ORDER):
    """Stage-A training: Stage-1 ERM + the four-method Stage-2 trajectories (identical to the monolithic
    path)."""
    engine_cfg = execution_cfg.engine_config_for(run_key)
    registry = InvocationRegistry()
    stage1 = run_stage1_once(run_key, level_population, level_plans, model_factory, model_spec, engine_cfg,
                             execution_cfg.execution_config_hash, registry, device)
    trained = train_four_methods(method_order, stage1, level_population, level_plans, support_state,
                                 fold_scope, execution_cfg, model_factory, engine_cfg, device)
    return stage1, trained


def prefetch_level_gpu_artifacts(run_key, stage1, trained, fold_data, support_state, level_population,
                                 fold_scope, level_plans, execution_cfg, model_spec, model_factory,
                                 device, store) -> dict:
    """GPU-extract every feasible candidate's source-train + source-audit features and the three
    prediction-logit roles into ``store`` (record mode). Returns the candidate {model_hash: state}.
    The keys MATCH the runtime select/audit/finalize keys exactly, so Stage B replays them."""
    tol = execution_cfg.engine_template.numerical_tol
    candidates = feasible_candidate_states(stage1, trained, tol)
    set_replay_store(store, "record")
    try:
        if level_plans.selection_status == "estimable":                  # selection features (source_train)
            sdesign, sdata = level_plans.selection_design, level_population.training_data
            sfseed = derive_seed(run_key.model_seed, "selection_feature", run_key.run_key_hash)
            for mh, ms in candidates.items():
                key = FeatureArtifactKey(mh, level_population.tensor_hash, sdesign.population_hash,
                                         model_spec.model_spec_hash, execution_cfg.feature_chunk_size)
                _resolve("feat:source_train", key, lambda s=ms, h=mh: extract_frozen_features(
                    s, h, model_factory, sdata, sdesign, factory_seed=sfseed,
                    chunk_size=execution_cfg.feature_chunk_size, device=device))

        audit = fold_scope.source_audit                                  # audit features (source_audit)
        if audit.status == "estimable" and audit.fold_plan is not None and audit.bootstrap_plan is not None:
            adesign = audit.design
            atd = build_training_data_for_design(fold_data, adesign)
            for mh, ms in candidates.items():
                afseed = derive_seed(run_key.model_seed, "audit_feature_factory", run_key.run_key_hash, mh)
                key = FeatureArtifactKey(mh, fold_data.source_audit_tensor_hash, adesign.population_hash,
                                         model_spec.model_spec_hash, execution_cfg.feature_chunk_size)
                _resolve("feat:source_audit", key, lambda s=ms, h=mh, fs=afseed: extract_frozen_features(
                    s, h, model_factory, atd, adesign, factory_seed=fs,
                    chunk_size=execution_cfg.feature_chunk_size, device=device))

        views = {"source_guard": fold_data.make_role_view("source_guard", support_state.source_train_idx),
                 "source_audit": fold_data.make_role_view("source_audit"),
                 "target_audit": fold_data.make_role_view("target_audit")}
        for role in _PRED_ROLES:                                         # prediction logits (3 roles)
            rv = views[role]
            for mh, ms in candidates.items():
                key = PredictionCacheKey(mh, role, rv.population_hash, rv.tensor_hash,
                                         model_spec.model_spec_hash, execution_cfg.prediction_chunk_size)
                pfseed = derive_seed(run_key.model_seed, "prediction_factory", run_key.run_key_hash, role, mh)
                _resolve(f"logits:{role}", key, lambda s=ms, h=mh, v=rv, fs=pfseed: predict_checkpoint(
                    s, h, model_factory, v, factory_seed=fs,
                    chunk_size=execution_cfg.prediction_chunk_size, device=device))
    finally:
        set_replay_store(None, "off")
    return candidates


def resume_level_from_store(run_key, fold_data, support_state, level_population, fold_scope, level_plans,
                            execution_cfg, model_spec, model_factory, stage1, trained, store,
                            device="cpu", method_order=DEFAULT_METHOD_ORDER, decision_ctx=None):
    """Stage-B CPU replay: select -> lock -> audit -> finalize entirely from ``store`` (no forward). This
    reproduces ``run_level_training_selection``'s assembly minus the (already-done) training."""
    set_replay_store(store, "replay")
    try:
        prov = RunProvenance()
        prov.record_fit("preprocess", fold_data.preprocess_fit_ids)
        prov.transition(RunnerPhase.TRAINING)
        prov.record_fit("optimization", support_state.source_train_sample_ids)
        prov.transition(RunnerPhase.SELECTION)
        selected, leak_stats, feat_stats = select_methods(
            run_key, trained, stage1, level_population, level_plans, support_state, fold_scope,
            execution_cfg, model_spec, model_factory, device, prov)
        inv = _level_invariants(stage1, trained, selected, level_plans, leak_stats)
        ts = LevelTrainingSelectionResult(run_key=run_key, stage1=stage1, trained_methods=trained,
                                          selected_methods=selected, leakage_cache_stats=leak_stats,
                                          feature_cache_stats=feat_stats, provenance=prov,
                                          phase=RunnerPhase.SELECTION, invariants=inv)
        ai = run_post_selection_audit(ts, fold_data, fold_scope, execution_cfg, model_spec, model_factory, device)
        return finalize_level_run(ai, fold_data, fold_scope, support_state, level_population, level_plans,
                                  execution_cfg, model_spec, model_factory, device, decision_ctx=decision_ctx)
    finally:
        set_replay_store(None, "off")


def run_level_staged(run_key, fold_data, support_state, level_population, fold_scope, level_plans,
                     execution_cfg, model_spec, model_factory, gpu_device, *, cpu_device="cpu",
                     method_order=DEFAULT_METHOD_ORDER, store=None, decision_ctx=None):
    """One-process staged level (Stage A on ``gpu_device``, Stage B on ``cpu_device``). For the true
    two-job split, call train_level + prefetch_level_gpu_artifacts (GPU job) and resume_level_from_store
    (CPU job) with a persisted store. Returns (LevelRunResult, store)."""
    stage1, trained = train_level(run_key, level_population, level_plans, support_state, fold_scope,
                                  execution_cfg, model_spec, model_factory, gpu_device, method_order)
    store = store if store is not None else ReplayStore()
    prefetch_level_gpu_artifacts(run_key, stage1, trained, fold_data, support_state, level_population,
                                 fold_scope, level_plans, execution_cfg, model_spec, model_factory,
                                 gpu_device, store)
    lr = resume_level_from_store(run_key, fold_data, support_state, level_population, fold_scope,
                                 level_plans, execution_cfg, model_spec, model_factory, stage1, trained,
                                 store, device=cpu_device, method_order=method_order, decision_ctx=decision_ctx)
    return lr, store
