"""Per-level training + source-train selection orchestrator (A2b-1b-i).

Advances PREPARED -> TRAINING -> SELECTION only; the selection lock, audit and predictions are
A2b-1b-ii. The InvocationRegistry is owned here (one per level) so Stage-1 is provably trained once.
This API deliberately never receives the source-audit or target tensors.
"""
from __future__ import annotations

from ..train.engine import InvocationRegistry
from .provenance import RunnerPhase, RunProvenance
from .results import LevelTrainingSelectionResult
from .selection import select_methods
from .stage1 import run_stage1_once
from .stage2 import train_four_methods

DEFAULT_METHOD_ORDER = ("ERM", "OACI", "global_lpc", "uniform")


def _level_invariants(stage1, trained, selected, level_plans, leakage_stats) -> dict:
    erm = {tm.shared_erm_hash for tm in trained.values()}
    tau = {round(tm.shared_tau, 12) for tm in trained.values()}
    s2 = {trained[m].shared_stage2_task_plan_hash for m in ("OACI", "global_lpc", "uniform")}
    inv = {
        "stage1_invocation_count": stage1.invocation_count,
        "stage1_total_claims": stage1.registry_total_claims,
        "shared_erm_unique": len(erm) == 1,
        "shared_tau_unique": len(tau) == 1,
        "stage2_task_plan_unique": len(s2) == 1,
        "lpc_uniform_alignment_same_object":
            trained["global_lpc"].train_result.alignment_plan_hash
            == trained["uniform"].train_result.alignment_plan_hash,
        "selection_status": level_plans.selection_status,
    }
    if trained["OACI"].active:
        inv["oaci_rejected_ineligible_rows"] = 0   # train_four_methods already asserted it is 0
    if level_plans.selection_status == "estimable":
        inv["erm_cache_request_compute_hit"] = (leakage_stats.get("erm_request"),
                                                leakage_stats.get("erm_compute"), leakage_stats.get("erm_hit"))
    return inv


def run_level_training_selection(run_key, fold_data, support_state, level_population, fold_scope,
                                 level_plans, execution_cfg, model_spec, model_factory, device,
                                 method_order=DEFAULT_METHOD_ORDER) -> LevelTrainingSelectionResult:
    prov = RunProvenance()
    prov.record_fit("preprocess", fold_data.preprocess_fit_ids)
    prov.transition(RunnerPhase.TRAINING)
    prov.record_fit("optimization", support_state.source_train_sample_ids)

    engine_cfg = execution_cfg.engine_config_for(run_key)
    registry = InvocationRegistry()
    stage1 = run_stage1_once(run_key, level_population, level_plans, model_factory, model_spec, engine_cfg,
                             execution_cfg.execution_config_hash, registry, device)
    trained = train_four_methods(method_order, stage1, level_population, level_plans, support_state,
                                 fold_scope, execution_cfg, model_factory, engine_cfg, device)

    prov.transition(RunnerPhase.SELECTION)
    selected, leak_stats, feat_stats = select_methods(run_key, trained, stage1, level_population, level_plans,
                                                      support_state, fold_scope, execution_cfg, model_spec,
                                                      model_factory, device, prov)
    inv = _level_invariants(stage1, trained, selected, level_plans, leak_stats)
    return LevelTrainingSelectionResult(run_key=run_key, stage1=stage1, trained_methods=trained,
                                        selected_methods=selected, leakage_cache_stats=leak_stats,
                                        feature_cache_stats=feat_stats, provenance=prov,
                                        phase=RunnerPhase.SELECTION, invariants=inv)
