"""Single source of truth for the method / level / fold logical payloads.

``finalize`` builds these from the live training objects; the artifact verifier rebuilds the SAME
payload from the persisted result objects and recomputes the hash. Both call the one ``*_payload``
core, so the hash formula is never duplicated.
"""
from __future__ import annotations

from ..runner.scientific_hash import leakage_result_hash, scientific_value_hash

_ROLES = ("source_guard", "source_audit", "target_audit")


def method_payload(*, name, active, inactive_reason, shared_erm, shared_tau, shared_stage2_task_plan_hash,
                   initial_erm, trajectory, selection, audit_status, audit_leakage_hash, pred_hashes,
                   metrics_hashes, training_diagnostics_hash) -> dict:
    return {"method": name, "active": active, "inactive_reason": inactive_reason, "shared_erm": shared_erm,
            "shared_tau": float(shared_tau), "shared_stage2_task_plan_hash": shared_stage2_task_plan_hash,
            "initial_erm": initial_erm, "trajectory": trajectory, "selection": selection,
            "audit_status": audit_status, "audit_leakage_hash": audit_leakage_hash, "preds": list(pred_hashes),
            "metrics": list(metrics_hashes), "training_diagnostics_hash": training_diagnostics_hash}


def _trajectory(records):
    return [{"epoch": int(c.epoch), "optimizer_step": int(c.optimizer_step), "model_hash": c.model_hash,
             "R_src": float(c.R_src), "balanced_err": float(c.balanced_err),
             "train_surrogate": float(c.train_surrogate), "lambda": float(c.lam)} for c in records]


def _selection_block(sel, selection_status, selection_leakage):
    return {"model_hash": sel.model_hash, "selected_epoch": int(sel.selected_epoch), "R_src": float(sel.R_src),
            "selection_score": None if sel.selection_score is None else float(sel.selection_score),
            "selected_erm": bool(sel.selected_erm), "used_erm_fallback": bool(sel.used_erm_fallback),
            "selection_reason": sel.selection_reason, "score_name": sel.score_name,
            "n_feasible": int(sel.n_feasible), "selection_status": selection_status,
            "selection_leakage_hash": None if selection_leakage is None else leakage_result_hash(selection_leakage)}


def method_payload_from_trained(name, tm, sm, audit_result, bundle_by_role, metrics_by_role) -> dict:
    return method_payload(
        name=name, active=tm.active, inactive_reason=tm.inactive_reason, shared_erm=tm.shared_erm_hash,
        shared_tau=tm.shared_tau, shared_stage2_task_plan_hash=tm.shared_stage2_task_plan_hash,
        initial_erm=tm.shared_erm_hash, trajectory=_trajectory(tm.train_result.trajectory),
        selection=_selection_block(sm.selection, sm.selection_status, sm.selection_leakage),
        audit_status=audit_result.status, audit_leakage_hash=audit_result.leakage_hash,
        pred_hashes=[bundle_by_role[r].prediction_content_hash() for r in _ROLES],
        metrics_hashes=[metrics_by_role[r].metrics_hash for r in _ROLES],
        training_diagnostics_hash=scientific_value_hash(dict(tm.training_diagnostics)))


def method_result_logical_payload(m) -> dict:
    """Rebuild the payload from a persisted MethodRunResult."""
    bundle = {"source_guard": m.source_guard_predictions, "source_audit": m.source_audit_predictions,
              "target_audit": m.target_predictions}
    mets = {"source_guard": m.source_guard_metrics, "source_audit": m.source_audit_metrics,
            "target_audit": m.target_metrics}
    return method_payload(
        name=m.method_name, active=m.active, inactive_reason=m.inactive_reason, shared_erm=m.shared_erm_hash,
        shared_tau=m.shared_tau, shared_stage2_task_plan_hash=m.shared_stage2_task_plan_hash,
        initial_erm=m.initial_erm_hash, trajectory=_trajectory(m.train_result.trajectory),
        selection=_selection_block(m.selection, m.selection_status, m.selection_leakage),
        audit_status=m.audit_status,
        audit_leakage_hash=None if m.audit_leakage is None else leakage_result_hash(m.audit_leakage),
        pred_hashes=[bundle[r].prediction_content_hash() for r in _ROLES],
        metrics_hashes=[mets[r].metrics_hash for r in _ROLES],
        training_diagnostics_hash=scientific_value_hash(dict(m.training_diagnostics_items)))


def level_payload(*, run_key_hash, support_hash, level_support_hash, level_plans_hash, execution_config_hash,
                  model_spec_hash, erm, phase, selection_snapshot_hash, method_hashes, selection_cache_hash,
                  audit_cache_hash, prediction_cache_hash, provenance_hash, invariant_items,
                  decision_hashes=None) -> dict:
    p = {"run_key": run_key_hash, "support_hash": support_hash, "level_support_hash": level_support_hash,
         "level_plans_hash": level_plans_hash, "execution_config_hash": execution_config_hash,
         "model_spec_hash": model_spec_hash, "erm": erm, "phase": phase,
         "selection_snapshot_hash": selection_snapshot_hash, "methods": list(method_hashes),
         "selection_cache": selection_cache_hash, "audit_cache": audit_cache_hash,
         "prediction_cache": prediction_cache_hash, "provenance": provenance_hash,
         "invariants": [[k, (v if isinstance(v, (bool, int)) else str(v))] for k, v in invariant_items]}
    if decision_hashes is not None:                       # C8a: bind K1/K2 ONLY when decisions are enabled
        p["decision"] = decision_hashes                   # (absent -> legacy payload byte-identical)
    return p


def _erm_block(erm_stage):
    return {"checkpoint": erm_stage.checkpoint.model_hash, "R_ERM_hat": float(erm_stage.R_ERM_hat),
            "tau": float(erm_stage.tau), "invocation_id": erm_stage.stage1_invocation_id,
            "task_plan_hash": erm_stage.task_plan_hash}


def level_result_logical_payload(lr) -> dict:
    # C8a: reconstruct the SAME decision binding finalize folded in, so the writer's re-hash of the payload
    # matches lr.level_result_hash (and deep verify recomputes it identically). Absent when no decision.
    decision_hashes = None
    if getattr(lr, "decision", None) is not None:
        from ..runner.decision import decision_binding_hashes
        decision_hashes = decision_binding_hashes(lr.decision)
    return level_payload(
        run_key_hash=lr.run_key.run_key_hash, support_hash=lr.support_state.support_hash,
        level_support_hash=lr.support_state.level_support_hash, level_plans_hash=lr.plans.level_plans_hash,
        execution_config_hash=lr.execution_config_hash, model_spec_hash=lr.model_spec_hash,
        erm=_erm_block(lr.erm_stage), phase=lr.phase.value, selection_snapshot_hash=lr.selection_snapshot_hash,
        method_hashes=[m.method_result_hash for _, m in lr.method_items],
        selection_cache_hash=scientific_value_hash(lr.selection_cache_stats),
        audit_cache_hash=lr.audit_cache_stats.stats_hash, prediction_cache_hash=lr.prediction_cache_stats.stats_hash,
        provenance_hash=lr.provenance.provenance_hash, invariant_items=lr.invariant_items,
        decision_hashes=decision_hashes)


def fold_payload(*, fold_scope_hash, level_hashes) -> dict:
    return {"scope": fold_scope_hash, "levels": [[int(lvl), h] for lvl, h in level_hashes]}


def fold_result_logical_payload(fr) -> dict:
    return fold_payload(fold_scope_hash=fr.fold_scope.fold_scope_hash,
                        level_hashes=[(lvl, lr.level_result_hash) for lvl, lr in fr.level_items])
