"""Finalize one level (predictions + metrics + COMPLETE) and assemble a fold (A2b-1b-ii-b).

Consumes a LevelAuditIntermediate; never re-trains, re-selects, or re-runs the audit estimator. The
selected checkpoints are forwarded for the three roles (one forward per role per unique model hash),
aggregated to eval units, and scored. The selection snapshot is re-checked unchanged after the
predictions; the target role is forward only. Phase reaches COMPLETE.
"""
from __future__ import annotations

from ..artifacts.result_payload import _erm_block, fold_payload, level_payload, method_payload_from_trained
from ..eval.calibration import fixed_bin_edges
from ..methods.activity import METHODS
from ..train.rng import derive_seed
from .audit import make_selection_snapshot
from .final_results import (FoldRunResult, LevelRunResult, MethodRunResult, PredictionCacheStats)
from .metrics import evaluate_prediction_bundle
from .predict import PredictionCacheKey, RowPredictionCache, aggregate_role_to_bundle, predict_checkpoint
from .provenance import RunnerPhase
from .scientific_hash import leakage_result_hash, scientific_value_hash

_ORDER = tuple(sorted(METHODS))
_ROLES = ("source_guard", "source_audit", "target_audit")


def _leak_hash(v):
    return None if v is None else leakage_result_hash(v)


def method_result_hash(name, trained_method, selected_method, audit_result, bundle_by_role, metrics_by_role) -> str:
    """Full logical identity of one method run (delegates to the shared payload builder)."""
    return scientific_value_hash(method_payload_from_trained(
        name, trained_method, selected_method, audit_result, bundle_by_role, metrics_by_role))


def finalize_level_run(audit_intermediate, fold_data, fold_scope, support_state, level_population,
                       level_plans, execution_cfg, model_spec, model_factory, device) -> LevelRunResult:
    if audit_intermediate.phase != RunnerPhase.AUDIT or audit_intermediate.provenance.phase != RunnerPhase.AUDIT:
        raise ValueError("finalize requires an AUDIT-phase audit intermediate")
    ts = audit_intermediate.training_selection
    selected, trained = ts.selected_methods, ts.trained_methods
    snap0 = audit_intermediate.selection_snapshot_before
    if audit_intermediate.selection_snapshot_after.snapshot_hash != snap0.snapshot_hash:
        raise RuntimeError("selection snapshot already changed before finalize")

    run_key, maps = ts.run_key, fold_scope.maps
    fold_key_hash = fold_scope.fold_key.fold_key_hash
    edges = fixed_bin_edges(execution_cfg.ece_bins)

    views = {
        "source_guard": fold_data.make_role_view("source_guard", support_state.source_train_idx),
        "source_audit": fold_data.make_role_view("source_audit"),
        "target_audit": fold_data.make_role_view("target_audit"),
    }
    role_domain_map = {"source_guard": maps.source_domain_to_index,
                       "source_audit": maps.evaluation_domain_to_index,
                       "target_audit": maps.evaluation_domain_to_index}
    caches = {r: RowPredictionCache() for r in _ROLES}

    # one forward per (role, unique model hash); aggregate to a method-specific eval-unit bundle
    bundles = {n: {} for n in _ORDER}
    metrics = {n: {} for n in _ORDER}
    for role in _ROLES:
        rv, dmap, cache = views[role], role_domain_map[role], caches[role]
        for name in _ORDER:
            sel = selected[name].selection
            key = PredictionCacheKey(sel.model_hash, role, rv.population_hash, rv.tensor_hash,
                                     model_spec.model_spec_hash, execution_cfg.prediction_chunk_size)
            fseed = derive_seed(run_key.model_seed, "prediction_factory", run_key.run_key_hash, role, sel.model_hash)
            row = cache.get_or_compute(key, lambda s=sel, f=fseed, v=rv: predict_checkpoint(
                s.model_state, s.model_hash, model_factory, v, factory_seed=f,
                chunk_size=execution_cfg.prediction_chunk_size, device=device))
            b = aggregate_role_to_bundle(
                row, rv, method_name=name, selected_model_hash=sel.model_hash, domain_map=dmap,
                class_names=maps.class_names, model_seed=run_key.model_seed, fold_key_hash=fold_key_hash,
                support_hash=support_state.support_hash, split_manifest_hash=fold_data.split_manifest_hash,
                preprocess_hash=fold_data.preprocess_hash, risk_metric=execution_cfg.engine_template.metric,
                prob_floor=execution_cfg.prediction_prob_floor, deletion_level=run_key.deletion_level)
            bundles[name][role] = b
            metrics[name][role] = evaluate_prediction_bundle(b, bin_edges=edges)

    # predictions must not have disturbed the selection
    snap_pred = make_selection_snapshot(selected)
    if snap_pred.snapshot_hash != snap0.snapshot_hash:
        raise RuntimeError("a prediction changed a selection field or model state")

    prov = audit_intermediate.provenance
    sa_ids = set(fold_data.role_ids("source_audit"))
    if sa_ids & (set(prov.optimization_fit_ids) | set(prov.selection_fit_ids)):
        raise RuntimeError("a source-audit id leaked into an optimization/selection fit")
    if prov.target_fit_ids:
        raise RuntimeError("target fits are forbidden")
    prov.transition(RunnerPhase.COMPLETE)
    prov_snap = prov.snapshot()

    audit_map = dict(audit_intermediate.audit_method_items)
    method_items = []
    for name in _ORDER:
        tm, sm = trained[name], selected[name]
        ar = audit_map[name]
        diag_items = tuple(sorted(tm.training_diagnostics.items(), key=lambda kv: str(kv[0])))
        mrh = method_result_hash(name, tm, sm, ar, bundles[name], metrics[name])
        method_items.append((name, MethodRunResult(
            method_name=name, active=tm.active, inactive_reason=tm.inactive_reason,
            shared_erm_hash=tm.shared_erm_hash, shared_tau=tm.shared_tau,
            shared_stage2_task_plan_hash=tm.shared_stage2_task_plan_hash, initial_erm_hash=tm.shared_erm_hash,
            train_result=tm.train_result, training_diagnostics_items=diag_items, selection=sm.selection,
            selection_status=sm.selection_status, selection_leakage=sm.selection_leakage,
            audit_status=ar.status, audit_leakage=ar.leakage,
            source_guard_predictions=bundles[name]["source_guard"],
            source_audit_predictions=bundles[name]["source_audit"],
            target_predictions=bundles[name]["target_audit"],
            source_guard_metrics=metrics[name]["source_guard"],
            source_audit_metrics=metrics[name]["source_audit"], target_metrics=metrics[name]["target_audit"],
            method_result_hash=mrh)))

    pc = _prediction_cache_stats(caches)
    inv = _level_invariants(prov, trained, selected, bundles, caches, snap0, snap_pred)
    inv_items = tuple(sorted(inv.items(), key=lambda kv: kv[0]))
    erm_stage = ts.stage1.erm_stage
    lvl_payload = level_payload(
        run_key_hash=run_key.run_key_hash, support_hash=support_state.support_hash,
        level_support_hash=support_state.level_support_hash, level_plans_hash=level_plans.level_plans_hash,
        execution_config_hash=execution_cfg.execution_config_hash, model_spec_hash=model_spec.model_spec_hash,
        erm=_erm_block(erm_stage), phase=prov.phase.value, selection_snapshot_hash=snap0.snapshot_hash,
        method_hashes=[m.method_result_hash for _, m in method_items],
        selection_cache_hash=scientific_value_hash(ts.leakage_cache_stats),
        audit_cache_hash=audit_intermediate.audit_cache_stats.stats_hash,
        prediction_cache_hash=pc.stats_hash, provenance_hash=prov_snap.provenance_hash, invariant_items=inv_items)
    return LevelRunResult(
        run_key=run_key, support_state=support_state, plans=level_plans, erm_stage=erm_stage,
        method_items=tuple(method_items), execution_config_hash=execution_cfg.execution_config_hash,
        model_spec_hash=model_spec.model_spec_hash, provenance=prov_snap, phase=RunnerPhase.COMPLETE,
        selection_snapshot_hash=snap0.snapshot_hash, selection_cache_stats=ts.leakage_cache_stats,
        audit_cache_stats=audit_intermediate.audit_cache_stats, prediction_cache_stats=pc,
        invariant_items=inv_items, level_result_hash=scientific_value_hash(lvl_payload))


def _prediction_cache_stats(caches) -> PredictionCacheStats:
    sg, sa, ta = caches["source_guard"].role_stats("source_guard"), \
        caches["source_audit"].role_stats("source_audit"), caches["target_audit"].role_stats("target_audit")
    s = dict(source_guard_requests=sg[0], source_guard_computes=sg[1], source_guard_hits=sg[2],
             source_audit_requests=sa[0], source_audit_computes=sa[1], source_audit_hits=sa[2],
             target_requests=ta[0], target_computes=ta[1], target_hits=ta[2])
    return PredictionCacheStats(**s, stats_hash=scientific_value_hash(s))


def _level_invariants(prov, trained, selected, bundles, caches, snap0, snap_pred) -> dict:
    hashes = [selected[n].selection.model_hash for n in _ORDER]
    K = len(set(hashes))
    erm = {trained[n].shared_erm_hash for n in _ORDER}
    tau = {round(float(trained[n].shared_tau), 12) for n in _ORDER}
    plan = {trained[n].shared_stage2_task_plan_hash for n in _ORDER}
    inv = {
        "phase_complete": prov.phase == RunnerPhase.COMPLETE,
        "selection_snapshot_unchanged": snap_pred.snapshot_hash == snap0.snapshot_hash,
        "target_fit_ids_empty": not prov.target_fit_ids,
        "shared_erm_unique": len(erm) == 1, "shared_tau_unique": len(tau) == 1,
        "shared_task_plan_unique": len(plan) == 1,
        "source_audit_signature_match": len({bundles[n]["source_audit"].audit_signature_hash for n in _ORDER}) == 1,
        "target_signature_match": len({bundles[n]["target_audit"].audit_signature_hash for n in _ORDER}) == 1,
        "n_unique_checkpoints": K,
    }
    for n in _ORDER:
        if not trained[n].active:                                          # inactive -> selected ERM checkpoint
            inv[f"{n}_inactive_is_erm"] = (selected[n].selection.selected_erm
                                           and selected[n].selection.model_hash == trained[n].shared_erm_hash)
    if trained["OACI"].active:
        inv["oaci_rejected_ineligible_rows"] = trained["OACI"].training_diagnostics["rejected_ineligible_rows"]
    for r in _ROLES:
        req, comp, hit = caches[r].role_stats(r)
        inv[f"{r}_cache_reuse_valid"] = (req == 4 and comp == K and hit == 4 - K)
    return inv


def assemble_fold_run(fold_scope, level_results) -> FoldRunResult:
    items = sorted(((int(lvl), lr) for lvl, lr in level_results.items()), key=lambda kv: kv[0])
    if len({lvl for lvl, _ in items}) != len(items):
        raise ValueError("duplicate deletion level")
    sa_sig = set(); ta_sig = set(); uni = set()
    for _, lr in items:
        if lr.phase != RunnerPhase.COMPLETE:
            raise ValueError("every level must be COMPLETE")
        if lr.run_key.fold_key.fold_key_hash != fold_scope.fold_key.fold_key_hash:
            raise ValueError("a level FoldKey disagrees with the fold scope")
        for _, m in lr.method_items:
            sa_sig.add(m.source_audit_predictions.audit_signature_hash)
            ta_sig.add(m.target_predictions.audit_signature_hash)
        um = lr.methods["uniform"].training_diagnostics.get("prior_matrix_hash")
        if um is not None:
            uni.add(um)
    if len(sa_sig) > 1:
        raise ValueError("source-audit signature changed across methods/levels")
    if len(ta_sig) > 1:
        raise ValueError("target signature changed across methods/levels")
    if len(uni) > 1:
        raise ValueError("uniform prior matrix changed across levels")
    payload = fold_payload(fold_scope_hash=fold_scope.fold_scope_hash,
                           level_hashes=[(lvl, lr.level_result_hash) for lvl, lr in items])
    return FoldRunResult(fold_scope=fold_scope, level_items=tuple(items),
                         fold_result_hash=scientific_value_hash(payload))
