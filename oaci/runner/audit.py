"""Selection snapshot + lock + fixed source-audit leakage (A2b-1b-ii-a).

Nothing reads the source-audit X before ``lock_selection()``. The audit replays the FROZEN AuditScope
design / fold / bootstrap plans; a selected model hash shared by several methods is feature-extracted
and scored exactly once. The selection snapshot is verified byte-exact unchanged across the audit.
Phase stops at AUDIT.
"""
from __future__ import annotations

import numpy as np
import torch

from ..methods.activity import METHODS
from ..train.checkpoint import state_hash
from ..train.data import TrainingData, population_signature_hash
from ..train.rng import derive_seed
from .audit_results import (AuditCacheStats, AuditMethodResult, LevelAuditIntermediate,
                            MethodSelectionSnapshot, SelectionSnapshot)
from .features import extract_frozen_features
from .provenance import RunnerPhase
from .scientific_hash import leakage_result_hash, scientific_value_hash
from .scoring import compute_leakage_score, overlap_probe_sample_ids
from .selection import FeatureArtifactCache, FeatureArtifactKey, make_leakage_score_key
from ..leakage.cache import LeakageScoreCache

_ORDER = tuple(sorted(METHODS))


def make_selection_snapshot(selected_methods) -> SelectionSnapshot:
    if set(selected_methods) != set(METHODS):
        raise ValueError(f"selection snapshot needs exactly {sorted(METHODS)}")
    snaps = []
    for name in _ORDER:                                          # canonical order -> order-invariant
        sm = selected_methods[name]; sel = sm.selection
        recomputed = state_hash(sel.model_state)
        if recomputed != sel.model_hash:
            raise ValueError(f"{name}: selected model state hash does not recompute")
        if sm.selection_status == "estimable":
            if sm.selection_leakage is None:
                raise ValueError(f"{name}: estimable selection must carry a leakage result")
        else:
            if not sel.selected_erm or sm.selection_leakage is not None:
                raise ValueError(f"{name}: non-estimable selection must be ERM with no leakage")
        lh = leakage_result_hash(sm.selection_leakage) if sm.selection_leakage is not None else None
        snaps.append(MethodSelectionSnapshot(
            method_name=name, model_hash=sel.model_hash, recomputed_state_hash=recomputed,
            selected_epoch=int(sel.selected_epoch), R_src=float(sel.R_src),
            selection_score=(None if sel.selection_score is None else float(sel.selection_score)),
            selected_erm=bool(sel.selected_erm), used_erm_fallback=bool(sel.used_erm_fallback),
            selection_reason=sel.selection_reason, score_name=sel.score_name, n_feasible=int(sel.n_feasible),
            selection_status=sm.selection_status, selection_leakage_hash=lh,
            feasible_unique_hashes=tuple(sm.feasible_unique_hashes)))
    payload = [{k: getattr(s, k) for k in s.__dataclass_fields__} for s in snaps]
    return SelectionSnapshot(tuple(snaps), scientific_value_hash(payload))


def build_training_data_for_design(fold_data, design, *, role="source_audit", n_classes=None) -> TrainingData:
    if role != "source_audit":
        raise ValueError("audit TrainingData is only built for source_audit")
    fold_data.assert_integrity()
    audit_ids = set(fold_data.role_ids("source_audit"))
    if set(design.sample_id) != audit_ids:
        raise ValueError("design sample ids != source_audit ids")
    if set(design.sample_id) & set(fold_data.role_ids("target_audit")):
        raise ValueError("a target id leaked into the audit design")
    sid2row = {fold_data.sample_id[i]: i for i in fold_data.source_audit_idx.tolist()}
    rows = [sid2row[s] for s in design.sample_id]
    for k, row in enumerate(rows):                              # metadata must match the design row-for-row
        if int(fold_data.y[row]) != int(design.y[k]) or fold_data.group_id[row] != str(design.group_id[k]) \
                or abs(float(fold_data.sample_mass[row]) - float(design.sample_mass[k])) > 1e-9:
            raise ValueError("audit design metadata disagrees with FoldData")
    X = fold_data.X[torch.as_tensor(rows, dtype=torch.long)]
    nc = n_classes if n_classes is not None else len(fold_data.class_names)
    td = TrainingData(X=X, y=torch.as_tensor(np.array(design.y)), sample_id=tuple(design.sample_id),
                      sample_mass=torch.as_tensor(np.array(design.sample_mass)), n_classes=nc,
                      d=torch.as_tensor(np.array(design.d)), group=tuple(str(g) for g in design.group_id)).validate()
    if population_signature_hash(td) != design.population_hash:
        raise ValueError("audit TrainingData population hash != design population hash")
    return td


def _cache_stats(feat_cache, score_cache, n_unique) -> AuditCacheStats:
    s = dict(n_methods=4, n_unique_selected_models=int(n_unique),
             feature_requests=feat_cache.total_requests(), feature_computes=feat_cache.total_computes(),
             feature_hits=feat_cache.total_hits(), score_requests=score_cache.total_requests(),
             score_computes=score_cache.total_computes(), score_hits=score_cache.total_hits())
    return AuditCacheStats(**s, stats_hash=scientific_value_hash(s))


def run_post_selection_audit(intermediate, fold_data, fold_scope, execution_cfg, model_spec,
                             model_factory, device) -> LevelAuditIntermediate:
    if intermediate.phase != RunnerPhase.SELECTION or intermediate.provenance.phase != RunnerPhase.SELECTION:
        raise ValueError("audit requires the intermediate result to be in SELECTION")
    selected = intermediate.selected_methods
    snap_before = make_selection_snapshot(selected)

    prov = intermediate.provenance
    prov.lock_selection()
    if prov.phase != RunnerPhase.SELECTION_LOCKED:
        raise RuntimeError("lock_selection did not reach SELECTION_LOCKED")
    prov.transition(RunnerPhase.AUDIT)

    audit = fold_scope.source_audit
    run_key = intermediate.run_key
    if audit.status == "estimable" and audit.fold_plan is not None and audit.bootstrap_plan is not None:
        design, sg, fold, boot = audit.design, audit.support_graph, audit.fold_plan, audit.bootstrap_plan
        critic = execution_cfg.critic
        td = build_training_data_for_design(fold_data, design)
        prov.record_fit("audit_estimator", overlap_probe_sample_ids(design, sg))
        feat_cache, score_cache = FeatureArtifactCache(), LeakageScoreCache()
        items, hashes, feat_items = [], set(), []
        for name in _ORDER:
            sel = selected[name].selection
            mh, mstate = sel.model_hash, sel.model_state
            hashes.add(mh)
            fseed = derive_seed(run_key.model_seed, "audit_feature_factory", run_key.run_key_hash, mh)
            fkey = FeatureArtifactKey(mh, fold_data.source_audit_tensor_hash, design.population_hash,
                                      model_spec.model_spec_hash, execution_cfg.feature_chunk_size)
            feat = feat_cache.get_or_extract(fkey, lambda ms=mstate, h=mh, s=fseed: extract_frozen_features(
                ms, h, model_factory, td, design, factory_seed=s,
                chunk_size=execution_cfg.feature_chunk_size, device=device), role="source_audit")
            skey = make_leakage_score_key(feat, sg, fold, boot, critic)
            leak = score_cache.get_or_compute(skey, lambda f=feat: compute_leakage_score(f.features, sg, fold, boot, critic))
            items.append((name, AuditMethodResult(name, "estimable", mh, leak, leakage_result_hash(leak))))
            feat_items.append((name, mh, feat))          # C8a: retain the selected source-audit feature (deduped)
        stats = _cache_stats(feat_cache, score_cache, len(hashes))
    else:
        feat_items = []
        items = [(n, AuditMethodResult(n, audit.status, selected[n].selection.model_hash, None, None)) for n in _ORDER]
        stats = AuditCacheStats(4, 0, 0, 0, 0, 0, 0, 0, scientific_value_hash({"nonestimable": audit.status}))

    snap_after = make_selection_snapshot(selected)
    if snap_after.snapshot_hash != snap_before.snapshot_hash:
        raise RuntimeError("the selection snapshot changed during audit")
    first_audit = next((ev.index for ev in prov.ordered_events if ev.kind == "fit:audit_estimator"), None)
    inv = {
        "selection_snapshot_unchanged": True,
        "selection_locked_event_index": prov.selection_locked_event_index,
        "first_audit_fit_event_index": first_audit,
        "all_audit_fits_after_lock": first_audit is None or first_audit > prov.selection_locked_event_index,
        "audit_support_hash": audit.support_graph.support_hash(),
        "audit_fold_plan_hash": None if audit.fold_plan is None else audit.fold_plan.plan_hash,
        "audit_bootstrap_plan_hash": None if audit.bootstrap_plan is None else audit.bootstrap_plan.plan_hash,
        "audit_fit_id_count": len(prov.audit_estimator_fit_ids),
        "n_unique_selected_models": stats.n_unique_selected_models,
        "feature_cache_reuse_valid": (stats.feature_requests == 4
                                      and stats.feature_computes == stats.n_unique_selected_models
                                      and stats.feature_hits == 4 - stats.n_unique_selected_models),
        "score_cache_reuse_valid": (stats.score_requests == 4
                                    and stats.score_computes == stats.n_unique_selected_models
                                    and stats.score_hits == 4 - stats.n_unique_selected_models),
        "target_fit_ids_empty": not prov.target_fit_ids,
    }
    return LevelAuditIntermediate(training_selection=intermediate, selection_snapshot_before=snap_before,
                                  selection_snapshot_after=snap_after, audit_method_items=tuple(items),
                                  audit_cache_stats=stats, provenance=prov, phase=RunnerPhase.AUDIT, invariants=inv,
                                  audit_feature_items=tuple(feat_items))
