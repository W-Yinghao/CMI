"""Two-phase staged fold runner (C4b-2c) -- the persistence boundary for the GPU/CPU two-job split.

* **Phase A (GPU)**: for each deletion level, train + GPU-extract every feasible candidate's features /
  logits into a per-level :class:`ReplayStore`, and persist the trained state (Stage-1 + the four-method
  Stage-2 trajectories) and the store to ``out_dir``. Release the GPU.
* **Phase B (CPU)**: REBUILD the deterministic non-training context (support / population / plans / run
  key) from the same fold + manifest, load the persisted trained state and store, and resume
  select -> lock -> audit -> finalize from the store (no GPU). Assemble the FoldRunResult.

Only the trained state and the GPU-extracted artifacts cross the boundary; the large fold tensors and the
plans are rebuilt deterministically in Phase B (re-loading the offline data is far cheaper than holding a
V100 through the leakage scoring). The result is bit-identical to the monolithic in-memory run.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pickle

from .finalize import assemble_fold_run
from .keys import RunKey
from .plans import build_level_plans
from .replay_store import ReplayStore
from .scope import build_level_population
from .staged import (DEFAULT_METHOD_ORDER, prefetch_level_gpu_artifacts, resume_level_from_store,
                     train_level)
from .support import build_level_support, level0_reference_prior

_PHASE_A_MANIFEST = "phase_a.json"
_FOLD_PKL = "fold.pkl"


def _persistable_fold(fold):
    """The fold without any heavy/unpicklable provisioning field (e.g. the MOABB load_result, used only
    during the initial build, never in the resume)."""
    return dataclasses.replace(fold, load_result=None) if hasattr(fold, "load_result") else fold


def _save_fold(fold, out_dir) -> None:
    with open(os.path.join(out_dir, _FOLD_PKL), "wb") as f:
        pickle.dump(_persistable_fold(fold), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_phase_a_fold(out_dir):
    """Load the EXACT fold Phase A used (data + scope) -- Phase B must NOT re-load the data: the offline
    MNE/scipy preprocessing is not bit-reproducible across nodes, so a rebuild gives a different
    target_tensor_hash / fold_scope_hash."""
    with open(os.path.join(out_dir, _FOLD_PKL), "rb") as f:
        return pickle.load(f)


def _level_contexts(fold, model_seed, dataset_id):
    """The deterministic per-level context (support / population / plans / run key) -- identical in both
    phases. Never trains."""
    fd, maps, schedule, fs = fold.fold_data, fold.maps, fold.deletion_schedule, fold.fold_scope
    support_m = int(fold.manifest.enabled_datasets()[dataset_id].support_m)
    ref = level0_reference_prior(fd, maps)
    for level in (0, 1):
        ss = build_level_support(fd, maps, level, schedule, ref, support_m=support_m)
        lp = build_level_population(fd, maps, ss)
        plans = build_level_plans(fs, level, ss, lp, fold.scope_config, model_seed=int(model_seed))
        rk = RunKey(fs.fold_key, level, int(model_seed))
        yield level, rk, ss, lp, plans


def staged_phase_a(fold, *, dataset_id, model_seed=0, method_order=DEFAULT_METHOD_ORDER,
                   gpu_device, out_dir) -> str:
    """GPU record stage: train + prefetch every level, persist the trained state + store to out_dir."""
    fd, fs = fold.fold_data, fold.fold_scope
    exec_cfg, model_spec = fold.execution_config, fold.model_spec
    os.makedirs(out_dir, exist_ok=True)
    _save_fold(fold, out_dir)                                       # Phase B loads this EXACT fold (no re-load)
    levels = []
    for level, rk, ss, lp, plans in _level_contexts(fold, model_seed, dataset_id):
        stage1, trained = train_level(rk, lp, plans, ss, fs, exec_cfg, model_spec, fold.model_factory(),
                                      gpu_device, method_order)
        store = ReplayStore()
        cands = prefetch_level_gpu_artifacts(rk, stage1, trained, fd, ss, lp, fs, plans, exec_cfg,
                                             model_spec, fold.model_factory(), gpu_device, store)
        with open(os.path.join(out_dir, f"level-{level}-trained.pkl"), "wb") as f:
            pickle.dump({"stage1": stage1, "trained": trained}, f, protocol=pickle.HIGHEST_PROTOCOL)
        store.save(os.path.join(out_dir, f"level-{level}-store.pkl"))
        levels.append({"level": int(level), "n_candidates": len(cands), "n_store": len(store)})
    with open(os.path.join(out_dir, _PHASE_A_MANIFEST), "w") as f:
        json.dump({"model_seed": int(model_seed), "method_order": list(method_order),
                   "dataset_id": str(dataset_id), "manifest_hash": fold.manifest_hash,
                   "fold_scope_hash": fs.fold_scope_hash, "levels": levels}, f, sort_keys=True)
    return out_dir


def staged_phase_b(out_dir, *, fold=None, cpu_device="cpu", decision_ctx=None):
    """CPU replay stage: LOAD the exact Phase-A fold (never re-load the data), rebuild the deterministic
    context, load the persisted trained state + store, resume each level from the store (no GPU), and
    assemble the FoldRunResult. ``fold`` may be passed (e.g. the in-process fake fixture); otherwise the
    fold persisted by Phase A is loaded."""
    with open(os.path.join(out_dir, _PHASE_A_MANIFEST)) as f:
        meta = json.load(f)
    if fold is None:
        fold = load_phase_a_fold(out_dir)
    if fold.manifest_hash != meta["manifest_hash"]:
        raise ValueError("Phase B fold manifest hash does not match Phase A")
    if fold.fold_scope.fold_scope_hash != meta["fold_scope_hash"]:
        raise ValueError("Phase B fold scope hash does not match Phase A (the data was re-loaded?)")
    fd, fs = fold.fold_data, fold.fold_scope
    exec_cfg, model_spec = fold.execution_config, fold.model_spec
    model_seed, method_order, dataset_id = meta["model_seed"], tuple(meta["method_order"]), meta["dataset_id"]
    levels = {}
    for level, rk, ss, lp, plans in _level_contexts(fold, model_seed, dataset_id):
        with open(os.path.join(out_dir, f"level-{level}-trained.pkl"), "rb") as f:
            t = pickle.load(f)
        store = ReplayStore.load(os.path.join(out_dir, f"level-{level}-store.pkl"))
        levels[level] = resume_level_from_store(rk, fd, ss, lp, fs, plans, exec_cfg, model_spec,
                                                fold.model_factory(), t["stage1"], t["trained"], store,
                                                device=cpu_device, method_order=method_order,
                                                decision_ctx=decision_ctx)
    return assemble_fold_run(fs, levels)
