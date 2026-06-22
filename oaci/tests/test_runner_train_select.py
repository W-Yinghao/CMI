"""A2b-1b-i: Stage-1 once, four-method training, source-train leakage selection (in memory).

Standalone (``python -m oaci.tests.test_runner_train_select``) and pytest-compatible.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from oaci.leakage.critic import CriticConfig
from oaci.models import build_model
from oaci.runner import (DEFAULT_METHOD_ORDER, DeletionCell, FoldData, ModelSpec, RunnerExecutionConfig,
                         ScopePlanConfig, build_fold_scope, build_frozen_maps, build_level_plans,
                         build_level_population, build_level_support, level0_reference_prior,
                         make_deletion_schedule, make_objective, run_level_training_selection,
                         unique_feasible_records)
from oaci.runner.keys import FoldKey, RunKey
from oaci.train.engine import EngineConfig


def _X(rows):
    def v(s):
        return np.random.default_rng(int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)).standard_normal(5).astype(np.float32)
    return torch.from_numpy(np.stack([v(r["sid"]) for r in rows]))


def _rows(src_dom=2, src_recs=4, single_src=None):
    rows = []

    def blk(p, nd, nr, role):
        for d in range(nd):
            dom = f"{p}d{d}"
            classes = [0] if single_src == (role, d) else [0, 1]
            for r in range(nr):
                for c in classes:
                    rows.append(dict(sid=f"{role}_{dom}_r{r}_c{c}", dom=dom, grp=f"{dom}-rec{r}",
                                     unit=f"{role}_{dom}_r{r}_c{c}", y=c, mass=1.0, role=role))
    blk("S", src_dom, src_recs, "source_train")
    blk("A", 2, 3, "source_audit")
    blk("T", 1, 2, "target_audit")
    return rows


def _fold(rows):
    ri = {"source_train": [], "source_audit": [], "target_audit": []}
    for i, r in enumerate(rows):
        ri[r["role"]].append(i)
    return FoldData.from_arrays(
        X=_X(rows), y=np.array([r["y"] for r in rows]), sample_id=[r["sid"] for r in rows],
        domain_id=[r["dom"] for r in rows], group_id=[r["grp"] for r in rows],
        support_unit_id=[r["unit"] for r in rows], mass_unit_id=[r["unit"] for r in rows],
        eval_unit_id=[r["unit"] for r in rows], sample_mass=np.ones(len(rows)), class_names=["c0", "c1"],
        source_train_idx=np.array(ri["source_train"]), source_audit_idx=np.array(ri["source_audit"]),
        target_audit_idx=np.array(ri["target_audit"]), preprocess_hash="pp", split_manifest_hash="sp",
        preprocess_fit_ids=frozenset())


def _scope_cfg(support_m=2):
    return ScopePlanConfig(support_m=support_m, leakage_alpha=0.2, probe_folds=2, probe_capacities=(0, 8),
                           l2_C=1.0, max_iter=50, prob_floor=1e-6, feature_seed_base=10000,
                           selection_bootstrap_replicates=4, audit_bootstrap_replicates=4,
                           max_candidate_multiplier=8, max_invalid_draw_rate=0.5, stage1_epochs=2,
                           stage2_epochs=2, stage1_steps_per_epoch=1, stage2_steps_per_epoch=1,
                           task_batch_size=4, warmup_steps=1, critic_steps=1, min_per_eligible_cell=2,
                           min_per_observed_cell=2, adv_microbatch_size=64, adv_accumulation_steps=4,
                           replacement_mode="auto", selection_seed=303, audit_seed=404)


def _exec_cfg():
    eng = EngineConfig(metric="ce", epsilon=0.03, numerical_tol=1e-4, stage1_epochs=2, stage1_steps_per_epoch=1,
                       stage2_epochs=2, steps_per_epoch=1, warmup_steps=1, critic_steps=1, checkpoint_every=1,
                       lr_stage1=0.05, lr_encoder=0.1, lr_critic=0.05)
    return RunnerExecutionConfig(eng, CriticConfig(capacities=(0, 8), max_iter=50), 8, 1.0, 1e-8, None, None, "exec1")


def _factory():
    return build_model("mlp", in_dim=5, n_classes=2, z_dim=8, hidden=8)


_MSPEC = ModelSpec.build("mlp", {"z_dim": 8, "hidden": 8}, (5,), 2)


def _run(rows=None, model_seed=0, order=DEFAULT_METHOD_ORDER, support_m=2, deletion=None, factory=None):
    rows = rows if rows is not None else _rows()
    fd, maps = _fold(rows), build_frozen_maps(["c0", "c1"], sorted({r["dom"] for r in rows if r["role"] == "source_train"}),
                                              ["Ad0", "Ad1", "Td0"])
    sch = make_deletion_schedule(deletion if deletion is not None else [DeletionCell("Sd0", "c1")], fd, maps)
    ref = level0_reference_prior(fd, maps)
    cfg = _scope_cfg(support_m)
    rk = RunKey(FoldKey("m", "ds", "f0", 1, 2), 0, model_seed)
    ss = build_level_support(fd, maps, 0, sch, ref, support_m=support_m)
    lp = build_level_population(fd, maps, ss)
    fs = build_fold_scope(rk.fold_key, maps, fd, sch, cfg)
    plans = build_level_plans(fs, 0, ss, lp, cfg, model_seed=model_seed)
    res = run_level_training_selection(rk, fd, ss, lp, fs, plans, _exec_cfg(), _MSPEC,
                                       factory or _factory, torch.device("cpu"), method_order=order)
    return res, (fd, maps, ss, lp, fs, plans)


_CACHE = {}


def _default():
    if "r" not in _CACHE:
        _CACHE["r"] = _run()
    return _CACHE["r"]


# ---------------- config / spec (fast) ----------------
def test_runner_config_consumes_every_manifest_field():
    import os
    from oaci.protocol.freeze import default_confirmatory_path
    from oaci.protocol.manifest_v2 import load_v2
    m = load_v2(os.path.join(os.path.dirname(default_confirmatory_path()), "smoke_v1.yaml"))
    ec = RunnerExecutionConfig.from_manifest(m)
    assert ec.method_critic_hidden == m.methods.critic_capacity
    assert ec.global_lpc_alpha == m.methods.global_lpc_laplace_smoothing
    assert ec.feature_chunk_size == m.training.feature_chunk_size and len(ec.execution_config_hash) == 64


def test_feature_and_prediction_chunk_sizes_enter_manifest_hash():
    import os
    from oaci.protocol.freeze import default_confirmatory_path
    from oaci.protocol.manifest_v2 import load_v2
    p = os.path.join(os.path.dirname(default_confirmatory_path()), "smoke_v1.yaml")
    base = load_v2(p).freeze()["sha256"]
    m = load_v2(p); m.training.feature_chunk_size = (m.training.feature_chunk_size or 1) + 1
    assert m.freeze()["sha256"] != base


def test_model_spec_hash_binds_input_shape_and_class_count():
    base = ModelSpec.build("mlp", {"z_dim": 8}, (5,), 2).model_spec_hash
    assert ModelSpec.build("mlp", {"z_dim": 8}, (6,), 2).model_spec_hash != base
    assert ModelSpec.build("mlp", {"z_dim": 8}, (5,), 3).model_spec_hash != base


def test_unique_feasible_records_dedup_and_erm_exclusion():
    from oaci.train.checkpoint import CheckpointRecord, ERMStage, TrainResult

    def ck(h, R):
        return CheckpointRecord(0, 0, {"w": torch.tensor([0.0])}, h, R, 0.0, 0.0, 0.0)
    erm = ck("erm", 0.4)
    stage = ERMStage(erm, 0.4, 0.5, "t", "i")
    traj = [ck("a", 0.45), ck("a", 0.45), ck("erm", 0.4), ck("b", 0.9), ck("c", 0.46)]   # dup a, erm, infeasible b
    tr = TrainResult("OACI", True, None, stage, erm, traj, "erm", "t", "a")
    feas = unique_feasible_records(tr, numerical_tol=1e-4)
    assert [c.model_hash for c in feas] == ["a", "c"]                  # dedup a, drop erm, drop infeasible b
    bad = TrainResult("OACI", True, None, stage, erm, [ck("a", 0.45), ck("a", 0.49)], "erm", "t", "a")
    try:
        unique_feasible_records(bad, numerical_tol=1e-4)
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent R_src for a duplicate hash must fail")


def test_uniform_prior_is_exact_and_global_lpc_deleted_cell_prior_positive_rows_zero():
    res, (fd, maps, ss0, lp, fs, plans) = _default()
    _, uspec = make_objective("uniform", ss0, fs, _exec_cfg())
    assert np.allclose(uspec.prior_matrix, 1.0 / len(maps.source_domain_ids))
    # LEVEL 1 actually deletes (Sd0, c1)
    sch = make_deletion_schedule([DeletionCell("Sd0", "c1")], fd, maps)
    ss1 = build_level_support(fd, maps, 1, sch, level0_reference_prior(fd, maps), support_m=2)
    _, gspec = make_objective("global_lpc", ss1, fs, _exec_cfg())
    d = maps.source_domain_to_index["Sd0"]
    assert ss1.cell_mass[d, 1] == 0 and gspec.prior_matrix[1, d] > 0     # deleted cell: 0 rows, positive prior


# ---------------- end-to-end (shared run) ----------------
def test_stage1_is_invoked_once_and_methods_share_erm_tau_and_task_plan():
    res, _ = _default()
    assert res.invariants["stage1_invocation_count"] == 1 and res.invariants["stage1_total_claims"] == 1
    assert res.invariants["shared_erm_unique"] and res.invariants["shared_tau_unique"]
    assert res.invariants["stage2_task_plan_unique"] and res.invariants["lpc_uniform_alignment_same_object"]


def test_stage1_registry_is_owned_by_level_orchestrator():
    from oaci.runner import run_stage1_once
    from oaci.train.engine import InvocationRegistry
    res, (fd, maps, ss, lp, fs, plans) = _default()
    reg = InvocationRegistry()
    rk = RunKey(FoldKey("m", "ds", "f0", 1, 2), 0, 0)
    run_stage1_once(rk, lp, plans, _factory, _MSPEC, _exec_cfg().engine_config_for(rk), "exec1", reg, torch.device("cpu"))
    try:
        run_stage1_once(rk, lp, plans, _factory, _MSPEC, _exec_cfg().engine_config_for(rk), "exec1", reg, torch.device("cpu"))
    except ValueError:
        pass
    else:
        raise AssertionError("a second Stage-1 on the same registry must fail")


def test_erm_cache_counts_are_exactly_4_1_3():
    res, _ = _default()
    assert res.leakage_cache_stats == {"erm_request": 4, "erm_compute": 1, "erm_hit": 3}
    assert res.invariants["erm_cache_request_compute_hit"] == (4, 1, 3)


def test_selected_leakage_retains_full_result():
    res, _ = _default()
    leak = res.selected_methods["OACI"].selection_leakage
    for k in ("extractable_LQ_ov", "bootstrap_ucl", "selected_capacity", "replicate_capacities",
              "fold_plan_hash", "bootstrap_plan_hash"):
        assert k in leak


def test_phase_remains_selection_and_fit_ids_are_overlap_only():
    res, (fd, maps, ss, lp, fs, plans) = _default()
    from oaci.runner.provenance import RunnerPhase
    assert res.phase == RunnerPhase.SELECTION
    # selection fit ids are only the overlap-probe rows (d in S_y), a strict subset of source_train
    st = set(ss.source_train_sample_ids)
    assert res.provenance.selection_fit_ids <= st and len(res.provenance.selection_fit_ids) > 0
    assert res.provenance.optimization_fit_ids == st and not res.provenance.target_fit_ids


def test_nonestimable_selection_returns_erm_without_cache_work():
    res, _ = _run(rows=_rows(src_recs=1), deletion=[])                 # 1 group/cell -> selection folds fail
    for m, sm in res.selected_methods.items():
        assert sm.selection.selected_erm and sm.selection_status.startswith("nonestimable")
        assert sm.selection_leakage is None and sm.selection.selection_score is None
    assert res.leakage_cache_stats == {} and res.feature_cache_stats == {}


def test_permuted_method_order_reproduces_training_and_selection():
    a, _ = _default()
    b, _ = _run(order=("uniform", "global_lpc", "OACI", "ERM"))
    for m in DEFAULT_METHOD_ORDER:
        assert a.selected_methods[m].selection.model_hash == b.selected_methods[m].selection.model_hash
        assert a.trained_methods[m].train_result.initial_model_hash == b.trained_methods[m].train_result.initial_model_hash
        assert ([c.model_hash for c in a.trained_methods[m].train_result.trajectory]
                == [c.model_hash for c in b.trained_methods[m].train_result.trajectory])


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} runner-train-select tests")


if __name__ == "__main__":
    _run_all()
