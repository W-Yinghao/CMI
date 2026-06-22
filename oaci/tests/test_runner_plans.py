"""A2b-1a pt.3: LevelPopulation / AuditScope / FoldScope / LevelPlans + the 3 identity blockers
(full SHA-256, unified level population identity, retention-policy in the schedule hash).

Standalone (``python -m oaci.tests.test_runner_plans``) and pytest-compatible.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from oaci.runner import (DeletionCell, FoldData, ScopePlanConfig, build_audit_scope, build_fold_scope,
                         build_frozen_maps, build_level_plans, build_level_population, build_level_support,
                         level0_reference_prior, make_deletion_schedule)
from oaci.runner.keys import FoldKey


def _X(rows, dim=5):
    def v(sid):
        s = int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(s).standard_normal(dim).astype(np.float32)
    return torch.from_numpy(np.stack([v(r["sid"]) for r in rows]))


def _rows(src_dom=2, src_recs=4, audit_recs=3, audit_dom=2, single_src=None):
    rows = []

    def blk(p, nd, nr, role):
        for d in range(nd):
            dom = f"{p}d{d}"
            classes = [0] if single_src == (role, d) else [0, 1]
            for r in range(nr):
                grp = f"{dom}-rec{r}"
                for c in classes:
                    sid = f"{role}_{dom}_r{r}_c{c}"
                    rows.append(dict(sid=sid, dom=dom, grp=grp, unit=sid, y=c, mass=1.0, role=role))
    blk("S", src_dom, src_recs, "source_train")
    blk("A", audit_dom, audit_recs, "source_audit")
    blk("T", 1, 2, "target_audit")
    return rows


def _fold(rows=None):
    rows = rows if rows is not None else _rows()
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


def _maps(src=("Sd0", "Sd1")):
    return build_frozen_maps(["c0", "c1"], list(src), ["Ad0", "Ad1", "Td0"])


def _cfg(**over):
    base = dict(support_m=2, leakage_alpha=0.2, probe_folds=2, probe_capacities=(0, 8), l2_C=1.0, max_iter=50,
                prob_floor=1e-6, feature_seed_base=10000, selection_bootstrap_replicates=4,
                audit_bootstrap_replicates=4, max_candidate_multiplier=8, max_invalid_draw_rate=0.5,
                stage1_epochs=2, stage2_epochs=2, stage1_steps_per_epoch=1, stage2_steps_per_epoch=1,
                task_batch_size=4, warmup_steps=1, critic_steps=1, min_per_eligible_cell=2,
                min_per_observed_cell=2, adv_microbatch_size=64, adv_accumulation_steps=4,
                replacement_mode="auto", selection_seed=303, audit_seed=404)
    base.update(over)
    return ScopePlanConfig(**base)


def _fk(**over):
    base = dict(manifest_hash="m", dataset_id="ds", outer_fold="f0", split_seed=1, deletion_seed=2)
    base.update(over)
    return FoldKey(**base)


def _build(rows=None, maps=None, cfg=None, model_seed=0, schedule_cells=None, level=0, **cfgover):
    fd = _fold(rows); maps = maps or _maps(); cfg = cfg or _cfg(**cfgover)
    sch = make_deletion_schedule(schedule_cells if schedule_cells is not None else [DeletionCell("Sd0", "c1")], fd, maps)
    ref = level0_reference_prior(fd, maps)
    ss = build_level_support(fd, maps, level, sch, ref, support_m=cfg.support_m)
    lp = build_level_population(fd, maps, ss)
    fs = build_fold_scope(_fk(), maps, fd, sch, cfg)
    plans = build_level_plans(fs, level, ss, lp, cfg, model_seed=model_seed)
    return fd, maps, cfg, sch, ss, lp, fs, plans


# ---------------- identity blockers ----------------
def test_all_runner_scientific_hashes_are_full_sha256():
    fd, maps, cfg, sch, ss, lp, fs, plans = _build()
    for h in (fd.data_contract_hash, fd.source_train_population_hash, sch.schedule_hash,
              ss.support_hash, ss.level_support_hash, lp.population_hash, fs.source_audit.audit_scope_hash,
              fs.scope_config_hash, fs.fold_scope_hash, plans.training_plans_hash,
              plans.selection_plans_hash, plans.level_plans_hash):
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_schedule_hash_binds_domain_retention_policy():
    fd, maps = _fold(), _maps()
    a = make_deletion_schedule([DeletionCell("Sd0", "c1")], fd, maps, require_deleted_domain_retained=True)
    b = make_deletion_schedule([DeletionCell("Sd0", "c1")], fd, maps, require_deleted_domain_retained=False)
    assert a.schedule_hash != b.schedule_hash


def test_stable_metadata_ids_must_be_nonempty():
    rows = _rows(); rows[0]["grp"] = ""
    try:
        _fold(rows)
    except ValueError:
        pass
    else:
        raise AssertionError("empty group id must be rejected")


# ---------------- level population ----------------
def test_level_training_and_leakage_population_hashes_match():
    fd, maps, cfg, sch, ss, lp, fs, plans = _build()
    from oaci.train.data import population_signature_hash
    assert population_signature_hash(lp.training_data) == lp.leakage_design.population_hash == lp.population_hash


def test_level_population_is_row_order_invariant():
    rows = _rows()
    a = _build(rows=list(rows))[5]
    b = _build(rows=list(reversed(rows)))[5]
    assert a.population_hash == b.population_hash and a.tensor_hash == b.tensor_hash


def test_task_and_alignment_plan_population_hashes_match_level_population():
    fd, maps, cfg, sch, ss, lp, fs, plans = _build()
    assert plans.stage1_task.population_signature_hash == lp.population_hash
    assert plans.stage2_task.population_signature_hash == lp.population_hash
    assert plans.oaci_alignment.population_signature_hash == lp.population_hash
    assert plans.full_domain_alignment.population_signature_hash == lp.population_hash


# ---------------- audit scope ----------------
def test_audit_scope_uses_local_contiguous_domain_map():
    fd, maps, cfg, sch, ss, lp, fs, plans = _build()
    a = fs.source_audit
    assert sorted(dict(a.domain_to_index_items).values()) == list(range(len(a.domain_ids)))
    assert set(a.domain_ids) == {"Ad0", "Ad1"} and "Sd0" not in dict(a.domain_to_index_items)


def test_audit_prior_is_computed_from_audit_mass():
    fd, maps, cfg, sch, ss, lp, fs, plans = _build()
    assert abs(float(fs.source_audit.reference_prior.sum()) - 1.0) < 1e-9
    assert np.allclose(fs.source_audit.reference_prior, fs.source_audit.support_graph.reference_prior)


def test_audit_design_survives_nonestimable_fold_or_bootstrap():
    # one recording group per audit cell -> cannot form >=2 grouped folds
    fd, maps, cfg = _fold(_rows(audit_recs=1)), _maps(), _cfg()
    a = build_audit_scope(fd, maps, cfg, _fk())
    assert a.status.startswith("nonestimable") and a.fold_plan is None and a.bootstrap_plan is None
    assert a.design is not None and a.support_graph is not None and a.reference_prior is not None


def test_audit_scope_is_model_seed_invariant():
    fd, maps, cfg = _fold(), _maps(), _cfg()
    assert build_audit_scope(fd, maps, cfg, _fk()).audit_scope_hash == build_audit_scope(fd, maps, cfg, _fk()).audit_scope_hash


# ---------------- fold scope ----------------
def test_fold_scope_hash_excludes_model_seed():
    # build_fold_scope takes no model seed; identical inputs -> identical hash (model seed cannot enter)
    fd, maps, cfg, sch = _fold(), _maps(), _cfg(), None
    sch = make_deletion_schedule([DeletionCell("Sd0", "c1")], fd, maps)
    assert build_fold_scope(_fk(), maps, fd, sch, cfg).fold_scope_hash == build_fold_scope(_fk(), maps, fd, sch, cfg).fold_scope_hash


def test_fold_scope_changes_with_support_m_or_probe_config():
    fd, maps = _fold(), _maps()
    sch = make_deletion_schedule([DeletionCell("Sd0", "c1")], fd, maps)
    base = build_fold_scope(_fk(), maps, fd, sch, _cfg()).fold_scope_hash
    assert build_fold_scope(_fk(), maps, fd, sch, _cfg(support_m=3)).fold_scope_hash != base
    assert build_fold_scope(_fk(), maps, fd, sch, _cfg(probe_folds=3)).fold_scope_hash != base


# ---------------- level plans ----------------
def test_training_plan_hash_changes_with_model_seed():
    a = _build(model_seed=0)[7]
    b = _build(model_seed=1)[7]
    assert a.training_plans_hash != b.training_plans_hash
    assert a.selection_plans_hash == b.selection_plans_hash       # selection is model-seed invariant


def test_selection_plan_hash_changes_with_level_or_selection_seed():
    fd = _fold(_rows())
    maps = _maps()
    sch = make_deletion_schedule([DeletionCell("Sd0", "c1")], fd, maps)
    ref = level0_reference_prior(fd, maps)

    def sel(level, seed):
        cfg = _cfg(selection_seed=seed)
        ss = build_level_support(fd, maps, level, sch, ref, support_m=cfg.support_m)
        lp = build_level_population(fd, maps, ss)
        fs = build_fold_scope(_fk(), maps, fd, sch, cfg)
        return build_level_plans(fs, level, ss, lp, cfg, model_seed=0).selection_plans_hash
    base = sel(0, 303)
    assert sel(1, 303) != base and sel(0, 999) != base


def test_selection_design_always_exists_when_plans_are_nonestimable():
    # only ONE recording group per source cell -> selection folds non-estimable, design still present
    fd, maps, cfg, sch, ss, lp, fs, plans = _build(rows=_rows(src_recs=1), schedule_cells=[])
    assert plans.selection_status.startswith("nonestimable")
    assert plans.selection_fold_plan is None and plans.selection_design is not None


def test_inactive_methods_have_no_alignment_plan():
    # a single source domain -> OACI and full-domain both inactive
    fd = _fold(_rows(src_dom=1)); maps = _maps(src=("Sd0",))
    cfg = _cfg(); sch = make_deletion_schedule([], fd, maps); ref = level0_reference_prior(fd, maps)
    ss = build_level_support(fd, maps, 0, sch, ref, support_m=cfg.support_m)
    lp = build_level_population(fd, maps, ss); fs = build_fold_scope(_fk(), maps, fd, sch, cfg)
    plans = build_level_plans(fs, 0, ss, lp, cfg, model_seed=0)
    assert plans.oaci_alignment is None and plans.full_domain_alignment is None


def test_oaci_plan_contains_only_estimable_cells_and_full_domain_includes_low_sample():
    # a low-sample (Sd1,c1) cell: 1 unit < m -> not eligible (OACI excludes), but observed (full-domain includes)
    rows = _rows(src_recs=4)
    rows = [r for r in rows if not (r["dom"] == "Sd1" and r["y"] == 1 and r["sid"] not in
                                    ("source_train_Sd1_r0_c1",))]            # keep 1 unit in (Sd1,c1)
    fd, maps, cfg, sch, ss, lp, fs, plans = _build(rows=rows, schedule_cells=[], support_m=2)
    sg = ss.support_graph
    elig = {(int(d), int(y)) for y in sg.comparable_classes for d in sg.support_of_class[y]}
    d_sd1 = maps.source_domain_to_index["Sd1"]
    sid2dy = {fd.sample_id[i]: (maps.source_domain_to_index[fd.domain_id[i]], int(fd.y[i]))
              for i in ss.source_train_idx.tolist()}
    for lb in plans.oaci_alignment.warmup_batches:
        for mb in lb.microbatches:
            for s in mb.sample_ids:
                assert sid2dy[s] in elig                            # OACI eligible cells only
    assert (d_sd1, 1) not in elig                                   # the low-sample cell is NOT eligible
    full_cells = {sid2dy[s] for lb in plans.full_domain_alignment.warmup_batches
                  for mb in lb.microbatches for s in mb.sample_ids}
    assert (d_sd1, 1) in full_cells                                 # ...but IS in the full-domain plan


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} runner-plans tests")


if __name__ == "__main__":
    _run_all()
