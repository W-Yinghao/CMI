"""A2b residual 2: string-preserving mass-unit samplers + sampler-driven plan materialisers.

Standalone (``python -m oaci.tests.test_plan_sampler``) and pytest-compatible.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from oaci.data.plan_materialize import (materialize_full_domain_alignment_plan,
                                        materialize_oaci_alignment_plan, materialize_stage1_task_plan,
                                        materialize_stage2_task_plan)
from oaci.data.plan_sampler import (MassUnitTaskSampler, ObservedCellSampler,
                                    RareEligibleCellSampler, UnitIndex)
from oaci.support_graph import build_support_graph, empirical_class_prior


def _setup(per_cell=4, n_dom=3, nc=2, m=4, drop=None, low=None):
    """Each (d,y) cell has `per_cell` single-row mass units (mass 1); `drop`=(d,y) removes a cell;
    `low`=(d,y) gives a cell only 2 units (observed but below the m-gate)."""
    sid, y, d, grp, unit, mass = [], [], [], [], [], []
    uid = 0
    counts = np.zeros((n_dom, nc), dtype=int)
    for dom in range(n_dom):
        for c in range(nc):
            if drop is not None and (dom, c) == drop:
                continue
            k = 2 if (low is not None and (dom, c) == low) else per_cell
            for u in range(k):
                sid.append(f"s{dom}_{c}_{u}"); y.append(c); d.append(dom)
                grp.append(f"rec{dom}-{u}"); unit.append(f"u{uid}"); mass.append(1.0); uid += 1
            counts[dom, c] = k
    idx = UnitIndex(tuple(sid), np.array(y), np.array(d), tuple(grp), tuple(unit), np.array(mass, float))
    sg = build_support_graph(counts, m=m, cell_mass=counts.astype(float),
                             reference_prior=empirical_class_prior(counts))
    sid2dy = {sid[i]: (d[i], y[i]) for i in range(len(sid))}
    return idx, sg, sid2dy


# -------------------------------- samplers --------------------------------
def test_task_sampler_works_without_comparable_classes():
    idx, sg, _ = _setup(n_dom=1)                       # single domain -> no comparable class
    assert sg.comparable_classes == []
    sids, ws, _ = MassUnitTaskSampler(idx, 6, base_seed=0).step(0)
    assert len(sids) == 6                              # task stream still produced


def test_sampler_preserves_string_group_and_unit_ids():
    idx, sg, _ = _setup()
    assert all(isinstance(g, str) for g in idx.group) and all(isinstance(u, str) for u in idx.units)
    assert idx.unit_cell[idx.units[0]][0] == 0          # mapping intact (no int(group) round-trip)


def test_sampler_plan_is_invariant_to_input_row_order():
    idx, sg, _ = _setup()
    perm = np.random.RandomState(0).permutation(len(idx.sample_id))
    idx2 = UnitIndex(tuple(idx.sample_id[i] for i in perm), idx.y[perm], idx.d[perm],
                     tuple(idx.group[i] for i in perm), tuple(idx.unit[i] for i in perm), idx.b[perm])
    a = MassUnitTaskSampler(idx, 6, base_seed=3).step(1)
    b = MassUnitTaskSampler(idx2, 6, base_seed=3).step(1)
    assert a == b                                       # same draw regardless of input row order


def test_task_batch_size_is_exact_and_covers_every_class():
    idx, sg, sid2dy = _setup()
    sids, ws, _ = MassUnitTaskSampler(idx, 7, base_seed=0).step(0)
    assert len(sids) == 7
    assert {sid2dy[s][1] for s in sids} == set(idx.present_classes())   # every class present (m_y>=1)


def test_task_weights_restore_class_mass():
    idx, sg, sid2dy = _setup()
    sids, ws, _ = MassUnitTaskSampler(idx, 6, base_seed=0).step(0)
    by_c = defaultdict(float)
    for s, w in zip(sids, ws):
        by_c[sid2dy[s][1]] += w
    for c in idx.present_classes():
        assert abs(by_c[c] - len(idx.class_units[c])) < 1e-9   # weighted class mass == U_y


def test_oaci_weights_restore_eligible_cell_mass():
    idx, sg, sid2dy = _setup()
    sampler = RareEligibleCellSampler(idx, sg, per_cell=3, base_seed=0)
    sids, ws, _ = sampler.step(0)
    by_cell = defaultdict(float)
    for s, w in zip(sids, ws):
        by_cell[sid2dy[s]] += w
    for cell in sampler.cells:
        assert abs(by_cell[cell] - len(idx.cell_units[cell])) < 1e-9


def test_full_domain_weights_restore_observed_cell_mass():
    idx, sg, sid2dy = _setup(low=(2, 1))
    sampler = ObservedCellSampler(idx, per_cell=2, base_seed=0)
    sids, ws, _ = sampler.step(0)
    by_cell = defaultdict(float)
    for s, w in zip(sids, ws):
        by_cell[sid2dy[s]] += w
    for cell in sampler.cells:
        assert abs(by_cell[cell] - len(idx.cell_units[cell])) < 1e-9


def test_full_domain_capacity_uses_observed_cell_count():
    idx, sg, _ = _setup(low=(2, 1))
    sampler = ObservedCellSampler(idx, per_cell=2, base_seed=0)
    assert len(sampler.cells) == 6                       # 3 domains x 2 classes all observed
    assert (2, 1) in sampler.cells                       # the low-sample cell IS observed


def test_missing_cells_never_enter_full_domain_plan():
    idx, sg, sid2dy = _setup(drop=(2, 1))
    sampler = ObservedCellSampler(idx, per_cell=2, base_seed=0)
    assert (2, 1) not in sampler.cells
    sids, _, _ = sampler.step(0)
    assert all(sid2dy[s] != (2, 1) for s in sids)


def test_low_sample_observed_cells_do_enter_full_domain_plan():
    idx, sg, _ = _setup(low=(2, 1), m=4)
    assert (2, 1) not in [(int(d), int(y)) for y in sg.comparable_classes for d in sg.support_of_class[y]]
    assert (2, 1) in ObservedCellSampler(idx, per_cell=2, base_seed=0).cells   # observed but not eligible


# -------------------------------- materialisers --------------------------------
POP = "pop-sig-fixed"


def test_stage1_and_stage2_task_plans_use_independent_streams():
    idx, sg, _ = _setup()
    s1 = materialize_stage1_task_plan(idx, POP, 2, 2, 6, base_seed=0)
    s2 = materialize_stage2_task_plan(idx, POP, 2, 2, 6, base_seed=0)
    assert s1.plan_hash != s2.plan_hash and s1.role == "stage1_task" and s2.role == "stage2_task"


def test_all_stage2_methods_share_task_plan_object_and_hash():
    idx, sg, _ = _setup()
    a = materialize_stage2_task_plan(idx, POP, 2, 2, 6, base_seed=0)
    b = materialize_stage2_task_plan(idx, POP, 2, 2, 6, base_seed=0)
    assert a.plan_hash == b.plan_hash                    # deterministic -> the three methods share it


def test_global_lpc_and_uniform_share_alignment_plan_object_and_hash():
    idx, sg, _ = _setup()
    a = materialize_full_domain_alignment_plan(idx, POP, 1, 2, 1, per_cell=2, micro_size=64, base_seed=0)
    b = materialize_full_domain_alignment_plan(idx, POP, 1, 2, 1, per_cell=2, micro_size=64, base_seed=0)
    assert a.plan_hash == b.plan_hash and a.sampling_design_hash == b.sampling_design_hash


def test_oaci_plan_contains_only_estimable_cells():
    idx, sg, sid2dy = _setup(low=(2, 1))
    plan = materialize_oaci_alignment_plan(idx, sg, POP, 1, 2, 1, per_cell=2, micro_size=64, base_seed=0)
    elig = {(int(d), int(y)) for y in sg.comparable_classes for d in sg.support_of_class[y]}
    for lb in plan.warmup_batches:
        for mb in lb.microbatches:
            for s in mb.sample_ids:
                assert sid2dy[s] in elig                # only eligible comparable cells
    assert plan.role == "oaci_alignment"


def test_alignment_plan_hash_binds_role_support_and_sampler_config():
    idx, sg, _ = _setup()
    base = materialize_oaci_alignment_plan(idx, sg, POP, 1, 2, 1, per_cell=2, micro_size=64, base_seed=0)
    diff_k = materialize_oaci_alignment_plan(idx, sg, POP, 1, 2, 1, per_cell=3, micro_size=64, base_seed=0)
    full = materialize_full_domain_alignment_plan(idx, POP, 1, 2, 1, per_cell=2, micro_size=64, base_seed=0)
    assert diff_k.sampling_design_hash != base.sampling_design_hash      # per_cell change
    assert full.sampling_design_hash != base.sampling_design_hash        # role/cell-set change


def test_method_order_does_not_advance_any_shared_sampler():
    idx, sg, _ = _setup()
    # materialising the OACI plan first, or the full-domain plan first, leaves each unchanged
    o1 = materialize_oaci_alignment_plan(idx, sg, POP, 1, 2, 1, per_cell=2, micro_size=64, base_seed=0)
    f1 = materialize_full_domain_alignment_plan(idx, POP, 1, 2, 1, per_cell=2, micro_size=64, base_seed=0)
    f2 = materialize_full_domain_alignment_plan(idx, POP, 1, 2, 1, per_cell=2, micro_size=64, base_seed=0)
    o2 = materialize_oaci_alignment_plan(idx, sg, POP, 1, 2, 1, per_cell=2, micro_size=64, base_seed=0)
    assert o1.plan_hash == o2.plan_hash and f1.plan_hash == f2.plan_hash


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} plan-sampler tests")


if __name__ == "__main__":
    _run_all()
