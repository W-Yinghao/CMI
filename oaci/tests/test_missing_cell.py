"""Controlled missing-cell harness — the shared artifacts of the stress test. These tests
pin the properties every downstream component relies on: monotone deletion, a FIXED reference
prior across the sweep, non-increasing identifiable mass, deterministic strategies, the
fragmentation readout, and exact sample masking.

Standalone (``python -m oaci.tests.test_missing_cell``) and pytest-compatible.
"""
from __future__ import annotations

import numpy as np

from oaci.data.missing_cell import (
    CellMask,
    apply_to_samples,
    make_group_ids,
    make_schedule,
)
from oaci.support_graph import empirical_class_prior


def _bridge_counts():
    # d2 connects to {d0,d1} ONLY via class 2; class-2 cells (1,2),(2,2) are the bridges.
    return np.array(
        [
            # y0   y1   y2
            [100, 100, 0],
            [100, 100, 200],
            [0, 0, 200],
        ]
    )


def test_deletions_are_monotone():
    sched = make_schedule(_bridge_counts(), m=10, strategy="bridge_first")
    prev_deleted = set()
    for s in sched.steps:
        cur = set(s.mask.deleted_cells())
        assert prev_deleted.issubset(cur)           # nothing un-deleted
        assert s.mask.n_deleted == s.level          # one cell per level
        prev_deleted = cur


def test_reference_prior_is_fixed_across_sweep():
    base = _bridge_counts()
    sched = make_schedule(base, m=10, strategy="rare_first")
    expected = empirical_class_prior(base)
    assert np.allclose(sched.reference_prior, expected)
    for s in sched.steps:
        assert np.allclose(s.support_graph.reference_prior, expected)  # every step shares it


def test_identifiable_mass_is_non_increasing():
    sched = make_schedule(_bridge_counts(), m=10, strategy="bridge_first")
    vals = [s.identifiable_mass_fraction for s in sched.steps]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-12


def test_rare_first_deletes_smallest_eligible_cell_first():
    counts = np.array([[100, 30], [40, 100]])  # smallest eligible is (0,1)=30
    sched = make_schedule(counts, m=10, strategy="rare_first", n_steps=1)
    assert sched.steps[1].deleted_cell == (0, 1)


def test_bridge_first_fragments_no_later_than_rare_first():
    counts = _bridge_counts()
    bridge = make_schedule(counts, m=10, strategy="bridge_first")
    rare = make_schedule(counts, m=10, strategy="rare_first")
    bf = bridge.first_fragmentation_level()
    rf = rare.first_fragmentation_level()
    assert bf == 1                                  # one targeted bridge deletion disconnects d2
    assert rf is not None and bf <= rf              # rare ordering takes at least as long


def test_base_step_is_unfragmented_and_full_support():
    sched = make_schedule(_bridge_counts(), m=10, strategy="bridge_first")
    base = sched.steps[0]
    assert base.level == 0 and base.deleted_cell is None
    assert base.mask.n_deleted == 0
    assert base.n_components == 1                   # connected at base


def test_apply_to_samples_drops_exactly_deleted_cells():
    # build per-sample labels consistent with a small count table
    d = np.array([0, 0, 0, 1, 1, 2])
    y = np.array([0, 0, 1, 0, 1, 1])
    keep = np.ones((3, 2), dtype=bool)
    keep[0, 1] = False                              # delete cell (0,1)
    mask = CellMask(keep)
    s = apply_to_samples(mask, d, y)
    # only the sample at index 2 (d=0,y=1) is dropped
    assert s.tolist() == [True, True, False, True, True, True]


def test_make_group_ids_combines_factors():
    subj = np.array([0, 0, 1, 1, 1])
    sess = np.array([0, 1, 0, 0, 1])
    g = make_group_ids(subj, sess)
    # distinct (subj,sess) tuples -> distinct ids; equal tuples -> equal ids
    assert g[2] == g[3]                             # (1,0) == (1,0)
    assert g[0] != g[1] != g[4]
    assert set(g.tolist()) == set(range(len(set(zip(subj.tolist(), sess.tolist())))))


def test_n_steps_cap_and_stop_when_no_comparable():
    sched = make_schedule(_bridge_counts(), m=10, strategy="bridge_first", n_steps=2)
    assert len(sched) == 3                          # base + 2 deletions
    # without a cap, the sweep stops once no comparable class remains
    full = make_schedule(_bridge_counts(), m=10, strategy="rare_first")
    assert full.steps[-1].n_comparable_classes == 0 or full.steps[-1].mask.n_deleted == int(
        (_bridge_counts() >= 10).sum()
    )


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} missing-cell harness tests")


if __name__ == "__main__":
    _run_all()
