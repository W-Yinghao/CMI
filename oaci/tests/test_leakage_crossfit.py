"""Strict grouped cross-fit guarantees: recording memorisation does NOT register as leakage
under a grouped probe; fold feasibility reduces then fails explicitly; each recording lands in
exactly one fold; duplicated bootstrap groups inherit one fold.

Standalone (``python -m oaci.tests.test_leakage_crossfit``) and pytest-compatible.
"""
from __future__ import annotations

import numpy as np

from oaci.leakage.critic import CriticConfig
from oaci.leakage.crossfit import FrozenFeatures, make_fold_plan, oof_nll_by_class
from oaci.leakage.estimate import estimate_extractable_leakage
from oaci.leakage.synthetic import make_group_memorization
from oaci.leakage.ucb import _group_to_rows, _rebuild, within_domain_group_bootstrap
from oaci.support_graph import build_support_graph, counts_from_labels, empirical_class_prior

FAST = CriticConfig(capacities=(0, 16))


def _arrays(recs_per_domain, per_cell, n_domains=2, n_classes=2):
    y, d, g = [], [], []
    gid = 0
    for dom in range(n_domains):
        for _ in range(recs_per_domain):
            for c in range(n_classes):
                y += [c] * per_cell; d += [dom] * per_cell; g += [gid] * per_cell
            gid += 1
    return np.array(y), np.array(d), np.array(g)


def _single_label_arrays(groups_per_cell=2, per_group=25, n_domains=2, n_classes=2):
    """Clinical-style layout: each recording carries ONE class (one (d,y) cell)."""
    y, d, g = [], [], []
    gid = 0
    for dom in range(n_domains):
        for c in range(n_classes):
            for _ in range(groups_per_cell):
                y += [c] * per_group; d += [dom] * per_group; g += [gid] * per_group
                gid += 1
    return np.array(y), np.array(d), np.array(g)


def test_cell_aware_grouped_folds():
    # Each recording is a single (d,y) cell. Cell-aware assignment must put EVERY eligible cell
    # in EVERY fold (domain-only round-robin could strand a whole cell in one fold).
    y, d, g = _single_label_arrays(groups_per_cell=2, per_group=25)
    counts = counts_from_labels(d, y, n_domains=2, n_classes=2)
    sg = build_support_graph(counts, m=20, reference_prior=empirical_class_prior(counts))
    feat = FrozenFeatures(np.zeros((y.size, 2)), y, d, g)
    for seed in range(4):
        plan = make_fold_plan(feat, sg, n_folds=2, seed=seed)
        # collect folds occupied by each eligible cell's groups
        cell_folds = {}
        for gid in np.unique(g):
            dom = int(d[g == gid][0]); cls = int(y[g == gid][0])
            cell_folds.setdefault((dom, cls), set()).add(plan.fold_of_group[str(int(gid))])
        for y_ in sg.comparable_classes:
            for d_ in sg.support_of_class[y_]:
                assert cell_folds[(d_, y_)] == {0, 1}, (seed, (d_, y_), cell_folds[(d_, y_)])


def test_group_memorization_does_not_register_as_leakage():
    # Z encodes recording identity only (no shared domain signal). A grouped probe must read ~0
    # because held-out recordings' identities are unseen; a sample-level split would over-flag.
    feat, sg = make_group_memorization(seed=1, recs_per_domain=4, per_cell=25, dim=16)
    plan = make_fold_plan(feat, sg, n_folds=2, seed=0)
    res = estimate_extractable_leakage(feat, sg, plan, FAST)
    assert res["L_abs"] < 0.15, res["L_abs"]


def test_fold_count_reduced_then_recorded():
    # each cell has exactly 2 recording groups -> requesting 4 folds reduces to 2 (and records it)
    y, d, g = _arrays(recs_per_domain=2, per_cell=25)
    counts = counts_from_labels(d, y, n_domains=2, n_classes=2)
    sg = build_support_graph(counts, m=20, reference_prior=empirical_class_prior(counts))
    plan = make_fold_plan(FrozenFeatures(np.zeros((y.size, 2)), y, d, g), sg, n_folds=4, seed=0)
    assert plan.n_folds == 2 and plan.reduced
    assert plan.notes and "reduced" in plan.notes[0]


def test_fold_infeasible_fails_explicitly():
    # a cell with only ONE recording group cannot form 2 grouped folds -> explicit error
    y = np.array([0] * 25 + [1] * 25 + [0] * 25 + [1] * 25)
    d = np.array([0] * 50 + [1] * 50)
    g = np.array([0] * 50 + [1] * 50)                      # one group per domain
    counts = counts_from_labels(d, y, n_domains=2, n_classes=2)
    sg = build_support_graph(counts, m=20, reference_prior=empirical_class_prior(counts))
    try:
        make_fold_plan(FrozenFeatures(np.zeros((y.size, 2)), y, d, g), sg, n_folds=2, seed=0)
    except ValueError as e:
        assert "grouped folds" in str(e) and "sample-level" in str(e)
    else:
        raise AssertionError("expected ValueError for an infeasible grouped split")


def test_each_recording_in_exactly_one_fold():
    y, d, g = _arrays(recs_per_domain=4, per_cell=10)
    counts = counts_from_labels(d, y, n_domains=2, n_classes=2)
    sg = build_support_graph(counts, m=8, reference_prior=empirical_class_prior(counts))
    plan = make_fold_plan(FrozenFeatures(np.zeros((y.size, 2)), y, d, g), sg, n_folds=2, seed=0)
    assert set(plan.fold_of_group) == {str(int(x)) for x in np.unique(g)}   # every group assigned (str keys)
    assert all(0 <= f < plan.n_folds for f in plan.fold_of_group.values())


def test_duplicated_bootstrap_groups_share_one_fold():
    y, d, g = _arrays(recs_per_domain=4, per_cell=10)
    counts = counts_from_labels(d, y, n_domains=2, n_classes=2)
    sg = build_support_graph(counts, m=8, reference_prior=empirical_class_prior(counts))
    feat = FrozenFeatures(np.zeros((y.size, 2)), y, d, g)
    plan = make_fold_plan(feat, sg, n_folds=2, seed=0)
    rng = np.random.default_rng(0)
    resampled = within_domain_group_bootstrap(plan, rng)
    feat_b = _rebuild(feat, _group_to_rows(feat), resampled)
    # every row of a given (original) group id maps to the same fold
    for gid in np.unique(feat_b.group):
        folds = {plan.fold_of_group[gg] for gg in feat_b.group[feat_b.group == gid]}   # str group keys
        assert len(folds) == 1
    # within-domain resampling holds each domain's group count fixed
    for dom in set(plan.domain_of_group.values()):
        n_dom_groups = sum(1 for v in plan.domain_of_group.values() if v == dom)
        assert sum(1 for gid in resampled if plan.domain_of_group[gid] == dom) == n_dom_groups


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} leakage-crossfit tests")


if __name__ == "__main__":
    _run_all()
