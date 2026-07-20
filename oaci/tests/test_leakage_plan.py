"""A2a-part2: leakage design / fold-plan hash / explicit bootstrap plan / score cache.

Standalone (``python -m oaci.tests.test_leakage_plan``) and pytest-compatible.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from oaci.leakage import (CriticConfig, FrozenFeatures, LeakageBootstrapPlan, LeakageScoreCache,
                          LeakageScoreKey, bootstrap_ucb, critic_config_hash, feat_population_hash,
                          frozen_feature_hash, make_fold_plan, make_fold_plan_from_design,
                          make_leakage_bootstrap_plan, make_leakage_design)
from oaci.leakage.design import population_hash
from oaci.leakage.plan import BootstrapDraw
from oaci.train.synthetic import make_covariate_shift

FAST = CriticConfig(capacities=(0, 8), max_iter=80)


def _bundle(seed=0):
    X, y, d, g, sg = make_covariate_shift(seed=seed)
    sid = tuple(f"r{i}" for i in range(len(y)))
    grp = tuple(str(int(x)) for x in g.tolist())
    feat = FrozenFeatures(Z=X, y=y, d=d, group=grp, sample_mass=np.ones(len(y)), sample_id=sid)
    design = make_leakage_design(sid, y, d, grp, np.ones(len(y)), sg)
    fold = make_fold_plan_from_design(design, sg, n_folds=4, seed=0)    # population_hash == design's
    return X, y, d, g, sg, feat, design, fold


# ---------------- design ----------------
def test_leakage_design_hash_is_row_order_invariant():
    X, y, d, g, sg, feat, design, fold = _bundle()
    n = len(y); perm = np.random.RandomState(0).permutation(n)
    sid = [f"r{i}" for i in range(n)]; grp = [str(int(x)) for x in g.tolist()]
    d2 = make_leakage_design(tuple(sid[i] for i in perm), y[perm], d[perm],
                             tuple(grp[i] for i in perm), np.ones(n)[perm], sg)
    assert d2.population_hash == design.population_hash


def test_leakage_design_hash_changes_with_mass_or_group_mapping():
    X, y, d, g, sg, feat, design, fold = _bundle()
    n = len(y); sid = tuple(f"r{i}" for i in range(n)); grp = [str(int(x)) for x in g.tolist()]
    mass = np.ones(n); mass[0] = 2.0
    sg2 = make_covariate_shift(seed=0)[4]
    sg2.cell_mass[d[0], y[0]] += 1.0                      # keep the design/sg cell-mass check consistent
    h_mass = make_leakage_design(sid, y, d, tuple(grp), mass, sg2).population_hash
    assert h_mass != design.population_hash               # mass change moves the hash
    j = next(i for i in range(n) if int(d[i]) == int(d[0]) and grp[i] != grp[0])   # same domain, other group
    grp2 = grp[:]; grp2[0] = grp[j]                       # remap row 0 to a different group (same domain)
    h_grp = make_leakage_design(sid, y, d, tuple(grp2), np.ones(n), sg).population_hash
    assert h_grp != design.population_hash


def test_design_rejects_duplicate_sample_ids():
    X, y, d, g, sg, feat, design, fold = _bundle()
    sid = ["r0"] * len(y); grp = [str(int(x)) for x in g.tolist()]
    try:
        make_leakage_design(tuple(sid), y, d, tuple(grp), np.ones(len(y)), sg)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate sample_id must be rejected")


def test_design_rejects_group_spanning_domains():
    X, y, d, g, sg, feat, design, fold = _bundle()
    grp = [str(int(x)) for x in g.tolist()]
    i0 = 0; i1 = int(np.where(d != d[0])[0][0])           # a row in a different domain
    grp[i1] = grp[i0]                                      # same group string across two domains
    try:
        make_leakage_design(tuple(f"r{i}" for i in range(len(y))), y, d, tuple(grp), np.ones(len(y)), sg)
    except ValueError:
        pass
    else:
        raise AssertionError("a group spanning two domains must be rejected")


def test_design_cell_mass_must_match_support_graph():
    X, y, d, g, sg, feat, design, fold = _bundle()
    mass = np.ones(len(y)); mass[0] = 3.0                  # now design cell mass != sg.cell_mass
    try:
        make_leakage_design(tuple(f"r{i}" for i in range(len(y))), y, d,
                            tuple(str(int(x)) for x in g.tolist()), mass, sg)
    except ValueError:
        pass
    else:
        raise AssertionError("design cell mass must match SupportGraph.cell_mass")


def test_support_hash_binds_counts_mass_prior_and_threshold():
    from oaci.support_graph import build_support_graph, counts_from_labels, empirical_class_prior
    X, y, d, g, sg = make_covariate_shift(seed=0)
    counts = counts_from_labels(d, y, n_domains=2, n_classes=2)
    base = build_support_graph(counts, m=20, reference_prior=empirical_class_prior(counts)).support_hash()
    assert build_support_graph(counts, m=10, reference_prior=empirical_class_prior(counts)).support_hash() != base
    cm = counts.astype(float).copy(); cm[0, 0] += 5
    assert build_support_graph(counts, m=20, cell_mass=cm,
                               reference_prior=empirical_class_prior(counts)).support_hash() != base


# ---------------- fold plan hash ----------------
def test_fold_plan_hash_is_order_invariant():
    X, y, d, g, sg, feat, design, fold = _bundle()
    perm = np.random.RandomState(1).permutation(len(y))
    sid = tuple(f"r{i}" for i in range(len(y)))
    feat2 = FrozenFeatures(Z=X[perm], y=y[perm], d=d[perm],
                           group=tuple(str(int(g[i])) for i in perm), sample_mass=np.ones(len(y)),
                           sample_id=tuple(sid[i] for i in perm))   # SAME rows, permuted order
    fold2 = make_fold_plan(feat2, sg, n_folds=4, seed=0)
    assert fold2.plan_hash == fold.plan_hash               # group->fold map independent of row order


def test_fold_plan_hash_changes_with_assignment():
    X, y, d, g, sg, feat, design, fold = _bundle()
    from oaci.leakage.crossfit import fold_plan_hash
    flipped = dict(fold.fold_of_group)
    gk = sorted(flipped)[0]; flipped[gk] = (flipped[gk] + 1) % fold.n_folds
    h = fold_plan_hash(fold.population_hash, fold.support_hash, fold.n_folds_requested,
                       fold.n_folds, flipped, fold.domain_of_group)
    assert h != fold.plan_hash


# ---------------- bootstrap plan ----------------
def test_bootstrap_plan_is_independent_of_Z_and_method():
    X, y, d, g, sg, feat, design, fold = _bundle()
    p1 = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=6, seed=0)
    p2 = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=6, seed=0)
    assert p1.plan_hash == p2.plan_hash                    # no Z, no method in the inputs
    assert not hasattr(design, "Z")


def test_bootstrap_draws_are_domain_stratified():
    X, y, d, g, sg, feat, design, fold = _bundle()
    plan = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=6, seed=0)
    dom_groups = defaultdict(list)
    for gg, dom in fold.domain_of_group.items():
        dom_groups[dom].append(str(int(gg)))
    for draw in plan.candidate_draws:
        mult = dict(draw.group_multiplicities)
        for dom, gs in dom_groups.items():
            assert sum(mult[gg] for gg in gs) == len(gs)   # each domain keeps its group count


def test_invalid_draws_are_fixed_at_plan_construction():
    X, y, d, g, sg, feat, design, fold = _bundle()
    plan = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=6, seed=0,
                                       max_candidate_multiplier=8)
    assert len(plan.accepted_candidate_ids) == 6
    assert set(plan.accepted_candidate_ids) <= {dr.candidate_id for dr in plan.candidate_draws}
    assert 0.0 <= plan.invalid_draw_rate < 1.0
    try:                                                   # impossible request -> fail at construction
        make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=10_000, seed=0,
                                    max_candidate_multiplier=1, max_invalid_draw_rate=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("an unsatisfiable request must fail at plan construction")


def test_bootstrap_plan_rejects_population_support_or_fold_mismatch():
    X, y, d, g, sg, feat, design, fold = _bundle()
    other_sg = make_covariate_shift(seed=0)[4]
    other_sg.cell_mass = other_sg.cell_mass.copy(); other_sg.cell_mass[0, 0] += 5.0   # different support_hash
    try:
        make_leakage_bootstrap_plan(design, other_sg, fold, alpha=0.1, requested_replicates=4, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("a support-hash mismatch must be rejected")


def test_bootstrap_plan_hash_binds_candidate_and_accepted_sequences():
    X, y, d, g, sg, feat, design, fold = _bundle()
    a = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=6, seed=0)
    b = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=6, seed=1)
    assert a.plan_hash != b.plan_hash                      # a different draw sequence -> different hash


def test_bootstrap_ucb_uses_no_rng_when_plan_is_supplied():
    X, y, d, g, sg, feat, design, fold = _bundle()
    plan = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=6, seed=0)
    st = np.random.get_state()
    r1 = bootstrap_ucb(feat, sg, fold, FAST, bootstrap_plan=plan)
    r2 = bootstrap_ucb(feat, sg, fold, FAST, bootstrap_plan=plan)
    assert np.array_equal(np.asarray(np.random.get_state()[1]), np.asarray(st[1]))   # global RNG untouched
    assert np.allclose(r1["replicates"], r2["replicates"])                            # deterministic replay
    assert r1["bootstrap_plan_hash"] == plan.plan_hash


def test_bootstrap_ucb_never_redraws_with_explicit_plan():
    X, y, d, g, sg, feat, design, fold = _bundle()
    plan = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=6, seed=0)
    r = bootstrap_ucb(feat, sg, fold, FAST, bootstrap_plan=plan)
    assert r["n_bootstrap"] == len(plan.accepted_candidate_ids)
    assert tuple(r["accepted_candidate_ids"]) == plan.accepted_candidate_ids


def test_same_plan_reproduces_replicate_capacity_sequence():
    X, y, d, g, sg, feat, design, fold = _bundle()
    plan = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=6, seed=0)
    r1 = bootstrap_ucb(feat, sg, fold, FAST, bootstrap_plan=plan)
    r2 = bootstrap_ucb(feat, sg, fold, FAST, bootstrap_plan=plan)
    assert r1["replicate_capacities"] == r2["replicate_capacities"]


def test_replicate_failure_reports_candidate_id():
    X, y, d, g, sg, feat, design, fold = _bundle()
    fold0 = [gg for gg, f in fold.fold_of_group.items() if f == 0]
    mult = {str(int(gg)): 0 for gg in fold.fold_of_group}
    for gg in fold0:
        mult[str(int(gg))] = 1                             # all mass in a single fold -> estimate fails
    bad = LeakageBootstrapPlan(population_hash=design.population_hash, support_hash=design.support_hash,
                               fold_plan_hash=fold.plan_hash, alpha=0.1, requested_replicates=1,
                               candidate_draws=(BootstrapDraw(77, tuple(sorted(mult.items()))),),
                               accepted_candidate_ids=(77,), invalid_draw_rate=0.0, plan_hash="x")
    try:
        bootstrap_ucb(feat, sg, fold, FAST, bootstrap_plan=bad)
    except ValueError as e:
        assert "77" in str(e)
    else:
        raise AssertionError("an accepted draw that fails must raise, naming the candidate id")


# ---------------- score cache ----------------
def test_cache_key_binds_actual_feature_bytes():
    X = np.random.RandomState(0).standard_normal((10, 4))
    assert frozen_feature_hash(X) != frozen_feature_hash(X * 2.0)
    k = dict(model_hash="m", population_hash="p", support_hash="s", fold_plan_hash="f",
             bootstrap_plan_hash="b", critic_config_hash="c")
    assert LeakageScoreKey(frozen_feature_hash=frozen_feature_hash(X), **k) != \
        LeakageScoreKey(frozen_feature_hash=frozen_feature_hash(X + 1), **k)


def test_cache_key_binds_critic_config():
    assert critic_config_hash(CriticConfig(capacities=(0, 8))) != critic_config_hash(CriticConfig(capacities=(0, 64)))
    assert critic_config_hash(CriticConfig(l2_C=1.0)) != critic_config_hash(CriticConfig(l2_C=2.0))


def test_erm_leakage_score_is_computed_once():
    cache = LeakageScoreCache()
    key = LeakageScoreKey("m", "z", "p", "s", "f", "b", "c")
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return {"extractable_LQ_ov": 0.5, "replicates": np.zeros(3)}

    for _ in range(3):                                     # three Stage-2 selectors read the ERM score
        cache.get_or_compute(key, fn)
    assert calls["n"] == 1 and cache.compute_count(key) == 1


def test_cached_result_cannot_be_mutated():
    cache = LeakageScoreCache()
    key = LeakageScoreKey("m", "z", "p", "s", "f", "b", "c")
    r = cache.get_or_compute(key, lambda: {"replicates": np.zeros(3)})
    try:
        r["replicates"][0] = 9.0
    except ValueError:
        pass
    else:
        raise AssertionError("cached arrays must be read-only")


# ---------------- residual 1: unified string-id identity ----------------
def test_frozen_features_preserves_string_group_ids():
    X, y, d, g, sg = make_covariate_shift(seed=0)
    grp = tuple(f"rec-{int(x)}" for x in g.tolist())
    feat = FrozenFeatures(Z=X, y=y, d=d, group=grp, sample_id=tuple(f"s{i}" for i in range(len(y))))
    assert feat.group.dtype == object and feat.group[0] == grp[0]        # stored as the STRING, not int
    assert feat.sample_id[0] == "s0"


def test_fold_and_bootstrap_plans_share_exact_population_hash():
    X, y, d, g, sg, feat, design, fold = _bundle()
    plan = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=4, seed=0)
    assert design.population_hash == fold.population_hash == plan.population_hash
    assert feat_population_hash(feat) == design.population_hash          # FrozenFeatures matches too


def test_bootstrap_ucb_rejects_sample_id_population_mismatch():
    X, y, d, g, sg, feat, design, fold = _bundle()
    plan = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=4, seed=0)
    feat2 = FrozenFeatures(Z=X, y=y, d=d, group=tuple(str(int(x)) for x in g.tolist()),
                           sample_mass=np.ones(len(y)), sample_id=tuple(f"X{i}" for i in range(len(y))))
    try:
        bootstrap_ucb(feat2, sg, fold, FAST, bootstrap_plan=plan)        # different sample_id -> different pop
    except ValueError:
        pass
    else:
        raise AssertionError("a sample_id population mismatch must be rejected")


def test_bootstrap_replay_never_casts_group_ids_to_int():
    X, y, d, g, sg = make_covariate_shift(seed=0)
    sid = tuple(f"r{i}" for i in range(len(y)))
    grp = tuple(f"rec-{int(x)}" for x in g.tolist())                     # NON-integer group ids
    feat = FrozenFeatures(Z=X, y=y, d=d, group=grp, sample_mass=np.ones(len(y)), sample_id=sid)
    design = make_leakage_design(sid, y, d, grp, np.ones(len(y)), sg)
    fold = make_fold_plan_from_design(design, sg, n_folds=4, seed=0)
    plan = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=4, seed=0)
    r = bootstrap_ucb(feat, sg, fold, FAST, bootstrap_plan=plan)         # would raise on int("rec-0")
    assert r["n_bootstrap"] == 4


def test_three_stage2_selectors_hit_same_erm_cache_entry():
    cache = LeakageScoreCache()
    key = LeakageScoreKey("m", "z", "p", "s", "f", "b", "c")
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return {"extractable_LQ_ov": 0.5, "nested": {"reps": np.zeros(3)}}

    for _ in range(3):                                                   # three Stage-2 selectors
        cache.get_or_compute(key, fn)
    assert cache.request_count(key) == 3 and cache.compute_count(key) == 1 and cache.hit_count(key) == 2
    r = cache.get_or_compute(key, fn)
    try:
        r["nested"]["reps"][0] = 1.0                                     # deep-frozen
    except ValueError:
        pass
    else:
        raise AssertionError("nested cached arrays must be read-only")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} leakage-plan tests")


if __name__ == "__main__":
    _run_all()
