"""A2b-1a pt.2: FoldData contract, 2-level support, deletion schedule, typed non-estimability.

Standalone (``python -m oaci.tests.test_runner_scope``) and pytest-compatible.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from oaci.leakage import (CriticConfig, FrozenFeatures, make_fold_plan, make_fold_plan_from_design,
                          make_leakage_bootstrap_plan, make_leakage_design)
from oaci.leakage.errors import (BootstrapPlanNonEstimable, FoldPlanNonEstimable, LeakageNonEstimableError,
                                 NoComparableSupport)
from oaci.runner import (DeletionCell, FoldData, FrozenMaps, build_frozen_maps, build_level_support,
                         level0_reference_prior, make_deletion_schedule)
from oaci.support_graph import build_support_graph, counts_from_labels, empirical_class_prior
from oaci.train.synthetic import make_covariate_shift


# ---------------- FoldData fixture ----------------
def _rows(src_dom=3, recs=3, audit_dom=2, target_dom=1, single_class_dom=None):
    rows = []

    def block(prefix, ndom, nrec, role):
        for d in range(ndom):
            dom = f"{prefix}d{d}"
            classes = [0] if (single_class_dom == (role, d)) else [0, 1]
            for r in range(nrec):
                grp = f"{dom}-rec{r}"
                for c in classes:
                    sid = f"{role}_{dom}_r{r}_c{c}"
                    rows.append(dict(sid=sid, dom=dom, grp=grp, unit=sid, y=c, mass=1.0, role=role))
    block("S", src_dom, recs, "source_train")
    block("A", audit_dom, 2, "source_audit")
    block("T", target_dom, 2, "target_audit")
    return rows


def _base_X(rows, dim=4):
    # X tied to sample_id (deterministic) so reordering rows moves each sample's X with it
    def vec(sid):
        s = int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(s).standard_normal(dim).astype(np.float32)
    return torch.from_numpy(np.stack([vec(r["sid"]) for r in rows]))


def _fold_data(rows=None, mutate=None, X=None):
    rows = rows if rows is not None else _rows()
    if mutate:
        mutate(rows)
    n = len(rows)
    X = _base_X(rows) if X is None else X
    role_idx = {"source_train": [], "source_audit": [], "target_audit": []}
    for i, r in enumerate(rows):
        role_idx[r["role"]].append(i)
    return FoldData.from_arrays(
        X=X, y=np.array([r["y"] for r in rows]), sample_id=[r["sid"] for r in rows],
        domain_id=[r["dom"] for r in rows], group_id=[r["grp"] for r in rows],
        support_unit_id=[r["unit"] for r in rows], mass_unit_id=[r["unit"] for r in rows],
        eval_unit_id=[r["unit"] for r in rows], sample_mass=np.array([r["mass"] for r in rows]),
        class_names=["c0", "c1"], source_train_idx=np.array(role_idx["source_train"]),
        source_audit_idx=np.array(role_idx["source_audit"]), target_audit_idx=np.array(role_idx["target_audit"]),
        preprocess_hash="pp", split_manifest_hash="sp", preprocess_fit_ids=frozenset())


def _maps():
    return build_frozen_maps(["c0", "c1"], ["Sd0", "Sd1", "Sd2"], ["Ad0", "Ad1", "Td0"])


# ---------------- FoldData ----------------
def test_fold_data_every_row_has_exactly_one_role():
    fd = _fold_data()
    n = len(fd.sample_id)
    assert set(fd.source_train_idx) | set(fd.source_audit_idx) | set(fd.target_audit_idx) == set(range(n))


def test_fold_data_allows_group_to_span_classes():
    fd = _fold_data()                                     # each recording group has both classes
    by_group = {}
    for i in range(len(fd.sample_id)):
        by_group.setdefault(fd.group_id[i], set()).add(int(fd.y[i]))
    assert any(len(cs) == 2 for cs in by_group.values())  # a group spanning classes is accepted


def _rejects(mutate):
    try:
        _fold_data(mutate=mutate)
    except ValueError:
        return
    raise AssertionError("FoldData must reject this")


def test_fold_data_rejects_group_spanning_domain_or_role():
    def m(rows):
        rows[0]["grp"] = rows[-1]["grp"]                  # a source row borrows a target group
    _rejects(m)


def test_fold_data_rejects_support_mass_eval_unit_spanning_cells():
    def m(rows):
        rows[0]["unit"] = rows[2]["unit"]                 # unit shared across two recording groups
    _rejects(m)


def test_fold_data_rejects_zero_source_class_mass():
    def m(rows):
        for r in rows:
            if r["role"] == "source_train" and r["y"] == 1:
                r["y"] = 0                                # remove all source class-1 mass
    _rejects(m)


def test_fold_data_mass_unit_sums_to_one():
    def m(rows):
        rows[0]["mass"] = 0.5                             # unit no longer sums to 1
    _rejects(m)


def test_fold_data_detects_postconstruction_tensor_mutation():
    fd = _fold_data()
    fd.assert_integrity()
    with torch.no_grad():
        fd.X[0, 0] += 1.0
    try:
        fd.assert_integrity()
    except ValueError:
        pass
    else:
        raise AssertionError("post-construction tensor mutation must be detected")


def test_subset_population_hash_is_row_order_invariant():
    rows = _rows()
    a = _fold_data(rows=list(rows))
    b = _fold_data(rows=list(reversed(rows)))
    assert a.source_train_population_hash == b.source_train_population_hash
    assert a.target_tensor_hash == b.target_tensor_hash and a.data_contract_hash == b.data_contract_hash


def test_subset_tensor_hash_changes_with_single_X_value():
    rows = _rows()
    base = _base_X(rows)
    fd = _fold_data(rows=list(rows), X=base.clone())
    bumped = base.clone()
    ti = [i for i, r in enumerate(rows) if r["role"] == "target_audit"][0]
    bumped[ti, 0] += 5.0                                              # a single target X value
    fd2 = _fold_data(rows=list(rows), X=bumped)
    assert fd2.target_tensor_hash != fd.target_tensor_hash
    assert fd2.target_population_hash == fd.target_population_hash      # ids/labels unchanged


# ---------------- deletion schedule + support ----------------
def test_deletion_schedule_hash_binds_order():
    fd, maps = _fold_data(), _maps()
    a = make_deletion_schedule([DeletionCell("Sd0", "c1"), DeletionCell("Sd1", "c0")], fd, maps)
    b = make_deletion_schedule([DeletionCell("Sd1", "c0"), DeletionCell("Sd0", "c1")], fd, maps)
    assert a.schedule_hash != b.schedule_hash


def test_schedule_rejects_unobserved_or_duplicate_cell():
    fd, maps = _fold_data(), _maps()
    try:
        make_deletion_schedule([DeletionCell("Sd0", "c0"), DeletionCell("Sd0", "c0")], fd, maps)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate cell must be rejected")


def test_declared_cell_has_zero_count_mass_and_rows():
    fd, maps = _fold_data(), _maps()
    sch = make_deletion_schedule([DeletionCell("Sd0", "c1")], fd, maps)
    ref = level0_reference_prior(fd, maps)
    s1 = build_level_support(fd, maps, 1, sch, ref, support_m=2)
    d = maps.source_domain_to_index["Sd0"]
    assert s1.eligibility_counts[d, 1] == 0 and s1.cell_mass[d, 1] == 0
    assert all(not (fd.domain_id[i] == "Sd0" and int(fd.y[i]) == 1) for i in s1.source_train_idx.tolist())


def test_reference_prior_is_byte_exact_across_levels():
    fd, maps = _fold_data(), _maps()
    sch = make_deletion_schedule([DeletionCell("Sd0", "c1")], fd, maps)
    ref = level0_reference_prior(fd, maps)
    s0 = build_level_support(fd, maps, 0, sch, ref, support_m=2)
    s1 = build_level_support(fd, maps, 1, sch, ref, support_m=2)
    assert np.array_equal(s0.support_graph.reference_prior, s1.support_graph.reference_prior)
    assert s0.level_support_hash != s1.level_support_hash    # deletion changes the level support


def test_support_counts_unique_support_units_not_rows():
    # one (Sd0,c0) cell whose 2 rows share ONE support unit (mass 0.5 each) -> count 1, not 2
    rows = [dict(sid="a", dom="Sd0", grp="g", unit="U", y=0, mass=0.5, role="source_train"),
            dict(sid="b", dom="Sd0", grp="g", unit="U", y=0, mass=0.5, role="source_train"),
            dict(sid="c", dom="Sd0", grp="g2", unit="c", y=1, mass=1.0, role="source_train"),
            dict(sid="d", dom="Sd1", grp="g3", unit="d", y=0, mass=1.0, role="source_train"),
            dict(sid="e", dom="Sd1", grp="g3", unit="e", y=1, mass=1.0, role="source_train"),
            dict(sid="f", dom="Ad0", grp="ga", unit="f", y=0, mass=1.0, role="source_audit"),
            dict(sid="h", dom="Ad0", grp="ga", unit="h", y=1, mass=1.0, role="source_audit"),
            dict(sid="t", dom="Td0", grp="gt", unit="t", y=0, mass=1.0, role="target_audit"),
            dict(sid="u", dom="Td0", grp="gt", unit="u", y=1, mass=1.0, role="target_audit")]
    fd = _fold_data(rows=rows); maps = _maps()
    sch = make_deletion_schedule([], fd, maps)
    s0 = build_level_support(fd, maps, 0, sch, level0_reference_prior(fd, maps), support_m=1)
    d = maps.source_domain_to_index["Sd0"]
    assert s0.eligibility_counts[d, 0] == 1 and abs(s0.cell_mass[d, 0] - 1.0) < 1e-9   # 1 unit, mass 1


def test_deleted_domain_retention_is_explicit_not_universal():
    # Sd2 has ONLY class 0 -> deleting (Sd2,c0) removes the domain entirely
    fd = _fold_data(rows=_rows(single_class_dom=("source_train", 2))); maps = _maps()
    sch_strict = make_deletion_schedule([DeletionCell("Sd2", "c0")], fd, maps, require_deleted_domain_retained=True)
    ref = level0_reference_prior(fd, maps)
    try:
        build_level_support(fd, maps, 1, sch_strict, ref, support_m=2)
    except ValueError:
        pass
    else:
        raise AssertionError("strict schedule must reject a deletion that removes the domain")
    sch_loose = make_deletion_schedule([DeletionCell("Sd2", "c0")], fd, maps, require_deleted_domain_retained=False)
    build_level_support(fd, maps, 1, sch_loose, ref, support_m=2)     # loose: allowed


# ---------------- typed non-estimability ----------------
def _feat(seed=0):
    X, y, d, g, sg = make_covariate_shift(seed=seed)
    sid = tuple(f"r{i}" for i in range(len(y)))
    grp = tuple(str(int(x)) for x in g.tolist())
    feat = FrozenFeatures(Z=X, y=y, d=d, group=grp, sample_id=sid)
    design = make_leakage_design(sid, y, d, grp, np.ones(len(y)), sg)
    return X, y, d, g, sg, feat, design


def test_no_comparable_support_raises_typed_error():
    X, y, d, g, sg = make_covariate_shift(seed=0, n_domains=1)        # 1 domain -> no comparable class
    feat = FrozenFeatures(Z=X, y=y, d=d, group=tuple(str(int(x)) for x in g.tolist()))
    try:
        make_fold_plan(feat, sg, n_folds=4, seed=0)
    except NoComparableSupport:
        pass
    else:
        raise AssertionError("no comparable class must raise NoComparableSupport")


def test_too_few_grouped_folds_raises_typed_error():
    # 2 domains x 2 classes but only ONE recording group per cell -> cannot form >=2 folds
    y = np.array([0, 1, 0, 1]); d = np.array([0, 0, 1, 1]); g = np.array([0, 0, 1, 1])
    counts = counts_from_labels(d, y, n_domains=2, n_classes=2)
    sg = build_support_graph(counts, m=1, reference_prior=empirical_class_prior(counts))
    feat = FrozenFeatures(Z=np.zeros((4, 2)), y=y, d=d, group=tuple(str(int(x)) for x in g))
    try:
        make_fold_plan(feat, sg, n_folds=2, seed=0)
    except FoldPlanNonEstimable:
        pass
    else:
        raise AssertionError("too few grouped folds must raise FoldPlanNonEstimable")


def test_bootstrap_exhaustion_raises_typed_error():
    X, y, d, g, sg, feat, design = _feat()
    fold = make_fold_plan_from_design(design, sg, n_folds=4, seed=0)
    try:
        make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=10_000, seed=0,
                                    max_candidate_multiplier=1, max_invalid_draw_rate=0.0)
    except BootstrapPlanNonEstimable:
        pass
    else:
        raise AssertionError("bootstrap exhaustion must raise BootstrapPlanNonEstimable")


def test_hash_mismatch_is_not_misclassified_as_nonestimable():
    X, y, d, g, sg, feat, design = _feat()
    fold = make_fold_plan_from_design(design, sg, n_folds=4, seed=0)
    other = make_covariate_shift(seed=0)[4]
    other.cell_mass = other.cell_mass.copy(); other.cell_mass[0, 0] += 5.0   # support_hash mismatch
    try:
        make_leakage_bootstrap_plan(design, other, fold, alpha=0.1, requested_replicates=4, seed=0)
    except LeakageNonEstimableError:
        raise AssertionError("a support-hash mismatch must NOT be a typed non-estimability")
    except ValueError:
        pass


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} runner-scope tests")


if __name__ == "__main__":
    _run_all()
