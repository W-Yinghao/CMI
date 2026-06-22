"""Support graph = the formal core of OACI (THEORY §1). These tests pin the corrected
bookkeeping: estimator-eligibility (not "identifiability") thresholding, singleton flagging,
**per-class** identifiable pairs (NO cross-class transitivity), coupling/decomposability as a
separate notion, and a fixed-reference-prior estimand that does not drift.

Standalone (``python -m oaci.tests.test_support_graph``) and pytest-compatible.
"""
from __future__ import annotations

import numpy as np

from oaci.support_graph import (
    build_support_graph,
    counts_from_labels,
    empirical_class_prior,
)


def test_eligibility_threshold_is_finite_sample_not_identifiability():
    # cell (d=1, y=1) has 3 samples < m=10 -> not eligible (but it IS present).
    counts = np.array([[50, 50], [50, 3]])
    sg = build_support_graph(counts, m=10)
    assert sg.is_eligible(0, 0) and sg.is_eligible(0, 1) and sg.is_eligible(1, 0)
    assert not sg.is_eligible(1, 1)
    assert sg.is_present(1, 1)            # present (n>0) but not eligible -> the §1 distinction
    assert (1, 1) in sg.ineligible_cells()


def test_singleton_support_class_is_flagged_not_comparable():
    counts = np.array([[40, 40], [40, 0], [40, 0]])
    sg = build_support_graph(counts, m=10)
    assert sg.comparable_classes == [0]
    assert sg.singleton_classes == [1]
    assert all(t["y"] != 1 for t in sg.overlap_terms())


def test_empty_class_reported():
    counts = np.array([[30, 0], [30, 0]])
    sg = build_support_graph(counts, m=10)
    assert sg.empty_classes == [1]
    assert sg.comparable_classes == [0]


def test_identifiable_pair_is_per_class():
    # class 0 eligible in domains {0,1}; class 1 eligible in domains {1,2}.
    counts = np.array(
        [
            # y0   y1
            [60, 0],   # d0
            [60, 60],  # d1
            [0, 60],   # d2
        ]
    )
    sg = build_support_graph(counts, m=10)
    assert sg.is_identifiable_pair(0, 1, y=0)        # both in S_0
    assert not sg.is_identifiable_pair(0, 2, y=0)    # d2 not in S_0
    assert sg.is_identifiable_pair(1, 2, y=1)        # both in S_1
    assert not sg.is_identifiable_pair(0, 1, y=1)    # d0 not in S_1


def test_no_cross_class_transitive_reach():
    """The v1 bug: 0~1 share class 0, 1~2 share class 1, so (0,2) are COUPLED, but NO fixed-y
    equality between 0 and 2 is identifiable (they co-observe no class)."""
    counts = np.array(
        [
            [60, 0],   # d0: class 0 only
            [60, 60],  # d1: bridge
            [0, 60],   # d2: class 1 only
        ]
    )
    sg = build_support_graph(counts, m=10)
    assert sg.coupled(0, 2)                          # one coupling component (decomposability)
    assert len(sg.coupling_components) == 1
    # but identifiable for NEITHER class:
    assert not sg.is_identifiable_pair(0, 2, y=0)
    assert not sg.is_identifiable_pair(0, 2, y=1)
    assert sg.decoupled_pairs() == []               # nothing fully decoupled here


def test_decoupled_components_share_no_class():
    counts = np.array(
        [
            [60, 0],
            [60, 0],
            [0, 60],
            [0, 60],
        ]
    )
    sg = build_support_graph(counts, m=10)
    assert len(sg.coupling_components) == 2
    assert not sg.coupled(0, 2)
    for pair in [(0, 2), (0, 3), (1, 2), (1, 3)]:
        assert pair in sg.decoupled_pairs()
    assert (0, 1) not in sg.decoupled_pairs()


def test_overlap_terms_use_fixed_reference_prior():
    counts = np.array([[100, 20], [100, 80]])  # both classes comparable
    p_ref = empirical_class_prior(counts)      # [200/300, 100/300]
    sg = build_support_graph(counts, m=10, reference_prior=p_ref)
    terms = {t["y"]: t for t in sg.overlap_terms()}
    assert abs(terms[0]["w_abs"] - 200 / 300) < 1e-9
    assert abs(terms[1]["w_abs"] - 100 / 300) < 1e-9
    # both comparable -> w_cond renormalises to sum 1; w_abs sums to identifiable mass (=1 here)
    assert abs(sum(t["w_abs"] for t in sg.overlap_terms()) - sg.identifiable_mass_fraction()) < 1e-9
    assert abs(sum(t["w_cond"] for t in sg.overlap_terms()) - 1.0) < 1e-9


def test_estimand_does_not_drift_when_prior_is_held_fixed():
    """Deleting a cell must not change the per-class weights if p_ref is passed fixed —
    this is the property the missing-cell sweep relies on (THEORY §Estimand)."""
    base = np.array([[100, 60], [100, 60], [100, 60]])
    p_ref = empirical_class_prior(base)
    sg_full = build_support_graph(base, m=10, reference_prior=p_ref)
    # delete class-1 cell of domain 2 -> class 1 still comparable (|S_1|=2), weights unchanged
    masked = base.copy()
    masked[2, 1] = 0
    sg_cut = build_support_graph(masked, m=10, reference_prior=p_ref)
    w_full = {t["y"]: t["w_abs"] for t in sg_full.overlap_terms()}
    w_cut = {t["y"]: t["w_abs"] for t in sg_cut.overlap_terms()}
    assert w_full == w_cut
    assert sg_full.identifiable_mass_fraction() == sg_cut.identifiable_mass_fraction()


def test_identifiable_mass_drops_when_a_class_loses_comparability():
    base = np.array([[100, 60], [100, 0]])     # class 1 only in domain 0 already? no: [60,0]
    p_ref = empirical_class_prior(base)        # class 1 singleton -> not comparable
    sg = build_support_graph(base, m=10, reference_prior=p_ref)
    assert sg.comparable_classes == [0]
    # identifiable mass = p_ref(class 0) only
    assert abs(sg.identifiable_mass_fraction() - p_ref[0]) < 1e-9
    assert sg.identifiable_mass_fraction() < 1.0


def test_counts_from_labels_roundtrip():
    rng = np.random.default_rng(0)
    d = rng.integers(0, 4, size=500)
    y = rng.integers(0, 3, size=500)
    counts = counts_from_labels(d, y, n_domains=4, n_classes=3)
    assert counts.shape == (4, 3) and counts.sum() == 500
    for dd in range(4):
        for yy in range(3):
            assert counts[dd, yy] == int(np.sum((d == dd) & (y == yy)))


def test_input_validation():
    bad = [
        dict(eligibility_counts=np.zeros((2, 2)), m=0),
        dict(eligibility_counts=np.zeros(3), m=5),
        dict(eligibility_counts=np.ones((2, 2)), m=1, reference_prior=np.array([1.0, 2.0, 3.0])),  # wrong len
        dict(eligibility_counts=np.ones((2, 2)), m=1, reference_prior=np.array([-1.0, 1.0])),      # negative
    ]
    for kw in bad:
        try:
            build_support_graph(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kw}")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} support-graph tests")


if __name__ == "__main__":
    _run_all()
