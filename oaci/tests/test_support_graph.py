"""Support graph = the formal core of OACI (THEORY §1). These tests pin the identifiability
bookkeeping: thresholding, singleton flagging, connected components / transitive reach, and
— most importantly — that unsupported mass is surfaced rather than smoothed.

Standalone (``python -m oaci.tests.test_support_graph``) and pytest-compatible.
"""
from __future__ import annotations

import numpy as np

from oaci.support_graph import build_support_graph, counts_from_labels


def test_threshold_marks_low_count_cells_non_identifiable():
    # cell (d=1, y=1) has 3 samples < m=10 -> NOT observed.
    counts = np.array([[50, 50], [50, 3]])
    sg = build_support_graph(counts, m=10)
    assert sg.is_observed(0, 0) and sg.is_observed(0, 1) and sg.is_observed(1, 0)
    assert not sg.is_observed(1, 1)
    assert (1, 1) in sg.unobserved_cells()


def test_singleton_support_class_is_flagged_not_comparable():
    # class 1 has support only in domain 0 -> singleton, contributes NO leakage term.
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


def test_disconnected_support_yields_two_components():
    # domains {0,1} co-observe class 0; domains {2,3} co-observe class 1; no shared class
    # links the two groups -> two components -> cross-group invariance is non-identifiable.
    counts = np.array(
        [
            [60, 0],
            [60, 0],
            [0, 60],
            [0, 60],
        ]
    )
    sg = build_support_graph(counts, m=10)
    assert len(sg.components) == 2
    assert sg.identifiable_pair(0, 1) and sg.identifiable_pair(2, 3)
    assert not sg.identifiable_pair(0, 2)
    nip = sg.non_identifiable_pairs()
    for pair in [(0, 2), (0, 3), (1, 2), (1, 3)]:
        assert pair in nip
    assert (0, 1) not in nip and (2, 3) not in nip


def test_chain_gives_transitive_reach():
    # 0~1 share class A, 1~2 share class B. 0 and 2 never co-observe a class, but the chain
    # through domain 1 makes cross-domain invariance between 0 and 2 reachable (one component).
    counts = np.array(
        [
            # A    B
            [60, 0],   # d0
            [60, 60],  # d1  (the bridge)
            [0, 60],   # d2
        ]
    )
    sg = build_support_graph(counts, m=10)
    assert len(sg.components) == 1
    assert sg.identifiable_pair(0, 2)            # transitive, even with no shared class
    assert sg.non_identifiable_pairs() == []


def test_overlap_terms_weights_sum_to_one_over_comparable_classes():
    counts = np.array([[100, 20], [100, 80]])  # both classes comparable here
    sg = build_support_graph(counts, m=10)
    w = [t["weight"] for t in sg.overlap_terms()]
    assert len(w) == 2
    assert abs(sum(w) - 1.0) < 1e-9
    # class 0 has more observed mass (200) than class 1 (100) -> larger weight
    by_y = {t["y"]: t["weight"] for t in sg.overlap_terms()}
    assert by_y[0] > by_y[1]


def test_unsupported_mass_is_surfaced_not_smoothed():
    # 80% of the data sits in cells that are NOT in any comparable term; the fraction must
    # reflect that honestly (no imputation pulls it toward 1.0).
    counts = np.array(
        [
            [10, 90],  # class-1 here is singleton (only domain 0) -> not comparable
            [10, 0],
        ]
    )
    sg = build_support_graph(counts, m=5)
    assert sg.comparable_classes == [0]           # only class 0 compares across domains
    frac = sg.identifiable_mass_fraction()
    assert abs(frac - (20 / 110)) < 1e-9          # only the 20 class-0 samples are constrained
    assert frac < 1.0


def test_counts_from_labels_roundtrip():
    rng = np.random.default_rng(0)
    d = rng.integers(0, 4, size=500)
    y = rng.integers(0, 3, size=500)
    counts = counts_from_labels(d, y, n_domains=4, n_classes=3)
    assert counts.shape == (4, 3)
    assert counts.sum() == 500
    for dd in range(4):
        for yy in range(3):
            assert counts[dd, yy] == int(np.sum((d == dd) & (y == yy)))


def test_input_validation():
    for bad in [dict(counts=np.zeros((2, 2)), m=0), dict(counts=np.zeros(3), m=5)]:
        try:
            build_support_graph(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} support-graph tests")


if __name__ == "__main__":
    _run_all()
