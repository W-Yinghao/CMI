"""Point-estimate behaviour of the extractable-leakage estimator: conditional null, explicit
(perfect) domain encoding, unsupported-cell exclusion, fixed-p_ref weighting, capacity
selection AFTER class-weighted aggregation, and no truncation of negative estimates.

Standalone (``python -m oaci.tests.test_leakage_estimate``) and pytest-compatible.
"""
from __future__ import annotations

import numpy as np

from oaci.leakage.critic import CriticConfig
from oaci.leakage.crossfit import FrozenFeatures, make_fold_plan, oof_nll_by_class
from oaci.leakage.estimate import estimate_extractable_leakage, reference_conditional_entropy
from oaci.leakage.synthetic import make_null, make_nonlinear, make_perfect
from oaci.support_graph import build_support_graph, counts_from_labels, empirical_class_prior

FAST = CriticConfig(capacities=(0, 16))


def _est(feat, sg, cfg=FAST, n_folds=2):
    plan = make_fold_plan(feat, sg, n_folds=n_folds, seed=0)
    return estimate_extractable_leakage(feat, sg, plan, cfg)


def test_conditional_null_is_near_zero():
    feat, sg = make_null(seed=1, recs_per_domain=4, per_cell=25)
    res = _est(feat, sg)
    # Z ⟂ D | Y  -> no extractable leakage; cross-fit makes it ~0 (small either sign)
    assert res["L_abs"] < 0.15, res["L_abs"]
    assert res["L_abs"] > -0.30, res["L_abs"]


def test_perfect_leakage_approaches_reference_entropy():
    feat, sg = make_perfect(seed=1, recs_per_domain=4, per_cell=25)
    res = _est(feat, sg)
    Hbar = sum(sg.reference_prior[y] * res["reference_entropy"][y] for y in sg.comparable_classes)
    assert Hbar > 0.6  # ~ln2 for balanced 2 domains
    assert res["L_abs"] > 0.6 * Hbar, (res["L_abs"], Hbar)          # recovers most of the max
    assert all(v < 0.25 for v in res["nll_by_class"].values()), res["nll_by_class"]


def test_unsupported_cell_excluded_from_labels_and_scoring():
    # domains 0,1 fully supported; domain 2 has only 5 class-0 samples (< m=20) -> ineligible.
    y, d, g = [], [], []
    gid = 0
    for dom in (0, 1):
        for _ in range(4):
            for c in (0, 1):
                y += [c] * 25; d += [dom] * 25; g += [gid] * 25
            gid += 1
    y += [0] * 5; d += [2] * 5; g += [gid] * 5            # tiny ineligible cell (2, class0)
    y, d, g = np.array(y), np.array(d), np.array(g)
    rng = np.random.default_rng(0)
    Z = (np.array([[2.0, 0], [0, 2.0]])[y] + 4.0 * np.array([[1, 0], [0, 1], [1, 1]])[d]
         + 0.2 * rng.standard_normal((y.size, 2)))
    counts = counts_from_labels(d, y, n_domains=3, n_classes=2)
    sg = build_support_graph(counts, m=20, reference_prior=empirical_class_prior(counts))
    assert 2 not in sg.support_of_class[0]                # domain 2 excluded from S_0
    feat = FrozenFeatures(Z, y, d, g)
    plan = make_fold_plan(feat, sg, n_folds=2, seed=0)
    nll = oof_nll_by_class(feat, sg, plan, capacity=0, cfg=FAST)
    assert nll[0]["n_rows"] == 200                        # 2 domains x 4 recs x 25, NOT the +5


def test_L_abs_uses_fixed_reference_prior_weighting():
    feat, sg = make_perfect(seed=2, recs_per_domain=4, per_cell=25)
    res = _est(feat, sg)
    recon = sum(sg.reference_prior[y] * res["L_y"][y] for y in sg.comparable_classes)
    assert abs(recon - res["L_abs"]) < 1e-9              # L_abs == Σ p_ref(y) L_y exactly


def test_capacity_selection_is_after_aggregation():
    feat, sg = make_nonlinear(seed=3, recs_per_domain=4, per_cell=40)
    res = estimate_extractable_leakage(
        feat, sg, make_fold_plan(feat, sg, n_folds=2, seed=0), CriticConfig(capacities=(0, 64))
    )
    by_c = res["L_abs_by_capacity"]
    assert by_c[64] > by_c[0] + 0.1                      # nonlinear probe extracts more
    assert res["selected_capacity"] == 64                # the sup picks it AFTER aggregation


def test_negative_estimates_are_not_truncated():
    # pure-noise Z + high capacity -> overfit -> OOF NLL > Ĥ_y -> negative L (kept, not clipped)
    y, d, g = [], [], []
    gid = 0
    for dom in (0, 1):
        for _ in range(4):
            for c in (0, 1):
                y += [c] * 8; d += [dom] * 8; g += [gid] * 8
            gid += 1
    y, d, g = np.array(y), np.array(d), np.array(g)
    Z = np.random.default_rng(0).standard_normal((y.size, 8))   # independent of d,y
    counts = counts_from_labels(d, y, n_domains=2, n_classes=2)
    sg = build_support_graph(counts, m=4, reference_prior=empirical_class_prior(counts))
    feat = FrozenFeatures(Z, y, d, g)
    res = estimate_extractable_leakage(
        feat, sg, make_fold_plan(feat, sg, n_folds=2, seed=0), CriticConfig(capacities=(0, 64))
    )
    assert res["L_abs_by_capacity"][64] < 0.0            # negative, not floored at 0


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} leakage-estimate tests")


if __name__ == "__main__":
    _run_all()
