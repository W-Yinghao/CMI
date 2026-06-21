"""Bootstrap UCB behaviour: in-replicate capacity reselection, fixed support/p_ref across the
bootstrap, the basic one-sided formula (kept distinct from the percentile endpoint), negatives
not truncated, and seed reproducibility.

Standalone (``python -m oaci.tests.test_leakage_ucb``) and pytest-compatible.
"""
from __future__ import annotations

import numpy as np

from oaci.leakage.critic import CriticConfig
from oaci.leakage.crossfit import make_fold_plan
from oaci.leakage.estimate import reference_conditional_entropy
from oaci.leakage.synthetic import make_nonlinear, make_null, make_perfect
from oaci.leakage.ucb import bootstrap_ucb

FAST = CriticConfig(capacities=(0, 32))


def _ucb(maker, seed=0, B=25, alpha=0.1, cfg=FAST, n_folds=2, **kw):
    feat, sg = maker(seed=seed, **kw)
    plan = make_fold_plan(feat, sg, n_folds=n_folds, seed=0)
    return bootstrap_ucb(feat, sg, plan, cfg, alpha=alpha, n_bootstrap=B, seed=0), sg


def test_capacity_is_reselected_inside_each_replicate():
    cfg = CriticConfig(capacities=(0, 64))
    res, _ = _ucb(make_nonlinear, seed=1, B=25, per_cell=40, cfg=cfg)
    assert len(res["replicate_capacities"]) == res["n_bootstrap"]      # one per replicate
    assert all(c in cfg.capacities for c in res["replicate_capacities"])
    # nonlinear leakage -> the nonlinear capacity should win in the replicates
    assert res["replicate_capacities"].count(64) >= res["replicate_capacities"].count(0)


def test_reference_entropy_is_from_fixed_support_not_resamples():
    res, sg = _ucb(make_perfect, seed=1, B=15, per_cell=25)
    H_fixed = reference_conditional_entropy(sg)
    assert res["reference_entropy"] == H_fixed                         # computed once, fixed


def test_basic_and_percentile_ucl_formulas_distinct():
    res, _ = _ucb(make_null, seed=2, B=40, per_cell=25)
    reps = res["replicates"]
    Lhat = res["extractable_LQ_ov"]
    assert abs(res["bootstrap_ucl"] - (2 * Lhat - np.quantile(reps, res["alpha"]))) < 1e-9
    assert abs(res["percentile_ucl"] - np.quantile(reps, 1 - res["alpha"])) < 1e-9
    assert res["bootstrap_ucl"] != res["percentile_ucl"]              # not mixed


def test_seed_reproducible_and_seed_sensitive():
    feat, sg = make_perfect(seed=1, per_cell=25)
    plan = make_fold_plan(feat, sg, n_folds=2, seed=0)
    a = bootstrap_ucb(feat, sg, plan, FAST, alpha=0.1, n_bootstrap=20, seed=7)
    b = bootstrap_ucb(feat, sg, plan, FAST, alpha=0.1, n_bootstrap=20, seed=7)
    c = bootstrap_ucb(feat, sg, plan, FAST, alpha=0.1, n_bootstrap=20, seed=8)
    assert np.allclose(a["replicates"], b["replicates"])              # same seed -> identical
    assert a["bootstrap_ucl"] == b["bootstrap_ucl"]
    assert a["extractable_LQ_ov"] == c["extractable_LQ_ov"]           # point estimate seed-independent
    assert not np.allclose(a["replicates"], c["replicates"])          # different seed -> different draws


def test_perfect_leakage_ucl_brackets_reference_ceiling():
    res, sg = _ucb(make_perfect, seed=3, B=25, per_cell=25)
    Hbar = sum(sg.reference_prior[y] * res["reference_entropy"][y] for y in sg.comparable_classes)
    assert res["L_abs"] > 0.6 * Hbar
    assert res["bootstrap_ucl"] >= res["L_abs"] - 1e-6                # an upper limit


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} leakage-ucb tests")


if __name__ == "__main__":
    _run_all()
