"""Eval package: distinct accuracy estimands, paired endpoints recomputed in-bootstrap, stable
NLL / fixed-bin ECE, no-target-calibration, strict pairing, fixed audit population, paired
clustered bootstrap (reuse / no-row-fallback / invalid-rate / too-few-clusters / seed), the
noninferiority rules, seed-blocking, and the CI fail-propagation.

Standalone (``python -m oaci.tests.test_eval``) and pytest-compatible.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from oaci.eval.artifacts import PredictionBundle, align_pair
from oaci.eval.bootstrap import (
    is_whole_group_resample,
    make_bootstrap_plan,
    paired_ci,
    point_delta_over_seeds,
)
from oaci.eval.calibration import fit_temperature, nll_per_sample, pooled_nll, top_label_ece
from oaci.eval.metrics import (
    domain_baccs,
    mean_domain_bacc,
    pooled_bacc,
    worst_domain_bacc,
    worst_paired_delta_bacc,
)
from oaci.eval.noninferiority import noninferiority, source_risk_noninferiority
from oaci.eval.sweep import assert_fixed_audit_population

CLS = [0, 1]


def _pop(sizes, per_group=20):
    y, dom, grp, gid = [], [], [], 0
    for d, ng in enumerate(sizes):
        for _ in range(ng):
            half = per_group // 2
            for lab in [0] * half + [1] * (per_group - half):
                y.append(lab); dom.append(d); grp.append(gid)
            gid += 1
    return np.arange(len(y)), np.array(y), np.array(dom), np.array(grp)


def _pred_with_acc(y, dom, acc_by_domain):
    pred = np.asarray(y).copy()
    for d, acc in acc_by_domain.items():
        for c in CLS:
            idx = np.where((dom == d) & (y == c))[0]
            nw = round((1 - acc) * len(idx))
            pred[idx[:nw]] = 1 - c
    return pred


def _bundle(sid, y, dom, grp, pred, level=0, role="target_audit", method="m"):
    logits = np.zeros((len(y), 2)); logits[np.arange(len(y)), pred] = 2.0
    return PredictionBundle(sid, logits, y, dom, grp, method, 0, "s", role, level, ["A", "B"])


# ---- accuracy estimands ----
def test_pooled_and_domain_macro_bacc_are_distinct():
    sid, y, dom, grp = _pop((8, 1), per_group=10)
    pred = y.copy(); pred[dom == 1] = 1 - y[dom == 1]          # big domain perfect, small all wrong
    assert abs(pooled_bacc(y, pred, CLS) - mean_domain_bacc(y, pred, dom, CLS)) > 0.2


def test_worst_domain_bacc_uses_minimum():
    sid, y, dom, grp = _pop((2, 2, 2), per_group=20)
    pred = _pred_with_acc(y, dom, {0: 0.9, 1: 0.6, 2: 0.8})
    assert abs(worst_domain_bacc(y, pred, dom, CLS) - min(domain_baccs(y, pred, dom, CLS).values())) < 1e-12
    assert abs(worst_domain_bacc(y, pred, dom, CLS) - 0.6) < 1e-9


def test_worst_paired_delta_detects_hidden_domain_harm():
    sid, y, dom, grp = _pop((2, 2), per_group=20)
    erm = _pred_with_acc(y, dom, {0: 0.6, 1: 0.9})
    oaci = _pred_with_acc(y, dom, {0: 0.95, 1: 0.7})           # domain 1 harmed, domain 0 improved
    diff_of_minima = worst_domain_bacc(y, oaci, dom, CLS) - worst_domain_bacc(y, erm, dom, CLS)
    wpd = worst_paired_delta_bacc(y, oaci, erm, dom, CLS)
    assert diff_of_minima >= 0 and wpd < 0                      # the difference-of-minima hides the harm


def test_worst_domain_is_recomputed_inside_each_bootstrap():
    # well-separated accuracies (so ERM/OACI predictions genuinely differ) with errors that vary
    # by group, so resampling moves both the delta and which domain is worst.
    sid, y, dom, grp = _pop((3, 3, 3), per_group=40)
    erm = _pred_with_acc(y, dom, {0: 0.55, 1: 0.80, 2: 0.82})
    oaci = _pred_with_acc(y, dom, {0: 0.85, 1: 0.70, 2: 0.70})  # d1,d2 tie as worst -> argmin flips
    plan = make_bootstrap_plan(dom, grp, y, CLS, n_boot=200, seed=0)
    deltas = [worst_paired_delta_bacc(y[i], oaci[i], erm[i], dom[i], CLS) for i in plan.replicates]
    worst = [min(b, key=b.get) for b in
             (domain_baccs(y[i], oaci[i], dom[i], CLS) for i in plan.replicates)]
    assert len(set(np.round(deltas, 6))) > 1                    # delta recomputed per replicate
    assert len(set(worst)) > 1                                  # the worst domain itself moves


# ---- calibration ----
def test_nll_from_extreme_logits_is_finite():
    logits = np.array([[1000.0, -1000.0], [-1e9, 1e9], [0.0, 0.0]])
    y = np.array([0, 1, 0])
    assert np.all(np.isfinite(nll_per_sample(logits, y))) and np.isfinite(pooled_nll(logits, y))


def test_fixed_bin_ece_matches_known_value():
    import math
    p = math.exp(2) / (math.exp(2) + 1)                        # confidence of every prediction
    logits = np.tile([2.0, 0.0], (10, 1)); y = np.zeros(10, int); y[:2] = 1   # 8/10 correct
    assert abs(top_label_ece(logits, y, n_bins=15) - abs(0.8 - p)) < 1e-9


def test_target_temperature_fitting_is_rejected():
    logits = np.tile([2.0, 0.0], (8, 1)); y = np.zeros(8, int)
    for bad in ("target", "test", "any"):
        try:
            fit_temperature(logits, y, role=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"target calibration role={bad!r} must be rejected")
    assert fit_temperature(logits, y, role="diagnostic") > 0    # diagnostic allowed, flagged


# ---- pairing / population ----
def test_pairing_requires_identical_sample_population():
    sid, y, dom, grp = _pop((2, 2), per_group=10)
    a = _bundle(sid, y, dom, grp, y, method="OACI")
    b = _bundle(sid + 1, y, dom, grp, y, method="ERM")         # shifted sample_id set
    try:
        align_pair(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("align_pair must reject a different sample_id population")


def test_missing_cell_sweep_keeps_audit_population_fixed():
    sid, y, dom, grp = _pop((2, 2), per_group=10)
    b0 = _bundle(sid, y, dom, grp, y, level=0, method="ERM")
    b1 = _bundle(sid, y, dom, grp, 1 - y, level=1, method="ERM")   # same population, different preds
    assert_fixed_audit_population({0: {"ERM": b0}, 1: {"ERM": b1}})   # ok (one population hash)
    bad = _bundle(sid, 1 - y, dom, grp, y, level=1, method="ERM")    # different y -> different population
    try:
        assert_fixed_audit_population({0: {"ERM": b0}, 1: {"ERM": bad}})
    except ValueError:
        pass
    else:
        raise AssertionError("a changed audit population must be rejected")


# ---- bootstrap ----
def test_same_bootstrap_plan_is_reused_across_methods_and_levels():
    sid, y, dom, grp = _pop((3, 3), per_group=20)
    plan = make_bootstrap_plan(dom, grp, y, CLS, n_boot=100, seed=0)
    fn = lambda i: float(np.mean(y[i]))                        # any per-resample statistic
    ci_a = paired_ci(plan, 0.0, fn)
    ci_b = paired_ci(plan, 0.0, fn)
    assert ci_a == ci_b                                        # same plan -> identical replicates reused


def test_cluster_bootstrap_never_falls_back_to_rows():
    sid, y, dom, grp = _pop((3, 3), per_group=20)
    plan = make_bootstrap_plan(dom, grp, y, CLS, n_boot=50, seed=0)
    for idx in plan.replicates:
        assert is_whole_group_resample(idx, grp, plan)        # whole groups only, never rows


def test_invalid_bootstrap_draw_rate_is_reported():
    # domain 0: class 1 lives only in group 0 -> replicates missing group 0 are invalid (redrawn)
    y, dom, grp, gid = [], [], [], 0
    for lab in [0] * 5 + [1] * 5:
        y.append(lab); dom.append(0); grp.append(0)
    gid = 1
    for g in range(3):
        for lab in [0] * 10:
            y.append(lab); dom.append(0); grp.append(gid)
        gid += 1
    for g in range(2):
        for lab in [0] * 5 + [1] * 5:
            y.append(lab); dom.append(1); grp.append(gid)
        gid += 1
    y, dom, grp = np.array(y), np.array(dom), np.array(grp)
    plan = make_bootstrap_plan(dom, grp, y, CLS, n_boot=100, seed=0, invalid_threshold=0.95)
    assert plan.estimable and plan.invalid_draw_rate > 0.05


def test_too_few_clusters_returns_nonestimable_ci():
    y, dom, grp = [], [], []
    for lab in [0] * 5 + [1] * 5:                              # domain 1 has a SINGLE group
        y.append(lab); dom.append(1); grp.append(99)
    for g in range(3):
        for lab in [0] * 5 + [1] * 5:
            y.append(lab); dom.append(0); grp.append(g)
    y, dom, grp = np.array(y), np.array(dom), np.array(grp)
    plan = make_bootstrap_plan(dom, grp, y, CLS, n_boot=50, seed=0)
    assert not plan.estimable and "cluster" in plan.reason
    assert paired_ci(plan, 0.0, lambda i: 0.0)["estimable"] is False


def test_seed_reproducibility():
    sid, y, dom, grp = _pop((3, 3), per_group=20)
    p1 = make_bootstrap_plan(dom, grp, y, CLS, n_boot=50, seed=3)
    p2 = make_bootstrap_plan(dom, grp, y, CLS, n_boot=50, seed=3)
    p3 = make_bootstrap_plan(dom, grp, y, CLS, n_boot=50, seed=4)
    assert all(np.array_equal(a, b) for a, b in zip(p1.replicates, p2.replicates))
    assert not all(np.array_equal(a, b) for a, b in zip(p1.replicates, p3.replicates))


def test_zero_paired_delta_has_zero_width_on_identical_predictions():
    sid, y, dom, grp = _pop((3, 3), per_group=20)
    pred = _pred_with_acc(y, dom, {0: 0.8, 1: 0.7})
    plan = make_bootstrap_plan(dom, grp, y, CLS, n_boot=50, seed=0)
    fn = lambda i: worst_paired_delta_bacc(y[i], pred[i], pred[i], dom[i], CLS)   # OACI == ERM
    ci = paired_ci(plan, 0.0, fn)
    assert ci["basic_lcl"] == 0.0 and ci["basic_ucl"] == 0.0
    assert ci["percentile_lcl"] == 0.0 and ci["percentile_ucl"] == 0.0


# ---- noninferiority ----
def test_higher_better_noninferiority_rule():
    assert noninferiority({"estimable": True, "basic_lcl": -0.01, "basic_ucl": 0.05}, 0.02, True) is True
    assert noninferiority({"estimable": True, "basic_lcl": -0.05, "basic_ucl": 0.00}, 0.02, True) is False


def test_lower_better_noninferiority_rule():
    assert noninferiority({"estimable": True, "basic_lcl": -0.1, "basic_ucl": 0.01}, 0.02, False) is True
    assert noninferiority({"estimable": True, "basic_lcl": -0.1, "basic_ucl": 0.05}, 0.02, False) is False


def test_epsilon_reuse_requires_matching_risk_metric():
    ci = {"estimable": True, "basic_ucl": 0.01}
    assert source_risk_noninferiority(ci, 0.05, "balanced_ce", "balanced_ce") is True
    try:
        source_risk_noninferiority(ci, 0.05, "ce", "balanced_ce")
    except ValueError:
        pass
    else:
        raise AssertionError("epsilon reuse across differing risk metrics must be rejected")


def test_repeated_seeds_are_not_counted_as_extra_trials():
    assert point_delta_over_seeds([0.2, 0.2]) == 0.2
    assert abs(point_delta_over_seeds([0.2, 0.4]) - 0.3) < 1e-12
    assert point_delta_over_seeds([0.2]) == point_delta_over_seeds([0.2, 0.2])   # duplicate seed != more N


# ---- CI must really fail ----
def test_ci_job_exits_nonzero_when_test_or_demo_fails():
    work = tempfile.mkdtemp()
    open(os.path.join(work, "a.rc"), "w").write("0")
    open(os.path.join(work, "b.rc"), "w").write("2")          # a failing job
    snip = 'fail=0; for f in "%s"/*.rc; do [ "$(cat "$f")" -ne 0 ] && fail=1; done; exit "$fail"' % work
    assert subprocess.call(["bash", "-c", snip]) != 0
    ok = tempfile.mkdtemp(); open(os.path.join(ok, "a.rc"), "w").write("0")
    snip2 = 'fail=0; for f in "%s"/*.rc; do [ "$(cat "$f")" -ne 0 ] && fail=1; done; exit "$fail"' % ok
    assert subprocess.call(["bash", "-c", snip2]) == 0
    script = open(os.path.join(os.path.dirname(__file__), "..", "slurm_ci.sh")).read()
    assert 'exit "$fail"' in script and "demo_eval.rc" in script   # demos folded into the exit code


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} eval tests")


if __name__ == "__main__":
    _run_all()
