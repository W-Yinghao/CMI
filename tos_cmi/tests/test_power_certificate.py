"""Phase 1.3.1b -- unit tests for the task-gate power floor (competence certificate).

Fast, training-free checks of the certificate machinery: the conservative + monotone lookup, the
3-way safety classification, and the matched positive-control effect-size tuning (Bayes-only).
The heavy end-to-end (gate abstains via TASK_POWER_INSUFFICIENT on a low-power injection while
factorized stays SAFE_ACCEPT) lives in the phase-diagram re-run / test_bayes_calibration."""
import numpy as np
import torch

torch.set_num_threads(1)

from tos_cmi.score_fisher import ScoreFisherConfig
from tos_cmi.eval.power_certificate import (lookup_power, make_control, tune_confound,
                                            wilson_lcb, assert_power_feasible,
                                            bayes_delta_of_geometry)


def _fake_table():
    return {"meta": {"delta_Y": 0.03}, "table": [
        {"n_eff": 2000, "d_base": 23, "d_extra": 1, "n_cls": 3, "power_ok": False, "mde": None},
        {"n_eff": 6000, "d_base": 23, "d_extra": 1, "n_cls": 3, "power_ok": True, "mde": 0.03},
        {"n_eff": 2000, "d_base": 22, "d_extra": 2, "n_cls": 3, "power_ok": True, "mde": 0.03},
    ]}


def test_fingerprint_mismatch_abstains():
    """A competence table built for one estimator config must NOT certify a different one: a
    fingerprint mismatch -> power NOT ok (abstain), even for a covered, power_ok cell."""
    from dataclasses import replace
    from tos_cmi.score_fisher import estimator_fingerprint
    cfg = ScoreFisherConfig(task_protect=True)
    t = {"meta": {"fingerprint": estimator_fingerprint(cfg)}, "table": [
        {"n_eff": 2000, "d_base": 23, "d_extra": 1, "n_cls": 3, "power_ok": True, "mde": 0.03}]}
    ok, _ = lookup_power(t, 6000, 23, 1, 3, cfg=cfg); assert ok                  # matching -> ok
    cfg2 = replace(cfg, task_gate_hidden=999)                                    # different config
    ok2, inf = lookup_power(t, 6000, 23, 1, 3, cfg=cfg2)
    assert not ok2 and inf["reason"] == "fingerprint_mismatch", inf
    print("test_fingerprint_mismatch_abstains: OK")


def test_exact_lookup_and_scope():
    """Phase 1.3.4: exact mode requires an EXACT (n_eff,d_base,d_extra,n_cls) cell AND a matching
    scope (n_dom, cluster_regime) -- no monotone-in-n extrapolation. Scope mismatch -> abstain."""
    t = {"meta": {"scope": {"n_dom": 6, "cluster_regime": "iid"}}, "table": [
        {"n_eff": 6000, "d_base": 23, "d_extra": 1, "n_cls": 3, "power_ok": True, "mde": 0.10}]}
    # exact hit (matching scope)
    ok, inf = lookup_power(t, 6000, 23, 1, 3, n_dom=6, cluster_regime="iid", mode="exact")
    assert ok and inf["mode"] == "exact", inf
    # exact mode does NOT extrapolate to a larger n_eff (no exact cell)
    ok2, inf2 = lookup_power(t, 9000, 23, 1, 3, n_dom=6, cluster_regime="iid", mode="exact")
    assert not ok2 and inf2["reason"] == "no_exact_cell", inf2
    # scope mismatch (different n_dom / cluster regime) -> abstain
    ok3, inf3 = lookup_power(t, 6000, 23, 1, 3, n_dom=8, cluster_regime="iid", mode="exact")
    assert not ok3 and inf3["reason"] == "scope_mismatch", inf3
    ok4, inf4 = lookup_power(t, 6000, 23, 1, 3, n_dom=6, cluster_regime="clustered", mode="exact")
    assert not ok4 and inf4["reason"] == "scope_mismatch", inf4
    # lower_n mode DOES extrapolate (exploratory) -> ok at 9000
    ok5, _ = lookup_power(t, 9000, 23, 1, 3, mode="lower_n"); assert ok5
    print("test_exact_lookup_and_scope: OK")


def test_lookup_conservative_and_monotone():
    t = _fake_table()
    # below the smallest calibrated n -> uncovered -> abstain
    ok, inf = lookup_power(t, 1000, 23, 1, 3); assert not ok and not inf["covered"], inf
    # at a covered low-n cell that is NOT power_ok -> not ok
    ok, _ = lookup_power(t, 2000, 23, 1, 3); assert not ok
    # between cells: only the n_eff=2000(False) cell qualifies (<=3000) -> not ok
    ok, _ = lookup_power(t, 3000, 23, 1, 3); assert not ok
    # above the power_ok cell: monotone -> ok
    ok, inf = lookup_power(t, 8000, 23, 1, 3); assert ok and inf["covered"], inf
    # different power_ok shape
    ok, _ = lookup_power(t, 4000, 22, 2, 3); assert ok
    # uncovered shape (dim not in table) -> abstain
    ok, inf = lookup_power(t, 8000, 17, 1, 3); assert not ok and not inf["covered"]
    print("test_lookup_conservative_and_monotone: OK")


def test_control_effect_tuning_exact_and_sample_invariant():
    """Fixed-geometry control hits the target Bayes Delta* via the confound scale, and the effect
    is INVARIANT to the sample seed (only y/d/noise resample) -- so power(Delta) is measured at a
    single effect size, not a mixture."""
    cfg = ScoreFisherConfig()
    for tgt in [cfg.delta_Y, 2 * cfg.delta_Y]:
        c = tune_confound(tgt, 23, 1, 3, 6, base_sep=1.5, sigma=1.0, geom_seed=7)
        b0 = bayes_delta_of_geometry(23, 1, 3, 6, 1.5, c, 1.0, 0.2, 7)
        assert abs(b0 - tgt) < 0.01, (tgt, c, b0)
        # same geometry, different sample seeds -> identical exact effect
        _, _, _, _, b1 = make_control(23, 1, 5000, 3, 6, 1.5, c, 1.0, sample_seed=1, geom_seed=7)
        _, _, _, _, b2 = make_control(23, 1, 9000, 3, 6, 1.5, c, 1.0, sample_seed=2, geom_seed=7)
        assert b1 == b2 == b0, (b0, b1, b2)
    print("test_control_effect_tuning_exact_and_sample_invariant: OK")


def test_power_ceiling_guard():
    """The R=8 ceiling bug: max Wilson LCB at det=R is R/(R+z^2); for R=8,z=1.64 it is ~0.748<0.8,
    so power_ok could NEVER be True. assert_power_feasible must REJECT (R=8,beta=0.2) and ACCEPT
    R=30; wilson_lcb(R,R) must equal the ceiling."""
    z = 1.64
    assert abs(wilson_lcb(8, 8, z) - 8 / (8 + z * z)) < 1e-9
    assert wilson_lcb(8, 8, z) < 0.8, wilson_lcb(8, 8, z)     # the bug: 8/8 still < 0.8
    try:
        assert_power_feasible(8, 0.2); raise AssertionError("R=8 should be infeasible")
    except ValueError:
        pass
    assert_power_feasible(30, 0.2)                            # ceiling 0.918 >= 0.8 -> OK
    assert wilson_lcb(28, 30, z) >= 0.8, wilson_lcb(28, 30, z)
    print("test_power_ceiling_guard: OK")


if __name__ == "__main__":
    test_fingerprint_mismatch_abstains()
    test_exact_lookup_and_scope()
    test_lookup_conservative_and_monotone()
    test_control_effect_tuning_exact_and_sample_invariant()
    test_power_ceiling_guard()
    print("ALL POWER-CERTIFICATE TESTS PASSED")
