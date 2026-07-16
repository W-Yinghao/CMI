"""Tests for the synthetic Bayes leakage-erasure oracle (Delta_Y*, Delta_D*, constrained search)."""
import numpy as np
import pytest
from tos_cmi.eval.erasure_oracle_bayes import (bayes_deltas, oracle_search, axis_projector,
                                               pareto_front, oracle_regret, safety_violation)


def _separable(C=2, ndom=3, Dm=4):
    mu = np.zeros((C, ndom, Dm))
    for y in range(C):
        for d in range(ndom):
            mu[y, d, 0] = y * 3.0          # task axis
            mu[y, d, 1] = d * 3.0          # pure conditional-subject axis
    return mu, np.ones(C) / C, np.ones((C, ndom)) / ndom


def test_pure_subject_axis_is_task_safe_and_removes_leakage():
    mu, py, pdy = _separable()
    r = bayes_deltas(mu, 1.0, py, pdy, axis_projector([1], 4), n_mc=8000)
    assert r["delta_Y"] == pytest.approx(0.0, abs=0.02)     # deleting subject axis costs ~0 task info
    assert r["delta_D"] > 0.3                                # and removes real conditional subject leakage


def test_task_axis_deletion_hurts_task():
    mu, py, pdy = _separable()
    r = bayes_deltas(mu, 1.0, py, pdy, axis_projector([0], 4), n_mc=8000)
    assert r["delta_Y"] > 0.1                                # deleting the task axis loses task info


def test_oracle_finds_subject_axis_when_separable():
    mu, py, pdy = _separable()
    res = oracle_search(mu, 1.0, py, pdy, k=2, delta=0.02, gamma_D=0.05, n_mc=8000)
    assert res["oracle"]["axes"] == [1] and not res["oracle_is_identity"]


def test_oracle_returns_identity_when_entangled():
    C, ndom, Dm = 2, 3, 4
    mu = np.zeros((C, ndom, Dm))
    for y in range(C):
        for d in range(ndom):
            mu[y, d, 0] = y * 3.0 + d * 3.0                  # task + subject entangled on one axis
    res = oracle_search(mu, 1.0, np.ones(C) / C, np.ones((C, ndom)) / ndom,
                        k=2, delta=0.02, gamma_D=0.05, n_mc=8000)
    assert res["oracle_is_identity"]                         # no safe deletion -> identity is correct


def test_pareto_and_regret():
    mu, py, pdy = _separable()
    res = oracle_search(mu, 1.0, py, pdy, k=2, delta=1.0, gamma_D=0.0, n_mc=6000)
    pf = pareto_front(res["frontier"])
    assert len(pf) >= 1 and all(pf[i]["delta_D"] <= pf[i + 1]["delta_D"] for i in range(len(pf) - 1))
    assert oracle_regret(0.8, 0.5) == pytest.approx(0.3)
    assert safety_violation(0.05, 0.02) == pytest.approx(0.03)
    assert safety_violation(0.01, 0.02) == 0.0
