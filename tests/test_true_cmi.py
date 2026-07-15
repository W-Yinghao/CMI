"""Tests for synthetic/true_cmi.py — the numerical ground-truth CMI.

The ground-truth Monte-Carlo machinery is exercised against SMALL DISCRETE
distributions where I(Z;D|Y) has a closed form (computed independently by
``discrete_cmi_exact``), plus the known Gaussian-mixture DGP. We verify:
  * an independence case where the true CMI == 0 (within MC SE);
  * a deterministic-dependence case with a known positive value (log 2);
  * an intermediate case where MC matches the closed form;
  * MC standard error shrinks ~1/sqrt(n);
  * determinism (same seed -> identical value).
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "synthetic"))
from true_cmi import (  # noqa: E402
    DGP,
    discrete_cmi_exact,
    discrete_cmi_mc,
    true_cmi_dgp,
)

LOG2 = math.log(2.0)


# --------------------------------------------------------------------------- joints
def _independence_joint():
    """P(z,d,y) = P(y) P(z|y) P(d|y)  =>  Z _|_ D | Y  =>  I(Z;D|Y) = 0."""
    nz, nd, ny = 3, 2, 2
    py = np.array([0.4, 0.6])
    pz_y = np.array([[0.2, 0.6], [0.5, 0.1], [0.3, 0.3]])  # [z, y]
    pd_y = np.array([[0.7, 0.25], [0.3, 0.75]])            # [d, y]
    p = np.zeros((nz, nd, ny))
    for y in range(ny):
        for z in range(nz):
            for d in range(nd):
                p[z, d, y] = py[y] * pz_y[z, y] * pd_y[d, y]
    return p


def _deterministic_joint():
    """D uniform binary, independent of uniform binary Y, and Z == D.
    Then I(Z;D|Y) = H(D|Y) = H(D) = log 2 exactly."""
    p = np.zeros((2, 2, 2))  # [z, d, y]
    for y in range(2):
        for d in range(2):
            p[d, d, y] = 0.25   # z == d
    return p


def _noisy_copy_joint(q=0.2):
    """Z is a noisy copy of a uniform binary D (flip prob q), Y uniform binary,
    D _|_ Y. Non-degenerate (positive-variance) intermediate CMI."""
    p = np.zeros((2, 2, 2))  # [z, d, y]
    for y in range(2):
        for d in range(2):
            base = 0.5 * 0.5   # P(y) P(d|y)
            for z in range(2):
                pzd = (1 - q) if z == d else q
                p[z, d, y] = base * pzd
    return p


# ------------------------------------------------------------------------ closed form
def test_independence_true_cmi_is_zero():
    p = _independence_joint()
    exact = discrete_cmi_exact(p)
    assert abs(exact) < 1e-9, f"closed-form independence CMI should be 0, got {exact}"
    mc = discrete_cmi_mc(p, seed=0, n_samples=200_000)
    tol = max(3.0 * mc["mc_se"], 1e-6)
    assert abs(mc["true_cmi_nats"]) <= tol, f"MC CMI {mc['true_cmi_nats']} not within {tol} of 0"


def test_deterministic_dependence_known_positive_value():
    p = _deterministic_joint()
    exact = discrete_cmi_exact(p)
    assert abs(exact - LOG2) < 1e-9, f"closed-form should be log2={LOG2}, got {exact}"
    mc = discrete_cmi_mc(p, seed=0, n_samples=100_000)
    # deterministic channel => every sample contributes exactly log2 => SE 0.
    assert abs(mc["true_cmi_nats"] - LOG2) < 1e-9


def test_intermediate_mc_matches_closed_form():
    p = _noisy_copy_joint(q=0.2)
    exact = discrete_cmi_exact(p)
    assert exact > 0.05  # genuinely positive, non-degenerate
    mc = discrete_cmi_mc(p, seed=1, n_samples=400_000)
    assert mc["mc_se"] > 0  # non-zero variance regime
    assert abs(mc["true_cmi_nats"] - exact) <= 4.0 * mc["mc_se"], (
        f"MC {mc['true_cmi_nats']} vs exact {exact} beyond 4 SE ({mc['mc_se']})"
    )


# ----------------------------------------------------------------------- MC behaviour
def test_mc_se_shrinks_as_one_over_sqrt_n():
    p = _noisy_copy_joint(q=0.25)
    se_n = discrete_cmi_mc(p, seed=2, n_samples=100_000)["mc_se"]
    se_4n = discrete_cmi_mc(p, seed=2, n_samples=400_000)["mc_se"]
    ratio = se_n / se_4n
    # 4x samples -> ~2x smaller SE.
    assert 1.7 < ratio < 2.3, f"SE ratio {ratio} not ~2 for a 4x sample increase"


def test_reported_se_present_and_positive_for_stochastic_case():
    p = _noisy_copy_joint(q=0.3)
    mc = discrete_cmi_mc(p, seed=0, n_samples=50_000)
    assert set(mc.keys()) == {"true_cmi_nats", "mc_se", "n_samples"}
    assert mc["n_samples"] == 50_000
    assert mc["mc_se"] > 0


# ------------------------------------------------------------------------ determinism
def test_determinism_same_seed_discrete():
    p = _noisy_copy_joint(q=0.2)
    a = discrete_cmi_mc(p, seed=7, n_samples=120_000)
    b = discrete_cmi_mc(p, seed=7, n_samples=120_000)
    assert a["true_cmi_nats"] == b["true_cmi_nats"]
    assert a["mc_se"] == b["mc_se"]


def test_determinism_different_seed_changes_value():
    p = _noisy_copy_joint(q=0.2)
    a = discrete_cmi_mc(p, seed=1, n_samples=80_000)
    b = discrete_cmi_mc(p, seed=2, n_samples=80_000)
    assert a["true_cmi_nats"] != b["true_cmi_nats"]


# -------------------------------------------------------------------- Gaussian DGP path
def test_dgp_default_positive_with_reported_se():
    r = true_cmi_dgp(DGP(), seed=0, n_samples=400_000)
    assert set(r.keys()) == {"true_cmi_nats", "mc_se", "n_samples"}
    assert r["true_cmi_nats"] > 0.5      # default DGP has strong domain leakage
    assert 0.0 < r["mc_se"] < 0.01       # small, reported standard error
    r2 = true_cmi_dgp(DGP(), seed=0, n_samples=400_000)
    assert r["true_cmi_nats"] == r2["true_cmi_nats"]  # determinism


def test_dgp_no_leakage_is_zero():
    """A DGP with identical spurious flip rate across domains and zero style
    scale has p(x|d,y) independent of d -> I(X;D|Y) = 0."""
    same_e = 0.2
    dgp = DGP(
        src=((0.15, same_e), (0.38, same_e), (0.62, same_e), (0.85, same_e)),
        style_scale=0.0,
    )
    r = true_cmi_dgp(dgp, seed=0, n_samples=300_000)
    tol = max(3.0 * r["mc_se"], 1e-6)
    assert abs(r["true_cmi_nats"]) <= tol, (
        f"no-leakage DGP CMI {r['true_cmi_nats']} not within {tol} of 0"
    )


def test_dgp_style_scale_monotone_increases_cmi():
    """More separated pure-style means -> strictly more domain leakage."""
    base = DGP(src=((0.5, 0.2),) * 4, style_scale=0.0)   # only style varies below
    lo = true_cmi_dgp(DGP(src=((0.5, 0.2),) * 4, style_scale=0.5), seed=0, n_samples=300_000)
    hi = true_cmi_dgp(DGP(src=((0.5, 0.2),) * 4, style_scale=2.0), seed=0, n_samples=300_000)
    z = true_cmi_dgp(base, seed=0, n_samples=300_000)
    assert z["true_cmi_nats"] <= lo["true_cmi_nats"] + 3 * lo["mc_se"]
    assert lo["true_cmi_nats"] < hi["true_cmi_nats"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
