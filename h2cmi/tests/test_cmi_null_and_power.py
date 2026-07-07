"""P0-2/P0-7: the cross-fitted CMI estimator must have null calibration AND power.

Construct latents directly:  Z = class_signal(Y) + alpha * domain_signal(D) + noise.
- alpha = 0  -> no conditional leakage -> excess above the permutation null ~ 0 (type-I).
- alpha up   -> I_hat increases monotonically and the excess becomes clearly positive (power).
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from h2cmi.domains import DomainDAG, DomainFactor, DomainLabels
from h2cmi.eval.leakage import crossfit_conditional_leakage


def _make(alpha, n=600, n_dom=3, n_cls=2, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_cls, n)
    d = rng.integers(0, n_dom, n)
    Z = np.zeros((n, 10), dtype=np.float32)
    Z[np.arange(n), y] = 2.0                          # class signal (dims 0..1)
    Z[np.arange(n), 2 + d] += alpha                   # domain signal (dims 2..4)
    Z += 0.4 * rng.standard_normal(Z.shape).astype(np.float32)
    dag = DomainDAG([DomainFactor("domain", n_dom, (), "invariant", 0.02)])
    dom = DomainLabels(dag, d.reshape(-1, 1))
    leak = crossfit_conditional_leakage(Z, y, dom, dag, n_cls, n_perm=12, seed=seed)
    return leak["domain"]


def test_null_calibration_and_power():
    r0 = _make(0.0, seed=0)
    r1 = _make(1.0, seed=0)
    r2 = _make(2.5, seed=0)
    # monotone power in the signed estimate
    assert r0["I_hat"] < r1["I_hat"] < r2["I_hat"], (r0["I_hat"], r1["I_hat"], r2["I_hat"])
    # null: no excess above the refit-permutation 95% quantile
    assert r0["excess"] < 0.3, f"type-I excess too large: {r0['excess']}"
    # power: strong planted leakage clears the null
    assert r2["excess"] > 0.3, f"failed to detect strong leakage: {r2['excess']}"


if __name__ == "__main__":
    test_null_calibration_and_power()
    print("test_cmi_null_and_power PASSED")
