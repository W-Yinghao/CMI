"""P0-3 / P0-4: TTA change-of-variables evidence, identity rollback, low-rank gradient."""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from h2cmi.config import DensityConfig, TTAConfig
from h2cmi.density.student_t_mixture import ClassConditionalDensity
from h2cmi.tta.class_conditional import ClassConditionalTTA, Transform


def _planted_density(d=8, K=3, sep=4.0):
    dens = ClassConditionalDensity(d, K, DensityConfig(n_components=1, cov_rank=2, df=8.0))
    with torch.no_grad():
        dens.mu.zero_()
        for c in range(K):
            dens.mu[c, 0, c % d] = sep                     # well-separated class means
        dens.log_s.fill_(-1.0)                             # tight-ish components
    return dens


def _acc(proba, y):
    return float((proba.argmax(1) == y).mean())


def test_change_of_variables_helps_under_diagonal_shift():
    torch.manual_seed(0)
    d, K = 8, 3
    dens = _planted_density(d, K)
    pi_S = np.full(K, 1.0 / K)
    rng = np.random.default_rng(0)
    yt = rng.integers(0, K, 400)
    zs = dens.mu[yt, 0] + 0.3 * torch.randn(400, d)        # points in SOURCE geometry
    a_true = torch.linspace(1.5, 0.6, d)                   # known diagonal shift
    b_true = 0.7 * torch.ones(d)
    U = (zs - b_true) / a_true                             # observed target (shifted)

    tta = ClassConditionalTTA(dens, pi_S, TTAConfig(em_iters=25), K)
    res = tta.fit(U, pseudo_labels=yt)
    assert res.adapted, "should adapt under a real recoverable shift"
    assert res.diagnostics["crossfit_evidence_gain"] > 0

    # change-of-variables recovery: the transform should pull each class's target mean back
    # toward its SOURCE mean (identity leaves them shifted). This is the right test even
    # when separation is large enough that identity classification is already perfect.
    z = res.apply(U).detach()

    def mean_dist_to_source(pts):
        tot = 0.0
        for c in range(K):
            m = yt == c
            tot += float(torch.norm(pts[m].mean(0) - dens.mu[c, 0]))
        return tot / K

    assert mean_dist_to_source(z) < 0.5 * mean_dist_to_source(U), \
        (mean_dist_to_source(U), mean_dist_to_source(z))


@torch.no_grad()
def _sample_from_density(dens, y, d):
    """Sample a true null target FROM the model (no fittable mean/cov shift)."""
    out = []
    for c in y:
        mu, L = dens.mu[c, 0], dens.L[c, 0]
        diag = torch.sqrt(F.softplus(dens.log_s[c, 0]) + dens.eig_floor)
        out.append(mu + L @ torch.randn(L.shape[1]) + diag * torch.randn(d))
    return torch.stack(out)


def test_identity_rollback_under_no_shift():
    torch.manual_seed(0)
    d, K = 8, 3
    dens = _planted_density(d, K)
    pi_S = np.full(K, 1.0 / K)
    rng = np.random.default_rng(1)
    yt = rng.integers(0, K, 800)
    U = _sample_from_density(dens, yt, d)                   # drawn from the model => no shift
    tta = ClassConditionalTTA(dens, pi_S, TTAConfig(em_iters=25, min_heldout_evidence=0.0), K)
    res = tta.fit(U, pseudo_labels=yt)
    # no held-out evidence improvement -> identity fallback
    assert not res.adapted, f"adapted under no shift (gain={res.diagnostics.get('crossfit_evidence_gain')})"
    assert res.diagnostics.get("reason") == "no_heldout_evidence"


def test_lowrank_transform_has_gradient_at_init():
    T = Transform(8, "lowrank_affine", lowrank=4)
    u = torch.randn(16, 8)
    (T.apply(u) ** 2).sum().backward()
    assert T.U.grad.norm() > 0 and T.V.grad.norm() > 0, "low-rank transform stuck at identity"
    assert T.b.grad.norm() > 0


if __name__ == "__main__":
    test_change_of_variables_helps_under_diagonal_shift()
    test_identity_rollback_under_no_shift()
    test_lowrank_transform_has_gradient_at_init()
    print("test_tta PASSED")
