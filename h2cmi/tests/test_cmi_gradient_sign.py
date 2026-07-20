"""P0-2: the encoder penalty must point in the leakage-REDUCING direction (no GRL).

With the penalty +lambda*(H_ref - CE) and critics frozen, a gradient-descent step on the
encoder latent z must INCREASE the critic's conditional cross-entropy (i.e. make the
domain harder to predict from z). A GRL would flip this; the estimator has no grl option.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from h2cmi.domains import DomainDAG, DomainFactor, DomainLabels
from h2cmi.cmi.hierarchical import HierarchicalCMI
from h2cmi.config import CMIConfig


def test_penalty_reduces_leakage():
    torch.manual_seed(0)
    n, n_dom, n_cls = 400, 3, 2
    rng = np.random.default_rng(0)
    y = rng.integers(0, n_cls, n)
    d = rng.integers(0, n_dom, n)
    Z = torch.zeros(n, 12)
    Z[torch.arange(n), torch.as_tensor(y)] = 2.0
    Z[torch.arange(n), 2 + torch.as_tensor(d)] = 1.5          # planted domain leakage
    Z += 0.3 * torch.randn(n, 12)

    dag = DomainDAG([DomainFactor("domain", n_dom, (), "invariant", 0.02)])
    dom = DomainLabels(dag, d.reshape(-1, 1))
    hcmi = HierarchicalCMI(12, n_cls, dag, dom, y, CMIConfig())
    yb = torch.as_tensor(y, dtype=torch.long)
    lev, pk = hcmi.batch_context(dom, np.arange(n))

    # fit the critic to near-optimality on the fixed z
    optc = torch.optim.Adam(hcmi.parameters(), lr=2e-3)
    for _ in range(150):
        optc.zero_grad(); hcmi.critic_loss(Z, yb, lev, pk).backward(); optc.step()

    def critic_ce(zz):
        with torch.no_grad():
            return float(F.cross_entropy(hcmi.critics["domain"](zz, yb, pk["domain"]),
                                         lev["domain"]))

    ce_before = critic_ce(Z)
    # freeze critics; take ONE gradient-descent step on z to minimise the penalty
    for p in hcmi.parameters():
        p.requires_grad_(False)
    z = Z.clone().requires_grad_(True)
    pen = hcmi.estimate(z, yb, lev, pk)["domain"]            # = H_ref - CE
    (grad,) = torch.autograd.grad(pen, z)
    z_step = (z - 5.0 * grad).detach()                       # descent on the penalty
    ce_after = critic_ce(z_step)

    assert ce_after > ce_before, f"penalty did not reduce leakage: {ce_before:.3f}->{ce_after:.3f}"
    # and there is genuinely no GRL knob to misconfigure
    import inspect
    assert "grl" not in inspect.signature(hcmi.estimate).parameters


if __name__ == "__main__":
    test_penalty_reduces_leakage()
    print("test_cmi_gradient_sign PASSED")
