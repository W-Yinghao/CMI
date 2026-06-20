"""Oracle ordering on a planted diagonal latent shift: with more oracle information the
held-out change-of-variable evidence does not get worse.

  supervised (labels, cross-fit)  >=  unsupervised  (held-out density NLL gain)
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from h2cmi.config import DensityConfig, TTAConfig
from h2cmi.density.student_t_mixture import ClassConditionalDensity
from h2cmi.tta.class_conditional import ClassConditionalTTA
from h2cmi.tta.oracles import crossfit_supervised_gain


def _planted(d=8, K=3):
    dens = ClassConditionalDensity(d, K, DensityConfig(n_components=1, cov_rank=2, df=8.0))
    with torch.no_grad():
        dens.mu.zero_()
        for c in range(K):
            dens.mu[c, 0, c % d] = 4.0
        dens.log_s.fill_(-1.0)
    return dens


def test_supervised_not_worse_than_unsupervised():
    torch.manual_seed(0)
    d, K = 8, 3
    dens = _planted(d, K)
    pi_S = np.full(K, 1.0 / K)
    rng = np.random.default_rng(0)
    yt = rng.integers(0, K, 500)
    zs = dens.mu[yt, 0] + 0.3 * torch.randn(500, d)
    a_true = torch.linspace(1.6, 0.5, d)
    b_true = 0.8 * torch.ones(d)
    U = ((zs - b_true) / a_true).detach()                       # recoverable diagonal shift
    tta = ClassConditionalTTA(dens, pi_S, TTAConfig(em_iters=25), K)
    g_unsup = tta._crossfit_evidence_gain(U)
    g_sup = crossfit_supervised_gain(tta, U, yt)
    assert g_sup > 0, f"supervised oracle found no recoverable evidence (g={g_sup:.3f})"
    assert g_sup >= g_unsup - 0.5, f"supervised < unsupervised (sup={g_sup:.3f}, unsup={g_unsup:.3f})"


if __name__ == "__main__":
    test_supervised_not_worse_than_unsupervised()
    print("test_oracle_ordering PASSED")
