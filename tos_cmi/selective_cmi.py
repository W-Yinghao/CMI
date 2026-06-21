"""Selective conditional-leakage penalty:  lambda * I( P_N Z ; D | Y ).

This applies leakage pressure ONLY to the nuisance component `P_N Z` chosen by
`SubspaceSelector`. The task-entangled component `(I - P_N) Z` is never pushed toward
invariance, so the classifier keeps whatever label-bearing domain structure lives there.

The estimator is the same label-prior-corrected posterior-KL plug-in used by the AAAI
core (`cmi.methods.regularizers.DomainPosteriors.reg("lpc_prior")`):

    I_hat(Z_N; D | Y) = E_i KL( q_psi(D | z_{N,i}, y_i) || pi_{y_i}(D) ),

tight at the Step-A critic optimum. We reimplement the small critic here (rather than
import `cmi`) so the package stays self-contained and testable without the MOABB /
braindecode import chain; see INTEGRATION.md for wiring the existing `DomainPosteriors`
onto `P_N Z` inside `cmi/train/trainer.py` instead.

When the selector returns identity (no safe nuisance subspace), `penalty()` is exactly 0
and `refresh()` leaves the critic untouched -- the method is a no-op, by design.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FisherConfig, SubspaceConfig, PenaltyConfig
from .subspace import SubspaceSelector


def label_prior(y, d, n_cls, n_dom, alpha=1.0, mode="empirical") -> np.ndarray:
    """pi_y(d) = p(d|y), Laplace-smoothed. mode:
      empirical : trial-weighted p(d|y)
      subject   : each domain counted once per class it appears in (presence)
      effective : uniform over the domains present in class y (class-balanced sampler)
    Matches cmi.methods.regularizers.{empirical,subject,effective}_priors[0]."""
    y = np.asarray(y); d = np.asarray(d)
    counts = np.zeros((n_cls, n_dom))
    for yi, di in zip(y, d):
        counts[yi, di] += 1
    if mode == "subject":
        counts = (counts > 0).astype(float)
        return (counts + alpha) / (counts.sum(1, keepdims=True) + alpha * n_dom)
    if mode == "effective":
        present = (counts > 0).astype(float)
        return (present + alpha) / (present.sum(1, keepdims=True) + alpha * n_dom)
    return (counts + alpha) / (counts.sum(1, keepdims=True) + alpha * n_dom)


class ConditionalDomainCritic(nn.Module):
    """q_psi(D | z, y): MLP over [z, onehot(y)] -> domain logits."""
    def __init__(self, z_dim, n_cls, n_dom, hidden=128):
        super().__init__()
        self.n_cls, self.n_dom = n_cls, n_dom
        self.net = nn.Sequential(nn.Linear(z_dim + n_cls, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_dom))

    def forward(self, z, y):
        y_oh = F.one_hot(y, self.n_cls).float()
        return self.net(torch.cat([z, y_oh], 1))


class SelectivePenalty(nn.Module):
    """Subspace selector + conditional-domain critic, exposing Step-A and Step-B hooks.

        sp = SelectivePenalty(z_dim, n_cls, n_dom, priors=(y_all, d_all))
        sp.refresh(Z, y, d)                       # recompute P_N (no grad)
        loss_A = sp.posterior_loss(Z.detach(), y, d)   # Step A: fit critic on P_N Z
        pen    = sp.penalty(Z, y, d)              # Step B: lambda * I(P_N Z; D|Y), grad to Z
    """
    def __init__(self, z_dim, n_cls, n_dom, priors,
                 fcfg: Optional[FisherConfig] = None,
                 scfg: Optional[SubspaceConfig] = None,
                 pcfg: Optional[PenaltyConfig] = None,
                 device="cpu"):
        super().__init__()
        self.pcfg = pcfg or PenaltyConfig()
        self.selector = SubspaceSelector(z_dim, n_cls, n_dom, fcfg, scfg, device=device)
        self.critic = ConditionalDomainCritic(z_dim, n_cls, n_dom)
        y_all, d_all = priors
        pi_y = label_prior(y_all, d_all, n_cls, n_dom, mode=self.pcfg.prior_mode)
        self.register_buffer("log_pi_y", torch.log(torch.tensor(pi_y, dtype=torch.float32)))
        self.to(device)

    @property
    def is_identity(self) -> bool:
        return self.selector.is_identity

    def refresh(self, Z, y, d):
        return self.selector.refresh(Z, y, d)

    def posterior_loss(self, Z_detached, y, d):
        """Step A: fit q(D | P_N Z, Y) to the current (projected) representation.
        Returns 0 when the subspace is identity (nothing to fit)."""
        if self.is_identity:
            return Z_detached.new_zeros(())
        zn = self.selector.project(Z_detached)
        return F.cross_entropy(self.critic(zn, y), d)

    def cmi(self, Z, y, d):
        """Signed plug-in I_hat(P_N Z; D | Y) = E KL(q(D|z_N,y) || pi_y(D)). Grad flows to Z;
        critic should be frozen (Step B)."""
        if self.is_identity:
            return Z.new_zeros(())
        zn = self.selector.project(Z)
        logits = self.critic(zn, y)
        logq = F.log_softmax(logits, dim=1)
        log_prior = self.log_pi_y[y]                       # [B, n_dom]
        kl = (logq.exp() * (logq - log_prior)).sum(1)
        return kl.mean()

    def penalty(self, Z, y, d, lam: Optional[float] = None):
        """Step B: lambda * I_hat(P_N Z; D|Y). 0 under identity."""
        lam = self.pcfg.lam if lam is None else lam
        return lam * self.cmi(Z, y, d)
