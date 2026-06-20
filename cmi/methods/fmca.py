"""FMCA / Normalized-Cross-Density dependence as a differentiable log-det surrogate — ROUTE 2.

Reimplemented from the PUBLIC NCD equations (Hu & Principe 2022, arXiv:2212.04631) — NOT vendored
from DY9910/MFMC or bohu615 (both all-rights-reserved). Matches the author's own prior implementation
(eeg2025/ICML_2026/loss.py::FMCA): COST = logdet(RFG) - logdet(RF) - logdet(RG) with eps=1e-5; here we
use slogdet (more stable) and FLIP the sign (T_K = -1/2*COST >= 0) because we SUPPRESS dependence
(their FMCA was a contrastive CSSL loss that MAXIMIZED view agreement). Richer variants exist in that
file (FMCA_exp eigenvalue form; return_cost_trace smoothed-trace surrogate) if the log-det destabilizes.

Total statistical dependence between continuous Phi [B,K] and a one-hot side V_oh [B,n]:
  T_K = -1/2 ( logdet R_joint - logdet R_F - logdet R_G ),  R_joint = [[R_F, P],[P^T, R_G]]
backprops through slogdet (no eigendecomposition in the hot loop). We use Phi = Z directly (closed-form
*linear* FMCA = canonical correlations between Z and the class indicators); minimizing T_K removes
that dependence. NOTE: T_K is a multivariate-dependence functional, equal to Shannon MI only in the
Gaussian case — so it is NOT additive in nats, which is exactly why the chain-rule route is fragile.

Used ONLY as appendix ablations + a leakage-spectrum probe; Route 1 (LPC-CMI posterior-KL) stays main:
  fmca_chain : min T_K(Z, S=(D,Y))           -> reproduces Y-ERASURE (the documented failure)
  fmca_diff  : T_K(Z,S) - mu*T_K(Z,Y)        -> corrected-in-expectation but high-variance/sign-unstable
  fmca_strat : mean_y T_K(Z, D | Y=y)        -> per-class; cannot erase Y by construction
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

FMCA_METHODS = {"fmca_chain", "fmca_diff", "fmca_strat"}


def fmca_logdet(Phi, V_oh, eps=1e-4):
    """T_K(>=0) between Phi [B,K] and one-hot V_oh [B,n]. Minimize to reduce dependence. float64 inside."""
    Phi = Phi.double(); V = V_oh.double(); B = Phi.shape[0]
    Fc = Phi - Phi.mean(0, keepdim=True)
    Gc = V - V.mean(0, keepdim=True)
    K, n = Fc.shape[1], Gc.shape[1]
    IK = torch.eye(K, dtype=torch.float64, device=Phi.device)
    In = torch.eye(n, dtype=torch.float64, device=Phi.device)
    RF = Fc.t() @ Fc / B + eps * IK
    RG = Gc.t() @ Gc / B + eps * In          # rank n-1 after centering -> jitter mandatory
    P = Fc.t() @ Gc / B
    Rj = torch.cat([torch.cat([RF, P], 1), torch.cat([P.t(), RG], 1)], 0)
    r = (torch.linalg.slogdet(Rj)[1] - torch.linalg.slogdet(RF)[1] - torch.linalg.slogdet(RG)[1])
    return (-0.5 * r).float()


@torch.no_grad()
def fmca_spectrum(Phi, V_oh, eps=1e-4):
    """Functional canonical correlations rho_k^2 (descending) — for the leakage-spectrum probe."""
    Phi = Phi.double(); V = V_oh.double(); B = Phi.shape[0]
    Fc = Phi - Phi.mean(0, keepdim=True); Gc = V - V.mean(0, keepdim=True)
    K, n = Fc.shape[1], Gc.shape[1]
    RF = Fc.t() @ Fc / B + eps * torch.eye(K, dtype=torch.float64, device=Phi.device)
    RG = Gc.t() @ Gc / B + eps * torch.eye(n, dtype=torch.float64, device=Phi.device)
    P = Fc.t() @ Gc / B
    M = torch.linalg.inv(RF) @ P @ torch.linalg.inv(RG) @ P.t()
    ev = torch.linalg.eigvalsh(M).real.clamp(0, 1 - 1e-6)
    return torch.sort(ev, descending=True).values


def fmca_reg(method, z, y, d, n_cls, n_dom, mu=1.0):
    """Route-2 regularizer on the representation z (closed-form linear FMCA)."""
    S = d * n_cls + y
    if method == "fmca_chain":
        return fmca_logdet(z, F.one_hot(S, n_dom * n_cls).float())
    if method == "fmca_diff":
        return (fmca_logdet(z, F.one_hot(S, n_dom * n_cls).float())
                - mu * fmca_logdet(z, F.one_hot(y, n_cls).float())).clamp(min=0)
    if method == "fmca_strat":
        tot = z.new_zeros(())
        for c in range(n_cls):
            m = y == c
            if m.sum() < n_dom + 2 or len(torch.unique(d[m])) < 2:
                continue
            tot = tot + m.float().mean() * fmca_logdet(z[m], F.one_hot(d[m], n_dom).float())
        return tot
    raise ValueError(method)
