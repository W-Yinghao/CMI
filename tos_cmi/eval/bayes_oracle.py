"""Exact-ish Bayes oracle for the CONDITIONAL task safety of a deletion (Phase 1.2.4).

The synthetic worlds are known Gaussian mixtures z | y,d ~ N(mu_{y,d}, sigma^2 I) with uniform
p(y), p(d). For a deployed projector P, the kept component in intrinsic coordinates is u = A z
with A = B_R^T (I-P) (B_R an orthobasis of range(I-P)). The quantity the task-risk gate is
trying to bound is the INCREMENTAL label information of the deleted component given the kept one,

    Delta_Y*(P) = H(Y|U) - H(Y|Z) = I(Y ; deleted | kept)  >= 0 .

This module computes it from the (empirical) mixture posteriors, with NO probe training -- the
ground truth against which the learned nested task gate is calibrated. The crucial subtlety it
settles: a geometrically task-orthogonal nuisance can still be conditionally task-useful through
a SHARED domain factor (explaining-away), so Delta_Y* can be > 0 even when I(Y;deleted)=0; in
that case the gate SHOULD refuse and there is no estimator bias to remove.

`bayes_conditional_task_delta` returns Delta_Y* (and H(Y|U), H(Y|Z)); `mixture_params` estimates
mu_{y,d}, sigma^2 empirically (cells with too few samples are dropped)."""
from __future__ import annotations
import numpy as np


def mixture_params(Z, y, d, n_cls, n_dom, min_cell=5):
    """Empirical mu_{y,d} [n_cls,n_dom,D], present-mask, pooled isotropic sigma^2."""
    Z = np.asarray(Z, dtype=np.float64)
    D = Z.shape[1]
    mu = np.zeros((n_cls, n_dom, D)); present = np.zeros((n_cls, n_dom), bool)
    sse = 0.0; cnt = 0
    for c in range(n_cls):
        for e in range(n_dom):
            m = (y == c) & (d == e)
            if m.sum() >= min_cell:
                Zc = Z[m]; mu[c, e] = Zc.mean(0); present[c, e] = True
                sse += ((Zc - mu[c, e]) ** 2).sum(); cnt += Zc.size
    sigma2 = sse / max(cnt, 1)
    return mu, present, float(sigma2)


def _cond_entropy(X, mu_cells, present, cov_inv, logdet, py):
    """H(Y|X) for x = (linear image of z); mu_cells [n_cls,n_dom,m], cov_inv [m,m] shared.
    p(y|x) ∝ p(y) * mean_d 1[present] N(x; mu_{y,d}, cov). Constants in |cov| cancel over y."""
    n_cls, n_dom = present.shape
    N = X.shape[0]
    logpyx = np.full((N, n_cls), -np.inf)
    for c in range(n_cls):
        comp = []
        for e in range(n_dom):
            if present[c, e]:
                diff = X - mu_cells[c, e]                      # [N,m]
                maha = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
                comp.append(-0.5 * maha)
        if comp:
            stak = np.stack(comp, 1)                           # [N, n_present]
            m = stak.max(1, keepdims=True)
            lse = (m[:, 0] + np.log(np.exp(stak - m).mean(1)))  # logmeanexp over present d
            logpyx[:, c] = np.log(py[c] + 1e-12) + lse
    mx = logpyx.max(1, keepdims=True)
    p = np.exp(logpyx - mx); p /= p.sum(1, keepdims=True)
    H = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)
    return float(H.mean())


def bayes_conditional_task_delta(Z, y, d, n_cls, n_dom, P, min_cell=5):
    """Delta_Y* = H(Y|U) - H(Y|Z) = I(Y; deleted | kept) under the (empirical) mixture, where
    U = B_R^T (I-P) Z is the kept component in intrinsic coords. Returns dict(delta, H_YU, H_YZ)."""
    Z = np.asarray(Z, dtype=np.float64); y = np.asarray(y); d = np.asarray(d)
    Dd = Z.shape[1]
    mu, present, s2 = mixture_params(Z, y, d, n_cls, n_dom, min_cell)
    py = np.array([(y == c).mean() for c in range(n_cls)])

    # H(Y|Z): full space, cov = s2 I
    covZ_inv = np.eye(Dd) / s2
    H_YZ = _cond_entropy(Z, mu, present, covZ_inv, 0.0, py)

    # U = A Z, A = B_R^T (I-P); cov_U = s2 A A^T
    Ur, sr, _ = np.linalg.svd(np.eye(Dd) - P)
    B_R = Ur[:, sr > 1e-8]                                     # [D, D-k]
    A = B_R.T @ (np.eye(Dd) - P)                              # [D-k, D]
    U = Z @ A.T
    muU = np.einsum("md,ced->cem", A, mu)                     # [n_cls,n_dom,m] = A mu_{y,d}
    covU = s2 * (A @ A.T)
    covU_inv = np.linalg.pinv(covU)
    H_YU = _cond_entropy(U, muU, present, covU_inv, 0.0, py)
    return {"delta": H_YU - H_YZ, "H_YU": H_YU, "H_YZ": H_YZ}
