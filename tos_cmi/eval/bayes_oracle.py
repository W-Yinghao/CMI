"""Exact Bayes oracle for the CONDITIONAL task safety of a deletion (Phase 1.2.4/1.2.5).

The synthetic worlds are KNOWN Gaussian mixtures z | y,d ~ N(mu_{y,d}, sigma^2 I), with priors
p(y), p(d|y). For a deployed projector P, the kept component in intrinsic coordinates is
u = A z, A = B_R^T (I-P) (B_R an orthobasis of range(I-P)). The quantity the task-risk gate is
trying to bound is the INCREMENTAL label information of the deleted component given the kept one,

    Delta_Y*(P) = H(Y|U) - H(Y|Z) = I(Y ; deleted | kept)  >= 0 .

`bayes_conditional_task_delta` (Phase 1.2.5) computes it from the TRUE params on an INDEPENDENT
Monte-Carlo draw (not a same-sample empirical plug-in) and returns a bootstrap CI, so it is a
proper certification. `classify_safety` turns the CI into SAFE / UNSAFE / BAYES_AMBIGUOUS vs a
threshold delta_Y. `mixture_params` (empirical fallback for data without known truth) is kept
but flagged as a plug-in (do NOT use for certification).

Settles the Phase 1.2.3 'bias' question: a geometrically task-orthogonal nuisance can still be
conditionally task-useful through a SHARED domain factor (explaining-away), so Delta_Y* can be
>> 0 even when I(Y;deleted)=0; the gate SHOULD refuse and there is no estimator bias to remove.
"""
from __future__ import annotations
import numpy as np


def _logjoint_cx(X, mu_cells, cov_inv, py):
    """[N, n_cls] log p(y=c) + log mean_d N(x; mu_{c,d}, cov)  (|cov| const dropped, cancels)."""
    n_cls, n_dom = mu_cells.shape[:2]
    out = np.full((X.shape[0], n_cls), -np.inf)
    for c in range(n_cls):
        comp = [(-0.5 * np.einsum("ij,jk,ik->i", X - mu_cells[c, e], cov_inv, X - mu_cells[c, e]))
                for e in range(n_dom)]
        stak = np.stack(comp, 1); m = stak.max(1, keepdims=True)
        out[:, c] = np.log(py[c] + 1e-12) + m[:, 0] + np.log(np.exp(stak - m).mean(1))
    return out


def _gaussian_logpost_entropy(X, mu_cells, cov_inv, py):
    """Per-sample H(Y|x) for x = (linear image of z); mu_cells [n_cls,n_dom,m], shared cov_inv."""
    lj = _logjoint_cx(X, mu_cells, cov_inv, py)
    mx = lj.max(1, keepdims=True)
    p = np.exp(lj - mx); p /= p.sum(1, keepdims=True)
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)       # [N]


def logpost_true_label(X, y, mu_cells, cov_inv, py):
    """Per-sample log p(y_i | x_i) under the true mixture (for the oracle info-density detector)."""
    lj = _logjoint_cx(X, mu_cells, cov_inv, py)
    logZ = lj.max(1) + np.log(np.exp(lj - lj.max(1, keepdims=True)).sum(1))
    return lj[np.arange(len(y)), y] - logZ                   # [N]


def bayes_conditional_task_delta(mu_yd, sigma, py, pdy, P, n_mc=20000, n_boot=300, seed=0):
    """EXACT Delta_Y* = H(Y|U) - H(Y|Z) on an INDEPENDENT MC draw from the true mixture, with a
    bootstrap CI. mu_yd [n_cls,n_dom,D] (true, rotated), `sigma` a scalar OR a per-dim std vector
    [D] (diagonal covariance diag(sigma^2)), py [n_cls], pdy [n_cls,n_dom].
    Returns dict(delta, ci_lo, ci_hi, H_YU, H_YZ)."""
    mu_yd = np.asarray(mu_yd, float); py = np.asarray(py, float); pdy = np.asarray(pdy, float)
    n_cls, n_dom, Dd = mu_yd.shape
    std = np.broadcast_to(np.asarray(sigma, float), (Dd,)).copy()             # per-dim std
    covZ = np.diag(std ** 2)
    rng = np.random.default_rng(seed)
    ys = rng.choice(n_cls, n_mc, p=py / py.sum())
    ds = np.array([rng.choice(n_dom, p=pdy[c] / pdy[c].sum()) for c in ys])
    Z = mu_yd[ys, ds] + std * rng.standard_normal((n_mc, Dd))

    H_YZ = _gaussian_logpost_entropy(Z, mu_yd, np.diag(1.0 / std ** 2), py)   # [N]
    Ur, sr, _ = np.linalg.svd(np.eye(Dd) - P)
    B_R = Ur[:, sr > 1e-8]
    A = B_R.T @ (np.eye(Dd) - P)
    U = Z @ A.T
    muU = np.einsum("md,ced->cem", A, mu_yd)
    covU_inv = np.linalg.pinv(A @ covZ @ A.T)
    H_YU = _gaussian_logpost_entropy(U, muU, covU_inv, py)                    # [N]

    dper = H_YU - H_YZ                                        # per-sample H(Y|u_i)-H(Y|z_i)
    brng = np.random.default_rng(seed + 1)
    boot = np.array([dper[brng.integers(0, n_mc, n_mc)].mean() for _ in range(n_boot)])
    return {"delta": float(dper.mean()), "ci_lo": float(np.quantile(boot, 0.05)),
            "ci_hi": float(np.quantile(boot, 0.95)),
            "H_YU": float(H_YU.mean()), "H_YZ": float(H_YZ.mean())}


def classify_safety(ci_lo, ci_hi, delta_Y):
    """Three-way Bayes safety verdict for a deletion vs the gate threshold delta_Y."""
    if ci_hi <= delta_Y:
        return "SAFE"
    if ci_lo > delta_Y:
        return "UNSAFE"
    return "BAYES_AMBIGUOUS"


def mixture_params(Z, y, d, n_cls, n_dom, min_cell=5):
    """EMPIRICAL plug-in params (mu_{y,d}, present-mask, pooled isotropic sigma^2) for data
    WITHOUT known truth. Plug-in only -- NOT a certification (use the true-param oracle above)."""
    Z = np.asarray(Z, dtype=np.float64); D = Z.shape[1]
    mu = np.zeros((n_cls, n_dom, D)); present = np.zeros((n_cls, n_dom), bool)
    sse = 0.0; cnt = 0
    for c in range(n_cls):
        for e in range(n_dom):
            m = (y == c) & (d == e)
            if m.sum() >= min_cell:
                Zc = Z[m]; mu[c, e] = Zc.mean(0); present[c, e] = True
                sse += ((Zc - mu[c, e]) ** 2).sum(); cnt += Zc.size
    return mu, present, float(sse / max(cnt, 1))
