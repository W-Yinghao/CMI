"""Monte-Carlo GROUND-TRUTH conditional mutual information I(Z;D|Y) for the
*known* synthetic data-generating process (DGP) in ``sanity_check.py``.

Motivation
----------
The validation lineage (``validate_proxy.py``) compares two *estimators* of the
leakage I(Z;D|Y): the neural posterior-KL "ruler"
``E KL(q(D|Z,Y) || pi_y(D))`` and an independent sklearn kNN estimator. Neither
is a ground truth. Because the DGP is fully specified (Gaussian-mixture
conditionals + a known label-conditional domain prior), the CMI of the RAW
generator features admits a *numerical ground truth*: we can evaluate the exact
per-sample densities and Monte-Carlo the estimand

    I(Z;D|Y) = E_{Z,D,Y}[ log p(D|Z,Y) - log p(D|Y) ]                         (1)

where p(D|Z,Y) is obtained by **Bayes** from the exact class-conditional feature
density p(Z|D,Y) and the exact label-conditional domain prior p(D|Y):

    p(D=d|z,y) = p(z|d,y) p(d|y) / sum_{d'} p(z|d',y) p(d'|y).                 (2)

Every density in (2) is a closed form of the DGP (diagonal Gaussians and, for
the spurious block, a **two-component Gaussian mixture over the domain-specific
flip rate**), evaluated with log-sum-exp. This is a ground truth, not another
density estimator.

Scope / honesty note
--------------------
The ground truth in (1)-(2) is defined for a variable whose conditional density
is *known in closed form*. That holds for the **raw generator features** X (the
canonical, information-preserving "representation"), whose p(X|D,Y) is the DGP
itself. It does NOT hold for the *learned* neural encoder Z=g(X): the pushforward
of a Gaussian mixture through a general neural network has no closed-form density,
so no numerical ground truth exists for a learned encoding. Truth-anchored
validation therefore operates on the known generator (raw features / a fixed
family of DGP parameter settings); the learned-encoder ruler-vs-kNN comparison in
``validate_proxy.py`` remains an independent cross-check. This module deliberately
avoids calling any *estimator* (neural or kNN) "unbiased"; it only provides the
exact-density Monte-Carlo truth together with its reported Monte-Carlo standard
error.

Public API
----------
- ``mc_cmi_from_logprobs(log_p_d_given_zy, log_p_d_given_y)``: shared MC core.
- ``true_cmi_dgp(dgp, seed, n_samples)``: ground-truth I(X;D|Y) for a ``DGP``.
- ``discrete_cmi_exact(p_zdy)`` / ``discrete_cmi_mc(p_zdy, seed, n_samples)``:
  closed-form and MC CMI for a small categorical joint P(Z,D,Y) (used by tests).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sanity_check import DGP  # noqa: E402

_LOG2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------------------- core
def mc_cmi_from_logprobs(
    log_p_d_given_zy: np.ndarray, log_p_d_given_y: np.ndarray
) -> Dict[str, float]:
    """Shared Monte-Carlo core for the estimand (1).

    Both arguments are 1-D arrays of length ``n``, each entry being the log
    probability *evaluated at the realized triple* ``(z_i, d_i, y_i)``:

        log_p_d_given_zy[i] = log p(D=d_i | z_i, y_i)   (exact Bayes posterior)
        log_p_d_given_y[i]  = log p(D=d_i | y_i)        (exact label-cond. prior)

    Returns the sample mean of ``log_p_d_given_zy - log_p_d_given_y`` (nats), its
    Monte-Carlo standard error (sample std / sqrt(n)), and ``n``.
    """
    log_p_d_given_zy = np.asarray(log_p_d_given_zy, dtype=np.float64)
    log_p_d_given_y = np.asarray(log_p_d_given_y, dtype=np.float64)
    if log_p_d_given_zy.shape != log_p_d_given_y.shape or log_p_d_given_zy.ndim != 1:
        raise ValueError("log-prob arrays must be 1-D of equal length")
    contrib = log_p_d_given_zy - log_p_d_given_y
    n = int(contrib.shape[0])
    mean = float(contrib.mean())
    # ddof=1 sample std for an honest standard-error estimate.
    se = float(contrib.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {"true_cmi_nats": mean, "mc_se": se, "n_samples": n}


def _diag_gauss_logpdf_sum(x: np.ndarray, mean: np.ndarray, var: float) -> np.ndarray:
    """Sum over the last axis of the log density of a diagonal Gaussian with a
    scalar variance ``var``. ``x`` and ``mean`` broadcast; returns the summed
    log-density over the feature axis (shape = broadcast shape without last dim)."""
    diff = x - mean
    per_dim = -0.5 * (_LOG2PI + math.log(var)) - 0.5 * (diff * diff) / var
    return per_dim.sum(axis=-1)


# --------------------------------------------------------------------- DGP priors
def dgp_label_conditional_domain_prior(dgp: DGP) -> np.ndarray:
    """Exact ``pi_y(d) = p(D=d | Y=y)`` for the source domains, shape ``[2, K]``.

    The source draw uses an equal number of samples per domain, so P(D=d)=1/K is
    uniform. With P(Y=1|D=d)=p_d,
        p(D=d|Y=y) ∝ P(Y=y|D=d) P(D=d) ∝ [p_d if y=1 else 1-p_d].
    """
    p = np.array([s[0] for s in dgp.src], dtype=np.float64)  # P(Y=1|D=d)
    pi = np.zeros((2, dgp.n_src), dtype=np.float64)
    pi[1] = p / p.sum()
    pi[0] = (1.0 - p) / (1.0 - p).sum()
    return pi


# ---------------------------------------------------------------- DGP ground truth
def true_cmi_dgp(dgp: DGP | None = None, seed: int = 0, n_samples: int = 800_000) -> Dict[str, float]:
    """Numerical ground-truth I(X;D|Y) (nats) for the known ``DGP`` over its
    SOURCE domains, using exact densities and the MC estimand (1).

    Parameters
    ----------
    dgp : DGP
        Generator whose parameters fully define p(X|D,Y) and p(D|Y).
    seed : int
        Seed for ``np.random.default_rng`` (deterministic).
    n_samples : int
        Total Monte-Carlo samples (rounded up to a multiple of the number of
        source domains so the per-domain draw is balanced, matching the DGP).

    Returns
    -------
    dict with ``true_cmi_nats``, ``mc_se`` (reported standard error), ``n_samples``.
    """
    dgp = dgp or DGP()
    K = dgp.n_src
    n_per = int(math.ceil(n_samples / K))
    rng = np.random.default_rng(seed)

    # Sample (X, Y, D) from the exact source joint used to define the truth.
    X, Y, D = dgp.sample(n_per, rng, target=False)
    N = X.shape[0]

    # Exact DGP constants.
    style = dgp._style_means()[:K]                 # [K, dm], deterministic (rng 12345)
    e = np.array([s[1] for s in dgp.src], np.float64)  # per-domain spurious flip rate
    log_e = np.log(e)
    log_1me = np.log1p(-e)
    log_pi = np.log(dgp_label_conditional_domain_prior(dgp))  # [2, K]

    # Split raw features into the three blocks.
    dc, ds = dgp.dc, dgp.ds
    xc = X[:, :dc].astype(np.float64)
    xs = X[:, dc:dc + ds].astype(np.float64)
    xst = X[:, dc + ds:].astype(np.float64)
    sy = (2 * Y - 1).astype(np.float64)            # +1 / -1 label sign, [N]

    # --- log p(x | D=d', Y=y) for every candidate domain d', shape [N, K] ------
    # (a) causal block: N(mu_c*(2y-1), 1), independent of d' -> cancels in the
    #     posterior softmax, but we include it for a transparent, general density.
    logp_xc = _diag_gauss_logpdf_sum(xc, dgp.mu_c * sy[:, None], 1.0)          # [N]

    # (b) spurious block: two-component Gaussian mixture over the flip.
    #     component + (weight 1-e_d'): N(+m_s*(2y-1), sig_s^2)
    #     component - (weight   e_d'): N(-m_s*(2y-1), sig_s^2)
    mean_plus = dgp.m_s * sy[:, None]                                          # [N,1]
    var_s = dgp.sig_s ** 2
    logN_plus = _diag_gauss_logpdf_sum(xs, mean_plus, var_s)                  # [N]
    logN_minus = _diag_gauss_logpdf_sum(xs, -mean_plus, var_s)               # [N]
    # logaddexp over the two mixture components, broadcast over domains via e_d'.
    t_plus = log_1me[None, :] + logN_plus[:, None]                            # [N,K]
    t_minus = log_e[None, :] + logN_minus[:, None]                           # [N,K]
    logp_xs = np.logaddexp(t_plus, t_minus)                                   # [N,K]

    # (c) pure-style block: N(style[d'], 1), depends on d'.
    diff = xst[:, None, :] - style[None, :, :]                                # [N,K,dm]
    logp_xst = (-0.5 * (_LOG2PI) - 0.5 * (diff * diff)).sum(axis=-1)          # [N,K]

    # Unnormalised log joint over d': log p(x|d',y) + log pi_y(d').
    L = logp_xc[:, None] + logp_xs + logp_xst + log_pi[Y]                     # [N,K]
    log_post = L - _logsumexp(L, axis=1, keepdims=True)                       # [N,K]

    idx = np.arange(N)
    log_p_d_given_zy = log_post[idx, D]
    log_p_d_given_y = log_pi[Y, D]
    out = mc_cmi_from_logprobs(log_p_d_given_zy, log_p_d_given_y)
    out["n_samples"] = N
    return out


def _logsumexp(a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


# ------------------------------------------------------------ discrete (for tests)
def _normalize_joint(p_zdy: np.ndarray) -> np.ndarray:
    p = np.asarray(p_zdy, dtype=np.float64)
    if p.ndim != 3:
        raise ValueError("p_zdy must be a 3-D array indexed [z, d, y]")
    s = p.sum()
    if s <= 0:
        raise ValueError("joint must have positive total mass")
    return p / s


def discrete_cmi_exact(p_zdy: np.ndarray) -> float:
    """Closed-form I(Z;D|Y) (nats) for a categorical joint ``P(Z=z,D=d,Y=y)``
    given as a 3-D array indexed ``[z, d, y]``.

        I(Z;D|Y) = sum_{z,d,y} p(z,d,y) log[ p(z,d|y) / (p(z|y) p(d|y)) ].
    """
    p = _normalize_joint(p_zdy)                     # [Z, D, Y]
    p_y = p.sum(axis=(0, 1))                         # [Y]
    total = 0.0
    for yi in range(p.shape[2]):
        py = p_y[yi]
        if py <= 0:
            continue
        pzd_y = p[:, :, yi] / py                     # p(z,d|y), [Z, D]
        pz_y = pzd_y.sum(axis=1)                     # p(z|y),   [Z]
        pd_y = pzd_y.sum(axis=0)                     # p(d|y),   [D]
        denom = np.outer(pz_y, pd_y)                 # p(z|y)p(d|y)
        mask = (pzd_y > 0) & (denom > 0)
        total += py * float(np.sum(pzd_y[mask] * np.log(pzd_y[mask] / denom[mask])))
    return total


def discrete_cmi_mc(p_zdy: np.ndarray, seed: int = 0, n_samples: int = 400_000) -> Dict[str, float]:
    """Monte-Carlo I(Z;D|Y) for a categorical joint, routed through the SAME
    ``mc_cmi_from_logprobs`` core as the Gaussian DGP path. Samples (z,d,y) from
    the joint, then evaluates the exact discrete posterior p(d|z,y) and prior
    p(d|y) from the (known) tables. Used to test the MC machinery against the
    closed form ``discrete_cmi_exact``.
    """
    p = _normalize_joint(p_zdy)                     # [Z, D, Y]
    nz, nd, ny = p.shape
    rng = np.random.default_rng(seed)

    flat = p.reshape(-1)
    draws = rng.choice(flat.size, size=int(n_samples), p=flat)
    zi, di, yi = np.unravel_index(draws, (nz, nd, ny))

    # Exact conditionals from the tables.
    p_y = p.sum(axis=(0, 1))                         # [Y]
    p_dy = p.sum(axis=0)                             # p(d,y), [D, Y]
    p_zy = p.sum(axis=1)                             # p(z,y), [Z, Y]
    with np.errstate(divide="ignore"):
        # p(d|y) = p(d,y)/p(y)
        pd_given_y = p_dy / p_y[None, :]             # [D, Y]
        # p(d|z,y) = p(z,d,y) / p(z,y)
        pd_given_zy = p / p_zy[:, None, :]           # [Z, D, Y]

    log_p_d_given_zy = np.log(pd_given_zy[zi, di, yi])
    log_p_d_given_y = np.log(pd_given_y[di, yi])
    return mc_cmi_from_logprobs(log_p_d_given_zy, log_p_d_given_y)


# ---------------------------------------------------------------------------- cli
def _demo():
    dgp = DGP()
    r = true_cmi_dgp(dgp, seed=0, n_samples=800_000)
    print(f"DGP default  I(X;D|Y) = {r['true_cmi_nats']:.4f} +/- {r['mc_se']:.4f} nats "
          f"(n={r['n_samples']})")


if __name__ == "__main__":
    _demo()
