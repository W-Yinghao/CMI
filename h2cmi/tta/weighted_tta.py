"""Weighted TTA estimators for V2P_WEIGHTED_PREVALENCE_INTERVENTION (REVIEW_P0 section D).

Same fixed adaptation reservoir for every prevalence ratio; only SAMPLE WEIGHTS change (the trial IDs
and temporal positions are identical). Weights are EFFECTIVE COUNTS: w_i=1 means one sample, so equal
weights (all ones) reproduce the unweighted estimator and integer weights reproduce explicit sample
replication (tests 1-2). Deployed estimators receive embeddings + weights, NEVER class labels; only the
oracle diagnostic uses true labels.

Weighted M-step convention (matches the original MEAN objective): geometry Q maximises the WEIGHTED MEAN
log-likelihood ll = sum_i w_i r_iy (log p(T u_i|y)+log pi) / sum_i w_i, with the SAME per-dim logdet /
trust regularisation; the prior M-step uses weighted responsibility sums counts_y = sum_i w_i r_iy with
the unchanged Dirichlet anchor. The caller rescales the canonical normalised weights (sum=1) to sum=N
(effective counts) before calling.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from h2cmi.tta.class_conditional import Transform


def _as_w(w, n, device):
    w = torch.as_tensor(np.asarray(w), dtype=torch.float32, device=device).reshape(-1)
    assert w.shape[0] == n, f"weight length {w.shape[0]} != n {n}"
    return w


def canonical_weights(y, q):
    """Normalised weights (sum=1): w_i = q/n0 if y_i==0 else (1-q)/n1. Labels used ONLY here (offline
    weight builder), never by an estimator."""
    y = np.asarray(y); n0 = int((y == 0).sum()); n1 = int((y == 1).sum())
    return np.where(y == 0, q / max(n0, 1), (1.0 - q) / max(n1, 1)).astype(np.float64)


def effective_weights(y, q):
    """Effective-count weights (sum=N): N * canonical. q=0.5 on balanced classes -> all ones (= the
    unweighted 1:1 reservoir)."""
    return canonical_weights(y, q) * len(np.asarray(y))


def weighted_pooled_moments(U, w):
    """Weighted classless pooled mean/std of the target embeddings (effective-count weights)."""
    w = _as_w(w, U.shape[0], U.device); W = w.sum().clamp_min(1e-8)
    mu = (w[:, None] * U).sum(0) / W
    var = (w[:, None] * (U - mu) ** 2).sum(0) / W
    return mu, var.clamp_min(1e-6).sqrt()


def fit_weighted_pooled(U, w, pooled_ref, device):
    """Weighted pooled_empirical_diag: match the weighted target moments to the source pooled_ref."""
    mu_S, sd_S = (torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device) for x in pooled_ref)
    mu_T, sd_T = weighted_pooled_moments(U.to(device), w)
    a = torch.log((sd_S / sd_T).clamp_min(1e-6))
    T = Transform(U.shape[1], "diag_affine", device=device)
    with torch.no_grad():
        T.a.copy_(a); T.b.copy_(mu_S - torch.exp(a) * mu_T)
    pi_S = None
    return T


def fit_weighted_em(density, U, w, pi_S, cfg, K, device, kind, oracle_labels=None, tta_seed=0):
    """Weighted EM for kind in {oneshot, iterative, joint, oracle}.
      oneshot   : gen_oneshot responsibilities (generated once on identity), prior pinned at pi_S.
      iterative : gen_iterative E-step each round, prior pinned at pi_S (fixed-prior geometry).
      joint     : gen_iterative E-step + WEIGHTED prior M-step (the joint).
      oracle    : true-label one-hot responsibilities (diagnostic), prior pinned at pi_S.
    Returns (Transform, pi_T)."""
    U = U.detach().to(device)
    n, d = U.shape
    w = _as_w(w, n, device); W = w.sum().clamp_min(1e-8)
    pi_S_t = torch.as_tensor(np.asarray(pi_S), dtype=torch.float32, device=device)
    T = Transform(d, "diag_affine", device=device)
    pi_T = pi_S_t.clone()
    anchor = (cfg.dirichlet + cfg.prior_anchor_strength) * pi_S_t
    frozen = [(p, p.requires_grad) for p in density.parameters()]
    for p, _ in frozen:
        p.requires_grad_(False)
    try:
        torch.manual_seed(tta_seed)
        opt = torch.optim.Adam(T.params, lr=cfg.em_lr)
        r_fixed = None
        if kind == "oneshot":
            with torch.no_grad():
                r_fixed = F.softmax(density.log_prob_all(U) + torch.log(pi_S_t.clamp_min(1e-8)).view(1, -1), dim=1)
        elif kind == "oracle":
            yl = torch.as_tensor(np.asarray(oracle_labels), dtype=torch.long, device=device)
            r_fixed = F.one_hot(yl, K).to(torch.float32)
        for _ in range(cfg.em_iters):
            with torch.no_grad():
                if r_fixed is not None:
                    r = r_fixed
                else:
                    z = T.apply(U)
                    r = F.softmax(density.log_prob_all(z) + torch.log(pi_T.clamp_min(1e-8)).view(1, -1), dim=1)
                if kind == "joint":
                    counts = (w[:, None] * r).sum(0)                  # weighted responsibility counts
                    pi_T = (counts + anchor) / (counts.sum() + anchor.sum())
            log_piT = torch.log(pi_T.clamp_min(1e-8))
            for _ in range(3):
                z = T.apply(U)
                ll = (w[:, None] * r * (density.log_prob_all(z) + log_piT.view(1, -1))).sum() / W
                obj = (ll + cfg.logdet_weight * T.logdet()
                       - cfg.trust_region * T.trust() / d
                       - cfg.trust_region_b * (T.b ** 2).sum() / d)
                opt.zero_grad(); (-obj).backward(); opt.step()
        return T, pi_T.detach()
    finally:
        for p, req in frozen:
            p.requires_grad_(req)
