"""C86D active policies + estimation (reconciled per PM C86D review).

Fixes: A1 entropy sign; C85U-exact composite plugin (Jeffreys bAcc, 15-bin ECE,
LURE-weighted NLL, oriented midrank percentile (r-1)/(n-1), equal-weight composite,
first-index argmax); uniform warm start (first 4) + nested-prefix acquisition path
with budget-specific LURE weights. Claim boundary: LURE unbiasedness only for the
linear moments; the composite plugin has no unbiasedness claim.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np

from .core import N_CANDIDATES, NONLINEAR_PLUGINS

PROB_FLOOR = 1e-7
CONF_BINS = 15
WARM_START = 4                     # first 4 queries uniform (locked contract)
CLASSES = (0, 1)


def load_pool(pool_root: str) -> dict:
    """target=(ds,subj) -> {trial: {context: probs[81,2]}}. Client reads pool ONLY."""
    out = {}
    for pf in sorted(glob.glob(os.path.join(pool_root, "*.npz"))):
        z = np.load(pf, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        tgt = (meta["dataset"], meta["subject"])
        ctx = f"panel={meta['panel']}|seed={meta['seed']}|level={meta['level']}"
        probs = z["probabilities"].astype(np.float64)
        for j, t in enumerate(z["trial_ids"]):
            out.setdefault(tgt, {}).setdefault(str(t), {})[ctx] = probs[j]
    return out


def _nll(probs):
    return -np.log(np.clip(probs, PROB_FLOOR, 1.0))


def _entropy(probs):               # [81,2] -> [81]  (non-negative Shannon entropy)
    return np.sum(probs * _nll(probs), axis=1)


def acquisition_score(target_pool: dict, method: str) -> dict:
    """Per-trial target-level acquisition score (equal-weight over the 8 contexts)."""
    scores = {}
    for trial, ctxs in target_pool.items():
        vals = []
        for probs in ctxs.values():
            if method == "A1":                         # mean expected candidate NLL = mean entropy
                vals.append(float(np.mean(_entropy(probs))))
            elif method == "A2H":                      # Σ_{k<k'} E_π|NLL_k − NLL_k'|
                p_ref = probs.mean(axis=0)
                nll = _nll(probs)
                s = sum(p_ref[y] * 0.5 * np.abs(nll[:, y][:, None] - nll[:, y][None, :]).sum()
                        for y in (0, 1))
                vals.append(float(s))
            else:
                raise ValueError(method)
        scores[trial] = float(np.mean(vals))
    return scores


def _avg_rank(x):
    """1..n average ranks with ties."""
    vals, cnt = np.unique(x, return_counts=True)
    start, avg = 0, {}
    for v, c in zip(vals, cnt):
        avg[v] = (start + 1 + start + c) / 2.0
        start += c
    return np.array([avg[xi] for xi in x])


def _midrank_pct(values, higher_is_better):
    x = np.asarray(values, dtype=np.float64) if higher_is_better else -np.asarray(values, dtype=np.float64)
    r = _avg_rank(x)
    n = len(x)
    return (r - 1) / (n - 1) if n > 1 else np.zeros(n)


def composite_from_metrics(bacc, nll, ece):
    """C85U-exact composite pipeline. Returns (composite[81], std_regret[81], selected)."""
    comp = (_midrank_pct(bacc, True) + _midrank_pct(nll, False) + _midrank_pct(ece, False)) / 3.0
    cmax, cmin = comp.max(), comp.min()
    spread = cmax - cmin
    std_regret = (cmax - comp) / spread if spread > 0 else np.zeros_like(comp)
    return comp, std_regret, int(np.argmax(comp))     # first-index tie


def estimate_metrics(labels, contribs, weights, full: bool, n_pool: int):
    """Locked LURE population-total estimator (NOT self-normalized).

    NLL   = (1/M) Σ v_m nll_m
    N̂_y  = (N/M) Σ v_m 1{y_m=y};   Ĉ_y = (N/M) Σ v_m 1{y_m=y} correct_m
    recall_y = (Ĉ_y + p)/(N̂_y + 2p)  with Jeffreys p=0.5 (finite) / p=0 (FULL, exact)
    ECE   = Σ_b |(1/M) Σ v_m 1{bin=b}(correct_m − conf_m)|
    """
    y = np.asarray(labels)
    nll = np.asarray(contribs["nll"]); correct = np.asarray(contribs["correct"])
    conf = np.asarray(contribs["confidence"]); cbin = np.asarray(contribs["conf_bin"])
    v = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    M = len(y); N = n_pool; K = nll.shape[1]
    est_nll = (v[:, None] * nll).sum(axis=0) / M                 # (1/M) Σ v_m nll_m
    pseudo = 0.0 if full else 0.5
    bacc = np.zeros(K)
    for cl in CLASSES:                                          # pre-registered class set (never drop)
        m = y == cl
        Ny = (N / M) * v[m].sum() if m.any() else 0.0
        Cy = (N / M) * (v[m][:, None] * correct[m]).sum(axis=0) if m.any() else np.zeros(K)
        bacc += (Cy + pseudo) / (Ny + 2 * pseudo)
    bacc /= len(CLASSES)
    ece = np.zeros(K)
    for c in range(K):
        e = 0.0
        for b in range(CONF_BINS):
            mb = cbin[:, c] == b
            if mb.any():
                e += abs((v[mb] * (correct[mb, c] - conf[mb, c])).sum() / M)
        ece[c] = e
    return bacc, est_nll, ece


def chain_seed(dataset, subject, chain_id, salt="C86_ACTIVE_CHAIN_V1"):
    """Target-bound seed: low64(SHA256(salt|dataset|subject|chain_id))."""
    h = hashlib.sha256(f"{salt}|{dataset}|{subject}|{chain_id}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def composite_select(labels, contribs, weights=None, full=False, n_pool=None):
    if n_pool is None:
        n_pool = len(labels)
    bacc, nll, ece = estimate_metrics(labels, contribs, weights, full, n_pool)
    comp, std_regret, sel = composite_from_metrics(bacc, nll, ece)
    return sel, {"balanced_accuracy": bacc, "nll": nll, "ece": ece,
                 "composite": comp, "std_regret_construction": std_regret}


def _lure_weights(q_seq, n_pool):
    M, N = len(q_seq), n_pool
    v = np.ones(M)
    for m in range(1, M + 1):
        q = max(q_seq[m - 1], 1e-12)
        if N - m > 0:
            v[m - 1] = 1.0 + (N - M) / (N - m) * (1.0 / ((N - m + 1) * q) - 1.0)
    return v


def acquisition_path(target_pool, method, seed, rho=0.05):
    """ONE full without-replacement path (warm start + active); returns (order, q_seq)."""
    trials = sorted(target_pool)
    N = len(trials)
    rng = np.random.default_rng(seed)
    if method == "P0":
        order = list(rng.permutation(trials))
        return order, [1.0 / (N - m) for m in range(N)]
    scores = acquisition_score(target_pool, method)
    remaining = list(trials); order, q_seq = [], []
    for step in range(N):
        if step < WARM_START or not remaining:
            p = np.ones(len(remaining)) / len(remaining)
        else:
            s = np.array([max(scores[t], 0.0) for t in remaining])
            p = s / s.sum() if s.sum() > 0 else np.ones(len(s)) / len(s)
            p = (1 - rho) * p + rho / len(p)
        idx = int(rng.choice(len(remaining), p=p))
        order.append(remaining[idx]); q_seq.append(float(p[idx])); remaining.pop(idx)
    return order, q_seq


def budget_prefix(order, q_seq, n_pool, budget):
    """Nested prefix for a budget with budget-specific LURE weights v_m^M."""
    M = n_pool if budget == "FULL" else min(int(budget), n_pool)
    pre = order[:M]
    w = _lure_weights(q_seq[:M], n_pool)
    return pre, w


def unbiasedness_claim(quantity: str) -> bool:
    if quantity in NONLINEAR_PLUGINS:
        return False
    from .core import LINEAR_MOMENTS
    if quantity in LINEAR_MOMENTS:
        return True
    raise ValueError(quantity)
