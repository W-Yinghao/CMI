"""C86D active policies + estimation.

Client-side: reads the unlabeled pool (probabilities), acquires physical labels via
the sealed server handle, and estimates candidate utility.

P0  uniform without replacement.
A1  registered Active Testing / LURE: acquisition ∝ mean expected candidate NLL
    (mean candidate predictive entropy), LURE weights correct the biased sampling.
A2H faithful Hara general-K: acquisition = Σ_{k<k'} E_π|NLL_k − NLL_k'|.
Candidate selection = composite plugin (bAcc/NLL/ECE → oriented midranks → equal-
weight composite → first-index argmax). Claim boundary: LURE unbiasedness applies
ONLY to linear moments; the composite plugin has no unbiasedness claim.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

from .core import N_CANDIDATES, NONLINEAR_PLUGINS, C86DClaimError

PROB_FLOOR = 1e-7
CONF_BINS = 15


def load_pool(pool_root: str) -> dict:
    """target=(ds,subj) -> {trial: {context: probs[81,2]}}. Client reads pool ONLY."""
    out = {}
    for pf in sorted(glob.glob(os.path.join(pool_root, "*.npz"))):
        z = np.load(pf, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        tgt = (meta["dataset"], meta["subject"])
        ctx = f"panel={meta['panel']}|seed={meta['seed']}|level={meta['level']}"
        probs = z["probabilities"].astype(np.float64)      # [n,81,2]
        for j, t in enumerate(z["trial_ids"]):
            out.setdefault(tgt, {}).setdefault(str(t), {})[ctx] = probs[j]
    return out


def _nll(probs):                       # [...,81,2] -> [...,81,2] class NLL
    return -np.log(np.clip(probs, PROB_FLOOR, 1.0))


def _entropy(probs):                   # [81,2] -> [81]
    return -np.sum(probs * _nll(probs), axis=1)


def acquisition_score(target_pool: dict, method: str) -> dict:
    """Per-trial target-level acquisition score (equal-weight over the 8 contexts)."""
    scores = {}
    for trial, ctxs in target_pool.items():
        vals = []
        for probs in ctxs.values():                        # probs [81,2]
            if method == "A1":                              # mean expected candidate NLL = mean entropy
                vals.append(float(np.mean(_entropy(probs))))
            elif method == "A2H":                           # Σ_{k<k'} E_π|NLL_k − NLL_k'|
                p_ref = probs.mean(axis=0)                  # [2] ensemble predictive
                nll = _nll(probs)                           # [81,2]
                s = 0.0
                for y in (0, 1):
                    d = np.abs(nll[:, y][:, None] - nll[:, y][None, :])
                    s += p_ref[y] * 0.5 * d.sum()
                vals.append(float(s))
            else:
                raise ValueError(method)
        scores[trial] = float(np.mean(vals))                # equal-weight context aggregation
    return scores


def _lure_weights(q_seq, n_pool):
    """LURE weights for without-replacement acquisition (Farquhar/Kossen)."""
    M = len(q_seq); N = n_pool
    v = np.ones(M)
    for m in range(1, M + 1):
        q = max(q_seq[m - 1], 1e-12)
        if N - m > 0:
            v[m - 1] = 1.0 + (N - M) / (N - m) * (1.0 / ((N - m + 1) * q) - 1.0)
    return v


def select_query_sequence(target_pool, method, budget, seed, rho=0.05):
    """Return ordered trial ids + LURE weights (uniform for P0)."""
    trials = sorted(target_pool)
    N = len(trials)
    cap = N if budget == "FULL" else min(int(budget), N)
    rng = np.random.default_rng(seed)
    if method == "P0" or cap >= N:
        order = list(rng.permutation(trials))[:cap]
        w = np.full(len(order), N / len(order)) if len(order) else np.array([])   # HT sample mean scale
        return order, (w / w.mean() if w.size else w)                              # normalized -> plain mean
    scores = acquisition_score(target_pool, method)
    remaining = list(trials)
    order, q_seq = [], []
    for _ in range(cap):
        s = np.array([max(scores[t], 0.0) for t in remaining], dtype=np.float64)
        p = s / s.sum() if s.sum() > 0 else np.ones(len(s)) / len(s)
        p = (1 - rho) * p + rho / len(p)                    # uniform mixing (LURE validity floor)
        idx = int(rng.choice(len(remaining), p=p))
        order.append(remaining[idx]); q_seq.append(float(p[idx]))
        remaining.pop(idx)
    return order, _lure_weights(q_seq, N)


def _oriented_midrank(values, higher_is_better):
    """Average-rank (midrank) oriented so a better metric => higher rank."""
    v = np.asarray(values, dtype=np.float64)
    order = v if higher_is_better else -v
    # average ranks with ties
    idx = np.argsort(order, kind="mergesort")
    ranks = np.empty(len(v)); ranks[idx] = np.arange(1, len(v) + 1)
    # midrank for ties
    uniq, inv, counts = np.unique(order, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    mid = {}
    start = 0
    for u, c in zip(uniq, counts):
        mid[u] = (start + 1 + start + c) / 2.0
        start += c
    return np.array([mid[order[i]] for i in range(len(v))])


def composite_select(labels, contribs, weights=None):
    """Per context: choose candidate via composite plugin (first-index argmax).

    labels: [m] queried true labels; contribs: {field: [m,81]} for queried trials.
    weights: [m] LURE weights (applied to the LINEAR moment; default uniform).
    Returns selected candidate index + the plugin metrics (all NONLINEAR — no claim).
    """
    y = np.asarray(labels)
    nll = np.asarray(contribs["nll"]); correct = np.asarray(contribs["correct"])
    conf = np.asarray(contribs["confidence"])
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum() * len(y)                                # normalize (mean-preserving)
    est_nll = (w[:, None] * nll).mean(axis=0)               # [81] LURE-weighted linear moment
    # balanced accuracy plugin (LURE-weighted per-class recall over present classes)
    bacc = np.zeros(N_CANDIDATES)
    for c in range(N_CANDIDATES):
        recs = []
        for cl in np.unique(y):
            m = y == cl
            if m.any():
                recs.append(float((w[m] * correct[m, c]).sum() / w[m].sum()))
        bacc[c] = float(np.mean(recs)) if recs else 0.0
    # ECE plugin (weighted mean |confidence − correctness|)
    ece = (w[:, None] * np.abs(conf - correct)).mean(axis=0)
    comp = (_oriented_midrank(bacc, True) + _oriented_midrank(est_nll, False)
            + _oriented_midrank(ece, False)) / 3.0
    sel = int(np.argmax(comp))                              # first-index tie
    return sel, {"balanced_accuracy": bacc, "ece": ece, "nll": est_nll, "composite": comp}


def unbiasedness_claim(quantity: str) -> bool:
    if quantity in NONLINEAR_PLUGINS:
        return False
    from .core import LINEAR_MOMENTS
    if quantity in LINEAR_MOMENTS:
        return True
    raise ValueError(quantity)
