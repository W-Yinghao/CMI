"""v2 subject-clustered split-conformal (notes/ACAR_FROZEN_v2.md A1). Replaces the v1 cohort-max heuristic.

Calibration unit = subject cluster. FIT / CAL subjects are disjoint. ĝ_a is fit on FIT batches; the conformal
quantile q is computed from ONE joint nonconformity score per CAL subject:

    s_i = max_{B in subject i}  max_{a in non-identity}  ( ΔR_a(B) - ĝ_a(φ_a(B)) )

q_{1-α} = the ⌈(m+1)(1-α)⌉-th smallest of {s_i} over the m CAL subjects; if the rank > m, q = +∞ (uninformative,
NEVER clipped). The same q is added to every action's ĝ_a -> simultaneous one-sided coverage over all actions for
an exchangeable new subject. Disease-stratified (PD/SCZ q computed separately by the caller).

Fallback batches (n < MIN_BATCH, forced identity) are excluded from FIT and from CAL scoring (they are never
adapted), but retained in EVAL deployment (routed to identity).
"""
from __future__ import annotations
import hashlib
from collections import defaultdict
import math
import numpy as np

from .regressor import ActionRegressor
from .deploy import Routers


def subject_fold(subject: str, k: int, seed: int) -> int:
    """Stable subject->fold assignment (no Math.random; hash-based, reproducible)."""
    h = hashlib.sha256(f"{seed}|{subject}".encode()).hexdigest()
    return int(h[:8], 16) % k


def split_fit_cal(subjects, fit_frac, seed):
    """Subject-disjoint FIT/CAL split of the given subjects (stable by hash)."""
    ordered = sorted(subjects, key=lambda s: hashlib.sha256(f"{seed}|fc|{s}".encode()).hexdigest())
    n_fit = int(round(fit_frac * len(ordered)))
    return set(ordered[:n_fit]), set(ordered[n_fit:])


def conformal_rank(m, alpha):
    """1-indexed split-conformal rank k = ceil((m+1)(1-α)). k>m => the (1-α) quantile is +inf (uninformative)."""
    return math.ceil((m + 1) * (1 - alpha))


def onesided_quantile(scores, alpha):
    """Conformal (1-α) one-sided upper quantile with finite-sample correction. HONEST: rank>m -> +inf.
    NB: m is the number of CALIBRATION SUBJECTS in THIS fold (not the pooled total); k varies by fold."""
    s = np.sort(np.asarray(scores, float))
    m = len(s)
    if m == 0:
        return float("inf"), 0
    k = conformal_rank(m, alpha)
    if k > m:
        return float("inf"), k                               # not enough calibration units -> uninformative
    return float(s[k - 1]), k


def fit_routers(fit_recs, cal_recs, actions, alpha, delta, seed):
    """Fit ĝ_a on FIT batches; calibrate shared joint q on CAL subjects. Returns (Routers, diag)."""
    regs = {}
    for a in actions:
        X = np.vstack([r["fvec"][a] for r in fit_recs]) if fit_recs else np.zeros((0, 1))
        dr = np.array([r["dr"][a] for r in fit_recs], float)
        regs[a] = ActionRegressor(seed=seed).fit(X, dr)
    # one joint nonconformity score per CAL subject (max over batches and actions)
    by_subj = defaultdict(list)
    for r in cal_recs:
        by_subj[r["subject"]].append(r)
    scores = []
    for subj, recs in by_subj.items():
        s_subj = -np.inf
        for r in recs:
            for a in actions:
                resid = r["dr"][a] - float(regs[a].predict(r["fvec"][a][None])[0])
                if resid > s_subj:
                    s_subj = resid
        scores.append(s_subj)
    q, k = onesided_quantile(scores, alpha)
    routers = Routers(regs=regs, q=q, delta=delta, actions=tuple(actions))
    diag = dict(n_fit=len(fit_recs), n_cal=len(scores), k=int(k), q=q, q_informative=bool(np.isfinite(q)),
                cal_scores=[float(s) for s in scores])
    return routers, diag
