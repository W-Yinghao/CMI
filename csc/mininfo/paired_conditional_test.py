"""
CSC Route B3 — paired conditional-change test (the PRIMARY minimal-information mechanism).

Target-INTERNAL: the target subject's OWN other condition is the reference, so NO source posterior is
needed (this sidesteps the B2 source-reference contamination + covariate-extrapolation confounds, and
the within-subject pairing cancels the subject random effect).

Tests whether P(Y|Z) depends on CONDITION beyond a condition intercept (which already absorbs a
condition-specific covariate offset AND a condition-specific class prior). On the queried paired audit:

  h0: logits ~ [Z_std, condition]                       # shared boundary + per-class condition intercept
  h1: logits ~ [Z_std, condition, condition x Z_pc(r)]  # condition-dependent LOW-RANK boundary

  T = vote(NLL_h0) - vote(NLL_h1)   >= 0    (subject-condition vote, the A-line estimand)

Null is parametric bootstrap under the FITTED h0 (Y* ~ h0(.|Z,condition)): refit h0*,h1*, recompute T*;
one-sided p = (1 + #{T* >= T}) / (B+1). A condition-dependent BOUNDARY (concept change) makes T exceed
the h0 null; a condition covariate offset or prior shift is absorbed by h0 and does NOT.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _features(Zs, cond, Vr):
    """[Z_std, condition, condition x (Z_std @ Vr)]."""
    cond = np.asarray(cond, float)[:, None]
    base = np.hstack([Zs, cond])
    if Vr is not None and Vr.shape[1]:
        base = np.hstack([base, cond * (Zs @ Vr)])
    return base


def _vote_nll(nll, groups, D):
    """Condition-first subject-vote mean of per-epoch nll."""
    nll = np.asarray(nll, float); g = np.asarray(groups); D = np.asarray(D)
    subj = []
    for s in np.unique(g):
        m = g == s
        subj.append(np.mean([nll[m][D[m] == c].mean() for c in np.unique(D[m])]))
    return float(np.mean(subj))


def _fit_nll(X, y, cl, C):
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(X, y)
    p = np.clip(clf.predict_proba(X), 1e-12, 1.0)
    order = [list(clf.classes_).index(c) for c in cl]
    p = p[:, order]
    yi = np.searchsorted(cl, y)
    return -np.log(p[np.arange(len(y)), yi]), clf


def paired_validity(Y, D, groups, min_subjects=4):
    """Pair structure must be sound: >=2 conditions overall, both classes present, and enough subjects
    that have BOTH conditions (a within-subject contrast needs paired subjects)."""
    D = np.asarray(D); Y = np.asarray(Y); g = np.asarray(groups)
    if len(np.unique(D)) < 2 or len(np.unique(Y)) < 2:
        return False, "needs >=2 conditions and >=2 classes"
    paired = [s for s in np.unique(g) if len(np.unique(D[g == s])) >= 2]
    if len(paired) < min_subjects:
        return False, f"only {len(paired)} paired subjects (<{min_subjects})"
    return True, f"{len(paired)} paired subjects"


def paired_conditional_change_test(Z, Y, D, groups, rank=3, C=0.5, n_boot=200, seed=0):
    """One-sided parametric-bootstrap test that the boundary depends on condition (concept change).
    Returns dict(T, p_value, n_paired_subjects, valid, reason)."""
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D); g = np.asarray(groups)
    ok, reason = paired_validity(Y, D, g)
    if not ok:
        return dict(T=float("nan"), p_value=1.0, n_paired_subjects=0, valid=False, reason=reason)
    mu = Z.mean(0); sd = Z.std(0) + 1e-8
    Zs = (Z - mu) / sd
    # low-rank interaction basis = top-r right singular vectors of the centered audit Z
    Zc = Zs - Zs.mean(0)
    Vt = np.linalg.svd(Zc, full_matrices=False)[2]
    Vr = Vt[:max(1, rank)].T
    cl = np.array(sorted(np.unique(Y)))
    X0 = _features(Zs, D, None)
    X1 = _features(Zs, D, Vr)
    nll0, clf0 = _fit_nll(X0, Y, cl, C)
    nll1, _ = _fit_nll(X1, Y, cl, C)
    T = _vote_nll(nll0, g, D) - _vote_nll(nll1, g, D)
    # parametric bootstrap under fitted h0: Y* ~ h0(.|Z,condition)
    p0 = np.clip(clf0.predict_proba(X0), 1e-12, 1.0)
    order = [list(clf0.classes_).index(c) for c in cl]
    p0 = p0[:, order]; cum = np.cumsum(p0, axis=1)
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(n_boot):
        u = rng.random(len(Y))
        ystar = cl[(u[:, None] > cum).sum(1)]
        if len(np.unique(ystar)) < 2:
            ge += 1; continue                       # degenerate -> charge as extreme (conservative)
        try:
            n0s, _ = _fit_nll(X0, ystar, cl, C)
            n1s, _ = _fit_nll(X1, ystar, cl, C)
            ge += int((_vote_nll(n0s, g, D) - _vote_nll(n1s, g, D)) >= T)
        except Exception:
            ge += 1
    return dict(T=float(T), p_value=ge / (n_boot + 1), n_paired_subjects=int(reason.split()[0])
                if reason[0].isdigit() else 0, valid=True, reason=reason)
