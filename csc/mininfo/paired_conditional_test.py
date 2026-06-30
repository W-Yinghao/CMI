"""
CSC Route B3 — paired conditional-change test (the PRIMARY minimal-information mechanism).

Target-INTERNAL: the target subject's OWN other condition is the reference, so NO source posterior is
needed (this sidesteps the B2 source-reference contamination + covariate-extrapolation confounds, and
the within-subject pairing cancels the subject random effect).

Tests whether P(Y|Z) depends on CONDITION beyond a condition intercept (which already absorbs a
condition-specific covariate offset AND a condition-specific class prior). On the queried paired audit:

  h0:     logits ~ [Z_std, condition]                       # shared boundary + per-class condition intercept
  h1 (full_z, B3-P2.2 R1, DEFAULT): [Z_std, condition, condition x Z_std]  # full condition-dependent boundary
  h1 (pc, P2.1 baseline):           [Z_std, condition, condition x Z_pc(r)]  # low-rank (misses low-var dirs)

  T = vote(NLL_h0) - vote(NLL_h1)   >= 0    (subject-condition vote, the A-line estimand)

Null is parametric bootstrap under the FITTED h0 (Y* ~ h0(.|Z,condition)): refit h0*,h1*, recompute T*;
one-sided p = (1 + #{T* >= T}) / (B+1). A condition-dependent BOUNDARY (concept change) makes T exceed
the h0 null; a condition covariate offset or prior shift is absorbed by h0 and does NOT.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _features(Zs, cond, basis, Vr=None):
    """h0:      [Z_std, condition]
    full_z:  [Z_std, condition, condition x Z_std]            (B3-P2.2 R1 — keeps ALL directions)
    pc:      [Z_std, condition, condition x (Z_std @ Vr)]     (P2.1 low-rank baseline)"""
    cond = np.asarray(cond, float)[:, None]
    base = np.hstack([Zs, cond])
    if basis == "h0":
        return base
    if basis == "full_z":
        return np.hstack([base, cond * Zs])
    if basis == "pc":
        return np.hstack([base, cond * (Zs @ Vr)])
    raise ValueError(f"unknown h1 basis {basis!r}")


def _vote_nll(nll, groups, D):
    """Condition-first subject-vote mean of per-epoch nll."""
    nll = np.asarray(nll, float); g = np.asarray(groups); D = np.asarray(D)
    subj = []
    for s in np.unique(g):
        m = g == s
        subj.append(np.mean([nll[m][D[m] == c].mean() for c in np.unique(D[m])]))
    return float(np.mean(subj))


def subject_condition_weights(groups, D):
    """RAW per-epoch weight 1/(|U_s| * n_su) for u=(subject,condition); sum == #subjects. sklearn lbfgs
    L2 = 1/(C*sum_w), so the fit/regularisation is invariant to epochs-per-condition (an epoch-heavy
    condition cannot dominate the boundary). Mirrors the A-line subject-condition estimand."""
    g = np.asarray(groups); D = np.asarray(D); w = np.ones(len(g), float)
    for s in np.unique(g):
        sm = g == s
        conds = np.unique(D[sm]); U = len(conds)
        for c in conds:
            cm = sm & (D == c); n = int(cm.sum())
            if n:
                w[cm] = 1.0 / (U * n)
    return w


def _fit_nll(X, y, cl, C, w=None):
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(X, y, sample_weight=w)
    p = np.clip(clf.predict_proba(X), 1e-12, 1.0)
    order = [list(clf.classes_).index(c) for c in cl]
    p = p[:, order]
    yi = np.searchsorted(cl, y)
    return -np.log(p[np.arange(len(y)), yi]), clf


def paired_validity(Y, D, groups, min_subjects=4):
    """Pair structure must be sound (B3-P2.1, fail closed): >=2 conditions overall; >=2 classes overall;
    EACH present condition has >=2 classes (a within-subject boundary contrast needs class spread in BOTH
    conditions); and enough subjects with BOTH conditions."""
    D = np.asarray(D); Y = np.asarray(Y); g = np.asarray(groups)
    if len(np.unique(D)) < 2 or len(np.unique(Y)) < 2:
        return False, "needs >=2 conditions and >=2 classes"
    for c in np.unique(D):                                   # per-condition class coverage
        if len(np.unique(Y[D == c])) < 2:
            return False, f"condition {c} has <2 classes"
    paired = [s for s in np.unique(g) if len(np.unique(D[g == s])) >= 2]
    if len(paired) < min_subjects:
        return False, f"only {len(paired)} paired subjects (<{min_subjects})"
    return True, f"{len(paired)} paired subjects"


def classes_by_condition(Y, D):
    Y = np.asarray(Y); D = np.asarray(D)
    return {int(c): int(len(np.unique(Y[D == c]))) for c in np.unique(D)}


def _resolve_C(h1_basis, d, rank, C):
    """Pre-declared regularisation (no sweep). full_z: C_full = 0.5*3/d (the rank-3 C=0.5 interaction
    budget, scaled to the full-Z interaction); pc baseline: C=0.5. ONE C is used for BOTH h0 and h1."""
    if C is not None:
        return float(C), (d if h1_basis == "full_z" else max(1, rank))
    if h1_basis == "full_z":
        return 0.5 * 3 / d, d
    return 0.5, max(1, rank)


def paired_conditional_change_test(Z, Y, D, groups, h1_basis="full_z", rank=3, C=None, n_boot=200,
                                   seed=0, invalid_frac_max=0.20):
    """One-sided parametric-bootstrap test that the boundary depends on condition (concept change).
    h1_basis="full_z" (B3-P2.2 R1) interacts condition with the FULL standardised Z under a fixed strong
    L2 -> keeps all directions incl. the low-variance discriminative one; "pc" is the P2.1 low-rank
    baseline. Subject-condition-WEIGHTED fits/standardise (epoch invariant); conservative null
    invalid-accounting (degenerate/fit-failed charged extreme; too many -> INVALID, fail closed).
    Returns T, p_value, valid, reason, n_pairs, classes_by_condition, n_boot_invalid, h1_basis, C_used,
    n_features_interaction."""
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D); g = np.asarray(groups)
    d = Z.shape[1]
    C_used, n_feat_int = _resolve_C(h1_basis, d, rank, C)
    n_pairs = int(len([s for s in np.unique(g) if len(np.unique(D[g == s])) >= 2]))
    cbc = classes_by_condition(Y, D)
    base = dict(T=float("nan"), p_value=1.0, valid=False, n_pairs=n_pairs, classes_by_condition=cbc,
                n_boot_invalid=0, h1_basis=h1_basis, C_used=float(C_used),
                n_features_interaction=int(n_feat_int))
    ok, reason = paired_validity(Y, D, g)
    if not ok:
        return {**base, "reason": reason}
    w = subject_condition_weights(g, D)                       # epoch-count-invariant weights
    W = w.sum()
    mu = (w[:, None] * Z).sum(0) / W                          # WEIGHTED standardise (epoch-invariant)
    sd = np.sqrt(np.clip((w[:, None] * (Z - mu) ** 2).sum(0) / W, 0, None)) + 1e-8
    Zs = (Z - mu) / sd                                       # weighted mean(Zs) == 0
    Vr = None
    if h1_basis == "pc":
        Vt = np.linalg.svd(np.sqrt(w)[:, None] * Zs, full_matrices=False)[2]   # weighted PCs (epoch-inv)
        Vr = Vt[:max(1, rank)].T
    cl = np.array(sorted(np.unique(Y)))
    X0 = _features(Zs, D, "h0")
    X1 = _features(Zs, D, h1_basis, Vr)
    nll0, clf0 = _fit_nll(X0, Y, cl, C_used, w)
    nll1, _ = _fit_nll(X1, Y, cl, C_used, w)
    T = _vote_nll(nll0, g, D) - _vote_nll(nll1, g, D)
    p0 = np.clip(clf0.predict_proba(X0), 1e-12, 1.0)
    order = [list(clf0.classes_).index(c) for c in cl]
    cum = np.cumsum(p0[:, order], axis=1)
    rng = np.random.default_rng(seed)
    ge, n_invalid = 1, 0
    for _ in range(n_boot):
        u = rng.random(len(Y))
        ystar = cl[(u[:, None] > cum).sum(1)]
        # a replicate is VALID only if it keeps the SAME pair-validity contract as the observed audit
        if not paired_validity(ystar, D, g)[0]:
            n_invalid += 1; ge += 1; continue                 # invalid -> charge extreme (conservative)
        try:
            n0s, _ = _fit_nll(X0, ystar, cl, C_used, w)
            n1s, _ = _fit_nll(X1, ystar, cl, C_used, w)
            ge += int((_vote_nll(n0s, g, D) - _vote_nll(n1s, g, D)) >= T)
        except Exception:
            n_invalid += 1; ge += 1
    # too many invalid replicates -> the null is not estimable -> test INVALID (fail closed)
    if n_invalid > invalid_frac_max * n_boot:
        return {**base, "reason": f"null not estimable: {n_invalid}/{n_boot} invalid replicates",
                "T": float(T), "n_boot_invalid": int(n_invalid)}
    return dict(T=float(T), p_value=ge / (n_boot + 1), valid=True, reason=reason, n_pairs=n_pairs,
                classes_by_condition=cbc, n_boot_invalid=int(n_invalid), h1_basis=h1_basis,
                C_used=float(C_used), n_features_interaction=int(n_feat_int))
