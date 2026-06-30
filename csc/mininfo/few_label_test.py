"""
CSC Route B2 — few-label conditional-risk test (the scientific core of the minimal-information
certificate). DEVELOPMENT prototype; simulator-only.

Idea (directly breaks the Z-only impossibility of Direction A by READING a few target labels):
fit the SOURCE-calibrated posterior `p_S(y|z)` on the labeled source; then, on a few queried target
subjects, ask whether the target labels are MORE surprising under `p_S` than `p_S` expects of itself.

Per audited epoch i with label Y_i at embedding Z_i:
    r_i = -log p_S(Y_i | Z_i)  -  H(p_S(. | Z_i))           # log-score residual
        = (observed NLL)        - (expected NLL under p_S = entropy)
Under H0: Y_i ~ p_S(.|Z_i)  (target conditional == source conditional)  =>  E[r_i] = 0.
A concept shift (P(Y|Z) moved) makes the target labels systematically more surprising => R_T > 0.

Aggregation is the SAME subject-condition estimand as the A line:
    R_T = mean_s ( mean_u ( mean_e r_{sue} ) )             # one vote per (subject,condition) then subject
Null is CONDITIONAL Monte Carlo: redraw Y*_i ~ p_S(.|Z_i) (fixing Z), recompute R*; one-sided p.

Optional prior correction (label-shift guard): reweight p_S to the target prior estimated from the
audit labels, p'(y|z) ∝ p_S(y|z)·π_T[y]/π_S[y]. This isolates the Y|Z change BEYOND a pure prior
(label) shift, so a label-shift-only target is not mistaken for a concept shift.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _standardise(Z):
    Z = np.asarray(Z, float)
    mu = Z.mean(0)
    sd = Z.std(0) + 1e-8
    return mu, sd


class SourcePosterior:
    """Source-calibrated p_S(y|z): standardised logistic regression on the labeled source pool."""

    def __init__(self, Z, Y, C=1.0):
        Z = np.asarray(Z, float); Y = np.asarray(Y)
        self.classes_ = np.array(sorted(np.unique(Y)))
        self.mu_, self.sd_ = _standardise(Z)
        Xs = (Z - self.mu_) / self.sd_
        self.clf_ = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(Xs, Y)
        # source prior (for optional label-shift correction)
        self.pi_S_ = np.array([(Y == c).mean() for c in self.classes_]) + 1e-12

    def proba(self, Z):
        Xs = (np.asarray(Z, float) - self.mu_) / self.sd_
        p = self.clf_.predict_proba(Xs)                      # columns follow self.clf_.classes_
        # re-order columns to self.classes_ (sorted) — robust if clf orders differently
        order = [list(self.clf_.classes_).index(c) for c in self.classes_]
        return np.clip(p[:, order], 1e-12, 1.0)


def _prior_corrected(p, classes, pi_S, Y_audit):
    """Reweight posterior columns to the target prior estimated from the audit labels."""
    pi_T = np.array([(np.asarray(Y_audit) == c).mean() for c in classes]) + 1e-12
    w = pi_T / pi_S
    pc = p * w[None, :]
    return pc / pc.sum(1, keepdims=True)


def log_score_residual(p, Y, classes):
    """r_i = -log p(Y_i) - H(p_i); E[r]=0 under Y~p (one-sided positive => more surprising)."""
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    yi = np.searchsorted(classes, np.asarray(Y))
    nll = -np.log(p[np.arange(len(yi)), yi])
    entropy = -(p * np.log(p)).sum(1)
    return nll - entropy


def _subject_condition_mean(vals, groups, D=None):
    """Condition-first subject-vote mean (matches the A-line estimand)."""
    vals = np.asarray(vals, float); g = np.asarray(groups)
    subj = []
    for s in np.unique(g):
        m = g == s
        if D is None:
            subj.append(vals[m].mean())
        else:
            Dm = np.asarray(D)[m]
            subj.append(np.mean([vals[m][Dm == c].mean() for c in np.unique(Dm)]))
    return float(np.mean(subj))


def few_label_conditional_risk_test(post: SourcePosterior, Z_audit, Y_audit, groups_audit,
                                    D_audit=None, n_mc=200, seed=0, prior_correct=True):
    """One-sided conditional-MC test that target P(Y|Z) deviates (more surprising) from p_S.
    Returns dict(R_T, p_value, n_subjects, n_epochs, prior_correct)."""
    Z_audit = np.asarray(Z_audit, float); Y_audit = np.asarray(Y_audit)
    g = np.asarray(groups_audit)
    cls = post.classes_
    p = post.proba(Z_audit)
    if prior_correct:
        p = _prior_corrected(p, cls, post.pi_S_, Y_audit)
    R_T = _subject_condition_mean(log_score_residual(p, Y_audit, cls), g, D_audit)
    rng = np.random.default_rng(seed)
    cum = np.cumsum(p, axis=1)
    ge = 1                                                   # +1 (observed) conservative p
    for _ in range(n_mc):
        u = rng.random(len(p))
        Ystar = cls[(u[:, None] > cum).sum(1)]
        Rs = _subject_condition_mean(log_score_residual(p, Ystar, cls), g, D_audit)
        ge += int(Rs >= R_T)
    return dict(R_T=R_T, p_value=ge / (n_mc + 1), n_subjects=int(len(np.unique(g))),
                n_epochs=int(len(Y_audit)), prior_correct=bool(prior_correct))
