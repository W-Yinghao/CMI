"""Track B -- source-OOD benefit gate (method-deepening phase). SOURCE-ONLY decision: given a frozen source
representation and a candidate erasure, decide ACCEPT / REJECT / ABSTAIN without ever looking at the held-out
target. Three fixed, pre-registered layers (see notes/METHOD_DEEPENING_PLAN.md):
  * safety : within-source (stratified) task-bAcc drop from erasing; REJECT if drop UCB > SAFETY_EPS.
  * benefit: source leave-one-source-subject-out pseudo-target ΔbAcc (erased-full); ACCEPT only if LCB > BENEFIT_LCB.
  * domain : subject-decode drop -- DIAGNOSTIC only, never sufficient for accept.
The target is used ONLY in the post-hoc audit (run_trackB), never here.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from tos_cmi.score_fisher import (ScoreFisherConfig, _metric, _cross_fit_fisher, _SplitPlan,
                                  candidate_order, _m_proj)
from tos_cmi.eeg.erasure_baselines import _ids, leace_eraser, inlp_eraser, rlace_eraser

SAFETY_EPS = 0.02      # task-drop UCB must be <= this to be "safe"
BENEFIT_LCB = 0.01     # source-LOSO benefit LCB must be > this to "accept"
METHODS = ["TOS_VD", "LEACE", "RLACE", "INLP", "random_k"]   # 'full' is the reference (benefit 0 by def)


def build_eraser(Zf, yf, subjf, n_cls, method, cfg, seed):
    """Fit ONE method's eraser on the given (source) subset. Returns apply(X)."""
    ns = len(set(subjf.tolist()))
    if method == "full":
        return (lambda X: X)
    if method == "LEACE":
        return leace_eraser(Zf, np.eye(ns)[_ids(subjf)[0]])
    if method == "INLP":
        return inlp_eraser(Zf, _ids(subjf)[0])
    if method == "RLACE":
        return rlace_eraser(Zf, _ids(subjf)[0], seed=seed)
    zdim = Zf.shape[1]
    d01 = _ids(subjf)[0]
    plan = _SplitPlan(len(yf), cfg.n_folds, 1)
    M = _metric(Zf, yf, n_cls, cfg)
    G_Y = _cross_fit_fisher(Zf, yf, None, n_cls, zdim, 0, cfg, plan, 0)
    G_DgY = _cross_fit_fisher(Zf, d01, np.eye(n_cls)[yf], ns, zdim, n_cls, cfg, plan, 100)
    V_D = candidate_order(G_DgY, G_Y, M, cfg, 0.0)[0]
    k = int(V_D.shape[1])
    if method == "TOS_VD":
        return (lambda X: X) if k == 0 else (lambda X: X - X @ _m_proj(V_D, M).T)
    if method == "random_k":
        Vr = np.random.default_rng(seed).standard_normal((zdim, max(k, 1)))
        return (lambda X: X) if k == 0 else (lambda X: X - X @ _m_proj(Vr, M).T)
    raise ValueError(method)


def _bacc(Ztr, ytr, Zte, yte):
    if len(np.unique(ytr)) < 2:
        return float("nan")
    h = LogisticRegression(max_iter=200, C=1.0).fit(Ztr, ytr)
    return float(balanced_accuracy_score(yte, h.predict(Zte)))


def _subj_acc(Ztr, dtr, Zte, dte):
    if len(np.unique(dtr)) < 2:
        return float("nan")
    return float((LogisticRegression(max_iter=200).fit(Ztr, dtr).predict(Zte) == dte).mean())


def gate_signals(Zs, ys, subj, n_cls, method, cfg, seed, n_pseudo=8):
    """SOURCE-ONLY signals for one dump+method. Returns dict:
      task_drop  : within-source stratified task-bAcc drop (safety; higher = worse)
      benefit    : list of source-LOSO pseudo-target ΔbAcc (erased-full) (benefit; higher = better)
      domain_gain: within-source subject-decode drop (diagnostic)"""
    subs = sorted(set(subj.tolist()))
    rng = np.random.default_rng(seed)
    # --- safety + domain: within-source STRATIFIED trial split (subjects in both halves) ---
    perm = rng.permutation(len(ys)); cut = len(ys) // 2
    A = np.zeros(len(ys), bool); A[perm[:cut]] = True; B = ~A
    E = build_eraser(Zs[A], ys[A], subj[A], n_cls, method, cfg, seed)
    task_drop = _bacc(Zs[A], ys[A], Zs[B], ys[B]) - _bacc(E(Zs[A]), ys[A], E(Zs[B]), ys[B])
    subj01, _ = _ids(subj)
    domain_gain = _subj_acc(Zs[A], subj01[A], Zs[B], subj01[B]) - _subj_acc(E(Zs[A]), subj01[A], E(Zs[B]), subj01[B])
    # --- benefit: source leave-one-source-subject-out pseudo-target (subsample n_pseudo subjects) ---
    pick = list(rng.permutation(subs)[:min(n_pseudo, len(subs))])
    benefit = []
    for s in pick:
        tr = subj != s; te = subj == s
        if len(np.unique(ys[te])) < 2 or len(np.unique(ys[tr])) < 2:
            continue
        Es = build_eraser(Zs[tr], ys[tr], subj[tr], n_cls, method, cfg, seed)
        full_b = _bacc(Zs[tr], ys[tr], Zs[te], ys[te])
        eras_b = _bacc(Es(Zs[tr]), ys[tr], Es(Zs[te]), ys[te])
        benefit.append(eras_b - full_b)
    return {"task_drop": float(task_drop), "benefit": benefit, "domain_gain": float(domain_gain),
            "n_pseudo": len(benefit)}


def _boot_bound(vals, clusters, side, B=2000, rng=None):
    """Cluster bootstrap one-sided bound. side='upper'->97.5 pct, side='lower'->2.5 pct of the mean."""
    vals = np.asarray(vals, float)
    clusters = np.asarray(clusters)[:len(vals)]           # align clusters to vals BEFORE NaN masking
    mask = ~np.isnan(vals)                                 # drop NaNs from vals AND their clusters together
    vals, clusters = vals[mask], clusters[mask]            # (prior code truncated clusters[:len] -> misalign if a NaN wasn't at the tail)
    if len(vals) == 0:
        return float("nan")
    rng = rng or np.random.default_rng(0)
    by = {}
    for i, c in enumerate(clusters):
        by.setdefault(c, []).append(i)
    uniq = sorted(by); means = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[c] for c in pick])
        means.append(vals[idx].mean())
    return float(np.percentile(means, 97.5 if side == "upper" else 2.5))


def gate_action(task_drop_ucb, benefit_lcb, safety_eps=SAFETY_EPS, benefit_thr=BENEFIT_LCB):
    """Pre-registered action. safety_eps/benefit_thr default to the module constants but may be supplied
    from a frozen config so the thresholds are authoritative rather than decorative. NaN safety is treated
    as UNCERTIFIABLE -> never ACCEPT/REJECT (closes the hole where a NaN task-drop bypassed the safety guard).
    All non-NaN behaviour is identical to the prior version (Track B results unchanged: its UCBs are non-NaN)."""
    if np.isnan(task_drop_ucb):
        return "ABSTAIN"           # safety undefined -> cannot certify safe; refuse
    if task_drop_ucb > safety_eps:
        return "REJECT"            # unsafe: erasing hurts the task
    if not np.isnan(benefit_lcb) and benefit_lcb > benefit_thr:
        return "ACCEPT"            # safe AND source-OOD benefit supported
    return "ABSTAIN"               # safe but no supported benefit
