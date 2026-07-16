"""CMI-Trace DG-identifiability — HARDENED Phase-0/Phase-1 protocol (post-Result-B rescue).

The first-pass DG-erasure oracle (`dg_erasure_oracle.py`) established, on real EEGNet features, that a
target-hindsight search finds candidate beneficial deletions in the source-subject span, yet the greedy
source-LOSO selector recovered ~0% (BNCI2014) / ~4% (BNCI2015) of that gain, while CMI-only selection was
harmful on both. Before upgrading that verdict to a *source-identifiability boundary* three protocol defects
must be fixed (PM directive, 2026-07-16):

  0.1  the target upper bound selected & scored on the SAME target trials  -> optimistic hindsight.
       FIX: CROSS-FITTED oracle: select subset on T_select, report on a DISJOINT T_query.
  0.2  the source-meta selector picked FIXED coordinate indices of a basis estimated from ALL source
       subjects (inner pseudo-target leaked into the candidate basis).
       FIX: TRULY NESTED source-meta over a REFITTABLE selection RULE (family, contested-flag, rank k,
       objective). Every inner pseudo-target is excluded from basis estimation, head fit, AND rank
       selection; the winning rule is then re-fit on all source subjects and applied to the outer target.
  0.3  CMI was minimized inside the greedy loop with a linear proxy (not the validated ruler).
       FIX: search with the cheap proxy; CERTIFY the FINAL selected subset with the posterior-KL ruler
       vs identity and matched-rank random (done by the runner, not here).

Phase-1 rescue compares FOUR candidate bases (marginal / label-conditional-subject / decision-rule-
disagreement / task-gradient-disagreement), each optionally restricted to the CONTESTED subspace (the part
of the basis inside the task head's row space = directions the source head actually uses; its complement,
the free ker(W_c) part, is functionally unused and already handled by the exact-head-null oracle), under
two robust source-meta objectives (mean paired-gain + one-SE smallest-subset, and lower-quartile paired
gain), each with a hard no-harm gate. Pure numpy + sklearn; firewall: target Y enters ONLY final scoring.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from tos_cmi.eeg.erasure_oracle import (head_nullspace_basis, marginal_subject_basis,
                                        label_conditional_subject_axes)


# ============================================================ heads + deletion
def _fit_logreg(Z, y, seed=0):
    """Standardized L2 logistic head. Returns (predict_fn, W[C,d], b[C]) with W/b in ORIGINAL (unstd) space."""
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    clf = LogisticRegression(max_iter=500, C=1.0).fit((Z - mu) / sd, y)
    classes = clf.classes_
    # fold standardization into W,b so logits = Z @ W.T + b in original coordinates
    Wc, bc = clf.coef_ / sd, clf.intercept_ - (clf.coef_ * (mu / sd)).sum(1)
    if len(classes) == 2:                                   # sklearn stores one row for binary -> expand to 2 rows
        W = np.vstack([-Wc[0], Wc[0]]); b = np.array([-bc[0], bc[0]])
    else:
        W, b = Wc, bc
    return clf, W, b, mu, sd


def _bacc(Ztr, ytr, Zte, yte, seed=0):
    """Fresh standardized logistic bAcc (the deployable DG readout)."""
    if len(np.unique(ytr)) < 2 or len(Zte) == 0:
        return float("nan")
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs").fit((Ztr - mu) / sd, ytr)
    return float(balanced_accuracy_score(yte, clf.predict((Zte - mu) / sd)))


def delete_topk(Z, B, k):
    """Remove the span of the FIRST k rows of an ordered orthonormal basis B [r,d]: Z (I - B_k^T B_k)."""
    if k <= 0 or B.shape[0] == 0:
        return Z
    Bk = B[:min(k, B.shape[0])]
    return Z - (Z @ Bk.T) @ Bk


# ============================================================ candidate bases (each -> B [r,d] orthonormal, ordered)
def basis_marginal(Zs, ys, ds, max_rank=None):
    """Between-subject mean span (what LEACE removes); ordered by subject-variance singular value."""
    return marginal_subject_basis(Zs, ds, max_rank=max_rank)


def basis_conditional(Zs, ys, ds, max_rank=None):
    """Label-conditional subject offsets sqrt(n_{d,y})(mu_{d,y}-mu_y); aligned to the estimand I(Z;D|Y)."""
    k = Zs.shape[1] if max_rank is None else int(max_rank)
    return label_conditional_subject_axes(Zs, ys, ds, k)    # already SVD-ordered, [k,d]


def _class_centered_head(Zd, yd, n_cls, seed=0):
    """Per-subject class-centered linear head W_d,c [C,d] (rows sum to ~0). None if a class is missing."""
    if len(np.unique(yd)) < n_cls or len(yd) < n_cls + 2:
        return None
    _, W, _, _, _ = _fit_logreg(Zd, yd, seed)
    if W.shape[0] != n_cls:
        return None
    return W - W.mean(0, keepdims=True)


def basis_rule(Zs, ys, ds, max_rank=None, seed=0):
    """Decision-rule DISAGREEMENT: per-subject class-centered head W_d,c; stack (W_d,c - mean_d W_d,c) over
    subjects+classes; top singular directions = axes along which subjects use DIFFERENT decision rules."""
    n_cls = len(np.unique(ys))
    heads = {u: _class_centered_head(Zs[ds == u], ys[ds == u], n_cls, seed) for u in np.unique(ds)}
    heads = {u: h for u, h in heads.items() if h is not None}
    if len(heads) < 2:
        return np.zeros((0, Zs.shape[1]))
    Wbar = np.mean(list(heads.values()), axis=0)            # [C,d]
    rows = np.vstack([h - Wbar for h in heads.values()])    # [(n_subj*C), d]
    Vt = np.linalg.svd(rows, full_matrices=False)[2]
    r = Zs.shape[1] if max_rank is None else int(max_rank)
    return Vt[:min(r, Vt.shape[0])]


def basis_grad(Zs, ys, ds, max_rank=None, seed=0):
    """Task-gradient DISAGREEMENT: shared head W on all source; per-subject mean representation gradient
    g_d = mean_d W^T(p - onehot(y)); stack (g_d - mean g) over subjects; top singular directions."""
    n_cls = len(np.unique(ys))
    if len(np.unique(ys)) < 2:
        return np.zeros((0, Zs.shape[1]))
    clf, W, b, mu, sd = _fit_logreg(Zs, ys, seed)
    P = clf.predict_proba((Zs - mu) / sd)                   # [N,C] in clf.classes_ order
    oh = np.eye(len(clf.classes_))[np.searchsorted(clf.classes_, ys)]
    G = (P - oh) @ W                                        # [N,d] representation gradient (W is [C,d])
    gm = G.mean(0)
    rows = np.vstack([G[ds == u].mean(0) - gm for u in np.unique(ds)])   # [n_subj, d]
    Vt = np.linalg.svd(rows, full_matrices=False)[2]
    r = min(len(np.unique(ds)) - 1, Zs.shape[1])
    if max_rank:
        r = min(r, int(max_rank))
    return Vt[:max(r, 0)]


BASES = {"marg": basis_marginal, "cond": basis_conditional, "rule": basis_rule, "grad": basis_grad}


def build_basis(family, Zs, ys, ds, max_rank=None, seed=0):
    fn = BASES[family]
    if family in ("rule", "grad"):
        return fn(Zs, ys, ds, max_rank=max_rank, seed=seed)
    return fn(Zs, ys, ds, max_rank=max_rank)


# ============================================================ free / contested split (section 4)
def contested_basis(B, Zs, ys, seed=0, tol=1e-6):
    """Restrict B to the CONTESTED subspace = its projection onto the row space of the class-centered source
    head W_c (directions the source head USES). The complement (inside ker(W_c)) is functionally FREE and is
    handled by the exact-head-null oracle. Returns an ordered orthonormal basis of the contested part."""
    if B.shape[0] == 0 or len(np.unique(ys)) < 2:
        return np.zeros((0, B.shape[1]))
    _, W, _, _, _ = _fit_logreg(Zs, ys, seed)
    N = head_nullspace_basis(W)                             # [d, r_null] = ker(W_c)
    Prow = np.eye(B.shape[1]) - (N @ N.T if N.shape[1] else 0.0)   # projector onto row space of W_c
    Bp = B @ Prow.T                                        # project each candidate direction into used space
    U, s, Vt = np.linalg.svd(Bp, full_matrices=False)
    keep = s > tol * (s.max() if s.size else 1.0)
    return Vt[keep]                                        # ordered contested directions


def get_candidate_basis(family, contested, Zs, ys, ds, max_rank=None, seed=0):
    B = build_basis(family, Zs, ys, ds, max_rank=max_rank, seed=seed)
    if contested:
        B = contested_basis(B, Zs, ys, seed=seed)
    return B


# ============================================================ 0.1 cross-fitted TARGET oracle (upper bound)
def _target_splits(yt, n_splits=5, select_frac=0.5, seed=0, temporal=False):
    """Stratified T_select/T_query masks within the single target subject. `temporal` = first/last block by
    within-class order (a session/drift proxy when no session labels exist)."""
    yt = np.asarray(yt); splits = []
    for j in range(n_splits):
        rng = np.random.default_rng(4242 + seed * 97 + j)
        sel = np.zeros(len(yt), bool)
        for c in np.unique(yt):
            idx = np.where(yt == c)[0]
            if temporal:
                cut = max(1, int(select_frac * len(idx)))
                sel[idx[:cut] if j % 2 == 0 else idx[cut:]] = True   # alternate which block selects
            else:
                rng.shuffle(idx); sel[idx[: max(1, int(select_frac * len(idx)))]] = True
        if sel.any() and (~sel).any():
            splits.append(sel)
    return splits


def _select_subset(Zs, ys, Zt, yt, B, mode, max_k, seed):
    """Return the deletion index set that maximizes T_select bAcc. mode='prefix' -> best top-k of the ordered
    basis; mode='greedy' -> forward-add the arbitrary coordinate that most improves bAcc (STRONGER upper
    bound, >= prefix). Selection uses target labels (hindsight) on the caller's T_select split only."""
    r = B.shape[0]
    ident = _bacc(Zs, ys, Zt, yt, seed)
    if mode == "prefix":
        best_k, best = 0, ident
        for k in range(1, max_k + 1):
            u = _bacc(delete_topk(Zs, B, k), ys, delete_topk(Zt, B, k), yt, seed)
            if np.isfinite(u) and u > best + 1e-6:
                best, best_k = u, k
        return list(range(best_k))
    S, cur = [], ident                                        # greedy arbitrary-coordinate
    for _ in range(max_k):
        cand = []
        for j in range(r):
            if j in S:
                continue
            Bj = B[S + [j]]
            u = _bacc(Zs - (Zs @ Bj.T) @ Bj, ys, Zt - (Zt @ Bj.T) @ Bj, yt, seed)
            cand.append((u, j))
        if not cand:
            break
        bm, bj = max(cand, key=lambda x: (x[0] if np.isfinite(x[0]) else -1))
        if not np.isfinite(bm) or bm <= cur + 1e-6:
            break
        S.append(bj); cur = bm
    return S


def crossfit_target_oracle(Zs, ys, Zt, yt, B, seed=0, n_splits=5, select_frac=0.5, max_k=None, mode="prefix"):
    """UPPER BOUND with hindsight, but honest: pick a deletion subset that maximizes T_select bAcc (target
    labels), report its gain on the DISJOINT T_query. mode 'prefix' (top-k of ordered basis) or 'greedy'
    (arbitrary coordinates, strongest existence test). Averaged over stratified + temporal splits; also
    returns the matched-rank random gain on T_query."""
    r = B.shape[0]; max_k = r if max_k is None else min(max_k, r)
    splits = _target_splits(yt, n_splits, select_frac, seed, temporal=False) + \
        _target_splits(yt, 2, select_frac, seed, temporal=True)
    if r == 0 or not splits:
        return {"delta_query": 0.0, "delta_query_random": 0.0, "k_selected": 0, "identity_query": float("nan"),
                "delta_query_per_split": [], "rank": int(r)}
    dq, dq_rand, ks, idq = [], [], [], []
    for sel in splits:
        Ztsel, ytsel, Ztqry, ytqry = Zt[sel], yt[sel], Zt[~sel], yt[~sel]
        ident_qry = _bacc(Zs, ys, Ztqry, ytqry, seed)
        S = _select_subset(Zs, ys, Ztsel, ytsel, B, mode, max_k, seed)     # select on T_select
        BS = B[S] if S else np.zeros((0, B.shape[1]))
        got_qry = _bacc(Zs - (Zs @ BS.T) @ BS if S else Zs, ys,
                        Ztqry - (Ztqry @ BS.T) @ BS if S else Ztqry, ytqry, seed)
        dq.append(got_qry - ident_qry); ks.append(len(S)); idq.append(ident_qry)
        if S:                                                 # matched-rank random on T_query
            rr = []
            for t in range(10):
                g = np.random.default_rng(9000 + seed + t)
                Br = B[g.choice(r, min(len(S), r), replace=False)]
                rr.append(_bacc(Zs - (Zs @ Br.T) @ Br, ys, Ztqry - (Ztqry @ Br.T) @ Br, ytqry, seed))
            dq_rand.append(float(np.mean(rr)) - ident_qry)
        else:
            dq_rand.append(0.0)
    return {"delta_query": float(np.nanmean(dq)), "delta_query_random": float(np.nanmean(dq_rand)),
            "k_selected": int(np.round(np.mean(ks))), "identity_query": float(np.nanmean(idq)),
            "delta_query_per_split": [float(x) for x in dq], "rank": int(r)}


# ============================================================ 0.2 + section 5: NESTED rule-based source-meta
def _paired_gains_for_rule(Zs, ys, ds, family, contested, max_rank, seed):
    """Inner LOSO over source subjects. For each pseudo-target d_v (excluded from basis, head, selection),
    return the vector of paired gains delta_{d_v}(k) = bAcc_{d_v}(delete top-k) - bAcc_{d_v}(identity) for
    k = 0..K. K is the min rank available across inner folds. Returns (deltas [n_v, K+1], ks list)."""
    subs = np.unique(ds)
    per_v, kmax_common = [], None
    for v in subs:
        tr = ds != v; te = ds == v
        if len(np.unique(ys[tr])) < 2 or te.sum() == 0:
            continue
        Bv = get_candidate_basis(family, contested, Zs[tr], ys[tr], ds[tr], max_rank=max_rank, seed=seed)
        r = Bv.shape[0]
        if r == 0:
            per_v.append(np.array([0.0])); kmax_common = 0 if kmax_common is None else min(kmax_common, 0); continue
        ident = _bacc(Zs[tr], ys[tr], Zs[te], ys[te], seed)
        row = [0.0]
        for k in range(1, r + 1):
            g = _bacc(delete_topk(Zs[tr], Bv, k), ys[tr], delete_topk(Zs[te], Bv, k), ys[te], seed)
            row.append(g - ident)
        per_v.append(np.array(row))
        kmax_common = r if kmax_common is None else min(kmax_common, r)
    if not per_v or kmax_common is None:
        return np.zeros((0, 1)), 0
    K = int(kmax_common)
    D = np.vstack([row[: K + 1] for row in per_v])           # [n_v, K+1]
    return D, K


def select_k_star(D, objective="mean_1se", eps=0.005, alpha=0.0):
    """Choose k* from inner paired-gain matrix D [n_v, K+1] under a robust objective + hard no-harm gate
    Q0.25(delta(k)) >= -eps. objective: 'mean_1se' (largest mean, then smallest k within 1 SE) or
    'cvar25' (largest lower-quartile). Returns (k_star, score, no_harm_ok, per_k dict)."""
    if D.shape[0] == 0 or D.shape[1] == 0:
        return 0, 0.0, True, {}
    n, Kp1 = D.shape
    mean_k = D.mean(0); se_k = D.std(0) / max(np.sqrt(n), 1.0)
    q25_k = np.quantile(D, 0.25, axis=0)
    penalty = alpha * np.arange(Kp1)
    gate = q25_k >= -eps                                     # hard no-harm gate per k
    if objective == "cvar25":
        obj = q25_k - penalty
    else:                                                    # mean + one-SE smallest-subset rule
        obj = mean_k - penalty
    feas = np.where(gate)[0]
    if feas.size == 0:
        return 0, float(obj[0]), False, {"mean": mean_k.tolist(), "q25": q25_k.tolist()}
    if objective == "cvar25":
        k_star = int(feas[np.argmax(obj[feas])])
    else:
        k_best = int(feas[np.argmax(obj[feas])])
        thr = obj[k_best] - se_k[k_best]                     # one-SE rule: smallest feasible k within 1 SE
        within = [k for k in feas if obj[k] >= thr]
        k_star = int(min(within)) if within else k_best
    return k_star, float(obj[k_star]), bool(gate[k_star]), {"mean": mean_k.tolist(), "q25": q25_k.tolist(),
                                                            "se": se_k.tolist()}


def nested_source_meta(Zs, ys, ds, family, contested, max_rank=None, seed=0, objective="mean_1se",
                       eps=0.005, alpha=0.0):
    """Full nested source-only selector for ONE (family, contested, objective). Returns the refittable rule
    (k*) + diagnostics (sign consistency, projector stability across inner folds). No target data touched."""
    D, K = _paired_gains_for_rule(Zs, ys, ds, family, contested, max_rank, seed)
    k_star, score, no_harm, per_k = select_k_star(D, objective, eps, alpha)
    # diagnostics
    sign_consistency = float(np.mean(np.sign(D[:, k_star]) == 1)) if (D.shape[0] and k_star > 0) else float("nan")
    # projector stability: principal-angle similarity of the top-k* subspaces across inner folds
    subs = np.unique(ds); tops = []
    for v in subs:
        tr = ds != v
        if len(np.unique(ys[tr])) < 2:
            continue
        Bv = get_candidate_basis(family, contested, Zs[tr], ys[tr], ds[tr], max_rank=max_rank, seed=seed)
        if Bv.shape[0] >= k_star and k_star > 0:
            tops.append(Bv[:k_star])
    stab = float("nan")
    if len(tops) >= 2 and k_star > 0:
        sims = []
        for i in range(len(tops)):
            for j in range(i + 1, len(tops)):
                s = np.linalg.svd(tops[i] @ tops[j].T, compute_uv=False)
                sims.append(float(np.mean(s)))               # mean cos(principal angle) in [0,1]
        stab = float(np.mean(sims)) if sims else float("nan")
    return {"family": family, "contested": bool(contested), "objective": objective, "k_star": int(k_star),
            "meta_score": float(score), "no_harm_ok": bool(no_harm), "inner_gain_mean": per_k.get("mean", []),
            "inner_gain_q25": per_k.get("q25", []), "sign_consistency": sign_consistency,
            "subspace_stability": stab, "n_inner": int(D.shape[0]), "K_available": int(K)}


def _inner_tops(Zs, ys, ds, family, contested, max_rank, seed, k):
    """Top-k inner-fold subspaces (one per pseudo-target) for the subspace-stability diagnostic."""
    tops = []
    for v in np.unique(ds):
        tr = ds != v
        if len(np.unique(ys[tr])) < 2:
            continue
        Bv = get_candidate_basis(family, contested, Zs[tr], ys[tr], ds[tr], max_rank=max_rank, seed=seed)
        if Bv.shape[0] >= k and k > 0:
            tops.append(Bv[:k])
    return tops


def nested_source_meta_multi(Zs, ys, ds, family, contested, max_rank=None, seed=0,
                             objectives=("mean_1se", "cvar25"), eps=0.005, alpha=0.0):
    """Cost-shared nested source-meta: compute the inner paired-gain matrix D ONCE, then evaluate several
    robust objectives from it (they differ only in select_k_star). Returns {objective: result-dict}."""
    D, K = _paired_gains_for_rule(Zs, ys, ds, family, contested, max_rank, seed)
    out = {}
    for obj in objectives:
        k_star, score, no_harm, per_k = select_k_star(D, obj, eps, alpha)
        sign = float(np.mean(np.sign(D[:, k_star]) == 1)) if (D.shape[0] and k_star > 0) else float("nan")
        tops = _inner_tops(Zs, ys, ds, family, contested, max_rank, seed, k_star) if k_star > 0 else []
        stab = float("nan")
        if len(tops) >= 2:
            sims = [float(np.mean(np.linalg.svd(tops[i] @ tops[j].T, compute_uv=False)))
                    for i in range(len(tops)) for j in range(i + 1, len(tops))]
            stab = float(np.mean(sims)) if sims else float("nan")
        out[obj] = {"family": family, "contested": bool(contested), "objective": obj, "k_star": int(k_star),
                    "meta_score": float(score), "no_harm_ok": bool(no_harm), "sign_consistency": sign,
                    "subspace_stability": stab, "n_inner": int(D.shape[0]), "K_available": int(K)}
    return out


def apply_rule_to_target_full(Zs, ys, ds, Zt, yt, family, contested, k_star, seed=0, n_splits=5, select_frac=0.5):
    """FINAL: re-estimate the basis on ALL source subjects, delete top-k*, fit a fresh head on all source,
    and report the gain on the SAME T_query splits as the cross-fitted oracle (target Y only in scoring)."""
    B = get_candidate_basis(family, contested, Zs, ys, ds, max_rank=None, seed=seed)
    splits = _target_splits(yt, n_splits, select_frac, seed, temporal=False) + \
        _target_splits(yt, 2, select_frac, seed, temporal=True)
    if B.shape[0] == 0 or k_star == 0 or not splits:
        # identity on the same splits -> gain 0 by construction; still report identity for context
        idq = [_bacc(Zs, ys, Zt[~sel], yt[~sel], seed) for sel in splits] if splits else []
        return {"delta_query": 0.0, "k_applied": 0, "identity_query": float(np.nanmean(idq)) if idq else float("nan"),
                "delta_query_per_split": [0.0] * len(splits)}
    dq, idq, dqr = [], [], []
    r = B.shape[0]
    for sel in splits:
        Ztq, ytq = Zt[~sel], yt[~sel]
        ident = _bacc(Zs, ys, Ztq, ytq, seed)
        got = _bacc(delete_topk(Zs, B, k_star), ys, delete_topk(Ztq, B, k_star), ytq, seed)
        dq.append(got - ident); idq.append(ident)
        rr = []                                             # matched-rank random control (same k*)
        for t in range(10):
            g = np.random.default_rng(5500 + seed + t)
            Br = B[g.choice(r, min(k_star, r), replace=False)]
            rr.append(_bacc(Zs - (Zs @ Br.T) @ Br, ys, Ztq - (Ztq @ Br.T) @ Br, ytq, seed))
        dqr.append(float(np.mean(rr)) - ident)
    return {"delta_query": float(np.nanmean(dq)), "k_applied": int(k_star),
            "identity_query": float(np.nanmean(idq)), "delta_query_per_split": [float(x) for x in dq],
            "delta_query_random": float(np.nanmean(dqr))}


# ============================================================ GREEDY source-only identifiability audit
# (the adversarial-verification gate: the nested prefix selector cannot express an arbitrary-coordinate ticket,
#  so it never tested whether the GREEDY target ticket is source-observable. This does — greedy-vs-greedy.)
def _source_loso_gain(Zs, ys, ds, B, S, seed=0):
    """Mean over source subjects of the paired held-out gain of deleting the arbitrary-coordinate set S: for
    each source subject v, fit a fresh head on source-minus-v (S-erased) and score v (S-erased), minus the
    identity baseline. SOURCE-ONLY (no target). This is the source analog of the target oracle's utility."""
    subs = np.unique(ds); gains = []
    BS = B[list(S)] if S else None
    for v in subs:
        tr = ds != v; te = ds == v
        if len(np.unique(ys[tr])) < 2 or te.sum() == 0:
            continue
        base = _bacc(Zs[tr], ys[tr], Zs[te], ys[te], seed)
        if BS is None:
            got = base
        else:
            got = _bacc(Zs[tr] - (Zs[tr] @ BS.T) @ BS, ys[tr], Zs[te] - (Zs[te] @ BS.T) @ BS, ys[te], seed)
        gains.append(got - base)
    return float(np.mean(gains)) if gains else float("nan")


def source_greedy_select(Zs, ys, ds, B, seed=0, max_k=None, tol=1e-4):
    """Greedy forward selection of an ARBITRARY-coordinate deletion set maximizing source-LOSO held-out bAcc
    (source-only; mechanism-matched to the greedy target oracle and to a differentiable supermask). Returns
    the selected index list (refittable rule = 'greedily delete directions that improve source-heldout risk')."""
    r = B.shape[0]; max_k = r if max_k is None else min(max_k, r)
    S, cur = [], 0.0
    for _ in range(max_k):
        cand = [(_source_loso_gain(Zs, ys, ds, B, S + [j], seed), j) for j in range(r) if j not in S]
        if not cand:
            break
        bm, bj = max(cand, key=lambda x: (x[0] if np.isfinite(x[0]) else -1))
        if not np.isfinite(bm) or bm <= cur + tol:
            break
        S.append(bj); cur = bm
    return S


def subspace_alignment(B, Sa, Sb):
    """Mean cos(principal angle) between span(B[Sa]) and span(B[Sb]) in [0,1]; nan if either set is empty."""
    if not Sa or not Sb:
        return float("nan")
    s = np.linalg.svd(B[list(Sa)] @ B[list(Sb)].T, compute_uv=False)
    return float(np.mean(s))


def source_greedy_audit(Zs, ys, ds, Zt, yt, B, seed=0, max_k=None):
    """Decisive identifiability audit for the GREEDY ticket. (1) source-greedy set S_src (source-only);
    (2) target-greedy ticket S_tgt on the FULL target (hindsight reference direction); (3) apply S_src to the
    TRUE target (fresh head on all-source-erased, score all-target-erased) vs identity -> delta_src (source
    selection uses NO target labels, so no cross-fit needed); (4) matched-rank random on target; (5) subspace
    alignment(S_src, S_tgt). Source is identifiable iff delta_src>0, beats random, and aligns with S_tgt."""
    r = B.shape[0]; max_k = r if max_k is None else min(max_k, r)
    if r == 0:
        return {"delta_src": 0.0, "delta_src_random": 0.0, "alignment": float("nan"),
                "k_src": 0, "k_tgt": 0, "rank": 0}
    S_src = source_greedy_select(Zs, ys, ds, B, seed=seed, max_k=max_k)
    S_tgt = _select_subset(Zs, ys, Zt, yt, B, "greedy", max_k, seed)     # hindsight ticket on full target
    ident = _bacc(Zs, ys, Zt, yt, seed)
    BS = B[S_src] if S_src else None
    got = ident if BS is None else _bacc(Zs - (Zs @ BS.T) @ BS, ys, Zt - (Zt @ BS.T) @ BS, yt, seed)
    if S_src:
        rr = []
        for t in range(15):
            g = np.random.default_rng(3300 + seed + t)
            Br = B[g.choice(r, min(len(S_src), r), replace=False)]
            rr.append(_bacc(Zs - (Zs @ Br.T) @ Br, ys, Zt - (Zt @ Br.T) @ Br, yt, seed))
        d_rand = float(np.mean(rr)) - ident
    else:
        d_rand = 0.0
    return {"delta_src": float(got - ident), "delta_src_random": float(d_rand),
            "alignment": subspace_alignment(B, S_src, S_tgt), "k_src": len(S_src), "k_tgt": len(S_tgt),
            "identity_target_bacc": float(ident), "rank": int(r)}


# ============================================================ 0.4 verdict
def recovery_verdict(oracle_delta, oracle_lcb, meta_delta, meta_lcb, meta_random_delta,
                     recovery_min=0.25, practical_min=0.005):
    """Map (cross-fit oracle, source-meta) into the 4 hardened states.
      SOURCE_IDENTIFIABLE_PRACTICAL : oracle LCB>0, meta LCB>0, meta beats random, recovery>=recovery_min
                                       OR meta_delta>=practical_min.
      SOURCE_DETECTABLE_TINY        : meta LCB>0 & beats random but recovery<recovery_min & meta<practical_min.
      TARGET_HINDSIGHT_ONLY         : oracle LCB>0 but meta not distinguishable from random/0.
      NO_CONFIRMED_TICKET           : oracle LCB<=0 (no confirmed beneficial deletion even with hindsight)."""
    recovery = (meta_delta / oracle_delta) if (oracle_delta and np.isfinite(oracle_delta) and oracle_delta > 0) else float("nan")
    meta_sig = (meta_lcb > 0) and (meta_delta > meta_random_delta + 1e-6)
    if oracle_lcb <= 0:
        state = "NO_CONFIRMED_TICKET"
    elif meta_sig and (np.isfinite(recovery) and recovery >= recovery_min or meta_delta >= practical_min):
        state = "SOURCE_IDENTIFIABLE_PRACTICAL"
    elif meta_sig:
        state = "SOURCE_DETECTABLE_TINY"
    else:
        state = "TARGET_HINDSIGHT_ONLY"
    return {"state": state, "recovery_ratio": float(recovery) if np.isfinite(recovery) else None,
            "meta_significant": bool(meta_sig)}
