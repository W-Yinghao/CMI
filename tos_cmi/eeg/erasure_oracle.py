"""CMI-Trace erasure ORACLE — redefine TOS around an existence oracle for safely-removable subject leakage.

The scientific question TOS never answered: does a subspace exist that (i) carries conditional subject
leakage I(PZ;D|Y,(I-P)Z) and (ii) can be removed with bounded task-information loss I(Y;PZ|(I-P)Z) <= delta?
If no non-zero such P exists, identity fallback is the CORRECT answer, not a method failure.

This module implements the cleanest real-EEG oracle: the **exact-head nullspace oracle**. For a stored linear
task head h(z)=softmax(Wz+b), softmax depends only on logit DIFFERENCES, so it is invariant to any direction
in ker(W_c) where W_c=(I-1 1^T/C)W is the class-centered head. Removing a subspace whose range lies in
ker(W_c) leaves EVERY logit shifted by the same constant -> softmax, probabilities, and predictions are
ALGEBRAICALLY unchanged (task-safety is exact, not statistical). Within ker(W_c) we then fit the
label-conditional subject subspace and delete it.

  P*_head = argmax_P  Delta_D(P)   s.t.   range(P) subseteq ker(W_c),  rank(P) <= k

Positive result = "there exists removable, functionally-UNUSED subject leakage" (the classifier's behaviour
is provably preserved) — NOT "erasure improves DG accuracy". Pure numpy.
"""
from __future__ import annotations
import numpy as np


# ------------------------------------------------------------------ head geometry
def centered_head(W):
    """W_c = (I - 1 1^T / C) W : center the head across classes. Softmax(Wz+b) is invariant to adding any
    class-constant to the logits, so it depends on z only through W_c z. W is [C, d]."""
    W = np.asarray(W, float)
    return W - W.mean(0, keepdims=True)


def head_nullspace_basis(W, tol=1e-8):
    """Orthonormal basis N [d, r] of ker(W_c): directions v with W_c v = 0. Deleting any subspace of span(N)
    leaves softmax(Wz+b) EXACTLY unchanged (r = d - rank(W_c) >= d - (C-1))."""
    Wc = centered_head(W)                                   # [C, d]
    # right singular vectors of Wc with ~zero singular value span the nullspace
    _, s, Vt = np.linalg.svd(Wc, full_matrices=True)        # Vt [d, d]
    smax = s.max() if s.size else 0.0
    rank = int((s > tol * max(smax, 1e-12)).sum()) if smax > 0 else 0
    return Vt[rank:].T                                      # [d, r]


# ------------------------------------------------------------------ subject subspace in a coordinate frame
def label_conditional_subject_axes(U, y, d, k):
    """Top-k label-conditional subject-offset directions in coordinate matrix U [N, r]: rows of the offset
    matrix are sqrt(n_{y,s})*(mean(U|y,s) - mean(U|y)); return their top-k right singular vectors Q [k, r]
    (orthonormal in U-space). k is capped at the available rank."""
    U = np.asarray(U, float); y = np.asarray(y); d = np.asarray(d)
    rows = []
    for c in np.unique(y):
        my = y == c; mu = U[my].mean(0)
        for s in np.unique(d[my]):
            m = my & (d == s)
            if m.sum() > 0:
                rows.append(np.sqrt(m.sum()) * (U[m].mean(0) - mu))
    M = np.vstack(rows) if rows else np.zeros((1, U.shape[1]))
    Vt = np.linalg.svd(M, full_matrices=False)[2]
    kk = int(min(k, Vt.shape[0], U.shape[1]))
    return Vt[:kk]                                          # [k, r]


def random_axes_in_span(N, k, seed):
    """Random orthonormal k-dim subspace expressed in the r-dim coordinate frame of span(N): Q [k, r]."""
    r = N.shape[1]
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((r, min(k, r))))
    return Q.T                                              # [k, r]


def _softmax(logits):
    m = logits - logits.max(1, keepdims=True); e = np.exp(m)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------ the exact-head nullspace oracle
def exact_head_null_projector(Z_fit, y_fit, d_fit, W, k):
    """Fit the label-conditional subject subspace of rank<=k INSIDE ker(W_c) on (Z_fit, y_fit, d_fit).
    Returns (P_HN [d,d], rank, N [d,r]). range(P_HN) subseteq ker(W_c), so removing it preserves softmax."""
    N = head_nullspace_basis(W)                             # [d, r]
    U = np.asarray(Z_fit, float) @ N                        # [N, r] head-null coordinates
    Q = label_conditional_subject_axes(U, y_fit, d_fit, k)  # [k, r]
    P_HN = N @ (Q.T @ Q) @ N.T                              # [d, d] projector within ker(W_c)
    return P_HN, int(Q.shape[0]), N


def random_head_null_projector(W, k, seed):
    """Matched-rank RANDOM subspace control, also constrained to ker(W_c) (so it too preserves softmax)."""
    N = head_nullspace_basis(W)
    Q = random_axes_in_span(N, k, seed)
    return N @ (Q.T @ Q) @ N.T, int(Q.shape[0])


def head_replay_error(Z, Zrm, W, b):
    """Max abs difference of softmax probabilities before/after deletion (algebraic guarantee -> ~0)."""
    W = np.asarray(W, float); b = np.asarray(b, float)
    p0 = _softmax(np.asarray(Z, float) @ W.T + b)
    p1 = _softmax(np.asarray(Zrm, float) @ W.T + b)
    return float(np.abs(p0 - p1).max()), float(np.abs((Z @ W.T) - (Zrm @ W.T)).max())


# ================================================================ EEGNet exhaustive subset oracle
def marginal_subject_basis(Z_fit, subj_fit, max_rank=None):
    """Orthonormal basis V [r, d] of the between-subject mean span (what LEACE removes): rows
    sqrt(n_s)(mean(Z|s)-mean(Z)); top-r right singular vectors, r <= #subjects-1."""
    Z = np.asarray(Z_fit, float); s = np.asarray(subj_fit)
    gm = Z.mean(0)
    rows = [np.sqrt((s == u).sum()) * (Z[s == u].mean(0) - gm) for u in np.unique(s)]
    M = np.vstack(rows) if rows else np.zeros((1, Z.shape[1]))
    Vt = np.linalg.svd(M, full_matrices=False)[2]
    r = min(len(np.unique(s)) - 1, Z.shape[1], Vt.shape[0])
    if max_rank:
        r = min(r, int(max_rank))
    return Vt[:max(r, 0)]                                   # [r, d]


def _subset_utility(V, S, Zs, ys, Zt, yt, seed, head="logreg"):
    """Target bAcc of retraining a fresh head on the S-erased SOURCE and scoring the S-erased TARGET."""
    from tos_cmi.eeg.relaxation_ladder import fresh_head_bacc
    if len(S) == 0:
        Ps = np.zeros((V.shape[1], V.shape[1]))
    else:
        Vs = V[list(S)]                                     # [|S|, d]
        Ps = Vs.T @ Vs
    I = np.eye(V.shape[1])
    return fresh_head_bacc(Zs @ (I - Ps).T, ys, Zt @ (I - Ps).T, yt, head=head, seed=seed)


def exhaustive_subset_oracle(feat, seed=0, head="logreg", max_rank=12, select_frac=0.5):
    """The non-deployable TARGET UPPER-BOUND oracle over ALL 2^r subsets of the SOURCE-fit subject basis.
    Selection on T_select (target labels), reported on T_eval (disjoint) to avoid selection optimism.
    Answers the rank question: is full-basis erasure OVER-REMOVAL (best subset >> full) or is there no useful
    subset (best subset ~ identity)? Reports identity / full-basis / best-prefix / best-subset / same-rank-random,
    each as T_eval bAcc − identity T_eval bAcc."""
    Zs = np.asarray(feat["Z_source"], float); ys = np.asarray(feat["y_source"]).astype(int)
    from tos_cmi.eeg.relaxation_ladder import _dense
    ds = _dense(feat["subj_source"])
    Zt = np.asarray(feat["Z_target"], float); yt = np.asarray(feat["y_target"]).astype(int)
    V = marginal_subject_basis(Zs, ds, max_rank=max_rank)   # [r, d] source-estimated candidate directions
    r = V.shape[0]
    # T_select / T_eval split (stratified by y), within the single target subject
    rng = np.random.default_rng(1234 + seed)
    sel = np.zeros(len(yt), bool)
    for c in np.unique(yt):
        idx = np.where(yt == c)[0]; rng.shuffle(idx)
        sel[idx[: max(1, int(select_frac * len(idx)))]] = True
    ev = ~sel
    if sel.sum() == 0 or ev.sum() == 0 or r == 0:
        return {"dataset": feat.get("dataset", ""), "backbone": feat.get("backbone", ""),
                "heldout_subject": str(feat.get("heldout_subject", "")), "seed": int(feat.get("seed", seed)),
                "rank_available": int(r), "note": "degenerate split/rank"}

    def util(S, mask):
        return _subset_utility(V, S, Zs, ys, Zt[mask], yt[mask], seed, head)

    identity_eval = util((), ev)
    full = tuple(range(r))
    full_eval = util(full, ev)
    # exhaustive subset sweep: select best on T_select
    from itertools import combinations, chain
    n_subsets = 2 ** r
    best_sel, best_S = -1.0, ()
    if r <= max_rank and n_subsets <= 4096:
        subsets = chain.from_iterable(combinations(range(r), m) for m in range(r + 1))
    else:  # beam fallback for large r (not expected here): greedy prefix + singletons
        subsets = [tuple(range(m)) for m in range(r + 1)] + [(i,) for i in range(r)]
    scored = []
    for S in subsets:
        u = util(S, sel)
        scored.append((S, u))
        if u > best_sel:
            best_sel, best_S = u, S
    best_subset_eval = util(best_S, ev)
    # best PREFIX (rank-ordered) selected on T_select
    best_pref_sel, best_pref = -1.0, ()
    for m in range(r + 1):
        u = util(tuple(range(m)), sel)
        if u > best_pref_sel:
            best_pref_sel, best_pref = u, tuple(range(m))
    best_prefix_eval = util(best_pref, ev)
    # matched-rank random (same |best_S|), averaged
    rand_evals = []
    krand = len(best_S)
    for j in range(20):
        rr = np.random.default_rng(7000 + seed + j)
        Sr = tuple(rr.choice(r, size=min(krand, r), replace=False)) if krand > 0 else ()
        rand_evals.append(util(Sr, ev))
    rand_eval = float(np.mean(rand_evals)) if rand_evals else identity_eval
    return {
        "dataset": feat.get("dataset", ""), "backbone": feat.get("backbone", ""),
        "training_method": feat.get("training_method", "erm"),
        "heldout_subject": str(feat.get("heldout_subject", "")), "seed": int(feat.get("seed", seed)),
        "rank_available": int(r), "n_subsets_swept": int(len(scored)),
        "best_subset": list(best_S), "best_subset_size": int(len(best_S)),
        "identity_eval_bacc": float(identity_eval),
        "delta_full_basis": float(full_eval - identity_eval),           # remove ALL r dims
        "delta_best_prefix": float(best_prefix_eval - identity_eval),   # rank-ordered prefix
        "delta_best_subset": float(best_subset_eval - identity_eval),   # arbitrary subset (upper bound)
        "delta_same_rank_random": float(rand_eval - identity_eval),
        "best_subset_beats_random": bool(best_subset_eval - rand_eval > 0),
        "best_subset_beats_full": bool(best_subset_eval - full_eval > 0),
    }


def run_exact_head_null_oracle(feat, k=None, n_perm=30, seed=0, epochs=80, device="cpu"):
    """One cell (a DGCNN audit-npz feat dict with head_W/head_b) -> exact-head-null oracle diagnostics:
    CMI (posterior-KL ruler) on full vs head-null-oracle-deleted vs matched-rank random-null, the replay
    error (must be ~0), and the head-replay task bAcc before/after (must be identical). k defaults to the
    full ker(W_c) subject-subspace rank (= min(#subjects-1, dim ker(W_c)))."""
    from cmi.eval.conditional_subject_leakage import (three_way_support_split, flat_conditional_cmi)
    Z = np.asarray(feat["Z_source"], float); y = np.asarray(feat["y_source"]).astype(int)
    from tos_cmi.eeg.relaxation_ladder import _dense
    d = _dense(feat["subj_source"])
    W = np.asarray(feat["head_W"], float); b = np.asarray(feat["head_b"], float)
    n_cls = int(feat.get("n_cls", len(np.unique(y)))); n_dom = int(len(np.unique(d)))
    if k is None:
        k = min(n_dom - 1, head_nullspace_basis(W).shape[1])

    er, pt, pe, _ = three_way_support_split(y, d, seed=seed)
    # fit the oracle projector on the eraser split ONLY (source)
    P_hn, rank, N = exact_head_null_projector(Z[er], y[er], d[er], W, k)
    P_rand, _ = random_head_null_projector(W, rank, seed=1000 + seed)
    Z_hn = Z @ (np.eye(Z.shape[1]) - P_hn).T
    Z_rn = Z @ (np.eye(Z.shape[1]) - P_rand).T

    # exact-softmax guarantee (both are within ker(W_c))
    p_err_hn, l_err_hn = head_replay_error(Z, Z_hn, W, b)
    p_err_rn, l_err_rn = head_replay_error(Z, Z_rn, W, b)

    # CMI ruler (same posterior-KL ruler as the ladder) with shared cross-fit split
    def cmi(Zx):
        r = flat_conditional_cmi(Zx, y, d, n_cls, n_dom, pt, pe, n_perm=n_perm, seed=seed,
                                 epochs=epochs, device=device, with_residual=False)
        return r["posterior_kl_nats"], r["excess_over_null"], r["perm_p"]
    kl_full, ex_full, p_full = cmi(Z)
    kl_hn, ex_hn, p_hn = cmi(Z_hn)
    kl_rn, ex_rn, p_rn = cmi(Z_rn)

    # task bAcc via the STORED head (must be identical for the oracle since predictions are unchanged)
    from sklearn.metrics import balanced_accuracy_score
    pred0 = (Z @ W.T + b).argmax(1); pred_hn = (Z_hn @ W.T + b).argmax(1)
    bacc0 = float(balanced_accuracy_score(y, pred0)); bacc_hn = float(balanced_accuracy_score(y, pred_hn))

    return {
        "dataset": feat.get("dataset", ""), "backbone": feat.get("backbone", ""),
        "training_method": feat.get("training_method", ""), "heldout_subject": str(feat.get("heldout_subject", "")),
        "seed": int(feat.get("seed", seed)), "oracle_rank": int(rank),
        "ker_Wc_dim": int(N.shape[1]), "feature_dim": int(Z.shape[1]),
        "cmi_full_kl": kl_full, "cmi_headnull_kl": kl_hn, "cmi_randomnull_kl": kl_rn,
        "delta_D_headnull": float(kl_full - kl_hn),          # conditional subject leakage removed by the oracle
        "delta_D_randomnull": float(kl_full - kl_rn),        # matched-rank random-null control
        "excess_full": ex_full, "excess_headnull": ex_hn, "excess_randomnull": ex_rn,
        "perm_p_full": p_full, "perm_p_headnull": p_hn,
        "softmax_replay_err_headnull": p_err_hn, "logit_replay_err_headnull": l_err_hn,
        "softmax_replay_err_randomnull": p_err_rn,
        "task_bacc_before": bacc0, "task_bacc_after_headnull": bacc_hn,
        "task_bacc_unchanged": bool(abs(bacc0 - bacc_hn) < 1e-9),
        "task_safety": "EXACT_ALGEBRAIC",                    # range(P) subseteq ker(W_c) -> softmax invariant
    }
