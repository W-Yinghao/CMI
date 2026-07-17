"""Mechanism-Subspace Oracle (M0.2 implementation contract; amendment 02). SPEC-frozen estimators + oracle for
the new objective: delete CROSS-SUBJECT TASK-MECHANISM disagreement (not subject identity, not min CMI). All
geometry in the source Ledoit-Wolf-whitened metric. PRIMARY object = class-contrast disagreement via a
generalized eigenproblem; secondaries = shared-plus-residual ridge rule + class-conditional gradient. Existence
oracle: source-only construction -> non-deployable Y_cal exhaustive-rank<=3 selection -> T_query session-macro
score, vs an equal-budget AMBIENT random and a SHARED-OVERLAP-MATCHED random dictionary. No science conclusion
from synthetic/smoke. Only the project owner may explicitly stop a scientific line. Pure numpy + scipy + sklearn.
"""
from __future__ import annotations
import hashlib
from itertools import combinations

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression, Ridge

from tos_cmi.eval.dg_identifiability import _bacc
from tos_cmi.eval.targetx_metric import _orthonormal, _hash

DICT_MAX_RANK = 8
MAX_SUBSET_RANK = 3
ETA_REL = 0.05
SHARED_MATCH_RMSE_TOL = 0.02
SHARED_MATCH_GAP_TOL = 0.01


def _del(Z, U):
    return Z if (U is None or U.shape[0] == 0) else Z - (Z @ U.T) @ U


# ============================================================ P0.1 Helmert contrast + contrast disagreement
def build_helmert_contrast_matrices(C):
    """Orthonormal Helmert contrast matrix H [(C-1),C]: H Hᵀ = I, H 1 = 0."""
    H = np.zeros((C - 1, C))
    for i in range(1, C):
        H[i - 1, :i] = 1.0 / i
        H[i - 1, i] = -1.0
    return H / np.linalg.norm(H, axis=1, keepdims=True)


def build_contrast_disagreement(Zs_w, ys, ds):
    """G_shared, G_dis from Helmert class-contrasts. FAILS_CLOSED if any source subject is missing a class."""
    classes = sorted(np.unique(ys).tolist()); C = len(classes)
    if C < 2:
        return dict(fail_closed=True, reason="fewer_than_2_classes")
    H = build_helmert_contrast_matrices(C); subs = np.unique(ds); Cds = []
    for d in subs:
        m = ds == d
        if any((m & (ys == c)).sum() == 0 for c in classes):
            return dict(fail_closed=True, reason=f"subject_{d}_missing_class")
        Md = np.vstack([Zs_w[m & (ys == c)].mean(0) for c in classes])     # C x p
        Cds.append(H @ Md)                                                 # (C-1) x p
    Cbar = np.mean(Cds, axis=0); m = len(subs)
    G_shared = Cbar.T @ Cbar / (C - 1)
    G_dis = sum((Cd - Cbar).T @ (Cd - Cbar) for Cd in Cds) / (m * (C - 1))
    return dict(fail_closed=False, G_shared=G_shared, G_dis=G_dis, C_d=Cds, Cbar=Cbar, C=C, m=m)


# ============================================================ P0.2 generalized eig construction
def solve_generalized_mechanism_basis(G_dis, G_shared, eta_rel=ETA_REL, max_rank=DICT_MAX_RANK, tol=1e-8):
    """Top-rho of G_dis v = rho (G_shared + eta I) v ; eta = eta_rel*tr(G_shared)/p. Below-resolution if the
    shared task mechanism is ~0 (do not manufacture eigenvectors from noise). Returns the M0.2 builder contract."""
    p = G_shared.shape[0]; tr = float(np.trace(G_shared))
    if tr < 1e-8:
        return dict(below_resolution=True, reason="TASK_MECHANISM_BELOW_RESOLUTION")
    eta = eta_rel * tr / p
    w, V = eigh(G_dis, G_shared + eta * np.eye(p))                          # ascending rho
    order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
    r = int(min((w > tol * max(w.max(), 1e-12)).sum(), max_rank))
    B = _orthonormal(V[:, :r].T) if r > 0 else np.zeros((0, p))            # ambient-orthonormal dictionary
    return dict(below_resolution=False, orthonormal_basis=B, generalized_eigenvalues=[float(x) for x in w[:max_rank]],
                raw_singular_values=[float(np.sqrt(max(x, 0))) for x in w[:max_rank]], numerical_rank=int(B.shape[0]), eta=float(eta))


# ============================================================ P0.7 secondary estimators
def fit_shared_residual_ridge(Zs_w, ys, ds, lam0=1.0, lamD=10.0):
    """B_rule via shared-plus-residual ridge W_d=W_0+dW_d on class-centered one-hot targets; G_rule=sum dW_d^T dW_d.
    Approx: fit W_0 = ridge(lam0) on pooled source; per subject dW_d = ridge(lamD) of residual. Returns builder."""
    classes = sorted(np.unique(ys).tolist()); C = len(classes)
    Y = np.eye(C)[np.searchsorted(classes, ys)]; Y = Y - Y.mean(0)         # class-centered one-hot
    W0 = Ridge(alpha=lam0, fit_intercept=False).fit(Zs_w, Y).coef_.T        # p x C
    dWs = []
    for d in np.unique(ds):
        m = ds == d
        if len(np.unique(ys[m])) < 2:
            continue
        resid = Y[m] - Zs_w[m] @ W0
        dWs.append(Ridge(alpha=lamD, fit_intercept=False).fit(Zs_w[m], resid).coef_.T)   # p x C
    if len(dWs) < 2:
        return dict(fail_closed=True, reason="too_few_subjects")
    G_rule = sum(dW @ dW.T for dW in dWs)                                  # p x p
    s = np.linalg.svd(G_rule, compute_uv=False)
    return dict(fail_closed=False, G_rule=G_rule, raw_singular_values=[float(x) for x in s[:DICT_MAX_RANK]])


def build_class_conditional_gradient_disagreement(Zs_w, ys, ds):
    """B_grad via class-CONDITIONAL gradients g_{d,y}=E[grad_z loss | D=d,Y=y] with ONE fresh source head;
    G_grad = sum_{d,y} (g_{d,y}-gbar_y)(.)^T. Class+subject balanced, numerical-rank thresholded."""
    classes = sorted(np.unique(ys).tolist())
    clf = LogisticRegression(max_iter=300).fit(Zs_w, ys)
    P = clf.predict_proba(Zs_w); oh = np.eye(len(clf.classes_))[np.searchsorted(clf.classes_, ys)]
    W = np.vstack([-clf.coef_[0], clf.coef_[0]]) if clf.coef_.shape[0] == 1 else clf.coef_
    G = (P - oh) @ W                                                       # N x p representation gradient
    gbar_y = {c: G[ys == c].mean(0) for c in classes}
    rows = []
    for d in np.unique(ds):
        for c in classes:
            m = (ds == d) & (ys == c)
            if m.sum() >= 2:
                rows.append(G[m].mean(0) - gbar_y[c])
    if len(rows) < 2:
        return dict(fail_closed=True, reason="too_few_dy_cells")
    Rmat = np.vstack(rows); G_grad = Rmat.T @ Rmat
    s = np.linalg.svd(Rmat, compute_uv=False)
    return dict(fail_closed=False, G_grad=G_grad, raw_singular_values=[float(x) for x in s[:DICT_MAX_RANK]])


def basis_from_gram(G, max_rank=DICT_MAX_RANK, tol=1e-8):
    """Top eigenvectors of a symmetric Gram G (for rule/grad), numerical-rank thresholded -> builder contract."""
    w, V = np.linalg.eigh(G); order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
    r = int(min((w > tol * max(w.max(), 1e-12)).sum(), max_rank))
    B = _orthonormal(V[:, :r].T) if r > 0 else np.zeros((0, G.shape[1]))
    return dict(orthonormal_basis=B, raw_singular_values=[float(x) for x in np.sqrt(np.clip(w[:max_rank], 0, None))],
                numerical_rank=int(B.shape[0]))


# ============================================================ P0.3 exhaustive action family + P0.4 safety
def build_exhaustive_action_family(r, max_subset_rank=MAX_SUBSET_RANK):
    """All subsets of rank<=max of a rank-r dictionary, identity included. r=8,max=3 -> 92 non-empty + identity."""
    acts = [[]]
    for k in range(1, min(max_subset_rank, r) + 1):
        acts += [list(c) for c in combinations(range(r), k)]
    return acts


def source_loso_safety(Zs_w, ys, ds, U):
    drops = []
    for v in np.unique(ds):
        tr, te = ds != v, ds == v
        if len(np.unique(ys[tr])) < 2 or te.sum() == 0:
            continue
        base = _bacc(Zs_w[tr], ys[tr], Zs_w[te], ys[te])
        got = _bacc(_del(Zs_w[tr], U), ys[tr], _del(Zs_w[te], U), ys[te])
        drops.append(base - got)
    d = np.array(drops)
    return dict(mean_drop=float(d.mean()) if d.size else float("nan"), median_drop=float(np.median(d)) if d.size else float("nan"),
                worst_drop=float(d.max()) if d.size else float("nan"), n_positive=int((d > 0).sum()), n_negative=int((d < 0).sum()))


# ============================================================ P0.5 random controls
def build_ambient_random_dictionaries(D, rank, n, block_seed):
    out = []
    for i in range(n):
        rng = np.random.default_rng(block_seed * 100003 + i)
        Q, _ = np.linalg.qr(rng.standard_normal((D, rank)))
        out.append(Q[:, :rank].T)
    return out


def _shared_profile(B, G_shared):
    tr = float(np.trace(G_shared)) + 1e-12
    return np.sort(np.array([float(B[j] @ G_shared @ B[j]) / tr for j in range(B.shape[0])]))[::-1]


def build_shared_profile_matched_dictionaries(B, G_shared, D, rank, n_keep, block_seed, n_pool=5000):
    """Keep the n_keep Haar random rank-`rank` dictionaries whose SORTED shared-overlap profile is closest to B's
    (NO target outcome). Fail-closed unless RMSE<=0.02 AND total-overlap gap<=0.01."""
    tgt = _shared_profile(B, G_shared)
    cand, dist = [], []
    for i in range(n_pool):
        rng = np.random.default_rng(block_seed * 7919 + i)
        Q, _ = np.linalg.qr(rng.standard_normal((D, rank)))
        Qd = Q[:, :rank].T; cand.append(Qd)
        dist.append(float(np.linalg.norm(_shared_profile(Qd, G_shared) - tgt)))
    idx = np.argsort(dist)[:n_keep]
    kept = [cand[i] for i in idx]
    rmse = float(np.sqrt(np.mean([dist[i] ** 2 for i in idx]))) if idx.size else float("inf")
    gap = float(abs(np.mean([_shared_profile(cand[i], G_shared).sum() for i in idx]) - tgt.sum())) if idx.size else float("inf")
    ok = rmse <= SHARED_MATCH_RMSE_TOL and gap <= SHARED_MATCH_GAP_TOL
    return dict(dictionaries=kept, matching_rmse=rmse, total_overlap_gap=gap, match_ok=bool(ok),
                verdict=("OK" if ok else "SHARED_MATCH_CONTROL_FAILED"))


# ============================================================ oracle: select on cal, score on query
def select_on_target_cal(Zs_w, ys, B, actions, Xcal_w, ycal, source_safe=False, ds=None, safety_max=0.02):
    """Exhaustive existence selection using Y_cal (non-deployable). source_safe filters actions whose source-LOSO
    mean drop > safety_max (requires ds). Returns the selected index subset (identity legal)."""
    best_s, best_S = _bacc(Zs_w, ys, Xcal_w, ycal), []
    for S in actions:
        if not S:
            continue
        U = _orthonormal(B[S])
        if source_safe and source_loso_safety(Zs_w, ys, ds, U)["mean_drop"] > safety_max:
            continue
        s = _bacc(_del(Zs_w, U), ys, _del(Xcal_w, U), ycal)
        if s > best_s + 1e-9:
            best_s, best_S = s, S
    return best_S


def score_on_target_query(Zs_w, ys, U, Xq_w, yq, sq):
    def gain(Zq, y):
        return _bacc(_del(Zs_w, U), ys, _del(Zq, U), y) - _bacc(Zs_w, ys, Zq, y)
    per = [gain(Xq_w[sq == s], yq[sq == s]) for s in np.unique(sq)
           if (sq == s).sum() >= 4 and len(np.unique(yq[sq == s])) >= 2]
    return float(np.mean(per)) if per else float(gain(Xq_w, yq))


# ============================================================ result routing (graded; no closeout)
def route_stage_result(dU_specific_lcb, dU_specific_ucb, holm_p, other_dataset_ucb, family, backbone):
    """M1 route A only if confirmatory (contrast/EEGNet) LCB>0 & Holm p<0.05 & other dataset UCB>-0.01. Every
    non-ENRICHED verdict carries a failure record (never a stop; only the project owner stops a line)."""
    if family == "contrast_disagreement" and backbone == "EEGNet" and dU_specific_lcb > 0 and holm_p < 0.05 and other_dataset_ucb > -0.01:
        return dict(verdict="MECHANISM_ENRICHED_OVER_RANDOM", next="M2_source_identifiability")
    if dU_specific_ucb <= 0:
        v = "NO_DETECTED_MECHANISM_ENRICHMENT"; nxt = "training_time_shaping"
    elif backbone == "DGCNN" and dU_specific_lcb > 0:
        v = "BACKBONE_SPECIFIC"; nxt = "analyze_capacity_consistency_confirm"
    elif family in ("rule_disagreement", "gradient_disagreement") and dU_specific_lcb > 0:
        v = "READOUT_OR_ESTIMATOR_DEPENDENT"; nxt = "confirmatory_rerun_before_M2"
    else:
        v = "INCONCLUSIVE"; nxt = "increase_power_or_random_draws"
    return dict(verdict=v, next=nxt, failure_layer=v, evidence="see cluster CIs + matched-random controls",
                learned_lesson="mechanism dictionary not enriched over matched random under current estimator/budget",
                next_hypothesis="revise mechanism object or shape mechanism at training time", next_experiment=nxt)


def config_hash(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
