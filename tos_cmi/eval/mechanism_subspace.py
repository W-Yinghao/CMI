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


# ============================================================ A03.1 shared-null projection (PRIMARY)
NULL_TAU = 1e-7


def _numrank(s, tau=NULL_TAU):
    s = np.asarray(s, float)
    return int((s > tau * max(float(s.max()) if s.size else 0.0, 1e-300)).sum()) if s.size else 0


def shared_null_projector(Cbar, tau=NULL_TAU):
    """N (p x q) with span(N) = numerical null space of the shared class-contrast mechanism row(Cbar), tau on the
    singular values of Cbar. shared_null_dim q = p - rank_tau(Cbar)."""
    p = Cbar.shape[1]
    _, s, Vt = np.linalg.svd(Cbar, full_matrices=True)                     # Vt is p x p
    r_shared = _numrank(s, tau)
    N = Vt[r_shared:].T                                                    # p x (p - r_shared)
    return N, int(N.shape[1]), int(r_shared)


def build_shared_null_contrast_basis(cd, tau=NULL_TAU, max_rank=DICT_MAX_RANK, tol=1e-8):
    """A03.1 PRIMARY: B_contrast = N @ TopEig_r(N^T G_dis N), r = min(max_rank, numrank(G_dis^N)); N = shared null
    of Cbar. Non-degeneracy gate: q>r AND numrank(G_dis^N)>0, else SHARED_NULL_CONTROL_LOW_DOF /
    TASK_MECHANISM_BELOW_RESOLUTION. Returns the builder contract + N + shared_null_dim."""
    N, q, r_shared = shared_null_projector(cd["Cbar"], tau)
    if q == 0:
        return dict(fail_closed=True, reason="TASK_MECHANISM_BELOW_RESOLUTION", shared_null_dim=0)
    Gdis_N = N.T @ cd["G_dis"] @ N                                         # q x q
    w, V = np.linalg.eigh(Gdis_N); order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
    nrk = int((w > tol * max(float(w.max()), 1e-12)).sum())
    if nrk == 0:
        return dict(fail_closed=True, reason="TASK_MECHANISM_BELOW_RESOLUTION", shared_null_dim=q)
    r = int(min(max_rank, nrk))
    if q <= r:                                                            # no room for a distinct random control
        return dict(fail_closed=True, reason="SHARED_NULL_CONTROL_LOW_DOF", shared_null_dim=q, dictionary_rank=r)
    B = _orthonormal((N @ V[:, :r]).T)                                    # p x r, ambient-orthonormal
    return dict(fail_closed=False, orthonormal_basis=B, N=N, shared_null_dim=q, shared_rank=r_shared,
                raw_matrix=(N @ V[:, :r]), generalized_eigenvalues=[float(x) for x in w[:max_rank]],
                raw_singular_values=[float(np.sqrt(max(x, 0))) for x in w[:max_rank]], numerical_rank=int(B.shape[0]))


def build_shared_null_gram_basis(G, N, max_rank=DICT_MAX_RANK, tol=1e-8):
    """A03.3 null-projected secondary basis for rule/grad: B = N @ TopEig_r(N^T G N) (reuses the contrast N)."""
    if N is None or N.shape[1] == 0:
        return dict(orthonormal_basis=np.zeros((0, G.shape[1])), numerical_rank=0, raw_singular_values=[])
    Gn = N.T @ G @ N
    w, V = np.linalg.eigh(Gn); order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
    r = int(min(max_rank, (w > tol * max(float(w.max()), 1e-12)).sum(), N.shape[1]))
    B = _orthonormal((N @ V[:, :r]).T) if r > 0 else np.zeros((0, G.shape[1]))
    return dict(orthonormal_basis=B, raw_matrix=(N @ V[:, :r]) if r > 0 else np.zeros((G.shape[1], 0)),
                raw_singular_values=[float(np.sqrt(max(x, 0))) for x in w[:max_rank]], numerical_rank=int(B.shape[0]))


# ============================================================ P0.7 secondary estimators
def fit_shared_residual_ridge(Zs_w, ys, ds, lam0=1.0, lamD=10.0, kkt_tol=1e-6):
    """B_rule via the EXACT joint shared-residual ridge (P0.2). Solve
        min_{W0,{dW_d}} sum_d ||Y_d - X_d(W0+dW_d)||^2 + lam0||W0||^2 + lamD sum_d||dW_d||^2
    by the joint normal equations (block elimination of each dW_d), NOT the pooled-then-residual approximation.
    G_rule = sum_d dW_d^T dW_d. Returns the joint solution + a KKT stationarity residual."""
    classes = sorted(np.unique(ys).tolist()); C = len(classes); p = Zs_w.shape[1]
    Y = np.eye(C)[np.searchsorted(classes, ys)]; Y = Y - Y.mean(0)         # class-centered one-hot
    subs = [d for d in np.unique(ds) if len(np.unique(ys[ds == d])) >= 2]
    if len(subs) < 2:
        return dict(fail_closed=True, reason="too_few_subjects")
    # Stationarity: for each d, (X_d^T X_d + lamD I) dW_d = X_d^T(Y_d - X_d W0)  => dW_d = M_d X_d^T(Y_d - X_d W0),
    # M_d = (X_d^T X_d + lamD I)^{-1}. Substitute into the W0 equation:
    #   [lam0 I + sum_d A_d(I - A_d? ...)] ... -> assemble exactly via S_d = X_d^T X_d, b_d = X_d^T Y_d.
    Sd, bd, Md = {}, {}, {}
    for d in subs:
        Xd = Zs_w[ds == d]; Sd[d] = Xd.T @ Xd; bd[d] = Xd.T @ Y[ds == d]
        Md[d] = np.linalg.inv(Sd[d] + lamD * np.eye(p))
    # W0 normal equation: (lam0 I + sum_d [S_d - S_d M_d S_d]) W0 = sum_d [b_d - S_d M_d b_d]
    LHS = lam0 * np.eye(p); RHS = np.zeros((p, C))
    for d in subs:
        SMS = Sd[d] @ Md[d]
        LHS += Sd[d] - SMS @ Sd[d]; RHS += bd[d] - SMS @ bd[d]
    W0 = np.linalg.solve(LHS, RHS)                                         # p x C
    dWs = {d: Md[d] @ (bd[d] - Sd[d] @ W0) for d in subs}
    # KKT stationarity residuals at the solution
    r0 = lam0 * W0 - sum(bd[d] - Sd[d] @ (W0 + dWs[d]) for d in subs)      # d/dW0 = 0
    rd = max(float(np.abs(lamD * dWs[d] - (bd[d] - Sd[d] @ (W0 + dWs[d]))).max()) for d in subs)
    kkt = float(max(np.abs(r0).max(), rd)) / (float(np.abs(W0).max()) + 1e-9)
    G_rule = sum(dWs[d] @ dWs[d].T for d in subs)                         # p x p representation-space Gram
    s = np.linalg.svd(G_rule, compute_uv=False)
    return dict(fail_closed=False, G_rule=G_rule, kkt_residual=kkt, kkt_ok=bool(kkt <= kkt_tol),
                raw_singular_values=[float(x) for x in s[:DICT_MAX_RANK]])


def build_class_conditional_gradient_disagreement(Zs_w, ys, ds):
    """B_grad via class-CONDITIONAL gradients g_{d,y}=E[grad_z loss | D=d,Y=y] with ONE fresh source head. EQUAL
    SUBJECT weighting (P0.3): gbar_y = mean over SUBJECTS of g_{d,y} (not a trial-count pooled mean), and all (d,y)
    residuals enter G_grad with equal weight -> invariant to per-subject trial-count duplication."""
    classes = sorted(np.unique(ys).tolist())
    clf = LogisticRegression(max_iter=300).fit(Zs_w, ys)
    P = clf.predict_proba(Zs_w); oh = np.eye(len(clf.classes_))[np.searchsorted(clf.classes_, ys)]
    W = np.vstack([-clf.coef_[0], clf.coef_[0]]) if clf.coef_.shape[0] == 1 else clf.coef_
    G = (P - oh) @ W                                                       # N x p representation gradient
    g_dy = {}                                                             # (d,y) -> per-cell mean gradient
    for d in np.unique(ds):
        for c in classes:
            m = (ds == d) & (ys == c)
            if m.sum() >= 2:
                g_dy[(d, c)] = G[m].mean(0)
    gbar_y = {}                                                          # EQUAL-subject mean over subjects present for class c
    for c in classes:
        gs = [g_dy[(d, c)] for d in np.unique(ds) if (d, c) in g_dy]
        if gs:
            gbar_y[c] = np.mean(gs, axis=0)
    rows = [g_dy[(d, c)] - gbar_y[c] for (d, c) in g_dy if c in gbar_y]
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


# ============================================================ P0.4/A03.2 random controls
def cell_seed(*parts):
    """Cell-specific deterministic seed: hash of the full cell identity so Monte-Carlo error is INDEPENDENT across
    (dataset, backbone, subject, model_seed, basis_family, control_family, block_id, random_id)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16)


def build_ambient_random_dictionaries(D, rank, n, seed):
    """Ambient Haar rank-`rank` dictionaries in R^D (SECONDARY control). `seed` should be a cell_seed(...) value."""
    out = []
    for i in range(n):
        rng = np.random.default_rng((int(seed) + i * 100003) % (2**63))
        Q, _ = np.linalg.qr(rng.standard_normal((D, rank)))
        out.append(Q[:, :rank].T)
    return out


def build_shared_null_haar_dictionaries(N, rank, n, seed):
    """A03.2 PRIMARY control: Haar rank-`rank` subspaces WITHIN the shared-null space span(N) -> B_rand = N @
    Haar(Gr(rank, q)), orthonormalized in ambient coords. Same shared_overlap=0 as the informed basis; differs
    only in G_dis alignment (conditional randomization). `seed` should be a cell_seed(...) value."""
    q = N.shape[1]; out = []
    for i in range(n):
        rng = np.random.default_rng((int(seed) + i * 100003) % (2**63))
        Qc, _ = np.linalg.qr(rng.standard_normal((q, rank)))              # Haar in Gr(rank, q)
        out.append(_orthonormal((N @ Qc[:, :rank]).T))                    # p x rank, ambient-orthonormal
    return out


# ============================================================ A03.4 diagnostics + stats
def shared_overlap(B, G_shared):
    """Fraction of B's directions' energy in the shared class-contrast mechanism (should be ~0 for shared-null B)."""
    tr = float(np.trace(G_shared)) + 1e-12
    return float(sum(B[j] @ G_shared @ B[j] for j in range(B.shape[0])) / tr)


def gdis_capture_fraction(U, G_dis, N=None):
    """tr(P_U G_dis) / tr(P_ref G_dis); ref = the shared-null projector N N^T if given, else full trace."""
    if U is None or U.shape[0] == 0:
        return 0.0
    num = float(np.trace(U @ G_dis @ U.T))
    den = float(np.trace(N.T @ G_dis @ N)) if (N is not None and N.shape[1]) else float(np.trace(G_dis))
    return num / (den + 1e-12)


def exact_sign_flip_p(vals, max_enum=1 << 22):
    """Exact one-sided sign-flip permutation p for H1: mean>0. Enumerate all 2^n sign patterns of the per-subject
    values when 2^n <= max_enum (BNCI2014 n=9 ->512, BNCI2015 n=12 ->4096); statistic = mean. p = fraction of sign
    patterns with mean >= observed. Falls back to a seeded 200k-sample MC if 2^n exceeds max_enum."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float); n = v.size
    if n == 0:
        return float("nan")
    obs = float(v.mean())
    if (1 << n) <= max_enum:
        signs = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1) * 2 - 1   # 2^n x n in {-1,+1}
        means = (signs * np.abs(v)).mean(1)
        return float((means >= obs - 1e-12).sum() / (1 << n))
    rng = np.random.default_rng(0); S = 200000
    signs = rng.integers(0, 2, size=(S, n)) * 2 - 1
    means = (signs * np.abs(v)).mean(1)
    return float((1 + (means >= obs - 1e-12).sum()) / (S + 1))


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
    mean drop > safety_max (requires ds). Returns the selected index subset (identity legal). Safety is checked
    LAZILY -- only for an action that would improve the current best cal score -- which yields the same
    best-safe-and-improving action while computing the (expensive) source-LOSO safety O(#improving) not O(#actions)."""
    best_s, best_S = _bacc(Zs_w, ys, Xcal_w, ycal), []
    for S in actions:
        if not S:
            continue
        U = _orthonormal(B[S])
        s = _bacc(_del(Zs_w, U), ys, _del(Xcal_w, U), ycal)
        if s <= best_s + 1e-9:
            continue
        if source_safe and source_loso_safety(Zs_w, ys, ds, U)["mean_drop"] > safety_max:
            continue
        best_s, best_S = s, S
    return best_S


def score_on_target_query(Zs_w, ys, U, Xq_w, yq, sq):
    def gain(Zq, y):
        return _bacc(_del(Zs_w, U), ys, _del(Zq, U), y) - _bacc(Zs_w, ys, Zq, y)
    per = [gain(Xq_w[sq == s], yq[sq == s]) for s in np.unique(sq)
           if (sq == s).sum() >= 4 and len(np.unique(yq[sq == s])) >= 2]
    return float(np.mean(per)) if per else float(gain(Xq_w, yq))


# ============================================================ result routing (graded; no closeout)
PRIMARY_CONTROL = "SHARED_NULL_HAAR"


def route_stage_result(dU_specific_lcb, dU_specific_ucb, holm_p, other_dataset_ucb, family, backbone,
                        specificity_control=PRIMARY_CONTROL):
    """M1 route A only if confirmatory (contrast/EEGNet) LCB>0 & Holm sign-flip p<0.05 & other dataset UCB>-0.01
    AND the PRIMARY (shared-null-Haar conditional-randomization) specificity control actually ran. If only a
    fallback ran (shared-null control fail-closed / degenerate), ENRICHMENT is NOT granted -- routing is gated, not
    merely flagged, so M2 cannot unlock on an ambient-only comparison. Every non-ENRICHED verdict carries a failure
    record (never a stop; only the project owner stops a line)."""
    enrich_stats = (family == "contrast_disagreement" and backbone == "EEGNet" and dU_specific_lcb > 0
                    and holm_p < 0.05 and other_dataset_ucb > -0.01)
    if enrich_stats and specificity_control == PRIMARY_CONTROL:
        return dict(verdict="MECHANISM_ENRICHED_OVER_RANDOM", next="M2_source_identifiability")
    if enrich_stats and specificity_control != PRIMARY_CONTROL:
        # statistics clear route A but the primary shared-null control did not run -> fail-closed, do NOT unlock M2.
        return dict(verdict="ENRICHED_VS_AMBIENT_ONLY_MATCHED_CONTROL_UNAVAILABLE",
                    next="resolve_specificity_control_before_M2", failure_layer="specificity_control_unavailable",
                    evidence="stats clear route A but shared-null-Haar control did not run; only a fallback ran",
                    learned_lesson="the primary conditional-randomization control did not run; fallback cannot certify specificity",
                    next_hypothesis="the shared-null-Haar control must run non-degenerately",
                    next_experiment="resolve_specificity_control_before_M2")
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
