"""Source-whitened metric + task-contested candidate basis + the shared, hashed selection rule (F2.1b, the
actual implementation of the subspace pivot). Everything geometric — basis, orthogonality, random projectors,
G1, deletion — is defined in the SOURCE Ledoit-Wolf-whitened metric, and the primary basis is restricted to
the TASK-CONTESTED subspace (row space of the class-centered whitened task head), so the selector can no longer
prefer functionally-unused free-cleaning directions. Deletions act as an affine map back to raw coordinates so
the task interface and the whitening-only baseline stay independent. Pure numpy + sklearn.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression

# ---- the frozen selection RULE (hashed; Gate 5 must re-run the SAME rule, not the same action) ----
RULE = {
    "version": "g1_safe_specific_v1",
    "observable": "G1_whitened_mean_discrepancy_reduction",
    "metric": "source_ledoitwolf_whitened",
    "primary_basis": "cond_contested_rowspace_Wc",
    "task_safety_max_drop": 0.02,
    "specificity_quantile": 0.95,
    "max_rank": 3,
    "identity_fallback": True,
}


def rule_hash():
    return hashlib.sha1(json.dumps(RULE, sort_keys=True).encode()).hexdigest()[:12]


def _hash(arr):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(arr, float)).tobytes()).hexdigest()[:12]


# ============================================================ source whitener (Ledoit-Wolf)
def source_whitener(Zs):
    """A_s = Sigma_s^{-1/2} from the source Ledoit-Wolf shrinkage covariance; A_s_inv = Sigma_s^{1/2}."""
    mu = Zs.mean(0)
    Sigma, _ = ledoit_wolf(Zs - mu)
    ev, V = np.linalg.eigh(Sigma); ev = np.clip(ev, 1e-8, None)
    A = V @ np.diag(ev ** -0.5) @ V.T
    A_inv = V @ np.diag(ev ** 0.5) @ V.T
    return {"mu": mu, "A": A, "A_inv": A_inv, "cond": float(ev.max() / ev.min()),
            "whitening_hash": _hash(A), "source_mean_hash": _hash(mu), "whitening_method": "LedoitWolf"}


def to_whitened(Z, W):
    return (np.asarray(Z, float) - W["mu"]) @ W["A"].T


def from_whitened(Zw, W):
    return Zw @ W["A_inv"].T + W["mu"]


def whitened_delete_fn(U, W):
    """Raw-space apply for deleting the whitened orthonormal directions U [k,D]:
    z -> mu + A_inv ( (I - U^T U) A (z-mu) )."""
    def f(Z):
        if U is None or U.shape[0] == 0:
            return np.asarray(Z, float)
        Zw = to_whitened(Z, W)
        return from_whitened(Zw - (Zw @ U.T) @ U, W)
    return f


# ============================================================ whitened bases + task-head anchor
def _orthonormal(rows, tol=1e-8):
    if np.atleast_2d(rows).shape[0] == 0:
        return np.zeros((0, np.atleast_2d(rows).shape[1]))
    U, s, Vt = np.linalg.svd(np.atleast_2d(rows), full_matrices=False)
    return Vt[s > tol * (s.max() if s.size else 1.0)]


def whitened_cond_basis(Zs_w, ys, ds, max_rank=None):
    """Label-conditional subject-offset directions in WHITENED coordinates: sqrt(n_{d,y})(mu_{d,y}-mu_y)."""
    rows = []
    for c in np.unique(ys):
        my = ys == c; mu = Zs_w[my].mean(0)
        for s in np.unique(ds[my]):
            m = my & (ds == s)
            if m.sum() > 0:
                rows.append(np.sqrt(m.sum()) * (Zs_w[m].mean(0) - mu))
    Vt = _orthonormal(np.vstack(rows) if rows else np.zeros((1, Zs_w.shape[1])))
    return Vt[: (Vt.shape[0] if max_rank is None else min(max_rank, Vt.shape[0]))]


def _svd_threshold(M_rows, max_rank=None, tau=1e-7):
    """SVD of M_rows keeping ONLY directions with singular value > tau*s_max (fixes the numerical-rank defect
    where zero-singular-value SVD-completion directions were returned). Returns (B [r,D], singular_values, r)."""
    if np.atleast_2d(M_rows).shape[0] == 0:
        return np.zeros((0, np.atleast_2d(M_rows).shape[1])), np.array([]), 0
    U, s, Vt = np.linalg.svd(np.atleast_2d(M_rows), full_matrices=False)
    r = int((s > tau * (s.max() if s.size else 1.0)).sum())
    if max_rank:
        r = min(r, int(max_rank))
    return Vt[:r], s, r


def whitened_rule_basis(Zs_w, ys, ds, max_rank=None, tau=1e-7, seed=0):
    """Decision-rule DISAGREEMENT in whitened coords, NUMERICAL-RANK thresholded. Returns (B, singular_values,
    numerical_rank). Per-subject class-centered head; stack (W_d,c - mean_d W_d,c)."""
    n_cls = len(np.unique(ys)); heads = {}
    for u in np.unique(ds):
        m = ds == u
        if len(np.unique(ys[m])) < n_cls or m.sum() < n_cls + 2:
            continue
        clf = LogisticRegression(max_iter=300).fit(Zs_w[m], ys[m])
        W = np.vstack([-clf.coef_[0], clf.coef_[0]]) if clf.coef_.shape[0] == 1 else clf.coef_
        if W.shape[0] == n_cls:
            heads[u] = W - W.mean(0)
    if len(heads) < 2:
        return np.zeros((0, Zs_w.shape[1])), np.array([]), 0
    Wbar = np.mean(list(heads.values()), axis=0)
    rows = np.vstack([h - Wbar for h in heads.values()])
    return _svd_threshold(rows, max_rank, tau)


def whitened_grad_basis(Zs_w, ys, ds, max_rank=None, tau=1e-7, seed=0):
    """Task-gradient DISAGREEMENT in whitened coords, NUMERICAL-RANK thresholded (gradients live in row(W) so
    the true nonzero rank is <= C-1). Returns (B, singular_values, numerical_rank)."""
    if len(np.unique(ys)) < 2:
        return np.zeros((0, Zs_w.shape[1])), np.array([]), 0
    clf = LogisticRegression(max_iter=300).fit(Zs_w, ys)
    W = np.vstack([-clf.coef_[0], clf.coef_[0]]) if clf.coef_.shape[0] == 1 else clf.coef_
    P = clf.predict_proba(Zs_w); oh = np.eye(len(clf.classes_))[np.searchsorted(clf.classes_, ys)]
    Gmat = (P - oh) @ W; gm = Gmat.mean(0)
    rows = np.vstack([Gmat[ds == u].mean(0) - gm for u in np.unique(ds)])
    return _svd_threshold(rows, max_rank, tau)


def head_overlap_enrichment(B, row_w, d):
    """Enrichment of a basis's task-head overlap over the isotropic random expectation rank(Wc)/d:
    [tr(P_B P_row)/rank(B)] / [rank(row)/d]. 1.0 == no enrichment; >1 task-aligned; <1 task-averse."""
    if B.shape[0] == 0 or row_w.shape[0] == 0:
        return float("nan")
    raw = float(np.sum((B @ row_w.T) ** 2) / B.shape[0])
    iso = row_w.shape[0] / d
    return float(raw / iso) if iso > 0 else float("nan")


def whitened_head_rowspace(Zs_w, ys, seed=0, W_stored=None, A_inv=None):
    """Orthonormal basis of the class-centered task-head ROW space (contested), and its complement (free/null),
    in whitened coordinates. EEGNet: fresh logistic on whitened source. DGCNN: stored head W transformed by
    A_inv (W~_c = W_c A_s^{-1})."""
    if W_stored is not None and A_inv is not None:
        Wc = (np.asarray(W_stored, float) - np.asarray(W_stored, float).mean(0)) @ A_inv     # W_c A_s^{-1}
    else:
        clf = LogisticRegression(max_iter=300).fit(Zs_w, ys)
        Wc = np.vstack([-clf.coef_[0], clf.coef_[0]]) if clf.coef_.shape[0] == 1 else clf.coef_
        Wc = Wc - Wc.mean(0)
    _, s, Vt = np.linalg.svd(Wc, full_matrices=True)
    rank = int((s > 1e-8 * (s.max() if s.size else 1.0)).sum())
    row = Vt[:rank]; null = Vt[rank:]
    return row, null


def project_basis(B_w, onto):
    """Project whitened basis rows onto span(`onto`) and re-orthonormalize (ordered by captured energy)."""
    if B_w.shape[0] == 0 or onto.shape[0] == 0:
        return np.zeros((0, B_w.shape[1]))
    P = onto.T @ onto
    return _orthonormal(B_w @ P.T)


def ambient_random_projectors_whitened(D, k, n, seed):
    out = []
    for t in range(n):
        rng = np.random.default_rng(90000 + seed * 131 + k * 17 + t)
        Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
        out.append(Q[:, :k].T)
    return out


# ============================================================ subspace stability metrics (Track B: NOT Jaccard)
def principal_angles_cos(A, B):
    """Cosines of principal angles between row-spaces of orthonormal A [ka,D], B [kb,D] (descending)."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.array([])
    return np.clip(np.linalg.svd(A @ B.T, compute_uv=False), 0, 1)


def normalized_projector_overlap(A, B):
    """||P_A P_B||_F^2 / min(k_a,k_b) in [0,1]; 1 == identical subspace."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float("nan")
    PA, PB = A.T @ A, B.T @ B
    return float(np.sum((PA @ PB) ** 2) / min(A.shape[0], B.shape[0]))     # ||P_A P_B||_F^2 / min(k_a,k_b)


def chordal_distance(A, B):
    """Chordal distance between equal-dim subspaces: sqrt(k - ||A B^T||_F^2)."""
    if A.shape[0] == 0 or B.shape[0] == 0 or A.shape[0] != B.shape[0]:
        return float("nan")
    return float(np.sqrt(max(A.shape[0] - np.sum((A @ B.T) ** 2), 0.0)))
