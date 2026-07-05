"""CIGL_66 — measurement→control gap diagnostics. WITHIN-RUN scalar diagnostics of a single run's frozen
representation + task head; cross-method comparison is done on the SCALARS, never on raw cross-model coordinates
(ERM and CIGL live in different, arbitrarily-rotated coordinate systems — comparing their raw axes is undefined).

Primary question: when CIGL reduced the measured label-conditional subject leakage, what changed in the
representation, and why did the task classifier's reliance on the residual subject subspace not fall?

All source-only (fit on d != target); target labels never enter a fit. Pure numpy.
"""
from __future__ import annotations
import numpy as np

DEFAULT_KS = (1, 2, 4, 8)


def subject_offset_matrix(z, y, d):
    """Label-conditional subject-offset matrix (same construction as R3's subspace fit): rows are
    sqrt(count)-weighted Δ_{y,d} = mean(z|y,d) − mean(z|y). Returns M [n_offsets, Zdim]. Source-only input."""
    z = np.asarray(z, dtype=float); y = np.asarray(y); d = np.asarray(d)
    rows = []
    for yy in np.unique(y):
        my = y == yy
        mu_y = z[my].mean(0)
        for dd in np.unique(d[my]):
            m = my & (d == dd)
            if m.sum() > 0:
                rows.append(np.sqrt(m.sum()) * (z[m].mean(0) - mu_y))
    return np.stack(rows) if rows else np.zeros((1, z.shape[1]))


def _svals(M):
    return np.linalg.svd(np.asarray(M, dtype=float), full_matrices=False)[1]


def effective_rank(svals):
    """Roy–Vetterli effective rank exp(−Σ p_i ln p_i), p_i = σ_i/Σσ. Finite for rank-deficient input; 0 if all
    singular values are ~0."""
    s = np.asarray(svals, dtype=float); s = s[s > 0]
    tot = s.sum()
    if tot <= 0 or s.size == 0:
        return 0.0
    p = s / tot
    return float(np.exp(-np.sum(p * np.log(p))))


def topk_energy_fraction(svals, k):
    """(Σ_{i<k} σ_i²)/(Σ σ_i²) — cumulative energy in the top-k directions. Monotonic non-decreasing in k, in
    [0,1]. 0 if there is no energy."""
    s = np.sort(np.asarray(svals, dtype=float))[::-1]
    e = s ** 2; tot = e.sum()
    if tot <= 0:
        return 0.0
    return float(e[:max(0, int(k))].sum() / tot)


def spectrum_diagnostics(M, ks=DEFAULT_KS):
    """Within-run subject-subspace spectrum of the offset matrix M. Returns singular values (list), total energy,
    top-k energy fractions, and effective rank."""
    s = _svals(M)
    out = {"singular_values": [float(x) for x in s], "total_subject_energy": float((s ** 2).sum()),
           "effective_rank": effective_rank(s), "n_offsets": int(np.asarray(M).shape[0])}
    for k in ks:
        out[f"top{k}_energy_fraction"] = topk_energy_fraction(s, k)
    return out


def subject_subspace(M, k):
    """Top-k right singular vectors of M -> [k, Zdim] orthonormal (the label-conditional subject subspace S_k)."""
    _, _, Vt = np.linalg.svd(np.asarray(M, dtype=float), full_matrices=False)
    kk = min(int(k), Vt.shape[0])
    return Vt[:kk]


def task_head_alignment(W, S_k):
    """Fraction of the task head's row-space energy lying in the subject subspace S_k:
    ||P_{S_k} Wᵀ||_F² / ||Wᵀ||_F², P_{S_k} = S_kᵀ S_k. In [0,1]; INVARIANT to the orthonormal basis of S_k (P is
    basis-independent). W [n_cls, Zdim], S_k [k, Zdim]. 0 if the head is all-zero."""
    W = np.asarray(W, dtype=float); S_k = np.asarray(S_k, dtype=float)
    Wt = W.T                                                    # [Zdim, n_cls]
    denom = float((Wt ** 2).sum())
    if denom <= 0:
        return 0.0
    P = S_k.T @ S_k                                             # [Zdim, Zdim] projector onto row(S_k)
    num = float(((P @ Wt) ** 2).sum())
    return float(num / denom)


def alignment_curve(M, W, ks=DEFAULT_KS):
    """task_head_alignment for each k -> dict k -> alignment. Monotonic non-decreasing in k (bigger subspace)."""
    return {int(k): task_head_alignment(W, subject_subspace(M, k)) for k in ks}
