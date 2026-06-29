"""CIGL Phase 2-real — seed-stability of the node/edge leakage maps.

A leakage map is only credible evidence if it points at the SAME electrodes / edges across independent
seeds. This module flattens the maps, measures mean pairwise rank (or linear) correlation across
seeds, and compares it to a random-map null (each map's entries independently shuffled). A map that is
no more consistent than the null is noise and must not be presented as a "subject-fingerprint" figure.
No scipy dependency (ranks computed inline).
"""
from __future__ import annotations
import numpy as np


def flatten_node_map(node_map):
    """Length-C per-channel map -> 1D vector of length C."""
    return np.asarray(node_map, dtype=np.float64).ravel()


def flatten_edge_map(edge_map):
    """C x C edge map -> strict upper-triangular vector (diagonal ignored), length C*(C-1)/2."""
    m = np.asarray(edge_map, dtype=np.float64)
    assert m.ndim == 2 and m.shape[0] == m.shape[1], "edge map must be square C x C"
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def _avg_rank(a):
    """Average ranks with tie handling (no scipy)."""
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=np.float64)
    ranks[order] = np.arange(a.shape[0], dtype=np.float64)
    # average ranks over tied groups
    sorted_a = a[order]
    i = 0
    n = a.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:                       # a constant map has no defined correlation -> treat as 0
        return 0.0
    return float((a * b).sum() / denom)


def _pairwise_corr(vectors, method):
    vecs = [np.asarray(v, dtype=np.float64) for v in vectors]
    if method == "spearman":
        vecs = [_avg_rank(v) for v in vecs]
    elif method != "pearson":
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    corrs = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            corrs.append(_pearson(vecs[i], vecs[j]))
    return np.asarray(corrs, dtype=np.float64)


def spearman_or_pearson_stability(vectors, method="spearman"):
    """Mean pairwise correlation across the per-seed map vectors (>=2 vectors of equal length)."""
    vectors = [np.asarray(v, dtype=np.float64).ravel() for v in vectors]
    if len(vectors) < 2:
        return dict(method=method, n_vectors=len(vectors), n_pairs=0,
                    mean_corr=float("nan"), min_corr=float("nan"), max_corr=float("nan"),
                    note="need >=2 seeds for a stability estimate")
    lengths = {v.shape[0] for v in vectors}
    assert len(lengths) == 1, f"all map vectors must have equal length, got {lengths}"
    corrs = _pairwise_corr(vectors, method)
    return dict(method=method, n_vectors=int(len(vectors)), n_pairs=int(corrs.size),
                mean_corr=float(corrs.mean()), min_corr=float(corrs.min()),
                max_corr=float(corrs.max()), pairwise=corrs.tolist())


def random_map_stability_null(vectors, n_perm=200, seed=0, method="spearman"):
    """Null: independently shuffle each map's entries, recompute mean pairwise correlation, repeat.

    The observed `spearman_or_pearson_stability` mean_corr should sit ABOVE this null if the maps
    consistently rank the same channels/edges across seeds. Two complementary verdicts are returned:
    `stability_p` is the (+1)-smoothed one-sided permutation p-value P(null >= observed); `above_random`
    is the coarser 95th-percentile threshold (observed > null_q95). They can disagree near the boundary
    — prefer `stability_p`. `degenerate=true` flags the uninformative case where the maps are (near-)
    constant so the null collapses to ~0 and neither verdict is meaningful.
    """
    vectors = [np.asarray(v, dtype=np.float64).ravel() for v in vectors]
    if len(vectors) < 2:
        return dict(method=method, n_perm=0, null_mean=float("nan"), null_std=float("nan"),
                    null_q95=float("nan"), degenerate=True, note="need >=2 seeds")
    rng = np.random.default_rng(seed)
    observed = float(_pairwise_corr(vectors, method).mean())
    null_means = np.empty(int(n_perm), dtype=np.float64)
    for k in range(int(n_perm)):
        shuffled = [v[rng.permutation(v.shape[0])] for v in vectors]
        null_means[k] = _pairwise_corr(shuffled, method).mean()
    q95 = float(np.quantile(null_means, 0.95))
    p = (1.0 + float((null_means >= observed).sum())) / (1.0 + int(n_perm))
    # near-constant maps -> all correlations ~0, null has ~zero spread: the test is uninformative
    degenerate = bool(null_means.std() < 1e-9 and abs(observed) < 1e-9)
    return dict(method=method, n_perm=int(n_perm), observed_mean_corr=observed,
                null_mean=float(null_means.mean()), null_std=float(null_means.std()),
                null_q95=q95, stability_p=p,
                above_random=bool(observed > q95) and not degenerate, degenerate=degenerate)
