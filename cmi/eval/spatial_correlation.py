"""CIGL R3 (descriptive) — spatial correlation between two per-node maps (e.g. the node leakage map vs a task
saliency map, or the same map across seeds). Spearman (primary, rank-based, robust) + Pearson (secondary) with
a cluster bootstrap CI over (fold, seed) groups. DESCRIPTIVE ONLY: a high leakage/saliency spatial correlation
is consistent with — but does not prove — task reliance on leakage. The flagship reliance claim is
leakage_removal.py. NaN-/constant-map safe (a constant map has undefined correlation -> nan, never a crash).
"""
from __future__ import annotations
import numpy as np


def _rankdata(a):
    """Average-rank (ties shared) — avoids a scipy dependency."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def _pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]
    if np.std(x) == 0 or np.std(y) == 0:                        # constant map -> undefined
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spatial_correlation(map_a, map_b, method="spearman"):
    """Correlation between two per-node maps. method in {spearman, pearson}. Returns nan for constant/degenerate
    maps rather than raising."""
    a = np.asarray(map_a, dtype=float); b = np.asarray(map_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"map shape mismatch {a.shape} vs {b.shape}")
    if method == "pearson":
        return _pearson(a, b)
    if method == "spearman":
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            return float("nan")
        return _pearson(_rankdata(a[ok]), _rankdata(b[ok]))
    raise ValueError(method)


def bootstrap_correlation_ci(pairs, method="spearman", n_boot=2000, alpha=0.05, seed=0):
    """Cluster bootstrap CI over groups. `pairs` = list of (map_a, map_b) per (fold, seed) group. Resamples
    GROUPS with replacement, correlates the concatenation each draw. Returns point estimate + [lo, hi] +
    n_groups. Degenerate draws (all-nan) are dropped from the CI."""
    pairs = [(np.asarray(a, dtype=float), np.asarray(b, dtype=float)) for a, b in pairs]
    G = len(pairs)
    point = spatial_correlation(np.concatenate([a for a, _ in pairs]),
                                np.concatenate([b for _, b in pairs]), method) if G else float("nan")
    if G < 2:
        return {"point": point, "ci": [float("nan"), float("nan")], "n_groups": G, "method": method}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, G, size=G)
        ca = np.concatenate([pairs[i][0] for i in idx]); cb = np.concatenate([pairs[i][1] for i in idx])
        r = spatial_correlation(ca, cb, method)
        if r == r:                                             # drop nan draws
            boots.append(r)
    if not boots:
        return {"point": point, "ci": [float("nan"), float("nan")], "n_groups": G, "method": method}
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci": [float(lo), float(hi)], "n_groups": G, "method": method}


def spatial_correlation_report(pairs, seed=0):
    """Both methods with bootstrap CIs — the descriptive spatial summary for CIGL_62."""
    return {m: bootstrap_correlation_ci(pairs, method=m, seed=seed) for m in ("spearman", "pearson")}
