"""Frozen latent masking for the P0 CEDAR gate."""

from __future__ import annotations

import numpy as np


def apply_diagonal_mask(z: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Apply a diagonal latent mask without changing feature dimensionality."""

    z = np.asarray(z, dtype=np.float64)
    keep_mask = np.asarray(keep_mask, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("z must be 2D")
    if keep_mask.shape != (z.shape[1],):
        raise ValueError("keep_mask must have shape [z_dim]")
    return z * keep_mask[None, :]


def mask_from_drop_dims(n_dims: int, drop_dims: list[int] | np.ndarray) -> np.ndarray:
    if n_dims <= 0:
        raise ValueError("n_dims must be positive")
    keep = np.ones(n_dims, dtype=bool)
    drop = np.asarray(drop_dims, dtype=np.int64)
    if len(drop):
        if drop.min() < 0 or drop.max() >= n_dims:
            raise ValueError("drop dimension out of range")
        keep[drop] = False
    return keep


def effective_rank(z: np.ndarray, eps: float = 1e-12) -> float:
    """Entropy effective rank of the feature covariance."""

    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("z must be 2D")
    if len(z) < 2:
        return 0.0
    zs = z - z.mean(axis=0, keepdims=True)
    eig = np.linalg.eigvalsh(np.cov(zs, rowvar=False)).clip(min=0.0)
    total = eig.sum()
    if total <= eps:
        return 0.0
    p = eig / total
    ent = -(p[p > eps] * np.log(p[p > eps])).sum()
    return float(np.exp(ent))


def _between_group_mean_variance(values: np.ndarray, groups: np.ndarray) -> float:
    means = []
    weights = []
    for g in np.unique(groups):
        idx = groups == g
        if idx.sum() == 0:
            continue
        means.append(float(values[idx].mean()))
        weights.append(float(idx.mean()))
    if len(means) < 2:
        return 0.0
    means_arr = np.asarray(means)
    weights_arr = np.asarray(weights)
    center = float(np.sum(weights_arr * means_arr) / weights_arr.sum())
    return float(np.sum(weights_arr * (means_arr - center) ** 2) / weights_arr.sum())


def latent_dimension_scores(z: np.ndarray, y: np.ndarray, d: np.ndarray) -> list[dict[str, float | int]]:
    """Score dimensions by conditional-domain signal versus task signal.

    The score is a cheap source-side localization heuristic, not a CMI estimate.
    It averages between-domain mean variance inside each class and divides it by
    between-class mean variance.
    """

    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y).astype(np.int64, copy=False)
    d = np.asarray(d).astype(np.int64, copy=False)
    if z.ndim != 2 or len(y) != len(z) or len(d) != len(z):
        raise ValueError("z, y, d shape mismatch")
    zs = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-8)
    out: list[dict[str, float | int]] = []
    for dim in range(z.shape[1]):
        x = zs[:, dim]
        domain_score = 0.0
        for cls in np.unique(y):
            idx = y == cls
            domain_score += float(idx.mean()) * _between_group_mean_variance(x[idx], d[idx])
        task_score = _between_group_mean_variance(x, y)
        cedar_score = domain_score / (1e-8 + max(0.0, task_score))
        out.append(
            {
                "dim": int(dim),
                "domain_score": float(domain_score),
                "task_score": float(task_score),
                "cedar_score": float(cedar_score),
            }
        )
    return out


def rank_latent_dimensions(z: np.ndarray, y: np.ndarray, d: np.ndarray) -> list[int]:
    scores = latent_dimension_scores(z, y, d)
    scores = sorted(scores, key=lambda r: (float(r["cedar_score"]), float(r["domain_score"])), reverse=True)
    return [int(r["dim"]) for r in scores]


def candidate_drop_sets(
    ranked_dims: list[int],
    fractions: tuple[float, ...],
    *,
    min_drop: int = 1,
) -> list[tuple[str, list[int]]]:
    """Build cumulative drop sets from a ranked latent-dimension list."""

    if not ranked_dims:
        raise ValueError("ranked_dims is empty")
    n_dims = len(ranked_dims)
    out: list[tuple[str, list[int]]] = []
    seen: set[int] = set()
    for frac in fractions:
        if frac <= 0.0 or frac >= 1.0:
            raise ValueError("fractions must be in (0, 1)")
        k = max(min_drop, int(round(frac * n_dims)))
        k = min(k, n_dims - 1) if n_dims > 1 else 1
        if k in seen:
            continue
        seen.add(k)
        out.append((f"drop_top_{k}_of_{n_dims}", list(ranked_dims[:k])))
    return out
