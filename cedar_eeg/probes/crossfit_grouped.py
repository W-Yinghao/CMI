"""Grouped cross-fit wrappers for conditional domain probes."""

from __future__ import annotations

import numpy as np

from .conditional_domain_heads import fit_conditional_domain_probe


def make_folds(
    n: int,
    *,
    groups: np.ndarray | None = None,
    n_splits: int = 3,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return train/eval indices, keeping groups disjoint when groups are given."""

    if n <= 1:
        raise ValueError("at least two rows are required")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.permutation(n)
        chunks = [c for c in np.array_split(idx, min(n_splits, n)) if len(c)]
        return [(np.setdiff1d(idx, ev, assume_unique=False), ev) for ev in chunks]

    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError("groups length mismatch")
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError("grouped cross-fit requires at least two groups")
    uniq = rng.permutation(uniq)
    chunks = [c for c in np.array_split(uniq, min(n_splits, len(uniq))) if len(c)]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(n)
    for ev_groups in chunks:
        ev = np.where(np.isin(groups, ev_groups))[0]
        tr = np.setdiff1d(all_idx, ev, assume_unique=False)
        if len(tr) and len(ev):
            folds.append((tr, ev))
    if not folds:
        raise ValueError("no non-empty folds could be made")
    return folds


def permute_domain_within_label(d: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Permutation null preserving p(D | Y) counts."""

    rng = np.random.default_rng(seed)
    d = np.asarray(d).astype(np.int64, copy=True)
    y = np.asarray(y).astype(np.int64, copy=False)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        d[idx] = d[idx][rng.permutation(len(idx))]
    return d


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def crossfit_conditional_domain_probe(
    z: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    *,
    n_classes: int,
    n_domains: int,
    groups: np.ndarray | None = None,
    n_splits: int = 3,
    probe: str = "linear",
    seed: int = 0,
    max_iter: int = 500,
    hidden: tuple[int, ...] = (64,),
    permutation: bool = False,
) -> dict[str, object]:
    """Cross-fit q(D | Z, Y), optionally under the within-label permutation null."""

    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y).astype(np.int64, copy=False)
    d = np.asarray(d).astype(np.int64, copy=False)
    if permutation:
        d_fit = permute_domain_within_label(d, y, seed + 7919)
    else:
        d_fit = d

    fold_results = []
    for fold_id, (tr, ev) in enumerate(make_folds(len(z), groups=groups, n_splits=n_splits, seed=seed)):
        if len(np.unique(d_fit[tr])) < 2 or len(np.unique(d_fit[ev])) < 2:
            continue
        res = fit_conditional_domain_probe(
            z[tr],
            y[tr],
            d_fit[tr],
            z[ev],
            y[ev],
            d_fit[ev],
            n_classes=n_classes,
            n_domains=n_domains,
            probe=probe,
            seed=seed + fold_id,
            max_iter=max_iter,
            hidden=hidden,
        )
        rec = res.to_dict()
        rec["fold_id"] = fold_id
        fold_results.append(rec)

    if not fold_results:
        raise ValueError("all folds were degenerate for the conditional domain probe")
    advantages = [float(r["advantage"]) for r in fold_results]
    return {
        "probe": probe,
        "permutation": permutation,
        "advantage_mean": _mean(advantages),
        "advantage_std": float(np.std(advantages, ddof=0)),
        "domain_bacc_mean": _mean([float(r["domain_bacc"]) for r in fold_results]),
        "prior_bacc_mean": _mean([float(r["prior_bacc"]) for r in fold_results]),
        "n_folds": len(fold_results),
        "folds": fold_results,
    }
