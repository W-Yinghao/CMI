"""Paired clustered bootstrap — one plan reused across methods / deletion levels / metrics.

Resample COMPLETE recording groups WITHIN each domain (never rows); a replicate's row multiset
is a union of whole groups, with the SAME group multiplicities applied to every method (paired).
A replicate that loses a pre-registered class in some eval domain is INVALID — it is redrawn and
``invalid_draw_rate`` is reported; if too many are invalid, or any eval domain has < 2 clusters,
the CI is non-estimable (we do NOT fall back to a row-level bootstrap).

One-sided paired **basic** limits (no clipping to the metric's natural range):
``LCL = 2Δ̂ − Q_{1−α}(Δ*)``, ``UCL = 2Δ̂ − Q_{α}(Δ*)``; percentile endpoints reported too.

Repeated seeds are blocked, not concatenated: compute Δ̂ per seed, then average over seeds
(``point_delta_over_seeds``) — seeds are paired-algorithm randomness, not extra trials.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BootstrapPlan:
    replicates: list                  # list of row-index arrays (whole-group, within-domain resample)
    n_boot: int
    mode: str
    seed: int
    invalid_draw_rate: float
    estimable: bool
    reason: str = ""
    group_sizes: dict = field(default_factory=dict)   # group -> #rows (for the whole-group check)


def make_bootstrap_plan(domain, group, y, reference_classes, n_boot=1000, seed=0,
                        mode="fixed_domain", invalid_threshold=0.2, min_clusters=2,
                        max_attempts_factor=10) -> BootstrapPlan:
    domain, group, y = np.asarray(domain, int), np.asarray(group, int), np.asarray(y, int)
    uniq = np.unique(group)
    grp_rows = {int(g): np.where(group == g)[0] for g in uniq}
    grp_dom = {int(g): int(domain[group == g][0]) for g in uniq}
    by_dom: dict[int, list[int]] = {}
    for g in uniq:
        by_dom.setdefault(grp_dom[int(g)], []).append(int(g))
    group_sizes = {g: int(len(r)) for g, r in grp_rows.items()}

    for dd, gs in by_dom.items():
        if len(gs) < min_clusters:
            return BootstrapPlan([], n_boot, mode, seed, 0.0, False,
                                 f"domain {dd} has {len(gs)} cluster(s) < {min_clusters}", group_sizes)

    rng = np.random.default_rng(seed)
    ref = list(reference_classes)

    def draw_idx():
        idx = []
        for dd, gs in by_dom.items():
            gs = np.array(gs)
            for gg in gs[rng.integers(0, len(gs), len(gs))]:
                idx.append(grp_rows[int(gg)])
        return np.concatenate(idx)

    def valid(idx):
        dsub, ysub = domain[idx], y[idx]
        for dd in by_dom:
            present = set(ysub[dsub == dd].tolist())
            if any(c not in present for c in ref):
                return False
        return True

    reps, attempts, invalid = [], 0, 0
    max_attempts = n_boot * max_attempts_factor
    while len(reps) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = draw_idx()
        if valid(idx):
            reps.append(idx)
        else:
            invalid += 1
    rate = invalid / attempts if attempts else 0.0
    if len(reps) < n_boot or rate > invalid_threshold:
        return BootstrapPlan(reps, n_boot, mode, seed, rate, False,
                             f"invalid_draw_rate {rate:.3f} > {invalid_threshold} or only "
                             f"{len(reps)}/{n_boot} valid replicates", group_sizes)
    return BootstrapPlan(reps, n_boot, mode, seed, rate, True, "", group_sizes)


def paired_ci(plan: BootstrapPlan, delta_full: float, delta_fn, alpha: float = 0.05) -> dict:
    """``delta_fn(idx)`` -> the paired Δ on resampled rows (worst-domain recomputed inside)."""
    if not plan.estimable:
        return {"estimable": False, "reason": plan.reason, "invalid_draw_rate": plan.invalid_draw_rate}
    deltas = np.array([delta_fn(idx) for idx in plan.replicates], dtype=float)
    return {
        "estimable": True,
        "delta": float(delta_full),
        "basic_lcl": float(2 * delta_full - np.quantile(deltas, 1 - alpha)),
        "basic_ucl": float(2 * delta_full - np.quantile(deltas, alpha)),
        "percentile_lcl": float(np.quantile(deltas, alpha)),
        "percentile_ucl": float(np.quantile(deltas, 1 - alpha)),
        "n_boot": len(deltas),
        "invalid_draw_rate": plan.invalid_draw_rate,
        "alpha": alpha,
    }


def is_whole_group_resample(idx, group, plan: BootstrapPlan) -> bool:
    """True iff every group's row-count in ``idx`` is an integer multiple of its size (i.e. the
    resample never split a group into rows)."""
    group = np.asarray(group, int)
    vals, counts = np.unique(group[idx], return_counts=True)
    return all(int(c) % plan.group_sizes[int(g)] == 0 for g, c in zip(vals, counts))


def point_delta_over_seeds(per_seed_deltas) -> float:
    """Block seeds: average per-seed point estimates (seeds are paired randomness, NOT extra
    trials — duplicating a seed does not inflate N or the estimate)."""
    return float(np.mean([float(x) for x in per_seed_deltas]))
