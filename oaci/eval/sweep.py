"""Controlled missing-cell sweep aggregation.

The pre-registered stress-test scalar is the POST-FRAGMENTATION worst-domain bAcc curve average
``A_post = mean_{ℓ≥ℓ_f} worst_domain_bAcc_ℓ`` (``ℓ_f`` = first fragmentation level), compared
between OACI and each baseline as ``ΔA_post``. This avoids cherry-picking the best deletion
level. Per-level CIs are still produced (for plotting) and a same-resample simultaneous band is
available; the formal conclusion uses ``ΔA_post`` (or the simultaneous band), not a hand-picked
level.

The eval (target/source-audit) population is FIXED and byte-identical across all deletion levels
and methods (the cell mask only removes source-TRAINING rows). ``assert_fixed_audit_population``
enforces it so a performance change cannot be confounded by a population change.
"""
from __future__ import annotations

import numpy as np


def assert_fixed_audit_population(bundles_by_level: dict) -> str:
    """Every (level, method) audit bundle must share one ``eval_population_hash``; returns it."""
    hashes = set()
    for level, methods in bundles_by_level.items():
        for method, b in methods.items():
            hashes.add(b.eval_population_hash)
    if len(hashes) != 1:
        raise ValueError(f"audit population is NOT fixed across levels/methods: {len(hashes)} distinct hashes")
    return next(iter(hashes))


def post_fragmentation_curve_average(levels, values, first_fragmentation_level) -> float:
    """``mean`` of ``values`` over levels ``ℓ ≥ ℓ_f`` (skipping NaN/non-estimable)."""
    if first_fragmentation_level is None:
        return float("nan")
    sel = [v for l, v in zip(levels, values) if l >= first_fragmentation_level and not np.isnan(v)]
    return float(np.mean(sel)) if sel else float("nan")


def simultaneous_band(plan, per_level_delta_fns, alpha: float = 0.05) -> dict:
    """Same-resample simultaneous (1−α) band for a per-level Δ curve: for each replicate build the
    vector of Δ across levels, take the max absolute deviation from the per-level bootstrap mean,
    and use its (1−α) quantile as a uniform half-width."""
    if not plan.estimable:
        return {"estimable": False, "reason": plan.reason}
    L = len(per_level_delta_fns)
    mat = np.array([[fn(idx) for fn in per_level_delta_fns] for idx in plan.replicates])  # [B, L]
    centre = mat.mean(axis=0)
    max_dev = np.max(np.abs(mat - centre), axis=1)                      # per replicate, across levels
    half = float(np.quantile(max_dev, 1 - alpha))
    return {"estimable": True, "centre": centre.tolist(), "half_width": half,
            "lower": (centre - half).tolist(), "upper": (centre + half).tolist(), "n_boot": len(mat)}
