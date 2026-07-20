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


def assert_fixed_audit_population(bundles_by_level: dict) -> tuple:
    """Every (level, method) audit bundle must share one full audit signature — population hash
    AND the actual tensor / split-manifest / preprocess hashes (identical sample IDs alone do not
    prove byte-identical inputs). Returns the shared signature."""
    sigs = set()
    for level, methods in bundles_by_level.items():
        for method, b in methods.items():
            sigs.add(b.audit_signature())
    if len(sigs) != 1:
        raise ValueError(f"audit population/tensor is NOT fixed across levels/methods: {len(sigs)} distinct signatures")
    return next(iter(sigs))


def post_fragmentation_curve_average(levels, values, first_fragmentation_level) -> float:
    """``mean`` of ``values`` over levels ``ℓ ≥ ℓ_f`` (skipping NaN/non-estimable)."""
    if first_fragmentation_level is None:
        return float("nan")
    sel = [v for l, v in zip(levels, values) if l >= first_fragmentation_level and not np.isnan(v)]
    return float(np.mean(sel)) if sel else float("nan")


def simultaneous_band(plan, per_level_delta_fns, point_curve, alpha: float = 0.05) -> dict:
    """Same-resample simultaneous (1−α) band CENTERED on the POINT estimates ``Δ̂_ℓ`` (NOT the
    bootstrap mean): ``q = Q_{1−α}[ max_ℓ |Δ*_ℓ − Δ̂_ℓ| ]``, band ``[Δ̂_ℓ − q, Δ̂_ℓ + q]``."""
    if not plan.estimable:
        return {"estimable": False, "reason": plan.reason}
    point = np.asarray(point_curve, dtype=float)
    if len(point) != len(per_level_delta_fns):
        raise ValueError("point_curve length must match the number of per-level delta functions")
    mat = np.array([[fn(idx) for fn in per_level_delta_fns] for idx in plan.replicates])  # [B, L]
    max_dev = np.max(np.abs(mat - point[None, :]), axis=1)             # deviation from the POINT curve
    q = float(np.quantile(max_dev, 1 - alpha))
    return {"estimable": True, "centre": point.tolist(), "half_width": q,
            "lower": (point - q).tolist(), "upper": (point + q).tolist(), "n_boot": len(mat)}
