"""Frozen continuous endpoint utilities for C34."""
from __future__ import annotations

import math

import numpy as np

from ..local_boundary.artifact_loader import units
from . import schema


def finite(v) -> bool:
    try:
        return v is not None and np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _safe_float(v, default=np.nan):
    return float(v) if finite(v) else default


def _dominates(a, b) -> bool:
    ge = all(x >= y for x, y in zip(a, b))
    gt = any(x > y for x, y in zip(a, b))
    return bool(ge and gt)


def _standardizer(rows, raw_key):
    vals = np.array([_safe_float(r.get(raw_key)) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    mu = float(vals.mean()) if len(vals) else 0.0
    sd = float(vals.std()) if len(vals) else 1.0
    if sd <= 1e-12:
        sd = 1.0
    return mu, sd


def attach_endpoint_utilities(rows):
    """Attach endpoint vectors and fixed scalar summaries in-place.

    Raw endpoint deltas are always reported first. Scalar summaries use a single
    global z-standardization over the read-only C34 candidate registry and are
    used only for diagnostics, not for score tuning or selection.
    """
    raw_map = {
        "target_bacc_delta": "bacc_delta",
        "target_nll_delta": "nll_improve",
        "target_ece_delta": "ece_improve",
    }
    stats = {out_key: _standardizer(rows, in_key) for out_key, in_key in raw_map.items()}
    for r in rows:
        for out_key, in_key in raw_map.items():
            raw = _safe_float(r.get(in_key))
            r[out_key] = raw
            mu, sd = stats[out_key]
            r[out_key.replace("_delta", "_z")] = (raw - mu) / sd if np.isfinite(raw) else np.nan
        z = endpoint_z_vector(r)
        r["continuous_joint_min_margin"] = float(np.min(z)) if np.all(np.isfinite(z)) else np.nan

    for _, cs in units(rows).items():
        valid = [c for c in cs if np.all(np.isfinite(endpoint_z_vector(c)))]
        if not valid:
            continue
        mat = np.array([endpoint_z_vector(c) for c in valid], dtype=float)
        ideal = mat.max(axis=0)
        pareto = []
        for i, c in enumerate(valid):
            if not any(_dominates(mat[j], mat[i]) for j in range(len(valid)) if j != i):
                pareto.append((c, mat[i]))
        pmat = np.array([p for _, p in pareto], dtype=float) if pareto else mat
        for c, vec in zip(valid, mat):
            deficit = np.maximum(ideal - vec, 0.0)
            c["endpoint_vector_norm_regret"] = float(np.linalg.norm(deficit))
            c["dominated_hypervolume_regret"] = float(np.prod(1.0 + deficit) - 1.0)
            if any(pc is c for pc, _ in pareto):
                c["pareto_distance"] = 0.0
            else:
                c["pareto_distance"] = float(np.min([np.linalg.norm(np.maximum(p - vec, 0.0)) for p in pmat]))
    return rows


def endpoint_raw_vector(r):
    return np.array([_safe_float(r.get(k)) for k in schema.ENDPOINT_RAW_KEYS], dtype=float)


def endpoint_z_vector(r):
    return np.array([_safe_float(r.get(k)) for k in schema.ENDPOINT_Z_KEYS], dtype=float)


def endpoint_delta(a, b) -> dict:
    """Return b - a endpoint deltas with higher-is-better orientation."""
    raw = endpoint_raw_vector(b) - endpoint_raw_vector(a)
    z = endpoint_z_vector(b) - endpoint_z_vector(a)
    return {
        "target_bacc_delta": float(raw[0]),
        "target_nll_delta": float(raw[1]),
        "target_ece_delta": float(raw[2]),
        "target_bacc_z_delta": float(z[0]),
        "target_nll_z_delta": float(z[1]),
        "target_ece_z_delta": float(z[2]),
        "endpoint_vector_delta_norm": float(np.linalg.norm(z)) if np.all(np.isfinite(z)) else math.nan,
        "endpoint_tradeoff": int(np.any(z > schema.STANDARDIZED_TINY_REGRET) and
                                 np.any(z < -schema.STANDARDIZED_TINY_REGRET)),
    }


def endpoint_registry(rows) -> list:
    out = []
    for r in rows:
        out.append({
            "seed": r["seed"], "target": r["target"], "level": r["level"], "regime": r.get("regime", ""),
            "order": r.get("order"), "epoch": r.get("epoch"), "selected_oaci": int(r.get("selected_oaci", 0)),
            "primary_joint_good": int(r.get("joint_good", 0)),
            "target_bacc_delta": r.get("target_bacc_delta"),
            "target_nll_delta": r.get("target_nll_delta"),
            "target_ece_delta": r.get("target_ece_delta"),
            "target_bacc_z": r.get("target_bacc_z"),
            "target_nll_z": r.get("target_nll_z"),
            "target_ece_z": r.get("target_ece_z"),
            "continuous_joint_min_margin": r.get("continuous_joint_min_margin"),
            "pareto_distance": r.get("pareto_distance"),
            "dominated_hypervolume_regret": r.get("dominated_hypervolume_regret"),
            "endpoint_vector_norm_regret": r.get("endpoint_vector_norm_regret"),
        })
    return out


def summarize_registry(rows) -> dict:
    selected = [r for r in rows if r.get("selected_oaci")]
    vals = lambda rs, k: [float(r[k]) for r in rs if finite(r.get(k))]
    out = {"n_candidates": len(rows), "n_selected": len(selected)}
    for key in schema.ENDPOINT_RAW_KEYS + ("continuous_joint_min_margin", "pareto_distance",
                                           "dominated_hypervolume_regret", "endpoint_vector_norm_regret"):
        all_vals = vals(rows, key)
        sel_vals = vals(selected, key)
        out[f"mean_{key}"] = float(np.mean(all_vals)) if all_vals else None
        out[f"mean_selected_{key}"] = float(np.mean(sel_vals)) if sel_vals else None
    return out
