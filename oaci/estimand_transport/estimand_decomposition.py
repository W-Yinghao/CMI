"""C22 Q1 — decompose pooled vs within-target AUC. If pooled << within-target, the score ranks candidates
within a target but its levels are not comparable across targets (a between-target offset problem)."""
from __future__ import annotations

import numpy as np

from ..identifiability.multivariate_probe import _auc


def _pooled_within(rows):
    y = np.array([r["label"] for r in rows]); s = np.array([r["score"] for r in rows], dtype=float)
    pooled = _auc(y, s) if (0 < y.sum() < len(y)) else None
    per_t = {}
    for t in sorted({r["target"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["target"] == t]
        yt, st = y[idx], s[idx]
        if 0 < yt.sum() < len(yt):
            per_t[t] = _auc(yt, st)
    wv = [v for v in per_t.values() if v is not None]
    within_mean = float(np.mean(wv)) if wv else None
    return pooled, within_mean, per_t


def decompose(rows) -> dict:
    """Per (mode, regime) group -> pooled AUC, within-target mean AUC, gap, per-target AUC spread."""
    out = {}
    groups = {}
    for r in rows:
        groups.setdefault((r["mode"], r["regime"]), []).append(r)
    for (mode, regime), g in groups.items():
        pooled, within, per_t = _pooled_within(g)
        vals = [v for v in per_t.values() if v is not None]
        out[f"{mode}:{regime}"] = {
            "mode": mode, "regime": regime, "n": len(g), "pooled_auc": pooled, "within_target_mean_auc": within,
            "pooled_minus_within": (pooled - within) if (pooled is not None and within is not None) else None,
            "within_target_min": (min(vals) if vals else None), "within_target_max": (max(vals) if vals else None),
            "per_target_auc": {str(t): v for t, v in per_t.items()}}
    return out


def summary(dec) -> dict:
    """Across groups: is within-target consistently above pooled (the offset signature)?"""
    gaps = [d["pooled_minus_within"] for d in dec.values() if d["pooled_minus_within"] is not None]
    withins = [d["within_target_mean_auc"] for d in dec.values() if d["within_target_mean_auc"] is not None]
    pooleds = [d["pooled_auc"] for d in dec.values() if d["pooled_auc"] is not None]
    return {"mean_pooled_minus_within": (float(np.mean(gaps)) if gaps else None),
            "mean_within_target_auc": (float(np.mean(withins)) if withins else None),
            "mean_pooled_auc": (float(np.mean(pooleds)) if pooleds else None),
            "within_exceeds_pooled_everywhere": bool(gaps and all(g < 0 for g in gaps))}
