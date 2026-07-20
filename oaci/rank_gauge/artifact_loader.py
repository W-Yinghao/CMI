"""C30 — read-only loader + the within-target (RANK) vs pooled (GAUGE) AUC primitives. within_target_auc grades
how well a factor ranks candidates BY competence WITHIN a target (the rank axis, gauge removed by construction);
pooled_auc grades it across the pooled population (where the per-target gauge miscalibrates). Reads the C22
score sidecar; nothing is refit or tuned."""
from __future__ import annotations

import numpy as np

from ..identifiability.multivariate_probe import _auc
from ..information_ladder import artifact_loader as il
from . import schema

load = il.load
_finite = il._finite


def _val(r, key):
    v = r.get(key)
    return float(v) if _finite(v) else None


def within_target_auc(rows, key, mode="in_regime") -> float:
    """Mean over targets of the within-target AUC (factor -> competence label). Rank axis (gauge-free)."""
    mr = [r for r in rows if r["mode"] == mode]
    per = []
    for t in sorted({r["target"] for r in mr}):
        g = [r for r in mr if r["target"] == t]
        y = np.array([r[schema.LABEL_KEY] for r in g])
        x = np.array([_val(r, key) for r in g], dtype=float)
        ok = np.isfinite(x)
        if ok.sum() > 2 and 0 < y[ok].sum() < ok.sum() and x[ok].std() > 1e-9:
            per.append(_auc(y[ok], x[ok]))
    return float(np.mean(per)) if per else None


def pooled_auc(rows, key, mode="in_regime") -> float:
    mr = [r for r in rows if r["mode"] == mode]
    y = np.array([r[schema.LABEL_KEY] for r in mr])
    x = np.array([_val(r, key) for r in mr], dtype=float)
    ok = np.isfinite(x)
    return _auc(y[ok], x[ok]) if (ok.sum() > 2 and 0 < y[ok].sum() < ok.sum()) else None


def rank_strength(auc):
    """Direction-agnostic ranking strength |AUC - 0.5| (a factor may predict competence with either sign)."""
    return None if auc is None else abs(auc - 0.5)


def per_target_aucs(rows, key, mode="in_regime") -> dict:
    """Per-target within-target AUC (factor -> label) for each target separately."""
    mr = [r for r in rows if r["mode"] == mode]
    out = {}
    for t in sorted({r["target"] for r in mr}):
        g = [r for r in mr if r["target"] == t]
        y = np.array([r[schema.LABEL_KEY] for r in g])
        x = np.array([_val(r, key) for r in g], dtype=float)
        ok = np.isfinite(x)
        if ok.sum() > 2 and 0 < y[ok].sum() < ok.sum() and x[ok].std() > 1e-9:
            out[t] = _auc(y[ok], x[ok])
    return out


def sign_consistency(rows, key, mode="in_regime") -> dict:
    """Does the factor rank competence in the SAME direction across targets? (red-team G5): the mean |AUC-0.5|
    can MASK per-target sign flips. Returns the fraction of targets on the majority side of 0.5 (1.0 => the
    within-target rank is direction-CONSISTENT and transfers; << 1 => target-LOCAL, non-transferable)."""
    aucs = per_target_aucs(rows, key, mode)
    if not aucs:
        return {"per_target_auc": {}, "n_targets": 0, "n_above_half": None, "sign_consistency": None, "transfers": None}
    above = sum(1 for a in aucs.values() if a > 0.5)
    frac = max(above, len(aucs) - above) / len(aucs)
    return {"per_target_auc": {int(t): float(a) for t, a in aucs.items()}, "n_targets": len(aucs),
            "n_above_half": above, "sign_consistency": float(frac), "transfers": bool(frac >= 0.8)}


def per_target_gauge(rows, mode="in_regime"):
    mr = [r for r in rows if r["mode"] == mode]
    return {t: float(np.mean([r["score"] for r in mr if r["target"] == t])) for t in {r["target"] for r in mr}}


def within_target_auc_residualized(rows, key, control_key, mode="in_regime") -> float:
    """within-target AUC of `key` after regressing out `control_key` per target (control the gauge/other axis)."""
    mr = [r for r in rows if r["mode"] == mode]
    per = []
    for t in sorted({r["target"] for r in mr}):
        g = [r for r in mr if r["target"] == t]
        y = np.array([r[schema.LABEL_KEY] for r in g])
        x = np.array([_val(r, key) for r in g], dtype=float)
        c = np.array([_val(r, control_key) for r in g], dtype=float)
        ok = np.isfinite(x) & np.isfinite(c)
        if ok.sum() > 3 and 0 < y[ok].sum() < ok.sum() and x[ok].std() > 1e-9 and c[ok].std() > 1e-9:
            beta = np.polyfit(c[ok], x[ok], 1)
            resid = x[ok] - (beta[0] * c[ok] + beta[1])
            if resid.std() > 1e-9:
                per.append(_auc(y[ok], resid))
    return float(np.mean(per)) if per else None
