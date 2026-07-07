"""C22 Q3 — post-hoc score-normalization DIAGNOSTICS (MECHANISM tests, NOT deployment). Does removing a
target/regime-specific score offset/scale RECOVER the pooled cross-regime AUC? If target-wise centering /
z-score / rank recovers it, the failure is a score-calibration / offset-transport problem (the signal is
rank-like). If it does not, the source->target relationship itself shifts by regime. These transforms need the
target/regime identity at score-time and are therefore NON-deployable; they are reported as mechanism only."""
from __future__ import annotations

import numpy as np

from ..identifiability.multivariate_probe import _auc
from . import schema


def _apply(rows, method):
    s = np.array([r["score"] for r in rows], dtype=float)
    if method == "none":
        return s
    if method in ("target_center", "target_zscore", "target_rank", "quantile"):
        key = "target"
    elif method == "regime_center":
        key = "regime"
    elif method == "target_regime_center":
        key = None
    else:
        raise ValueError(method)
    out = s.copy()
    if method == "target_regime_center":
        groups = {}
        for i, r in enumerate(rows):
            groups.setdefault((r["target"], r["regime"]), []).append(i)
    else:
        groups = {}
        for i, r in enumerate(rows):
            groups.setdefault(r[key], []).append(i)
    for idx in groups.values():
        v = s[idx]
        if method in ("target_center", "regime_center", "target_regime_center"):
            out[idx] = v - v.mean()
        elif method == "target_zscore":
            out[idx] = (v - v.mean()) / (v.std() + 1e-9)
        elif method in ("target_rank", "quantile"):
            r = np.argsort(np.argsort(v)).astype(float)
            out[idx] = r / max(len(v) - 1, 1)      # rank/quantile -> [0,1] within group
    return out


def normalization_diagnostics(rows) -> dict:
    """On the CROSS-regime pooled set (where pooling fails) and the in-regime set, pooled AUC after each
    post-hoc normalization. Recovery = target-normalized pooled AUC returns toward the within-target level."""
    out = {}
    for mode in ("in_regime", "cross_regime"):
        mr = [r for r in rows if r["mode"] == mode]
        y = np.array([r["label"] for r in mr])
        if not (0 < y.sum() < len(y)):
            continue
        per_method = {}
        for m in schema.NORMALIZATIONS:
            s = _apply(mr, m)
            per_method[m] = _auc(y, s)
        base = per_method.get("none")
        tgt_norm = max((per_method[m] for m in ("target_center", "target_zscore", "target_rank")
                        if per_method.get(m) is not None), default=None)
        out[mode] = {"pooled_auc_by_normalization": per_method, "pooled_none": base,
                     "best_target_normalized_pooled": tgt_norm,
                     "target_normalization_recovers": bool(base is not None and tgt_norm is not None
                                                           and (tgt_norm - base) >= schema.SIGNAL_MARGIN
                                                           and tgt_norm >= 0.5 + schema.SIGNAL_MARGIN)}
    return {"per_mode": out, "is_diagnostic": schema.NORMALIZATION_IS_DIAGNOSTIC,
            "note": ("post-hoc MECHANISM diagnostics only; target/regime-wise normalization needs the target/"
                     "regime identity at score time and is NON-deployable. Recovery => the failure is score "
                     "offset/calibration (rank-like signal); no recovery => regime-specific relationship shift.")}
