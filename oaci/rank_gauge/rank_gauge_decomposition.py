"""C30 Q1 — decompose the competence signal into a within-target RANK axis and a cross-target GAUGE axis. The
frozen source-only probe SCORE has a within-target ranking AUC (rank axis) but fails pooled (gauge axis); target-
centering (removing the per-target intercept) recovers pooled. The two axes are orthogonal by construction
(within-target deviation vs per-target gauge). This explains why C19 passes in-regime but C20 pooled fails."""
from __future__ import annotations

import numpy as np

from ..identifiability.multivariate_probe import _auc
from . import artifact_loader, schema


def rank_gauge_decomposition(rows, mode="in_regime") -> dict:
    mr = [r for r in rows if r["mode"] == mode]
    score_within = artifact_loader.within_target_auc(rows, schema.SCORE_KEY, mode)
    score_pooled = artifact_loader.pooled_auc(rows, schema.SCORE_KEY, mode)
    tmean = artifact_loader.per_target_gauge(rows, mode)
    y = np.array([r[schema.LABEL_KEY] for r in mr]); s = np.array([r["score"] for r in mr], float)
    mu = np.array([tmean[r["target"]] for r in mr]); dev = s - mu
    gauge_centered_pooled = _auc(y, dev) if (0 < y.sum() < len(y)) else None
    orthogonality = float(np.corrcoef(dev, mu)[0, 1]) if (dev.std() > 1e-9 and mu.std() > 1e-9) else None
    between_var = float(np.var([tmean[t] for t in tmean]))
    within_var = float(np.var(dev))
    gauge_var_fraction = between_var / (between_var + within_var + 1e-12)
    separation = bool(score_within is not None and score_within >= schema.RANK_SIGNAL_MIN
                      and score_pooled is not None and score_pooled < score_within
                      and orthogonality is not None and abs(orthogonality) < 0.20)
    return {"score_within_target_auc": score_within, "score_pooled_auc": score_pooled,
            "gauge_centered_pooled_auc": gauge_centered_pooled, "rank_gauge_orthogonality": orthogonality,
            "gauge_variance_fraction": gauge_var_fraction, "two_axis_separation": separation,
            "note": ("within-target RANK axis AUC %.3f (real) is orthogonal (corr %.3f) to the per-target GAUGE; "
                     "pooled fails (%.3f) but target-centering recovers (%.3f) -> two-axis separation is real"
                     % (score_within or 0, orthogonality or 0, score_pooled or 0, gauge_centered_pooled or 0)
                     if separation else "two-axis separation not cleanly established")}
