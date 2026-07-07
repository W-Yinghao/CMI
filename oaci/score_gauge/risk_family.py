"""C23 — SECONDARY risk-family gauge audit. C22 found R_src (source risk) is a strong within-target predictor.
Here we test whether a risk-family gauge (R_src per-target mean ONLY) predicts the per-target offset, LOTO, and
how it compares to the robust-core gauge. R_src is a training-realized STATIC scalar, so this is SECONDARY and
clearly labelled; it must not be merged into the robust-core primary."""
from __future__ import annotations

import numpy as np

from . import ceiling_ladder, schema
from .offset_model import _ridge_fit_predict


def risk_family_gauge(gauge_table, rows, mode) -> dict:
    targets = sorted(gauge_table)
    X = np.array([[gauge_table[t]["R_src_mean"]] for t in targets], dtype=np.float64)
    y = np.array([gauge_table[t]["offset"] for t in targets], dtype=np.float64)
    loto = {}
    for i, t in enumerate(targets):
        tr = [j for j in range(len(targets)) if j != i]
        loto[t] = float(_ridge_fit_predict(X[tr], y[tr], X[i:i + 1], schema.RIDGE_L2)[0])
    ss_res = sum((gauge_table[t]["offset"] - loto[t]) ** 2 for t in targets)
    ss_tot = sum((v - y.mean()) ** 2 for v in y)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    ladder = ceiling_ladder.ceiling_ladder(rows, mode, loto)
    return {"is_secondary": True, "feature": "R_src_mean_only", "loto_r2": r2,
            "source_gauge_loto_auc": ladder["source_gauge_loto"], "gap_closed": ladder["gap_closed_source_gauge"],
            "note": "SECONDARY: R_src is a training-realized static scalar, not a robust-core deletion-robust observable."}
