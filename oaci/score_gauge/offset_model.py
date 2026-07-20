"""C23 — fixed low-freedom offset model. Predicts the per-target score OFFSET from TARGET-ANONYMOUS source-only
gauge summaries with a fixed-regularization ridge, evaluated LEAVE-ONE-TARGET-OUT (the target-free test:
predict a held-out target's offset from the OTHER targets). No grid search, no feature selection. Target labels
enter only via the offset (a diagnostic quantity); no selector is produced. With only 9 targets this is a small-
N extrapolation -- reported honestly."""
from __future__ import annotations

import numpy as np

from . import gauge_feature_registry as gfr
from . import schema


def _design(gauge_table, names=None):
    targets = sorted(gauge_table)
    names = names if names is not None else gfr.gauge_feature_names()
    X = np.array([[gauge_table[t]["gauge"][n] for n in names] for t in targets], dtype=np.float64)
    y = np.array([gauge_table[t]["offset"] for t in targets], dtype=np.float64)
    return targets, X, y, names


def _ridge_fit_predict(Xtr, ytr, Xte, l2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ym = ytr.mean()
    d = Xtr.shape[1]
    w = np.linalg.solve(Xtr.T @ Xtr + l2 * np.eye(d), Xtr.T @ (ytr - ym))
    return Xte @ w + ym


def fit_offsets(gauge_table, names=None) -> dict:
    """LOTO offset prediction (target-free) + in-sample (identity-available). Returns offset_hat per target.
    `names` selects the gauge feature columns; defaults to the C23 source gauge names (pass the target-unlabeled
    names for C24 R3/R4)."""
    targets, X, y, names = _design(gauge_table, names)
    n = len(targets)
    loto = {}
    for i, t in enumerate(targets):
        tr = [j for j in range(n) if j != i]
        loto[t] = float(_ridge_fit_predict(X[tr], y[tr], X[i:i + 1], schema.RIDGE_L2)[0])
    insample = _ridge_fit_predict(X, y, X, schema.RIDGE_L2)
    insample = {t: float(insample[i]) for i, t in enumerate(targets)}

    def _r2(true, pred):
        true = np.array([true[t] for t in targets]); pred = np.array([pred[t] for t in targets])
        ss_res = ((true - pred) ** 2).sum(); ss_tot = ((true - true.mean()) ** 2).sum()
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    true_off = {t: gauge_table[t]["offset"] for t in targets}
    return {"targets": targets, "offset_true": true_off, "offset_hat_loto": loto, "offset_hat_insample": insample,
            "loto_r2": _r2(true_off, loto), "insample_r2": _r2(true_off, insample), "n_targets": n,
            "n_gauge_features": len(names)}
