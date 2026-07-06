"""C19 — the fixed low-freedom probe: L2 logistic, fixed regularization, per-fold standardization, leave-one-
target-out (+ leave-one-seed-out sensitivity) validation, within-(seed,target,level) permutation baseline.
No grid search, no feature selection. Reuses the C17/C18 probe primitives (_auc/_fit_logit/_loto/_finite) so
the machinery is identical; only the pre-registered feature set differs."""
from __future__ import annotations

import numpy as np

from ..identifiability.multivariate_probe import _auc, _finite, _fit_logit, _loto
from . import schema
from .permutation import permutation_null


def _matrix(rows, feature_cols, label=schema.DIAGNOSTIC_LABEL):
    cols = [c for c in feature_cols if any(_finite(r.get(c)) for r in rows)]     # drop entirely-non-finite cols
    keep = [r for r in rows if cols and all(_finite(r.get(c)) for c in cols)]
    X = (np.array([[float(r[c]) for c in cols] for r in keep], dtype=np.float64) if keep
         else np.zeros((0, len(cols)), dtype=np.float64))
    y = np.array([1 if r[label] else 0 for r in keep], dtype=int)
    gt = np.array([r["target"] for r in keep]); gs = np.array([r["seed"] for r in keep])
    fold = np.array([hash((r["seed"], r["target"], r["level"])) for r in keep])
    return X, y, gt, gs, fold, cols, keep


def _degenerate(cols, y):
    return {"n_used": int(len(y)), "n_features": len(cols), "base_rate": (float(y.mean()) if len(y) else None),
            "loto_auc": None, "loso_auc": None, "per_target_auc": {}, "permutation_mean_auc": None,
            "permutation_p": None, "beats_permutation": False, "meets_margin": False, "non_deployable": True,
            "note": "degenerate feature matrix"}


def run_probe(rows, feature_cols, *, n_perm=schema.N_PERM, perm_seed=schema.PERM_SEED) -> dict:
    X, y, gt, gs, fold, cols, keep = _matrix(rows, feature_cols)
    if len(y) == 0 or len(cols) == 0 or len(np.unique(gt)) < 2 or y.sum() == 0 or y.sum() == len(y):
        return _degenerate(cols, y)
    loto, _ = _loto(X, y, gt); loso, _ = _loto(X, y, gs)
    per_target = {}
    for g in np.unique(gt):
        te = gt == g; tr = ~te
        if 0 < y[tr].sum() < len(y[tr]):
            per_target[str(int(g))] = _auc(y[te], _fit_logit(X[tr], y[tr].astype(float), X[te]))
    null = permutation_null(X, y, fold, gt, _loto, n_perm=n_perm, perm_seed=perm_seed)
    p = float((np.sum(null >= loto) + 1) / (len(null) + 1)) if loto is not None else None
    pm = float(null.mean()) if len(null) else None
    beats = bool(loto is not None and p is not None and p < schema.SUCCESS_P)
    margin = bool(loto is not None and pm is not None and (loto - pm) >= schema.SUCCESS_AUC_MARGIN)
    return {"n_used": int(len(y)), "n_features": len(cols), "features_used": cols, "base_rate": float(y.mean()),
            "loto_auc": loto, "loso_auc": loso, "per_target_auc": per_target, "permutation_mean_auc": pm,
            "permutation_p": p, "beats_permutation": beats, "meets_margin": margin,
            "passes": bool(beats and margin), "non_deployable": schema.NON_DEPLOYABLE,
            "note": "DIAGNOSTIC-ONLY fixed L2-logistic LOTO probe; no selector emitted."}
