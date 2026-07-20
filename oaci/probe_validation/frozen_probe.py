"""C20 — the frozen probe applied cross-regime. Uses the EXACT C19 fixed L2-logistic (_fit_logit: standardize
on train, l2=1.0, iters=800, lr=0.3) and the C19 finite-filter. Fits on development-regime rows and scores
held-out-regime rows; emits NO selector. Nothing here is tuned."""
from __future__ import annotations

import numpy as np

from ..competence_probe import schema as c19
from ..identifiability.multivariate_probe import _auc, _finite, _fit_logit


def _finite_rows(rows, cols):
    keep = [r for r in rows if all(_finite(r.get(c)) for c in cols)]
    X = (np.array([[float(r[c]) for c in cols] for r in keep], dtype=np.float64) if keep
         else np.zeros((0, len(cols)), dtype=np.float64))
    y = np.array([1 if r[c19.DIAGNOSTIC_LABEL] else 0 for r in keep], dtype=int)
    return X, y, keep


def fit_predict(train_rows, test_rows, cols):
    """Fit the frozen logistic on finite train rows; return (test_scores, test_labels, n_train, n_test).
    Standardization is C19's (fit on train inside _fit_logit). No selector, no threshold."""
    Xtr, ytr, ktr = _finite_rows(train_rows, cols)
    Xte, yte, kte = _finite_rows(test_rows, cols)
    if len(ytr) == 0 or len(yte) == 0 or ytr.sum() == 0 or ytr.sum() == len(ytr):
        return None, None, len(ytr), len(yte)
    scores = _fit_logit(Xtr, ytr.astype(float), Xte)
    return scores, yte, len(ytr), len(yte)


def auc(y, s):
    return _auc(np.asarray(y), np.asarray(s)) if (y is not None and s is not None and len(y)) else None
