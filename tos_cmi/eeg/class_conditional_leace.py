"""Phase 2 -- class-conditional LEACE. Fit a separate LEACE eraser of subject D within each task class Y=y.
Two variants:
  * predicted-route (DEPLOYABLE): a source-trained router assigns yhat; the eraser routes ANY X (source or
    target) by yhat, so the target never needs true labels. When fit on a training split and applied to a
    held-out split (as the Phase 2 evaluator does), the routing is honest (held-out).
  * oracle-route (DIAGNOSTIC upper bound, NOT deployable): routes by the TRUE labels -- answers "if class
    routing were perfect, could conditional erasure preserve the task?" Only for the source/target diagnostic.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from tos_cmi.eeg.erasure_baselines import leace_eraser, _ids


def _per_class_leace(Zf, yf, subjf, n_cls):
    subj01, _ = _ids(subjf)
    Es = {}
    for y in range(n_cls):
        m = yf == y
        ns = len(set(subj01[m].tolist())) if m.any() else 0
        if m.sum() < 8 or ns < 2:
            Es[y] = (lambda X: X)                       # too few -> identity for that class
        else:
            Es[y] = leace_eraser(Zf[m], np.eye(ns)[_ids(subjf[m])[0]])
    return Es


def _apply_by_labels(X, labels, Es, n_cls):
    out = X.copy()
    for y in range(n_cls):
        m = labels == y
        if m.any():
            out[m] = Es[y](X[m])
    return out


def cc_leace_factory_predicted(Zf, yf, subjf, n_cls, seed=0):
    """DEPLOYABLE: returns apply(X) that routes by a source-trained predictor (no target labels needed).

    CAVEAT (Phase 2 adversarial review, verified np.array_equal on all target points): because the routing
    predictor R and the downstream task probe are both linear task classifiers, routing features by R and
    then re-probing the task reproduces R's own decision boundary -> the target argmax (hence target bAcc)
    is STRUCTURALLY pinned to the full-Z baseline (exact +0.000 ΔbAcc), independent of whether conditional
    erasure helps. So this variant is NOT a clean test of conditional erasure; tp_leace is. For V2 a fair
    deployable conditional eraser must route by a predictor INDEPENDENT of the evaluation probe (e.g. a fixed
    pretrained head or a disjoint-split router). Kept here only to exhibit the tautology honestly."""
    Es = _per_class_leace(Zf, yf, subjf, n_cls)
    R = LogisticRegression(max_iter=200).fit(Zf, yf)
    return lambda X: _apply_by_labels(X, R.predict(X), Es, n_cls)


def cc_leace_apply_oracle(Zfit, yfit, subjfit, n_cls):
    """DIAGNOSTIC upper bound: returns apply(X, Ytrue) that routes by TRUE labels. NOT deployable."""
    Es = _per_class_leace(Zfit, yfit, subjfit, n_cls)
    return lambda X, Ytrue: _apply_by_labels(X, np.asarray(Ytrue), Es, n_cls)
