"""C23 — target-identity-leakage HARD GATE (runs BEFORE any positive calibration claim). In LOSO the source
composition nearly identifies the target, so a gauge that merely re-encodes target identity is NOT a target-
free calibration. This audit asks: can a candidate's raw SOURCE features predict its TARGET id (9-way)? High
accuracy => source features carry target identity, and any apparent calibration must be treated as identity-
laden (G3) unless the offset relationship still GENERALIZES leave-one-target-out."""
from __future__ import annotations

import numpy as np

from ..competence_probe import schema as c19
from . import schema
from .artifact_loader import _finite


def _matrix(rows):
    cols = ["feat__" + s for s in c19.ROBUST_CORE_FEATURES]
    keep = [r for r in rows if all(_finite(r.get(c)) for c in cols)]
    X = np.array([[float(r[c]) for c in cols] for r in keep], dtype=np.float64)
    y = np.array([r["target"] for r in keep])
    return X, y


def _nearest_centroid_cv(X, y, k=5, seed=0):
    """Stratified k-fold nearest-centroid accuracy for target-id prediction from source features."""
    rng = np.random.RandomState(seed)
    n = len(y); idx = rng.permutation(n); folds = np.array_split(idx, k)
    correct = 0; total = 0
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(k) if j != f])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        cents = {}
        for t in np.unique(y[tr]):
            cents[t] = Xtr[y[tr] == t].mean(0)
        ts = list(cents); C = np.stack([cents[t] for t in ts])
        for i, xi in zip(te, Xte):
            d = ((C - xi) ** 2).sum(1)
            pred = ts[int(np.argmin(d))]
            correct += int(pred == y[i]); total += 1
    return correct / total if total else None


def identity_leakage_audit(rows, mode="in_regime") -> dict:
    mr = [r for r in rows if r["mode"] == mode]
    X, y = _matrix(mr)
    acc = _nearest_centroid_cv(X, y)
    chance = schema.IDENTITY_LEAKAGE_CHANCE
    return {"target_id_accuracy_from_source_features": acc, "chance": chance,
            "n_candidates": int(len(y)), "n_targets": int(len(np.unique(y))),
            "source_features_identity_separable": bool(acc is not None and acc > schema.IDENTITY_LEAKAGE_CEILING),
            "note": ("If source features predict target id far above chance (1/9), the per-target gauge is "
                     "identity-laden; a positive calibration then only counts if the offset relationship "
                     "GENERALIZES leave-one-target-out (offset_model.loto), else it is G3.")}
