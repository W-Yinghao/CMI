"""C17-C — DIAGNOSTIC-ONLY multivariate competence probe. If no single source signal identifies target-good
checkpoints, does any COMBINATION of source-only descriptors? A low-freedom logistic probe is trained on
source-only features to predict the post-hoc target-good label, evaluated leave-one-TARGET-out (and
leave-one-SEED-out for sensitivity), against a within-fold-level permutation baseline.

THIS IS NOT A DEPLOYABLE SELECTOR. It trains on target labels (diagnostic-only) purely to measure whether
source observables CONTAIN recoverable information about target-good checkpoints. It emits no selector
artifact and makes no deployment claim.
"""
from __future__ import annotations

import numpy as np

from .signal_atlas import SOURCE_SIGNALS

_N_PERM = 120
NON_DEPLOYABLE = True                       # hard flag: this module never returns a selector


def _finite(v):
    return v is not None and not (isinstance(v, float) and v != v)     # drop None AND NaN


def _matrix(rows, label="tgt__target_bacc_good"):
    allcols = ["src__" + s for s in SOURCE_SIGNALS]
    # Drop columns that are ENTIRELY non-finite (an absent / estimability-dropped signal); otherwise the
    # all-columns-finite row filter below would drop every ROW instead of removing the column.
    cols = [c for c in allcols if any(_finite(r.get(c)) for r in rows)]
    keep = [r for r in rows if cols and all(_finite(r.get(c)) for c in cols)]
    X = (np.array([[float(r[c]) for c in cols] for r in keep], dtype=np.float64) if keep
         else np.zeros((0, len(cols)), dtype=np.float64))
    y = np.array([1 if r[label] else 0 for r in keep], dtype=int)
    groups_t = np.array([r["target"] for r in keep])
    groups_s = np.array([r["seed"] for r in keep])
    fold = np.array([hash((r["seed"], r["target"], r["level"])) for r in keep])
    return X, y, groups_t, groups_s, fold, cols


def _auc(y, s):
    # rank-based AUC (Mann-Whitney); no sklearn dependency, deterministic
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    avg = np.zeros(len(cnt)); pos_c = 0
    for i, c in enumerate(cnt):
        avg[i] = pos_c + (c + 1) / 2.0; pos_c += c
    ranks = avg[inv]
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _fit_logit(Xtr, ytr, Xte, *, l2=1.0, iters=800, lr=0.3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    n, d = Xtr.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        z = Xtr @ w + b; p = 1 / (1 + np.exp(-z))
        gw = Xtr.T @ (p - ytr) / n + l2 * w / n; gb = (p - ytr).mean()
        w -= lr * gw; b -= lr * gb
    return 1 / (1 + np.exp(-(Xte @ w + b)))


def _loto(X, y, groups):
    scores = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        if y[tr].sum() == 0 or y[tr].sum() == len(y[tr]):
            scores[te] = 0.5
        else:
            scores[te] = _fit_logit(X[tr], y[tr].astype(float), X[te])
    return _auc(y, scores), scores


def multivariate_probe(rows, *, perm_seed=0, n_perm=_N_PERM) -> dict:
    X, y, gt, gs, fold, cols = _matrix(rows)
    loto_auc, _ = _loto(X, y, gt)
    loso_auc, _ = _loto(X, y, gs)
    per_target = {}
    for g in np.unique(gt):
        te = gt == g; tr = ~te
        if 0 < y[tr].sum() < len(y[tr]):
            s = _fit_logit(X[tr], y[tr].astype(float), X[te])
            per_target[str(int(g))] = _auc(y[te], s)
    # within-fold-level permutation baseline for LOTO AUC
    rng = np.random.RandomState(perm_seed)
    null = []
    for _ in range(int(n_perm)):
        yp = y.copy()
        for fv in np.unique(fold):
            idx = np.where(fold == fv)[0]
            yp[idx] = rng.permutation(yp[idx])
        a, _ = _loto(X, yp, gt)
        if a is not None:
            null.append(a)
    null = np.array(null)
    p = float((np.sum(null >= loto_auc) + 1) / (len(null) + 1)) if loto_auc is not None else None
    beats = loto_auc is not None and p is not None and p < 0.05 and loto_auc > 0.55
    return {"n_used": int(len(y)), "n_features": len(cols), "base_rate": float(y.mean()),
            "loto_auc": loto_auc, "loso_auc": loso_auc, "per_target_auc": per_target,
            "permutation_mean_auc": float(null.mean()) if len(null) else None,
            "permutation_p": p, "beats_permutation": bool(beats), "non_deployable": NON_DEPLOYABLE,
            "note": ("DIAGNOSTIC-ONLY information-content probe; NOT a deployable selector. Trains on post-hoc "
                     "target labels solely to test whether source observables contain recoverable target-good "
                     "information. Leave-one-target-out AUC vs a within-fold-level permutation null.")}
