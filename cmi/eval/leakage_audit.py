"""Leakage audit — verify the conditional-domain-leakage estimate on a frozen Z, to tell apart:
  (1) estimator inaccurate, (2) penalizing harmless info, (3) sampler/prior mismatch.

For a frozen representation Z and discrete (Y, D), we measure conditional domain leakage as the
*advantage* of predicting D from [Z, one-hot(Y)] over predicting D from Y alone (the
label-conditional prior), using an ENSEMBLE of probes. Plus:
  - permutation null: shuffle D within each Y stratum -> a valid probe must give advantage ~0.
  - cross-fit: probe trained on split A, evaluated on disjoint split B (no encoder/probe overlap).
  - grouped variant: split by `groups` (e.g. session) so probe-eval is on different recordings
    (controls the non-iid-window contamination).
If all probes agree (ERM high, lpc_prior low) AND the permutation null is ~0, the estimate is
trustworthy -> a flat accuracy then points to harmless leakage (2), not estimator error (1).
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_selection import mutual_info_classif

PROBES = {
    "linear": lambda: LogisticRegression(max_iter=500),
    "mlp_s": lambda: MLPClassifier((64,), max_iter=250, early_stopping=True),
    "mlp_l": lambda: MLPClassifier((256, 128), max_iter=300, early_stopping=True),
    "rf": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1),
    "hgbm": lambda: HistGradientBoostingClassifier(max_iter=150),
}


def _prior_bacc(y_tr, d_tr, y_ev, d_ev, n_dom, n_cls):
    counts = np.zeros((n_cls, n_dom))
    for yi, di in zip(y_tr, d_tr):
        counts[yi, di] += 1
    pred = counts.argmax(1)[y_ev]                       # most likely domain given Y
    return balanced_accuracy_score(d_ev, pred)


def _feat(Z, y, n_cls):
    return np.concatenate([Z, np.eye(n_cls)[y]], axis=1)


def _hsic(A, B):                                        # numpy RBF-vs-linear HSIC
    d2 = ((A[:, None] - A[None]) ** 2).sum(-1)
    sig = np.median(d2[d2 > 0]) + 1e-8
    K = np.exp(-d2 / (2 * sig)); L = B @ B.T
    n = len(A); H = np.eye(n) - 1.0 / n
    return float(np.trace(K @ H @ L @ H) / (n - 1) ** 2)


def _chsic(Z, y, d, n_cls, n_dom, cap, rng):
    tot = 0.0
    for c in range(n_cls):
        m = np.where(y == c)[0]
        if len(m) < 8 or len(np.unique(d[m])) < 2:
            continue
        if len(m) > cap:
            m = rng.choice(m, cap, replace=False)
        tot += (len(m) / len(Z)) * _hsic(Z[m], np.eye(n_dom)[d[m]])
    return tot


def _knn_cmi(Z, y, d, n_cls):
    tot = 0.0
    for c in range(n_cls):
        m = y == c
        if m.sum() < 20 or len(np.unique(d[m])) < 2:
            continue
        tot += m.mean() * float(mutual_info_classif(Z[m], d[m], discrete_features=False,
                                                     random_state=0).sum())
    return tot


def audit(Z, y, d, n_cls, n_dom, seed=0, groups=None, hsic_cap=1500):
    rng = np.random.default_rng(seed)
    Z = (Z.astype("float64") - Z.mean(0)) / (Z.std(0) + 1e-8)
    n = len(Z)
    if groups is not None:                              # grouped split (e.g. by session)
        g = np.unique(groups); rng.shuffle(g)
        trg = set(g[:max(1, int(0.7 * len(g)))])
        tr = np.where(np.isin(groups, list(trg)))[0]; ev = np.where(~np.isin(groups, list(trg)))[0]
        if len(ev) < 20 or len(tr) < 20:               # fall back to random if split degenerate
            groups = None
    if groups is None:
        idx = rng.permutation(n); cut = int(0.7 * n); tr, ev = idx[:cut], idx[cut:]
    F = _feat(Z, y, n_cls)
    prior = _prior_bacc(y[tr], d[tr], y[ev], d[ev], n_dom, n_cls)
    out = {"prior_bacc": prior, "n_eval": int(len(ev))}
    for name, mk in PROBES.items():
        try:
            pred = mk().fit(F[tr], d[tr]).predict(F[ev])
            out[name + "_adv"] = float(balanced_accuracy_score(d[ev], pred) - prior)
        except Exception:
            out[name + "_adv"] = float("nan")
    out["hsic"] = _chsic(Z[ev], y[ev], d[ev], n_cls, n_dom, hsic_cap, rng)
    out["knn_cmi"] = _knn_cmi(Z, y, d, n_cls)
    # permutation null: shuffle D within each Y stratum -> advantage must be ~0
    dp = d.copy()
    for c in range(n_cls):
        m = np.where(y == c)[0]; dp[m] = d[m][rng.permutation(len(m))]
    pred = PROBES["mlp_s"]().fit(F[tr], dp[tr]).predict(F[ev])
    out["perm_null_adv"] = float(balanced_accuracy_score(dp[ev], pred)
                                 - _prior_bacc(y[tr], dp[tr], y[ev], dp[ev], n_dom, n_cls))
    return out
