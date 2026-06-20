"""Metric panel + domain-clustered uncertainty (review section 3 'statistics to report').

Beyond mean balanced accuracy, every evaluation reports macro-F1, NLL, Brier, ECE, the
worst-domain balanced accuracy and the domain-level CVaR; uncertainty is a cluster
(domain) paired bootstrap, NOT a per-trial CI, because trials within a recording are
strongly autocorrelated.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def _ece(prob, y_true, n_bins=15):
    conf = prob.max(1); pred = prob.argmax(1); acc = (pred == y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def classification_metrics(prob: np.ndarray, y_true: np.ndarray) -> dict:
    prob = np.asarray(prob, dtype=np.float64)
    y_true = np.asarray(y_true)
    n_cls = prob.shape[1]
    pred = prob.argmax(1)
    p_true = np.clip(prob[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    onehot = np.eye(n_cls)[y_true]
    brier = float(((prob - onehot) ** 2).sum(1).mean())
    return dict(
        balanced_acc=float(balanced_accuracy_score(y_true, pred)),
        macro_f1=float(f1_score(y_true, pred, average="macro")),
        nll=float(-np.log(p_true).mean()),
        brier=brier,
        ece=_ece(prob, y_true),
    )


def per_domain_balanced_acc(y_true, pred, domain) -> dict:
    out = {}
    for d in np.unique(domain):
        m = domain == d
        if m.sum() == 0:
            continue
        out[int(d)] = float(balanced_accuracy_score(y_true[m], pred[m]))
    return out


def worst_domain_balanced_acc(y_true, pred, domain) -> float:
    pd = per_domain_balanced_acc(np.asarray(y_true), np.asarray(pred), np.asarray(domain))
    return float(min(pd.values())) if pd else float("nan")


def domain_cvar(y_true, pred, domain, alpha: float = 0.25) -> float:
    """Mean balanced acc over the worst ``alpha`` fraction of domains (lower-tail CVaR)."""
    pd = per_domain_balanced_acc(np.asarray(y_true), np.asarray(pred), np.asarray(domain))
    vals = np.sort(np.array(list(pd.values())))
    if len(vals) == 0:
        return float("nan")
    k = max(1, int(np.ceil(alpha * len(vals))))
    return float(vals[:k].mean())


def cluster_bootstrap_ci(per_domain_delta: dict, n_boot: int = 2000, alpha: float = 0.05,
                         seed: int = 0) -> dict:
    """Domain-clustered paired bootstrap CI on a per-domain delta (e.g. adapt - identity).

    ``per_domain_delta`` maps domain id -> scalar delta.  Resamples DOMAINS (clusters) with
    replacement.  Returns mean, [lo,hi] CI and the fraction of bootstrap means > 0.
    """
    vals = np.array(list(per_domain_delta.values()), dtype=np.float64)
    if len(vals) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), p_gt0=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                      for _ in range(n_boot)])
    return dict(mean=float(vals.mean()),
                lo=float(np.quantile(boots, alpha / 2)),
                hi=float(np.quantile(boots, 1 - alpha / 2)),
                p_gt0=float((boots > 0).mean()), n=int(len(vals)))


def metric_panel(prob: np.ndarray, y_true: np.ndarray, domain: np.ndarray) -> dict:
    """Full panel: classification metrics + worst-domain + CVaR."""
    pred = prob.argmax(1)
    m = classification_metrics(prob, y_true)
    m["worst_domain_bacc"] = worst_domain_balanced_acc(y_true, pred, domain)
    m["domain_cvar25"] = domain_cvar(y_true, pred, domain, 0.25)
    m["per_domain_bacc"] = per_domain_balanced_acc(np.asarray(y_true), pred, np.asarray(domain))
    return m


def panel_delta(panel_a: dict, panel_b: dict, keys=("balanced_acc", "macro_f1", "nll",
                                                     "brier", "ece", "worst_domain_bacc")) -> dict:
    """Delta panel (a - b) on the scalar metrics -- the review's Delta-metric table."""
    return {f"d_{k}": float(panel_a[k] - panel_b[k]) for k in keys if k in panel_a and k in panel_b}
