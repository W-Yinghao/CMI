"""Calibration — NLL (the formal K2 endpoint) and fixed-bin top-label ECE (auxiliary).

NLL is computed from a numerically stable ``log_softmax`` (finite for extreme logits). ECE uses
FIXED, pre-registered equal-width bin edges (default 15) shared across all methods/domains/seeds/
levels. **No** target-fit temperature/binning/calibration parameters — ``fit_temperature`` raises
unless explicitly flagged as a diagnostic (never in the main table).
"""
from __future__ import annotations

import numpy as np


def log_softmax(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def softmax(logits) -> np.ndarray:
    return np.exp(log_softmax(logits))


def nll_per_sample(logits, y) -> np.ndarray:
    ls = log_softmax(logits)
    y = np.asarray(y, dtype=int)
    return -ls[np.arange(len(y)), y]


def pooled_nll(logits, y) -> float:
    return float(nll_per_sample(logits, y).mean())


def domain_nlls(logits, y, domain) -> dict:
    logits, y, domain = np.asarray(logits), np.asarray(y), np.asarray(domain)
    per = nll_per_sample(logits, y)
    return {int(dd): float(per[domain == dd].mean()) for dd in np.unique(domain)}


def mean_domain_nll(logits, y, domain) -> float:
    return float(np.mean(list(domain_nlls(logits, y, domain).values())))


def worst_domain_nll(logits, y, domain) -> float:
    return float(np.max(list(domain_nlls(logits, y, domain).values())))   # higher NLL = worse


def fixed_bin_edges(n_bins: int = 15) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def top_label_ece(logits, y, n_bins: int = 15, bin_edges=None) -> float:
    """Multiclass top-label ECE ``Σ_b (n_b/N) |acc(b) − conf(b)|`` with FIXED equal-width bins."""
    edges = fixed_bin_edges(n_bins) if bin_edges is None else np.asarray(bin_edges, float)
    p = softmax(logits)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == np.asarray(y, int)).astype(float)
    n = len(y)
    ece = 0.0
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() > 0:
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def domain_eces(logits, y, domain, n_bins: int = 15, bin_edges=None) -> dict:
    """Per-domain top-label ECE with the SAME fixed equal-width bins for every domain."""
    logits, y, d = np.asarray(logits), np.asarray(y), np.asarray(domain)
    edges = fixed_bin_edges(n_bins) if bin_edges is None else np.asarray(bin_edges, float)
    return {int(dd): top_label_ece(logits[d == dd], y[d == dd], bin_edges=edges) for dd in np.unique(d)}


def mean_domain_ece(logits, y, domain, n_bins: int = 15, bin_edges=None) -> float:
    return float(np.mean(list(domain_eces(logits, y, domain, n_bins, bin_edges).values())))


def worst_domain_ece(logits, y, domain, n_bins: int = 15, bin_edges=None) -> float:
    return float(np.max(list(domain_eces(logits, y, domain, n_bins, bin_edges).values())))   # higher = worse


def fit_temperature(logits, y, role: str = "target") -> float:
    """REFUSED outside an explicit diagnostic. Source-only evaluation forbids target-fit
    calibration parameters (temperature/binning); an oracle target temperature is a labelled
    diagnostic only, never in the main table."""
    if role != "diagnostic":
        raise ValueError(
            f"target-fit calibration is forbidden in the source-only protocol (role={role!r}); "
            f"only role='diagnostic' is permitted, and it must never enter the main table."
        )
    # diagnostic-only 1-D temperature search (kept minimal; flagged, off the main path)
    from .metrics import np as _np  # noqa
    grid = np.linspace(0.25, 4.0, 16)
    return float(min(grid, key=lambda t: pooled_nll(np.asarray(logits) / t, y)))
