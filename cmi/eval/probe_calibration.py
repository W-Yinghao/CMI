"""CIGL R1 — probe calibration: expected calibration error (ECE) + temperature scaling. The leakage proxy is
a KL of the domain-probe posterior to the label-conditional prior; an over/under-confident probe inflates or
deflates that KL. Reporting ECE (and the leakage KL before/after temperature scaling on a held-out split)
shows the leakage signal is not a calibration artifact. Pure numpy.
"""
from __future__ import annotations
import numpy as np


def _softmax(logits, T=1.0):
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs, labels, n_bins=15):
    """Top-label ECE: |accuracy - confidence| averaged over confidence bins, weighted by bin population."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels).astype(np.int64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _nll(logits, labels, T):
    p = _softmax(logits, T)
    idx = np.arange(len(labels))
    return float(-np.log(np.clip(p[idx, labels], 1e-12, 1.0)).mean())


def fit_temperature(logits, labels, grid=None):
    """Fit the scalar temperature T>0 that minimizes NLL on (logits, labels) — the standard 1-parameter
    calibration. Coarse grid + local refine (no torch dependency). Returns T."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels).astype(np.int64)
    if grid is None:
        grid = np.geomspace(0.25, 8.0, 40)
    nlls = [_nll(logits, labels, T) for T in grid]
    T0 = float(grid[int(np.argmin(nlls))])
    # local refine around T0
    fine = np.linspace(max(0.05, T0 * 0.6), T0 * 1.6, 40)
    T = float(fine[int(np.argmin([_nll(logits, labels, t) for t in fine]))])
    return T


def calibration_report(logits, labels, n_bins=15):
    """ECE before/after temperature scaling + the fitted T. `logits` [N, n_dom] pre-softmax domain-probe
    outputs on a HELD-OUT split; `labels` = true domains on that split."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels).astype(np.int64)
    ece_raw = expected_calibration_error(_softmax(logits, 1.0), labels, n_bins)
    T = fit_temperature(logits, labels)
    ece_cal = expected_calibration_error(_softmax(logits, T), labels, n_bins)
    return {"ece_raw": ece_raw, "ece_calibrated": ece_cal, "temperature": T,
            "nll_raw": _nll(logits, labels, 1.0), "nll_calibrated": _nll(logits, labels, T),
            "improved": ece_cal <= ece_raw + 1e-9}
