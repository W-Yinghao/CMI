"""Metrics for TALOS preflight and final-only evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from talos_eeg.adapters.trust_region import stable_payload_hash


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> float:
    y_true = np.asarray(y_true).astype(np.int64, copy=False)
    y_pred = np.asarray(y_pred).astype(np.int64, copy=False)
    labels = np.arange(n_classes) if n_classes is not None else np.unique(y_true)
    recalls = []
    for label in labels:
        idx = y_true == int(label)
        if np.any(idx):
            recalls.append(float((y_pred[idx] == int(label)).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def negative_log_likelihood(y_true: np.ndarray, proba: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64, copy=False)
    proba = np.asarray(proba, dtype=np.float64)
    clipped = np.clip(proba[np.arange(len(y_true)), y_true], 1e-8, 1.0)
    return float(-np.log(clipped).mean())


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, *, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(np.int64, copy=False)
    proba = np.asarray(proba, dtype=np.float64)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = (pred == y_true).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            keep = (conf >= lo) & (conf <= hi)
        else:
            keep = (conf >= lo) & (conf < hi)
        if np.any(keep):
            ece += float(keep.mean()) * abs(float(correct[keep].mean()) - float(conf[keep].mean()))
    return float(ece)


def final_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float | int]:
    proba = np.asarray(proba, dtype=np.float64)
    pred = proba.argmax(axis=1)
    return {
        "bacc": balanced_accuracy(y_true, pred, n_classes=proba.shape[1]),
        "nll": negative_log_likelihood(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
        "n_eval": int(len(pred)),
    }


def unlabeled_diagnostics(proba: np.ndarray) -> dict[str, Any]:
    proba = np.asarray(proba, dtype=np.float64)
    entropy = -np.sum(proba * np.log(np.maximum(proba, 1e-8)), axis=1)
    pred = proba.argmax(axis=1)
    counts = np.bincount(pred, minlength=proba.shape[1]).astype(np.int64)
    return {
        "entropy_mean": float(entropy.mean()),
        "entropy_min": float(entropy.min()),
        "confidence_mean": float(proba.max(axis=1).mean()),
        "mean_prediction": proba.mean(axis=0).astype(float).tolist(),
        "predicted_label_counts": counts.astype(int).tolist(),
        "n_samples": int(len(proba)),
        "n_classes": int(proba.shape[1]),
    }


def metrics_hash(payload: dict[str, Any]) -> str:
    return stable_payload_hash(payload)
