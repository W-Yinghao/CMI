"""Euclidean Alignment (EA) comparator for V2 (review V2_FROZEN §3).

Standard EEG-transfer covariance reference alignment. STRICT protocol: the source reference
covariance is estimated from SOURCE TRAINING trials only; the target reference from the UNLABELED
ADAPT split only; the resulting transform is FROZEN and applied to EVALUATION trials, which never
enter any covariance estimate. The frozen raw-trained source model then embeds the EA-transformed
eval trials. Recolour-to-source convention (M = R_s^{1/2} R_t^{-1/2}) so a model trained on raw
source sees target trials whose average channel covariance matches the source reference. EA is an
always-align comparator -- NOT routed by metadata.
"""
from __future__ import annotations

import numpy as np


def reference_cov(X: np.ndarray) -> np.ndarray:
    """Mean per-trial channel covariance R = (1/N) sum_i X_i X_i^T / T. X: [n, ch, t]."""
    n, c, t = X.shape
    covs = np.einsum("nct,ndt->ncd", X, X) / t
    R = covs.mean(0)
    R = R + 1e-6 * np.eye(c)                          # ridge for invertibility
    return R


def _sym_pow(R: np.ndarray, p: float) -> np.ndarray:
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-10, None)
    return (V * (w ** p)) @ V.T


def ea_transform(R_src: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    """Frozen recolour-to-source map M = R_s^{1/2} R_t^{-1/2}."""
    return _sym_pow(R_src, 0.5) @ _sym_pow(R_tgt, -0.5)


def apply_ea(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply the frozen channel-space transform to every trial: X'_i = M X_i. X: [n, ch, t]."""
    return np.einsum("cd,ndt->nct", M, X).astype(np.float32)
