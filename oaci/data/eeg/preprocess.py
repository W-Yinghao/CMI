"""Preprocessing spec + normalization fit/apply with a hard target-leakage guard.

Cross-trial normalization statistics may be fit ONLY on ``source_train``. A per-window/per-trial
z-score is allowed as a fixed sample-wise transform but is recorded in the manifest. The main
clinical protocol does NOT do target-data-driven channel interpolation (header-scanned common
native channel list is frozen; interpolation is a labelled sensitivity analysis only).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class PreprocessSpec:
    l_freq: float = 4.0
    h_freq: float = 40.0
    resample_sfreq: float = 128.0
    window_s: float = 2.0
    window_stride_s: float = 1.0
    epoch_tmin: float | None = None           # trial epoch window (s) — passed to MOABB
    epoch_tmax: float | None = None
    channels: list | None = None              # frozen common native channel ORDER (no interpolation)
    normalization: str = "zscore_sample"      # 'none' | 'zscore_sample' | 'zscore_channel_source'
    channel_interpolation: bool = False       # main protocol: False (sensitivity analysis only)
    code_version: str = "oaci-eeg-1"

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:24]


def assert_fit_excludes_target(fit_idx, target_idx) -> None:
    if set(np.asarray(fit_idx).tolist()) & set(np.asarray(target_idx).tolist()):
        raise ValueError("normalization fit set overlaps the sealed target audit population")


def fit_normalization(X_source_train, spec: PreprocessSpec):
    """Cross-trial stats fit on source_train ONLY (X passed is already restricted to it)."""
    if spec.normalization != "zscore_channel_source":
        return {"kind": spec.normalization}
    X = np.asarray(X_source_train, dtype=np.float64)         # [N,C,T]
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return {"kind": "zscore_channel_source", "mu": mu, "sd": sd}


def apply_normalization(X, stats, spec: PreprocessSpec) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if spec.normalization == "none":
        return X
    if spec.normalization == "zscore_sample":                # sample-wise, no cross-trial stats
        mu = X.mean(axis=2, keepdims=True)
        sd = X.std(axis=2, keepdims=True) + 1e-8
        return ((X - mu) / sd).astype(np.float32)
    mu, sd = stats["mu"], stats["sd"]                        # source-fit channel stats
    return ((X - mu) / sd).astype(np.float32)
