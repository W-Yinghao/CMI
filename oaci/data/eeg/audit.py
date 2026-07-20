"""Audit hashes proving inputs are byte-identical, and prediction-bundle validation.

Identical sample IDs are NOT enough to prove the input tensors match — also pin
``audit_tensor_hash`` (the actual X), ``split_manifest_hash`` (the role index sets + seeds), and
``preprocess_hash``. These travel on the ``PredictionBundle`` and are checked by the sweep.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def tensor_hash(X) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(X, dtype=np.float32)).tobytes()).hexdigest()


def split_manifest_hash(split) -> str:
    payload = json.dumps({
        "source_train": sorted(int(i) for i in split.source_train),
        "source_audit": sorted(int(i) for i in split.source_audit),
        "target_audit": sorted(int(i) for i in split.target_audit),
        "target_domain": int(split.target_domain),
        "split_seed": int(split.split_seed),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def validate_prediction_bundle(pb) -> None:
    """Unique sample IDs, finite logits, in-range labels, consistent shapes."""
    if np.unique(pb.sample_id).size != pb.n:
        raise ValueError("PredictionBundle.sample_id must be unique")
    if pb.logits.shape != (pb.n, pb.n_classes):
        raise ValueError(f"logits shape {pb.logits.shape} != ({pb.n},{pb.n_classes})")
    if not np.isfinite(pb.logits).all():
        raise ValueError("PredictionBundle.logits contains non-finite values")
    if int(pb.y.min()) < 0 or int(pb.y.max()) >= pb.n_classes:
        raise ValueError("PredictionBundle.y out of class range")
