"""Processed-cache key + atomic write. Raw datalake is read-only; processed cache goes to a
separate scratch dir. The cache key folds the dataset fingerprint, channel ORDER, the full
preprocessing spec, and the code version — so any of those changing invalidates the cache.
Writes go to a temp file + ``os.replace`` (atomic; no torn/partial cache file).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile


def cache_key(dataset_fingerprint: str, ch_names, preprocess_spec: dict, code_version: str) -> str:
    payload = json.dumps({
        "dataset_fingerprint": dataset_fingerprint,
        "ch_names": list(ch_names),                 # ORDER-sensitive on purpose
        "preprocess": preprocess_spec,
        "code_version": code_version,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def atomic_write_bytes(path: str, data: bytes) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-cache-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)                       # atomic rename
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
