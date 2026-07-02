"""ACAR V5 Stage-1B source-state (numpy imported LAZILY; nothing heavy at module load). A class-conditional Gaussian over the frozen
ENCODER features of the FIT-train subjects (training_config.source_state = 'class_conditional_gaussian_tangent'): per-class mean, a
pooled (shared) covariance with a small ridge, and class priors. Serialized to DETERMINISTIC, pickle-free bytes so it is hashable and
reloadable from the frozen artifacts. This is the source distribution the TTA actions (matched_coral / spdim / t3a) will use later; at
Stage-1B8 we only need it deterministic + serializable + loadable.
"""
from __future__ import annotations
import json
import struct

_MAGIC = b"ACARV5SS"
_RIDGE = 1e-4


class SourceStateError(RuntimeError):
    pass


def fit_source_state(features, labels):
    """features: (N, D) float; labels: (N,) in {0,1}. Returns a dict of numpy arrays (means[2,D], cov[D,D], priors[2], classes[2])."""
    import numpy as np
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise SourceStateError("features must be (N,D) aligned with labels (N,)")
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise SourceStateError("labels must be in {0,1}")
    if X.shape[0] < 2:
        raise SourceStateError("need >=2 FIT-train windows to fit the source state")
    D = X.shape[1]
    means = np.zeros((2, D), dtype=np.float64)
    priors = np.zeros(2, dtype=np.float64)
    centered = np.empty_like(X)
    for c in (0, 1):
        mask = y == c
        n_c = int(mask.sum())
        if n_c == 0:
            raise SourceStateError(f"source state requires both classes present in FIT-train (class {c} absent)")
        means[c] = X[mask].mean(axis=0)
        priors[c] = n_c / X.shape[0]
        centered[mask] = X[mask] - means[c]
    cov = (centered.T @ centered) / max(X.shape[0] - 2, 1)
    cov = cov + _RIDGE * np.eye(D)                            # ridge → invertible / stable
    return {"means": means, "cov": cov, "priors": priors, "classes": np.asarray([0, 1], dtype=np.int64)}


def serialize_source_state(state):
    """Deterministic, pickle-free bytes: magic + json header (sorted names/shapes) + little-endian float64/int64 blobs."""
    import numpy as np
    order = sorted(state.keys())
    header, blobs = [], []
    for name in order:
        arr = np.ascontiguousarray(state[name])
        kind = "<i8" if arr.dtype.kind == "i" else "<f8"
        arr = arr.astype(kind, copy=False)
        header.append({"name": name, "shape": list(arr.shape), "dtype": kind})
        blobs.append(arr.tobytes())
    hjson = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out = bytearray(_MAGIC)
    out += struct.pack("<I", len(hjson))
    out += hjson
    for b in blobs:
        out += b
    return bytes(out)


def load_source_state(blob):
    import numpy as np
    if bytes(blob[:len(_MAGIC)]) != _MAGIC:
        raise SourceStateError("bad source-state blob (magic mismatch)")
    off = len(_MAGIC)
    (hlen,) = struct.unpack("<I", blob[off:off + 4])
    off += 4
    header = json.loads(bytes(blob[off:off + hlen]).decode("utf-8"))
    off += hlen
    out = {}
    for entry in header:
        shape = tuple(entry["shape"])
        dt = entry["dtype"]
        itemsize = 8
        n = 1
        for s in shape:
            n *= s
        arr = np.frombuffer(blob[off:off + itemsize * n], dtype=dt).reshape(shape).copy()
        off += itemsize * n
        out[entry["name"]] = arr
    return out
