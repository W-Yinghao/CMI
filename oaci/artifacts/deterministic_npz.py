"""Deterministic, pickle-free NPZ.

A hand-rolled ZIP of per-array .npy entries with canonical-sorted keys, a fixed 1980-01-01 timestamp,
ZIP_STORED (no DEFLATE version drift), fixed create_system / permissions and no comment/extra fields,
so identical arrays produce a byte-identical archive. Object dtype is rejected. Every entry carries an
``array_content_hash`` (dtype + shape + contiguous bytes) verified on read.
"""
from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np

_FIXED_DT = (1980, 1, 1, 0, 0, 0)


def to_unicode_array(seq) -> np.ndarray:
    """Fixed-width little-endian unicode array of stable string ids (never an object array)."""
    return np.array([str(s) for s in seq], dtype=np.str_)


def array_content_hash(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(a), allow_pickle=False)
    return buf.getvalue()


def _check(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == object:
        raise TypeError("deterministic_npz: object-dtype array is not allowed")
    if a.dtype.kind not in ("i", "u", "f", "b", "U"):
        raise TypeError(f"deterministic_npz: unsupported dtype {a.dtype!r}")
    return np.ascontiguousarray(a)


def write_deterministic_npz(path, arrays: dict) -> dict:
    """Write ``{key: ndarray}`` deterministically. Returns the per-key metadata block."""
    meta = {}
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as zf:
        for key in sorted(arrays):
            a = _check(arrays[key])
            zi = zipfile.ZipInfo(f"{key}.npy", date_time=_FIXED_DT)
            zi.compress_type = zipfile.ZIP_STORED
            zi.create_system = 0
            zi.external_attr = 0
            zf.writestr(zi, _npy_bytes(a))
            meta[key] = {"dtype": str(a.dtype), "shape": list(a.shape), "array_content_hash": array_content_hash(a)}
    data = payload.getvalue()
    with open(path, "wb") as f:
        f.write(data); f.flush()
        import os
        os.fsync(f.fileno())
    return meta


def deterministic_npz_bytes(arrays: dict) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as zf:
        for key in sorted(arrays):
            a = _check(arrays[key])
            zi = zipfile.ZipInfo(f"{key}.npy", date_time=_FIXED_DT)
            zi.compress_type = zipfile.ZIP_STORED; zi.create_system = 0; zi.external_attr = 0
            zf.writestr(zi, _npy_bytes(a))
    return payload.getvalue()


def read_verified_npz(path, metadata: dict) -> dict:
    out = {}
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        if names != {f"{k}.npy" for k in metadata}:
            raise ValueError(f"npz keys {sorted(names)} disagree with metadata {sorted(metadata)}")
        for key, m in metadata.items():
            a = np.lib.format.read_array(io.BytesIO(zf.read(f"{key}.npy")), allow_pickle=False)
            if str(a.dtype) != m["dtype"] or list(a.shape) != list(m["shape"]):
                raise ValueError(f"npz '{key}' dtype/shape disagrees with metadata")
            if array_content_hash(a) != m["array_content_hash"]:
                raise ValueError(f"npz '{key}' content hash mismatch (corruption)")
            out[key] = a
    return out
