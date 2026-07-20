"""Stable, type-tagged hash of nested scientific values (leakage mappings, cache stats, future
JSON identity). Rejects unknown objects; never uses repr() or an object-array's pointer bytes.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np

from .keys import feed_float64, feed_int64, feed_string


def _feed_float(h, x: float) -> None:
    """Tag non-finite floats semantically (never depend on the platform's NaN payload bits)."""
    if x != x:
        h.update(b"Fnan")
    elif x == float("inf"):
        h.update(b"F+inf")
    elif x == float("-inf"):
        h.update(b"F-inf")
    else:
        h.update(b"F"); feed_float64(h, float(x))


def _feed(h, v) -> None:
    if v is None:
        h.update(b"N")
    elif isinstance(v, bool):                         # before int (bool is an int subclass)
        h.update(b"B1" if v else b"B0")
    elif isinstance(v, (int, np.integer)):
        h.update(b"I"); feed_int64(h, int(v))
    elif isinstance(v, (float, np.floating)):
        _feed_float(h, float(v))
    elif isinstance(v, str):
        h.update(b"S"); feed_string(h, v)
    elif isinstance(v, np.ndarray):
        if v.dtype == object:                         # object arrays would pointer-hash -> hard fail
            raise TypeError("scientific_value_hash: object-dtype ndarray is not hashable")
        a = np.ascontiguousarray(v); h.update(b"A")
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); feed_int64(h, a.ndim)
        if a.dtype.kind == "f":                       # element-wise so non-finite tags are consistent
            h.update(b"f"); feed_int64(h, a.size)
            for x in a.ravel(order="C").tolist():
                _feed_float(h, float(x))
        else:
            h.update(b"b"); h.update(a.tobytes())
    elif isinstance(v, Mapping):
        h.update(b"M")
        for k in sorted(v.keys()):
            if not isinstance(k, str):                # str(1)==str("1") collision: forbid non-string keys
                raise TypeError(f"scientific_value_hash: mapping key must be str, got {type(k)!r}")
            feed_string(h, k); _feed(h, v[k])
    elif isinstance(v, (list, tuple)):
        h.update(b"L"); feed_int64(h, len(v))
        for x in v:
            _feed(h, x)
    else:
        raise TypeError(f"scientific_value_hash: unsupported type {type(v)!r}")


def scientific_value_hash(value) -> str:
    h = hashlib.sha256(); _feed(h, value); return h.hexdigest()


def _key_to_str(k) -> str:
    if isinstance(k, str):
        return k
    if isinstance(k, bool):
        return f"bool:{k}"
    if isinstance(k, (int, np.integer)):
        return f"int:{int(k)}"
    raise TypeError(f"normalize_keys: unsupported mapping key type {type(k)!r}")


def normalize_keys(v):
    """JSON-normalize nested mapping keys to (type-tagged) strings so a structure with legitimate
    int keys (e.g. a per-class reference_entropy) is hashable/serialisable. ``int:0`` never collides
    with the string ``"0"``; a real post-normalisation collision is a hard failure."""
    if isinstance(v, Mapping):
        out = {}
        for k, x in v.items():
            ks = _key_to_str(k)
            if ks in out:
                raise ValueError(f"normalize_keys: key collision on {ks!r}")
            out[ks] = normalize_keys(x)
        return out
    if isinstance(v, (list, tuple)):
        return type(v)(normalize_keys(x) for x in v)
    return v


def leakage_result_hash(value) -> str:
    """Scientific identity of a leakage result mapping (whose per-class sub-maps may be int-keyed)."""
    return scientific_value_hash(normalize_keys(value))
