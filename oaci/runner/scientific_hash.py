"""Stable, type-tagged hash of nested scientific values (leakage mappings, cache stats, future
JSON identity). Rejects unknown objects; never uses repr() or an object-array's pointer bytes.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np

from .keys import feed_float64, feed_int64, feed_string


def _feed(h, v) -> None:
    if v is None:
        h.update(b"N")
    elif isinstance(v, bool):                         # before int (bool is an int subclass)
        h.update(b"B1" if v else b"B0")
    elif isinstance(v, (int, np.integer)):
        h.update(b"I"); feed_int64(h, int(v))
    elif isinstance(v, (float, np.floating)):
        h.update(b"F"); feed_float64(h, float(v))
    elif isinstance(v, str):
        h.update(b"S"); feed_string(h, v)
    elif isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v); h.update(b"A")
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    elif isinstance(v, Mapping):
        h.update(b"M")
        for k in sorted(v, key=str):
            feed_string(h, str(k)); _feed(h, v[k])
    elif isinstance(v, (list, tuple)):
        h.update(b"L"); feed_int64(h, len(v))
        for x in v:
            _feed(h, x)
    else:
        raise TypeError(f"scientific_value_hash: unsupported type {type(v)!r}")


def scientific_value_hash(value) -> str:
    h = hashlib.sha256(); _feed(h, value); return h.hexdigest()
