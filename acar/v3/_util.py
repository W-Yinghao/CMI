"""Shared utilities for ACAR v3. DESIGN/DEV stage."""
from __future__ import annotations
import numpy as np


def frozen_array(a, dtype=None):
    """Return a STRONGLY-immutable ndarray backed by an immutable `bytes` buffer: `flags.writeable` is False AND
    cannot be re-enabled (re-enabling raises because the backing buffer is read-only). Use for all stored arrays so
    a 'frozen' dataclass cannot be mutated in place via `arr.flags.writeable = True`."""
    a = np.ascontiguousarray(a if dtype is None else np.asarray(a, dtype))
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
