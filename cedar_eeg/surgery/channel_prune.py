"""Structured channel/filter pruning helpers for the blocked P1 phase."""

from __future__ import annotations

import numpy as np


def channel_keep_mask(n_channels: int, drop_channels: list[int] | np.ndarray) -> np.ndarray:
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    keep = np.ones(n_channels, dtype=bool)
    drop = np.asarray(drop_channels, dtype=np.int64)
    if len(drop):
        if drop.min() < 0 or drop.max() >= n_channels:
            raise ValueError("drop channel out of range")
        keep[drop] = False
    return keep


def apply_channel_mask(x: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Zero dropped channels while preserving tensor shape."""

    x = np.asarray(x)
    keep_mask = np.asarray(keep_mask, dtype=x.dtype)
    if x.ndim < 2:
        raise ValueError("x must include a channel dimension")
    if keep_mask.shape != (x.shape[1],):
        raise ValueError("keep_mask must match x.shape[1]")
    shape = [1] * x.ndim
    shape[1] = len(keep_mask)
    return x * keep_mask.reshape(shape)
