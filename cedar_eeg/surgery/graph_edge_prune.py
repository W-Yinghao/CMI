"""Graph node/edge surgery helpers for the blocked graph phase."""

from __future__ import annotations

import numpy as np


def symmetric_edge_keep_mask(n_nodes: int, drop_edges: list[tuple[int, int]]) -> np.ndarray:
    if n_nodes <= 1:
        raise ValueError("n_nodes must be > 1")
    keep = np.ones((n_nodes, n_nodes), dtype=bool)
    for i, j in drop_edges:
        if i < 0 or j < 0 or i >= n_nodes or j >= n_nodes:
            raise ValueError(f"edge {(i, j)} out of range")
        keep[i, j] = False
        keep[j, i] = False
    np.fill_diagonal(keep, True)
    return keep


def apply_edge_mask(edge_logits: np.ndarray, keep_mask: np.ndarray, fill: float = 0.0) -> np.ndarray:
    edge_logits = np.asarray(edge_logits)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if edge_logits.shape[-2:] != keep_mask.shape:
        raise ValueError("keep_mask must match the last two edge dimensions")
    return np.where(keep_mask, edge_logits, fill)
