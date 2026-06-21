"""Subspace-stability diagnostics -- the project's *termination gate*.

The direction is only worth pursuing if the selected domain-rich / label-light subspace
is the SAME thing across seeds, probes and folds. If the selection is unstable we are
just fitting noise and should stop (per the stated termination condition). These
functions quantify that stability via principal angles between the selected subspaces.

    principal_angles(B1, B2)   -> angles (radians), ascending
    subspace_overlap(B1, B2)   -> mean cos^2(angle) in [0,1] (1 == identical span)
    grassmann_distance(B1, B2) -> sqrt(sum angle^2) (0 == identical span)
    selection_stability(bases) -> mean pairwise overlap over a list of bases
"""
from __future__ import annotations
from typing import List

import numpy as np


def _orthonormalize(B: np.ndarray) -> np.ndarray:
    if B.size == 0 or B.shape[1] == 0:
        return B.reshape(B.shape[0], 0) if B.ndim == 2 else np.zeros((0, 0))
    Q, _ = np.linalg.qr(B)
    return Q


def principal_angles(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """Principal angles between span(B1) and span(B2). Bases [d,k1],[d,k2]."""
    Q1, Q2 = _orthonormalize(B1), _orthonormalize(B2)
    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return np.array([np.pi / 2])           # one empty span: maximally distant
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def subspace_overlap(B1: np.ndarray, B2: np.ndarray) -> float:
    """Mean cos^2 of principal angles in [0,1]; symmetric, 1 iff spans coincide.
    Compares over the smaller dimension so unequal-k subspaces are handled gracefully."""
    ang = principal_angles(B1, B2)
    return float(np.mean(np.cos(ang) ** 2))


def grassmann_distance(B1: np.ndarray, B2: np.ndarray) -> float:
    ang = principal_angles(B1, B2)
    return float(np.sqrt(np.sum(ang ** 2)))


def selection_stability(bases: List[np.ndarray]) -> dict:
    """Mean / min pairwise subspace overlap over a list of selected bases (one per
    seed/fold/probe). Also reports the spread of selected dimensions. Identity (k=0)
    selections are kept: a run that picks identity while others pick a subspace is
    *unstable*, which is exactly what we want flagged."""
    ks = [int(b.shape[1]) for b in bases]
    overlaps = []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            overlaps.append(subspace_overlap(bases[i], bases[j]))
    overlaps = np.array(overlaps) if overlaps else np.array([1.0])
    return {
        "mean_overlap": float(overlaps.mean()),
        "min_overlap": float(overlaps.min()),
        "k_values": ks,
        "k_mean": float(np.mean(ks)),
        "k_std": float(np.std(ks)),
        "n_identity": int(sum(k == 0 for k in ks)),
    }
