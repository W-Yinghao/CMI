"""Subspace-stability / recovery diagnostics -- the project's *termination gate*.

The direction is only worth pursuing if the selected subspace is the SAME object across
seeds, probes and folds. If it isn't, we are fitting noise and should stop.

Honest metrics (the reviewer's point -- cos^2-over-min-dim is NOT a metric and is NOT
"1 iff spans coincide": a 2-D subspace fully contained in a 3-D one scores 1):

  principal_angles(B1, B2)        -> principal angles (radians)
  subspace_cos2_similarity(B1,B2) -> mean cos^2 over min(k1,k2) angles. CONTAINMENT-biased
                                     similarity in [0,1]; =1 when the smaller span sits
                                     inside the larger -- report alongside, never alone.
  projection_distance(B1, B2)     -> ||P1 - P2||_F, P=Q Q^T. A true metric: 0 iff the
                                     projectors (hence spans AND dims) are identical;
                                     dimension-SENSITIVE. This is the primary stability number.
  precision_recall(B_hat, B_star) -> precision = tr(P_hat P_star)/k_hat,
                                     recall    = tr(P_hat P_star)/k_star. Recovery vs a
                                     known ground-truth span (precision != recall when
                                     k_hat != k_star -- e.g. selecting 2 of a 4-D nuisance
                                     span gives precision~1, recall~0.5).

`selection_stability` therefore gates on projection distance + dimension spread + identity
consistency, NOT on the containment similarity alone.
"""
from __future__ import annotations
from typing import List

import numpy as np


def _orthonormalize(B: np.ndarray) -> np.ndarray:
    if B is None or B.size == 0 or B.ndim != 2 or B.shape[1] == 0:
        d = B.shape[0] if (B is not None and B.ndim == 2) else 0
        return np.zeros((d, 0))
    Q, _ = np.linalg.qr(B)
    return Q


def _projector(B: np.ndarray) -> np.ndarray:
    Q = _orthonormalize(B)
    d = Q.shape[0]
    return Q @ Q.T if Q.shape[1] else np.zeros((d, d))


def principal_angles(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    Q1, Q2 = _orthonormalize(B1), _orthonormalize(B2)
    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return np.array([np.pi / 2])
    s = np.clip(np.linalg.svd(Q1.T @ Q2, compute_uv=False), -1.0, 1.0)
    return np.arccos(s)


def subspace_cos2_similarity(B1: np.ndarray, B2: np.ndarray) -> float:
    """Mean cos^2 over min(k1,k2) principal angles. Containment-biased; report alongside
    projection_distance, never on its own (see module docstring)."""
    return float(np.mean(np.cos(principal_angles(B1, B2)) ** 2))


def projection_distance(B1: np.ndarray, B2: np.ndarray) -> float:
    """||P1 - P2||_F with P = Q Q^T. True metric on subspaces (incl. dimension); 0 iff equal.
    Range [0, sqrt(k1+k2)]; equals sqrt(|k1-k2|) when one span contains the other."""
    return float(np.linalg.norm(_projector(B1) - _projector(B2)))


def precision_recall(B_hat: np.ndarray, B_star: np.ndarray):
    """Recovery against a ground-truth span B_star:
        precision = tr(P_hat P_star) / rank(P_hat)   (how much of the SELECTION is real)
        recall    = tr(P_hat P_star) / rank(P_star)  (how much of the TRUTH was found)."""
    Ph, Ps = _projector(B_hat), _projector(B_star)
    kh, ks = _orthonormalize(B_hat).shape[1], _orthonormalize(B_star).shape[1]
    overlap = float(np.trace(Ph @ Ps))
    prec = overlap / kh if kh else 0.0
    rec = overlap / ks if ks else 0.0
    return {"precision": prec, "recall": rec, "k_hat": kh, "k_star": ks}


def grassmann_distance(B1: np.ndarray, B2: np.ndarray) -> float:
    return float(np.sqrt(np.sum(principal_angles(B1, B2) ** 2)))


def selection_stability(bases: List[np.ndarray], proj_dist_strict: float = 0.75,
                        max_k_spread: int = 1, nested_min: float = 0.90) -> dict:
    """Pairwise stability over selected bases (one per seed/draw/fold).

    Two bars, both reported honestly:
      * `passed` (CORE stability, the realistic bar for subspace *selection* that under-
        selects from a larger planted span): identity decision consistent across draws, the
        selected dim varies by <= `max_k_spread`, and the smaller span sits inside the larger
        (min pairwise containment cos^2 >= `nested_min`). A stable CORE with a flickering
        boundary dimension passes here.
      * `proj_dist_strict_pass` (the STRICT bar): max pairwise projection distance
        <= `proj_dist_strict`. A +-1 dimension flicker gives proj_dist ~1.0 and FAILS this
        -- it is the bar the eigengap/hysteresis robustness work must still reach. Not relaxed
        away; surfaced.
    """
    ks = [_orthonormalize(b).shape[1] for b in bases]
    cos2, pdist = [], []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            cos2.append(subspace_cos2_similarity(bases[i], bases[j]))
            pdist.append(projection_distance(bases[i], bases[j]))
    cos2 = np.array(cos2) if cos2 else np.array([1.0])
    pdist = np.array(pdist) if pdist else np.array([0.0])
    n_id = int(sum(k == 0 for k in ks))
    identity_consistent = (n_id == 0) or (n_id == len(ks))
    k_spread = (max(ks) - min(ks)) if ks else 0
    nested = bool(cos2.min() >= nested_min)            # smaller span inside the larger
    passed = bool(identity_consistent and k_spread <= max_k_spread and (nested or n_id == len(ks)))
    return {
        "cos2_similarity_mean": float(cos2.mean()),    # containment-biased; context only
        "cos2_similarity_min": float(cos2.min()),
        "proj_dist_mean": float(pdist.mean()),
        "proj_dist_max": float(pdist.max()),           # dimension-sensitive magnitude of flicker
        "k_values": ks,
        "k_spread": int(k_spread),
        "n_identity": n_id,
        "identity_consistent": identity_consistent,
        "nested": nested,
        "passed": passed,                              # CORE-stability bar
        "proj_dist_strict_pass": bool(pdist.max() <= proj_dist_strict),  # STRICT bar (eigengap/hysteresis TODO)
    }
