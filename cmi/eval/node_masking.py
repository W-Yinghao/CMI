"""CIGL R3 (secondary) — node masking. Zero the top-k / bottom-k / random-k leakage nodes (by the per-node
leakage map) in the frozen node features and measure the task drop. SUPPORTIVE evidence only — the flagship
reliance result is the source-fit label-conditional subspace removal (leakage_removal.py). Artifact-driven
(consumes node_z + node_leakage_map from .audit.npz); a source-fit task probe over flattened node_z evaluates
the drop (no backbone retrain). FIREWALL: probe fit on source, target eval-only, random baseline deterministic.
"""
from __future__ import annotations
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score


def _mask_task_bacc(node_z, y, src, tgt, mask_nodes):
    """Zero the given nodes in node_z [N,C,Zn], fit a task probe on flattened SOURCE, eval on target."""
    nz = np.array(node_z, dtype=float)
    if len(mask_nodes):
        nz[:, list(mask_nodes), :] = 0.0
    F = nz.reshape(nz.shape[0], -1)
    if np.allclose(F[src].std(0), 0):                          # every node masked -> constant -> chance
        return 1.0 / max(1, len(np.unique(y[src])))
    clf = LDA().fit(F[src], y[src])
    ev = tgt if tgt.sum() else src
    return float(balanced_accuracy_score(y[ev], clf.predict(F[ev])))


def node_masking_curve(data, target_domain, mask_k=3, n_random=20, seed=0):
    """top-k / bottom-k / random-k leakage-node masking drops. `data` = .audit.npz dict with node_z +
    node_leakage_map [C]; target_domain = held-out subject's d value. Returns drops + random CI."""
    node_z = np.asarray(data["node_z"], dtype=float)
    y = np.asarray(data["y"]); d = np.asarray(data["d"])
    C = node_z.shape[1]
    lm = np.asarray(data["node_leakage_map"], dtype=float)
    src = d != target_domain; tgt = d == target_domain
    k = min(mask_k, C)
    order = np.argsort(lm)                                      # ascending leakage
    base = _mask_task_bacc(node_z, y, src, tgt, [])
    top = _mask_task_bacc(node_z, y, src, tgt, order[-k:])      # highest-leakage nodes
    bot = _mask_task_bacc(node_z, y, src, tgt, order[:k])       # lowest-leakage nodes
    rng = np.random.default_rng(seed)
    rand = [_mask_task_bacc(node_z, y, src, tgt, rng.choice(C, k, replace=False)) for _ in range(n_random)]
    rand = np.asarray(rand)
    return {"mask_k": k, "base_task_bacc": base,
            "top_leak_mask_drop": float(base - top), "bottom_leak_mask_drop": float(base - bot),
            "random_mask_drop_mean": float(base - rand.mean()),
            "random_mask_drop_ci": [float(np.percentile(base - rand, 2.5)),
                                    float(np.percentile(base - rand, 97.5))],
            # supportive iff removing the HIGH-leakage nodes hurts the task more than random removal
            "top_exceeds_random": bool((base - top) > np.percentile(base - rand, 97.5))}
