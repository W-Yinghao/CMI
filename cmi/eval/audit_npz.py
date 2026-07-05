"""CIGL R2->R3 — per-fold .audit.npz sidecar. Today only means are saved; R3 (reliance / subspace-removal /
node-masking / topomaps) needs the frozen features + probe outputs + leakage map per fold. This defines that
schema so every R2 run can export it and R3 can consume it. Pure numpy/npz — no torch dependency to load.
"""
from __future__ import annotations
import numpy as np

# arrays (per fold): [N, .] over the audit-eval trials
ARRAY_KEYS = ("graph_z", "node_z", "y", "d", "model_logits", "probe_logits", "probe_predictions")
# scalars / small metadata
META_KEYS = ("fold", "seed", "target_subject", "method", "dataset")
# optional analysis products
OPTIONAL_KEYS = ("node_leakage_map", "task_saliency_map")


def save_audit_npz(path, *, graph_z, node_z, y, d, model_logits, fold, seed, target_subject,
                   method="", dataset="", probe_logits=None, probe_predictions=None,
                   node_leakage_map=None, task_saliency_map=None):
    """Save a per-fold audit sidecar. graph_z [N,Zg], node_z [N,C,Zn], y/d [N], model_logits [N,n_cls];
    probe_logits [N,n_dom] / probe_predictions [N] from the domain probe; node_leakage_map [C] per-node KL."""
    a = lambda x: None if x is None else np.asarray(x)
    payload = dict(graph_z=a(graph_z), node_z=a(node_z), y=a(y).astype("int64"), d=a(d).astype("int64"),
                   model_logits=a(model_logits), fold=np.asarray(fold), seed=np.asarray(seed),
                   target_subject=np.asarray(str(target_subject)), method=np.asarray(str(method)),
                   dataset=np.asarray(str(dataset)))
    for k, v in (("probe_logits", probe_logits), ("probe_predictions", probe_predictions),
                 ("node_leakage_map", node_leakage_map), ("task_saliency_map", task_saliency_map)):
        if v is not None:
            payload[k] = np.asarray(v)
    p = str(path)
    if not p.endswith(".audit.npz"):
        p = p + (".audit.npz" if not p.endswith(".npz") else "")
    np.savez_compressed(p, **payload)
    return p


def load_audit_npz(path):
    """Load a sidecar back to a dict (arrays + unwrapped scalar metadata)."""
    z = np.load(str(path), allow_pickle=False)
    out = {k: z[k] for k in z.files}
    for k in ("fold", "seed", "target_subject", "method", "dataset"):
        if k in out and out[k].ndim == 0:
            out[k] = out[k].item()
    return out


def validate_audit_npz(data):
    """Required arrays present + N consistent across per-trial arrays. Returns [] if valid else list of issues."""
    issues = []
    for k in ("graph_z", "node_z", "y", "d", "model_logits"):
        if k not in data:
            issues.append(f"missing required key {k}")
    for k in ("fold", "seed", "target_subject"):
        if k not in data:
            issues.append(f"missing meta key {k}")
    if not issues:
        n = len(data["y"])
        for k in ("graph_z", "node_z", "d", "model_logits"):
            if data[k].shape[0] != n:
                issues.append(f"{k} first dim {data[k].shape[0]} != N={n}")
    return issues
