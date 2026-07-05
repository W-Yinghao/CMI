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
# R3 optional task-head export -> enables HEAD-REPLAY (classifier-reliance) counterfactuals. If absent, R3
# falls back to a source-fit task probe (representation-reliance). Optional for the scaffold; required for the
# first real-EEG R3 gate. task_head_replay_ok is FAIL-CLOSED: only true when the exported linear head
# reconstructs the saved model_logits within tolerance (dropout/BN/nonlinear/fusion heads -> false).
TASK_HEAD_KEYS = ("task_head_weight", "task_head_bias", "task_head_kind", "task_head_input",
                  "task_head_replay_ok", "task_head_replay_max_abs_diff", "task_head_replay_mean_abs_diff")
# R3 firewall verification: which trials fed the fit vs eval. Lets a checker PROVE source-only fitting.
INDEX_KEYS = ("source_indices", "target_indices", "source_val_indices")
DEFAULT_REPLAY_TOL = 1e-5


def pack_task_head_fields(task_head_weight, task_head_bias, model_logits, head_input,
                          kind="linear", input_repr="graph_z", tol=DEFAULT_REPLAY_TOL):
    """Verify a linear task head reconstructs model_logits and pack the task_head_* fields. FAIL-CLOSED:
    task_head_replay_ok is True ONLY when kind=='linear' AND max|model_logits - (head_input @ Wᵀ + b)| <= tol.
    A nonlinear/unknown head, a missing weight, or missing logits/features to check against -> replay_ok=False
    (never a fabricated head-replay). Returns {} when no head weight is provided. Pure numpy."""
    if task_head_weight is None:
        return {}
    W = np.asarray(task_head_weight, dtype=float)
    b = np.zeros(W.shape[0]) if task_head_bias is None else np.asarray(task_head_bias, dtype=float)
    fields = {"task_head_weight": W, "task_head_bias": b,
              "task_head_kind": str(kind), "task_head_input": str(input_repr)}
    if kind == "linear" and model_logits is not None and head_input is not None:
        replay = np.asarray(head_input, dtype=float) @ W.T + b
        diff = np.abs(np.asarray(model_logits, dtype=float) - replay)
        mx, mn = float(diff.max()), float(diff.mean())
        fields.update(task_head_replay_ok=bool(mx <= tol),
                      task_head_replay_max_abs_diff=mx, task_head_replay_mean_abs_diff=mn)
    else:                                                    # unsupported head or nothing to verify -> fail closed
        fields.update(task_head_replay_ok=False,
                      task_head_replay_max_abs_diff=float("nan"), task_head_replay_mean_abs_diff=float("nan"))
    return fields


def has_task_head(data):
    """True iff the sidecar carries a task head over a named representation (fields present)."""
    return "task_head_weight" in data and "task_head_input" in data


def head_replay_ok(data):
    """True iff a VERIFIED replayable linear head is present (has_task_head AND fail-closed replay passed).
    R3 uses head-replay only when this is True; otherwise it falls back to the source-fit probe."""
    return has_task_head(data) and bool(data.get("task_head_replay_ok", False))


def replay_head(data, z):
    """Recompute task logits from a modified representation z using the exported linear head:
    logits = z @ W^T + b. Only valid when has_task_head(data) and z matches task_head_input's dim."""
    import numpy as _np
    W = _np.asarray(data["task_head_weight"]); b = _np.asarray(data.get("task_head_bias", 0.0))
    return z @ W.T + b


def save_audit_npz(path, *, graph_z, node_z, y, d, model_logits, fold, seed, target_subject,
                   method="", dataset="", probe_logits=None, probe_predictions=None,
                   node_leakage_map=None, task_saliency_map=None,
                   task_head_weight=None, task_head_bias=None, task_head_kind="linear",
                   task_head_input="graph_z", task_head_replay_tol=DEFAULT_REPLAY_TOL,
                   source_indices=None, target_indices=None, source_val_indices=None):
    """Save a per-fold audit sidecar. graph_z [N,Zg], node_z [N,C,Zn], y/d [N], model_logits [N,n_cls];
    probe_logits [N,n_dom] / probe_predictions [N] from the domain probe; node_leakage_map [C] per-node KL.
    OPTIONAL task head (task_head_weight [n_cls,Zin], task_head_bias [n_cls], task_head_input name) enables
    R3 head-replay; it is verified FAIL-CLOSED here against model_logits (task_head_replay_ok/_max/_mean written
    only when a head is exported). source/target/source_val_indices let R3's firewall be proven from the file."""
    a = lambda x: None if x is None else np.asarray(x)
    payload = dict(graph_z=a(graph_z), node_z=a(node_z), y=a(y).astype("int64"), d=a(d).astype("int64"),
                   model_logits=a(model_logits), fold=np.asarray(fold), seed=np.asarray(seed),
                   target_subject=np.asarray(str(target_subject)), method=np.asarray(str(method)),
                   dataset=np.asarray(str(dataset)))
    for k, v in (("probe_logits", probe_logits), ("probe_predictions", probe_predictions),
                 ("node_leakage_map", node_leakage_map), ("task_saliency_map", task_saliency_map)):
        if v is not None:
            payload[k] = np.asarray(v)
    for k, v in (("source_indices", source_indices), ("target_indices", target_indices),
                 ("source_val_indices", source_val_indices)):
        if v is not None:
            payload[k] = np.asarray(v).astype("int64")
    # task head: verify replay fail-closed against the head's declared input (graph_z here) + model_logits
    head_in = graph_z if task_head_input == "graph_z" else None
    for k, v in pack_task_head_fields(task_head_weight, task_head_bias, model_logits, head_in,
                                      kind=task_head_kind, input_repr=task_head_input,
                                      tol=task_head_replay_tol).items():
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
    for k in ("fold", "seed", "target_subject", "method", "dataset", "task_head_kind", "task_head_input",
              "task_head_replay_ok", "task_head_replay_max_abs_diff", "task_head_replay_mean_abs_diff"):
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
