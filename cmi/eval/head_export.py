"""CIGL R2.5 — export the task classifier's linear head + verified head-replay artifacts into the .audit.npz
sidecar, so R3 can make a CLASSIFIER-reliance claim (head-replay), not only a representation-reliance claim
(source-fit probe). Torch lives here; cmi/eval/audit_npz.py stays numpy-only (its load path must not import torch).

The graph-task backbones classify with a single nn.Linear over graph_z (DGCNN adapter -> net.head; graph-stem
nets -> gh.head), so logits = graph_z @ Wᵀ + b EXACTLY. We extract (W, b) and let save_audit_npz verify the
reconstruction FAIL-CLOSED against the saved model_logits (tol 1e-5); a nonlinear/unknown head -> replay_ok=False
and R3 falls back to the probe. We do NOT fabricate a replayable head.

FIREWALL: features/logits/head are all read in eval() with no grad; target trials are captured EVAL-ONLY (their
labels are stored for a final reported metric, never used to fit the subspace/probe). save_fold_audit records
source/target/source_val indices so the source-only fit can be proven from the file.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from cmi.eval.audit_npz import save_audit_npz


def extract_task_head(model):
    """Return (W [n_cls,Zin], b [n_cls], kind, input_repr) for a graph-task backbone's task classifier.
    Supported == a single nn.Linear consuming graph_z -> kind='linear'. Anything else -> kind='unsupported'
    with W=b=None so the caller fails closed (replay_ok=False)."""
    head = None
    if hasattr(model, "net") and isinstance(getattr(model.net, "head", None), nn.Linear):
        head = model.net.head                                # DGCNNForwardGraphAdapter
    elif hasattr(model, "gh") and isinstance(getattr(model.gh, "head", None), nn.Linear):
        head = model.gh.head                                 # Shallow/EEGNet graph-stem nets
    elif isinstance(getattr(model, "head", None), nn.Linear):
        head = model.head
    if head is None:
        return None, None, "unsupported", "graph_z"
    W = head.weight.detach().cpu().numpy().astype("float64")
    b = (head.bias.detach().cpu().numpy().astype("float64") if head.bias is not None
         else np.zeros(W.shape[0], dtype="float64"))
    return W, b, "linear", "graph_z"


@torch.no_grad()
def forward_graph_capture(model, X, device, bs=256):
    """Eval-mode forward_graph over X -> (model_logits, graph_z, node_z) as numpy. Unlike _forward_graph_feats,
    this KEEPS the logits (needed to verify head-replay)."""
    model.eval()
    L, G, N = [], [], []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
        lo, g, n, _ = model.forward_graph(xb)
        L.append(lo.cpu().numpy()); G.append(g.cpu().numpy()); N.append(n.cpu().numpy())
    return np.concatenate(L), np.concatenate(G), np.concatenate(N)


def save_fold_audit(path, *, model, X_source, y_source, d_source, device, fold, seed, target_subject,
                    method="", dataset="", X_target=None, y_target=None, target_domain=None,
                    source_indices=None, target_indices=None, source_val_indices=None,
                    node_leakage_map=None, probe_logits=None, probe_predictions=None,
                    task_head_replay_tol=1e-5, capture_bs=256):
    """Per-fold R3 export hook. Captures SOURCE (multi-subject) features + logits, optionally appends the LOSO
    TARGET subject's features EVAL-ONLY (tagged with a distinct `target_domain` id so R3 can hold it out), extracts
    the linear task head, and writes a verified .audit.npz. Returns (path, replay_ok, max_abs_diff).

    Firewall: X_target is forwarded in eval() only; y_target is stored for a final reported target metric and is
    NEVER used to fit anything. `target_domain` must differ from every value in d_source."""
    logits_s, gz_s, nz_s = forward_graph_capture(model, X_source, device, capture_bs)
    y = np.asarray(y_source).astype("int64"); d = np.asarray(d_source).astype("int64")
    gz, nz, logits = gz_s, nz_s, logits_s
    if X_target is not None and len(X_target) > 0:
        if target_domain is None or target_domain in np.unique(d):
            raise ValueError(f"target_domain={target_domain} must be a distinct id not present in d_source")
        logits_t, gz_t, nz_t = forward_graph_capture(model, X_target, device, capture_bs)
        gz = np.concatenate([gz_s, gz_t]); nz = np.concatenate([nz_s, nz_t])
        logits = np.concatenate([logits_s, logits_t])
        y = np.concatenate([y, np.asarray(y_target).astype("int64")])
        d = np.concatenate([d, np.full(len(X_target), int(target_domain), dtype="int64")])
    W, b, kind, input_repr = extract_task_head(model)
    out = save_audit_npz(path, graph_z=gz, node_z=nz, y=y, d=d, model_logits=logits,
                         fold=fold, seed=seed, target_subject=target_subject, method=method, dataset=dataset,
                         node_leakage_map=node_leakage_map, probe_logits=probe_logits,
                         probe_predictions=probe_predictions,
                         task_head_weight=W, task_head_bias=b, task_head_kind=kind, task_head_input=input_repr,
                         task_head_replay_tol=task_head_replay_tol,
                         source_indices=source_indices, target_indices=target_indices,
                         source_val_indices=source_val_indices)
    # re-derive the verification numbers for the caller/log (fail-closed if head unsupported)
    if kind == "linear":
        diff = np.abs(logits - (gz @ W.T + b))
        return out, bool(diff.max() <= task_head_replay_tol), float(diff.max())
    return out, False, float("nan")
