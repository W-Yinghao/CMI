#!/usr/bin/env python
"""CIGL Phase-1 CPU smoke test: one tiny end-to-end GraphCMI training run on synthetic tensors.

Builds GraphCMINet directly (NO braindecode / NO moabb / NO PyG), runs a few epochs of
method="graphcmi" (graph + node + edge CMI all on), then checks every output tensor is finite
and correctly shaped and that the trainer reports the lambda_g / lambda_node / lambda_edge
diagnostics. Also asserts the fail-closed guard fires for a non-graph backbone.

Exits 0 on success; exits 1 (with a FAIL line) on any NaN/Inf, shape mismatch, or missing
diagnostic. Small enough for a laptop CPU.

    python scripts/smoke_graphcmi.py --device cpu

See docs/CIGL_03_IMPLEMENTATION_PLAN.md Phase 1 and docs/CIGL_05_ACCEPTANCE_CRITERIA.md Gate 1.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import cmi.*
from cmi.models.gnn import GraphCMINet
from cmi.train.trainer import train_model, predict


class _NonGraphBackbone(nn.Module):
    """Minimal (logits, Z) backbone with NO forward_graph -> must trip the graphcmi guard."""
    def __init__(self, n_chans, n_times, n_classes, z_dim=8):
        super().__init__()
        self.lin = nn.Linear(n_chans * n_times, z_dim)
        self.head = nn.Linear(z_dim, n_classes)
        self.z_dim = z_dim

    def forward(self, x):
        z = torch.relu(self.lin(x.flatten(1)))
        return self.head(z), z


def _synthetic(n=48, n_chans=22, n_times=128, n_cls=3, n_dom=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_chans, n_times)).astype("float32")
    y = rng.integers(0, n_cls, n).astype("int64")
    d = rng.integers(0, n_dom, n).astype("int64")
    y[:n_cls] = np.arange(n_cls)          # every class present
    d[:n_dom] = np.arange(n_dom)          # every domain present
    return X, y, d, n_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("FAIL: --device cuda requested but CUDA unavailable"); return 1

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
            print(f"  [x] {msg}")
        else:
            print(f"  [ok] {msg}")

    torch.manual_seed(args.seed)
    X, y, d, n_cls = _synthetic(seed=args.seed)
    n_chans, n_times = X.shape[1], X.shape[2]
    print(f"[smoke] X={X.shape} n_cls={n_cls} n_dom={int(d.max())+1} device={device} epochs={args.epochs}")

    # --- 1. build backbone + check both contracts on a tiny batch ----------------------------------
    net = GraphCMINet(n_chans, n_times, n_cls).to(device)
    xb = torch.tensor(X[:4]).to(device)
    with torch.no_grad():
        f2 = net.forward(xb)
        f4 = net.forward_graph(xb)
    check(isinstance(f2, tuple) and len(f2) == 2, "forward returns 2-tuple (logits, graph_Z)")
    check(isinstance(f4, tuple) and len(f4) == 4, "forward_graph returns 4-tuple")
    logits, gz, nz, el = f4
    check(tuple(logits.shape) == (4, n_cls), f"logits shape {tuple(logits.shape)} == (4,{n_cls})")
    check(tuple(gz.shape) == (4, net.z_dim), f"graph_Z shape {tuple(gz.shape)} == (4,{net.z_dim})")
    check(tuple(nz.shape) == (4, n_chans, net.z_dim), f"node_Z shape {tuple(nz.shape)}")
    check(tuple(el.shape) == (4, n_chans, n_chans), f"edge_logits shape {tuple(el.shape)}")
    for name, t in [("logits", logits), ("graph_Z", gz), ("node_Z", nz), ("edge_logits", el)]:
        check(bool(torch.isfinite(t).all()), f"{name} finite (pre-train)")

    # --- 2. one tiny graphcmi training run (graph + node + edge CMI all active) --------------------
    net, _post, diag = train_model(
        net, X, y, d, n_cls, method="graphcmi",
        lam=0.3, gamma=0.3, lam_edge=0.1,          # lambda_g / lambda_node / lambda_edge
        epochs=args.epochs, bs=16, warmup=1, n_inner=2, device=device, seed=args.seed)
    print(f"[smoke] diag: lambda_g={diag.get('lambda_g')} lambda_node={diag.get('lambda_node')} "
          f"lambda_edge={diag.get('lambda_edge')}")
    print(f"[smoke] loss_ce={diag.get('loss_ce'):.4f} reg_graph={diag.get('reg_graph'):.4f} "
          f"reg_node={diag.get('reg_node'):.4f} reg_edge={diag.get('reg_edge'):.4f} "
          f"stepA_dom_acc={diag.get('stepA_dom_acc'):.3f}")

    for k in ("lambda_g", "lambda_node", "lambda_edge", "loss_ce", "reg_graph", "reg_node", "reg_edge"):
        check(k in diag, f"diagnostic '{k}' reported")
    check(diag.get("lambda_g") == 0.3 and diag.get("lambda_node") == 0.3 and diag.get("lambda_edge") == 0.1,
          "lambda_g/lambda_node/lambda_edge echo the requested weights")
    for k in ("loss_ce", "reg_graph", "reg_node", "reg_edge", "inloop_reg", "stepA_dom_acc"):
        v = diag.get(k, float("nan"))
        check(np.isfinite(v), f"diagnostic '{k}'={v} is finite")

    # --- 3. predict path + post-train output finiteness --------------------------------------------
    prob = predict(net, X[:8], device=device)
    check(tuple(prob.shape) == (8, n_cls), f"predict() prob shape {tuple(prob.shape)}")
    check(bool(np.isfinite(prob).all()), "predict() probabilities finite")
    check(np.allclose(prob.sum(1), 1.0, atol=1e-4), "predict() rows sum to 1")
    with torch.no_grad():
        _, gz2, nz2, el2 = net.forward_graph(torch.tensor(X[:8]).to(device))
    for name, t in [("graph_Z", gz2), ("node_Z", nz2), ("edge_logits", el2)]:
        check(bool(torch.isfinite(t).all()), f"{name} finite (post-train)")

    # --- 4. fail-closed guard: non-graph backbone + method='graphcmi' must raise --------------------
    try:
        train_model(_NonGraphBackbone(n_chans, n_times, n_cls).to(device),
                    X, y, d, n_cls, method="graphcmi", epochs=1, bs=16, warmup=1, device=device)
        check(False, "graphcmi guard raised for non-graph backbone")
    except ValueError as e:
        check("forward_graph" in str(e), "graphcmi guard raises clear ValueError for non-graph backbone")

    if failures:
        print(f"\nFAIL ({len(failures)} check(s)):")
        for m in failures:
            print(f"  - {m}")
        return 1
    print("\nPASS: GraphCMI CPU smoke test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
