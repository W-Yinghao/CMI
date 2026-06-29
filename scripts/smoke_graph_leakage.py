#!/usr/bin/env python
"""CIGL Phase-2 CPU smoke: probe-only leakage audit on a synthetic DGP with KNOWN leakage.

Builds frozen graph_z / node_z / edge_logits with an injected label-conditional domain signal in
(i) the graph embedding, (ii) a subset of channels, (iii) one edge — with Y ⟂ D so a label-only
prior cannot explain it. Runs audit_graph_objects with a small RETRAINED within-label permutation
null and asserts every object's observed held-out KL clears its own permutation mean. Writes
results/cigl/smoke_graph_leakage.json (+ map .npy sidecars). Exits nonzero on any failure.

    python scripts/smoke_graph_leakage.py --device cpu

No target data, no regularization, no real EEG. See docs/CIGL_09_PHASE2_LEAKAGE_AUDIT.md.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import cmi.*
from cmi.eval.graph_leakage import audit_graph_objects

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "cigl"


def _synthetic(seed, n=160, C=6, Dg=6, Dn=4, n_cls=2, n_dom=3, strength=2.5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_cls, n).astype("int64")
    d = rng.integers(0, n_dom, n).astype("int64")          # Y ⟂ D  -> leakage is conditional
    y[:n_cls] = np.arange(n_cls); d[:n_dom] = np.arange(n_dom)
    graph_z = rng.standard_normal((n, Dg)).astype("float32")
    graph_z[:, 0] += strength * d                          # (i) graph leakage
    node_z = rng.standard_normal((n, C, Dn)).astype("float32")
    leak_chans = list(range(1, C // 2 + 1))                # (ii) a SUBSET of channels leak
    for c in leak_chans:
        node_z[:, c, 0] += strength * d
    base = rng.standard_normal((n, C, C)).astype("float32")
    edge = 0.5 * (base + base.transpose(0, 2, 1))
    edge[:, 0, 2] += strength * d; edge[:, 2, 0] = edge[:, 0, 2]   # (iii) one leaking edge
    for k in range(C):
        edge[:, k, k] = 0.0
    return graph_z, node_z, edge, y, d, n_cls, n_dom, leak_chans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_perm", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--margin", type=float, default=0.03, help="required observed-minus-null KL margin")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("FAIL: --device cuda requested but CUDA unavailable"); return 1

    gz, nz, el, y, d, n_cls, n_dom, leak_chans = _synthetic(args.seed)
    N, C = el.shape[0], el.shape[1]
    print(f"[smoke] N={N} C={C} n_cls={n_cls} n_dom={n_dom} leak_chans={leak_chans} "
          f"n_perm={args.n_perm} epochs={args.epochs} device={args.device}")

    audit = audit_graph_objects(gz, nz, el, y, d, n_cls, n_dom,
                                n_perm=args.n_perm, epochs=args.epochs, seed=args.seed, device=args.device)

    failures = []
    for obj in ("graph", "node", "edge"):
        b = audit[obj]
        margin = b["kl_mean"] - b["permutation_mean"]
        ok = margin > args.margin and np.isfinite(b["permutation_p"])
        print(f"  {obj:5s}: kl_mean={b['kl_mean']:.4f} perm_mean={b['permutation_mean']:.4f} "
              f"margin={margin:+.4f} p={b['permutation_p']:.3f} "
              f"dom_acc={b['domain_acc']:.3f} prior_acc={b['prior_acc']:.3f} "
              f"-> {'OK' if ok else 'CLEARED-NULL FAIL'}")
        if not ok:
            failures.append(f"{obj}: observed KL did not clear permutation null by >{args.margin}")

    # node map should peak on the injected channels; edge map on the (0,2) edge
    node_map = np.asarray(audit["node"]["node_leakage_map"])
    edge_map = np.asarray(audit["edge"]["edge_leakage_map"])
    top_chan = int(node_map.argmax())
    iu = np.triu_indices(C, 1)
    top_edge = (int(iu[0][edge_map[iu].argmax()]), int(iu[1][edge_map[iu].argmax()]))
    print(f"  node_leakage_map peak channel = {top_chan} (injected {leak_chans})")
    print(f"  edge_leakage_map peak edge    = {top_edge} (injected (0, 2))")
    if top_chan not in leak_chans:
        failures.append(f"node map peak channel {top_chan} not in injected {leak_chans}")
    if top_edge != (0, 2):
        failures.append(f"edge map peak {top_edge} != injected (0,2)")
    if not np.allclose(np.diag(edge_map), 0.0):
        failures.append("edge map diagonal not zero")

    # ---- persist artifacts ------------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    node_path = OUT_DIR / "smoke_node_leakage_map.npy"
    edge_path = OUT_DIR / "smoke_edge_leakage_matrix.npy"
    np.save(node_path, node_map); np.save(edge_path, edge_map)
    audit["node"]["node_leakage_map_path"] = str(node_path.relative_to(OUT_DIR.parent.parent))
    audit["edge"]["edge_leakage_map_path"] = str(edge_path.relative_to(OUT_DIR.parent.parent))
    audit["meta"] = dict(n_samples=N, n_channels=C, n_classes=n_cls, n_domains=n_dom,
                         seed=args.seed, n_perm=args.n_perm, epochs=args.epochs,
                         strict_source_only=True, used_target_labels=False, used_target_covariates=False,
                         injected_leak_channels=leak_chans, injected_leak_edge=[0, 2],
                         setting="strict_source_only_DG", note="synthetic Phase-2 smoke (diagnostic only)")
    out_path = OUT_DIR / "smoke_graph_leakage.json"
    json.dump(audit, open(out_path, "w"), indent=2)
    print(f"[smoke] wrote {out_path}")

    if failures:
        print(f"\nFAIL ({len(failures)}):")
        for m in failures:
            print(f"  - {m}")
        return 1
    print("\nPASS: graph leakage audit smoke OK (graph/node/edge leakage all clear the retrained null)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
