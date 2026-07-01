"""Graph-DualCMI scaffolding — Part 1 tests (CPU only, no GPU/training).

Covers the two foundational grounding fixes from docs/CIGL_46:
  (a) the static DGCNN graph adapter is reachable via build_backbone('DGCNNGraph') and exposes
      forward_graph(x) -> (logits, graph_z, node_z, edge_logits=None);
  (c) NodePosterior conditions on a per-node embedding e_v  ->  q(D | Z_v, e_v, Y)  (manuscript formula),
      with a byte-compatible fallback to q(D | Z_v, Y) when n_chans is not supplied.
"""
import math

import numpy as np
import pytest
import torch

from cmi.models.backbones import build_backbone
from cmi.methods.graph_regularizers import NodePosterior
from cmi.train.trainer import train_model, ALL_METHODS


def _synth(n_per_cell=8, C=8, T=64, n_cls=2, n_dom=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            X.append(rng.standard_normal((n_per_cell, C, T)).astype("float32"))
            y += [yi] * n_per_cell
            d += [di] * n_per_cell
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def _priors(n_cls, n_dom):
    # (pi_y=p(D|Y), p_d=p(D), p_dy=p(D,Y)) — matches empirical_priors() output shape
    return (torch.full((n_cls, n_dom), 1.0 / n_dom),
            torch.full((n_dom,), 1.0 / n_dom),
            torch.full((n_cls, n_dom), 1.0 / (n_cls * n_dom)))


def test_dgcnn_graph_backbone_registered_and_static():
    B, C, T, n_cls = 4, 8, 128, 2
    bb = build_backbone("DGCNNGraph", C, T, n_cls, device="cpu")
    assert callable(getattr(bb, "forward_graph", None)), "DGCNNGraph must expose forward_graph"
    x = torch.randn(B, C, T)
    logits, gz, nz, el = bb.forward_graph(x)
    assert logits.shape == (B, n_cls)
    assert gz.shape[0] == B and gz.dim() == 2                      # graph readout [B, z_dim]
    assert nz.shape[0] == B and nz.shape[1] == C and nz.dim() == 3  # node features [B, C, node_z_dim]
    assert el is None, "static-adjacency adapter must have edge_logits=None"
    assert hasattr(bb, "z_dim") and hasattr(bb, "node_z_dim")
    # plain DGCNN must NOT be rerouted (it has no forward_graph)
    assert not callable(getattr(build_backbone("DGCNN", C, T, n_cls), "forward_graph", None))


def test_nodeposterior_conditions_on_node_id():
    B, C, d, n_cls, n_dom = 3, 6, 5, 2, 4
    post = NodePosterior(d, n_dom, n_cls, _priors(n_cls, n_dom), n_chans=C)
    assert post.use_node_id and hasattr(post, "node_emb")
    y = torch.zeros(B, dtype=torch.long)
    # identical node features across ALL channels: only the node embedding e_v can differ per channel.
    nz = torch.ones(B, C, d)
    lg = post._logits(nz, y)                                   # [B, C, n_dom]
    assert lg.shape == (B, C, n_dom)
    per_node = lg[0]                                           # [C, n_dom]
    # at least one pair of channels must differ -> the head genuinely uses node identity
    assert not torch.allclose(per_node[0], per_node[1], atol=1e-5), "node-id conditioning had no effect"
    assert torch.isfinite(post.reg(nz, y)) and post.reg(nz, y) >= 0


def test_nodeposterior_backward_compatible_without_n_chans():
    B, C, d, n_cls, n_dom = 3, 6, 5, 2, 4
    post = NodePosterior(d, n_dom, n_cls, _priors(n_cls, n_dom))   # no n_chans -> q(D|Z_v,Y)
    assert not post.use_node_id and not hasattr(post, "node_emb")
    y = torch.zeros(B, dtype=torch.long)
    nz = torch.ones(B, C, d)                                    # identical feats -> identical per-node logits
    per_node = post._logits(nz, y)[0]
    assert torch.allclose(per_node[0], per_node[1], atol=1e-6), "id-agnostic head must be node-invariant"
    # body input dim is d + n_cls (no embedding)
    assert post.body[0].in_features == d + n_cls


# --- Part 2: graphdualpc method (encoder graph/node/edge I(Z;D|Y) [GLS] + decoder residual I(Y;D|Z)) ---

def test_graphdualpc_registered():
    assert "graphdualpc" in ALL_METHODS
    assert {"graphcmi", "dualpc"} <= ALL_METHODS   # existing methods still present


def test_graphdualpc_cpu_tiny_run():
    X, y, d = _synth()
    C, T, n_cls = X.shape[1], X.shape[2], 2
    bb = build_backbone("DGCNNGraph", C, T, n_cls, device="cpu")
    # lambda_g=lam=0.01, lambda_node=beta=0.01, lambda_edge=lam_edge=0 (static->must be 0), gamma_dec=gamma=0.1
    bb, post, out = train_model(bb, X, y, d, n_cls, method="graphdualpc",
                                lam=0.01, beta=0.01, lam_edge=0.0, gamma=0.1,
                                epochs=2, bs=16, n_inner=1, warmup=1, device="cpu", seed=0)
    for k in ("lambda_g", "lambda_node", "lambda_edge", "gamma_dec",
              "reg_graph_gls", "reg_node_gls", "dec_js_res", "dec_ce_res", "loss_ce"):
        assert k in out, f"missing diagnostic {k}"
        assert math.isfinite(out[k]), f"{k} not finite: {out[k]}"
    assert out["lambda_g"] == 0.01 and out["lambda_node"] == 0.01 and out["gamma_dec"] == 0.1
    assert out["dec_js_res"] >= 0.0                      # JS residual is non-negative


def test_graphdualpc_edge_fail_closed_on_static_adjacency():
    X, y, d = _synth()
    C, T, n_cls = X.shape[1], X.shape[2], 2
    bb = build_backbone("DGCNNGraph", C, T, n_cls, device="cpu")   # static adjacency -> edge_logits=None
    with pytest.raises(ValueError, match="edge_logits"):
        train_model(bb, X, y, d, n_cls, method="graphdualpc",
                    lam=0.01, beta=0.01, lam_edge=0.01, gamma=0.1,   # lambda_edge>0 must fail closed
                    epochs=1, bs=16, n_inner=1, warmup=1, device="cpu")


def test_graphcmi_still_runs_after_refactor():
    # regression: the uses_graph refactor must not break the existing graphcmi method
    X, y, d = _synth()
    C, T, n_cls = X.shape[1], X.shape[2], 2
    bb = build_backbone("DGCNNGraph", C, T, n_cls, device="cpu")
    bb, post, out = train_model(bb, X, y, d, n_cls, method="graphcmi",
                                lam=0.01, gamma=0.01, lam_edge=0.0,
                                epochs=2, bs=16, n_inner=1, warmup=1, device="cpu")
    assert math.isfinite(out["reg_graph"]) and out["lambda_g"] == 0.01


def test_graphdualpc_requires_forward_graph_backbone():
    # non-graph backbone must fail closed with a clear message
    from cmi.models.backbones import HookedBackbone  # noqa: F401 (import path check only)

    class _Plain(torch.nn.Module):
        z_dim = 8
        def __init__(self): super().__init__(); self.head = torch.nn.Linear(8, 2); self.enc = torch.nn.Linear(64, 8)
        def forward(self, x): z = self.enc(x.flatten(1)); return self.head(z), z

    X, y, d = _synth(C=8, T=8)
    with pytest.raises(ValueError, match="forward_graph"):
        train_model(_Plain(), X, y, d, 2, method="graphdualpc", epochs=1, device="cpu")
