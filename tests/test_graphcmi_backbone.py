"""CIGL Phase-1 / Gate-1 tests: GraphCMINet backbone validity (CPU only, tiny synthetic tensors).

Verifies the two backbone contracts the CIGL trainer/diagnostics depend on:
  forward(x)        -> (logits, graph_Z)                          [2-tuple, harness-compatible]
  forward_graph(x)  -> (logits, graph_Z, node_Z, edge_logits)    [4-tuple, CIGL-only]
plus finiteness, one backward pass, edge_logits symmetry, a non-degenerate per-sample
adjacency, and that the shared-adjacency baselines (DGCNN/RGNN) still return (logits, Z).

No EEG data, no braindecode, no PyG. See docs/CIGL_05_ACCEPTANCE_CRITERIA.md Gate 1.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import cmi.*
from cmi.models.gnn import GraphCMINet, DGCNNBackbone, RGNNBackbone  # noqa: E402

CHANNEL_COUNTS = [19, 22, 32, 62]   # 10-20 clinical, BCI-IV-2a, biosemi-32, SEED-62
N_TIMES = 128
N_CLASSES = 4
BATCH = 6


def _rand_x(n_chans, n_times=N_TIMES, batch=BATCH):
    torch.manual_seed(0)
    return torch.randn(batch, n_chans, n_times)


@pytest.mark.parametrize("n_chans", CHANNEL_COUNTS)
def test_forward_contract_2tuple(n_chans):
    net = GraphCMINet(n_chans, N_TIMES, N_CLASSES).eval()
    x = _rand_x(n_chans)
    out = net.forward(x)
    assert isinstance(out, tuple) and len(out) == 2, "forward must return exactly (logits, graph_Z)"
    logits, graph_Z = out
    assert logits.shape == (BATCH, N_CLASSES)
    assert graph_Z.shape == (BATCH, net.z_dim)
    assert torch.isfinite(logits).all() and torch.isfinite(graph_Z).all()


@pytest.mark.parametrize("n_chans", CHANNEL_COUNTS)
def test_forward_graph_contract_4tuple(n_chans):
    net = GraphCMINet(n_chans, N_TIMES, N_CLASSES).eval()
    x = _rand_x(n_chans)
    out = net.forward_graph(x)
    assert isinstance(out, tuple) and len(out) == 4, \
        "forward_graph must return exactly (logits, graph_Z, node_Z, edge_logits)"
    logits, graph_Z, node_Z, edge_logits = out
    assert logits.shape == (BATCH, N_CLASSES)
    assert graph_Z.shape == (BATCH, net.z_dim)
    assert node_Z.shape == (BATCH, n_chans, net.z_dim)
    assert edge_logits.shape == (BATCH, n_chans, n_chans)
    for name, t in [("logits", logits), ("graph_Z", graph_Z),
                    ("node_Z", node_Z), ("edge_logits", edge_logits)]:
        assert torch.isfinite(t).all(), f"{name} has non-finite values"


@pytest.mark.parametrize("n_chans", CHANNEL_COUNTS)
def test_forward_and_forward_graph_agree(n_chans):
    """forward(x) must be the (logits, graph_Z) head of forward_graph(x)."""
    net = GraphCMINet(n_chans, N_TIMES, N_CLASSES).eval()
    x = _rand_x(n_chans)
    logits1, gz1 = net.forward(x)
    logits2, gz2, _, _ = net.forward_graph(x)
    assert torch.allclose(logits1, logits2, atol=1e-5)
    assert torch.allclose(gz1, gz2, atol=1e-5)


@pytest.mark.parametrize("n_chans", [22, 62])
def test_edge_logits_symmetric(n_chans):
    """edge_logits = (B+B^T) + g g^T is symmetric by construction (documented in Adjacency)."""
    net = GraphCMINet(n_chans, N_TIMES, N_CLASSES).eval()
    _, _, _, edge_logits = net.forward_graph(_rand_x(n_chans))
    assert torch.allclose(edge_logits, edge_logits.transpose(1, 2), atol=1e-5), \
        "edge_logits expected symmetric (pre-ReLU base+similarity)"


@pytest.mark.parametrize("n_chans", [22, 62])
def test_adjacency_non_degenerate(n_chans):
    """Gate 1: learned adjacency is data-dependent, not all-zero, not constant across samples."""
    net = GraphCMINet(n_chans, N_TIMES, N_CLASSES).eval()
    x = _rand_x(n_chans)
    _, _, _, edge_logits = net.forward_graph(x)
    assert torch.isfinite(edge_logits).all()
    assert edge_logits.abs().sum() > 0, "adjacency collapsed to all-zero"
    # per-sample similarity term must make the graph vary across distinct inputs
    spread = edge_logits.std(dim=0).max()
    assert spread > 1e-6, "edge_logits identical across samples -> not per-sample (data-dependent)"


@pytest.mark.parametrize("n_chans", CHANNEL_COUNTS)
def test_one_backward_pass(n_chans):
    net = GraphCMINet(n_chans, N_TIMES, N_CLASSES).train()
    x = _rand_x(n_chans)
    y = torch.randint(0, N_CLASSES, (BATCH,))
    logits, graph_Z, node_Z, edge_logits = net.forward_graph(x)
    # scalar loss that touches every CIGL output so gradient must reach node_Z and edge_logits paths
    loss = (torch.nn.functional.cross_entropy(logits, y)
            + 1e-3 * (node_Z.pow(2).mean() + edge_logits.pow(2).mean()))
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    # the per-sample adjacency projection must receive gradient (edge path is alive)
    assert net.adj.We.weight.grad is not None and torch.isfinite(net.adj.We.weight.grad).all()


@pytest.mark.parametrize("Backbone", [DGCNNBackbone, RGNNBackbone])
@pytest.mark.parametrize("n_chans", [22, 62])
def test_shared_adjacency_baselines_return_2tuple(Backbone, n_chans):
    """DGCNN/RGNN are the shared-adjacency baselines: they must still be plain (logits, Z) backbones."""
    net = Backbone(n_chans, N_TIMES, N_CLASSES).eval()
    out = net.forward(_rand_x(n_chans))
    assert isinstance(out, tuple) and len(out) == 2
    logits, z = out
    assert logits.shape == (BATCH, N_CLASSES)
    assert z.shape == (BATCH, net.z_dim)
    assert torch.isfinite(logits).all() and torch.isfinite(z).all()
    assert not hasattr(net, "forward_graph"), \
        "shared-adjacency baselines must NOT expose forward_graph (only GraphCMINet does)"
