"""Graph-DualCMI scaffolding — Part 1 tests (CPU only, no GPU/training).

Covers the two foundational grounding fixes from docs/CIGL_46:
  (a) the static DGCNN graph adapter is reachable via build_backbone('DGCNNGraph') and exposes
      forward_graph(x) -> (logits, graph_z, node_z, edge_logits=None);
  (c) NodePosterior conditions on a per-node embedding e_v  ->  q(D | Z_v, e_v, Y)  (manuscript formula),
      with a byte-compatible fallback to q(D | Z_v, Y) when n_chans is not supplied.
"""
import torch

from cmi.models.backbones import build_backbone
from cmi.methods.graph_regularizers import NodePosterior


def _priors(n_cls, n_dom):
    return (torch.full((n_cls, n_dom), 1.0 / n_dom),)


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
