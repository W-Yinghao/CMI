"""CIGL_47 P3-B — FB-LGG-DualCMI backbone tests (CPU only, no GPU/training beyond a tiny ERM run).

Verifies the new FBLGGGraph backbone:
  * registered via build_backbone('FBLGGGraph');
  * forward_graph(x) -> 5-tuple (logits, graph_z, node_z, edge_logits=None, fused_z) with a DISTINCT
    fused_z (the classifier input) — the property that lets graphdualpc run a real encoder/decoder split;
  * shapes correct for the real channel counts (22 = BNCI2014_001, 13 = BNCI2015_001) and a tiny case;
  * ablate(zero_graph / zero_temporal / permute_nodes) each change the logits (both branches contribute,
    no hidden single-branch bypass);
  * channel-group builder is name-aware with a crash-proof index-partition fallback;
  * a tiny ERM run trains through the whole pipeline and stays finite.
"""
import math

import numpy as np
import pytest
import torch

from cmi.models.backbones import build_backbone
from cmi.models.fb_lgg_dualcmi import FBLGGDualCMIBackbone, build_channel_groups


def _synth(n_per_cell=8, C=8, T=128, n_cls=2, n_dom=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            X.append(rng.standard_normal((n_per_cell, C, T)).astype("float32"))
            y += [yi] * n_per_cell
            d += [di] * n_per_cell
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


@pytest.mark.parametrize("C", [8, 13, 22])
def test_fblgg_forward_graph_5tuple_and_shapes(C):
    B, T, n_cls = 4, 128, 2
    torch.manual_seed(0)
    bb = build_backbone("FBLGGGraph", C, T, n_cls, device="cpu")
    bb.eval()                                          # dropout OFF: forward vs forward_graph must agree
    assert isinstance(bb, FBLGGDualCMIBackbone)
    assert callable(getattr(bb, "forward_graph", None))
    x = torch.randn(B, C, T)
    out = bb.forward_graph(x)
    assert len(out) == 5, "FBLGGGraph must return a 5-tuple (with fused_z)"
    logits, gz, nz, el, fz = out
    assert logits.shape == (B, n_cls)
    assert gz.shape == (B, bb.z_dim) and gz.dim() == 2
    assert nz.shape == (B, C, bb.node_z_dim) and nz.dim() == 3
    assert el is None, "v1 uses static shared adjacency -> edge_logits must be None"
    assert fz.shape == (B, bb.fused_z_dim) and fz.dim() == 2
    # fused_z is the classifier input and a DISTINCT object from graph_z
    assert fz is not gz and fz.data_ptr() != gz.data_ptr(), "fused_z must be distinct from graph_z"
    assert bb.meta.get("distinct_fused_z") is True
    # forward() returns (logits, fused_z) and matches forward_graph logits
    l2, z2 = bb(x)
    assert torch.allclose(l2, logits) and z2.shape == (B, bb.fused_z_dim)


def test_fblgg_ablations_each_change_logits():
    B, C, T, n_cls = 16, 22, 128, 2
    torch.manual_seed(1)
    bb = build_backbone("FBLGGGraph", C, T, n_cls, device="cpu")
    bb.eval()                                          # isolate the ablation effect from dropout noise
    x = torch.randn(B, C, T)
    full = bb.forward_graph(x)[0]
    for mode in ("zero_graph", "zero_temporal", "permute_nodes"):
        torch.manual_seed(7)                       # fix permute_nodes randomness
        ab = bb.ablate(x, mode)
        assert ab.shape == (B, n_cls), f"ablate({mode}) must return [B,n_cls] logits"
        assert not torch.allclose(full, ab, atol=1e-4), f"ablate({mode}) did not change logits"


def test_fblgg_zero_graph_and_zero_temporal_differ():
    # the two branch-drops must not be identical (proves both branches feed the fusion distinctly)
    B, C, T, n_cls = 8, 13, 128, 2
    torch.manual_seed(2)
    bb = build_backbone("FBLGGGraph", C, T, n_cls, device="cpu")
    bb.eval()                                          # isolate branch-drop difference from dropout noise
    x = torch.randn(B, C, T)
    zg = bb.ablate(x, "zero_graph")
    zt = bb.ablate(x, "zero_temporal")
    assert not torch.allclose(zg, zt, atol=1e-4)


def test_channel_group_builder_name_aware_and_fallback():
    # name-aware: a 10-20 montage splits into >=2 functional regions, covering every channel exactly once
    names = ["Fp1", "Fz", "FC3", "C3", "Cz", "CP4", "P3", "Pz", "O1", "Oz", "T7", "T8"]
    groups = build_channel_groups(len(names), ch_names=names, max_groups=6)
    covered = sorted(c for g in groups for c in g)
    assert covered == list(range(len(names))) and len(groups) >= 2
    # unknown names -> index-partition fallback, never crashes
    weird = [f"X{i}" for i in range(10)]
    g2 = build_channel_groups(len(weird), ch_names=weird, max_groups=6)
    assert sorted(c for g in g2 for c in g) == list(range(10)) and 1 <= len(g2) <= 6
    # no names -> index partition into min(max_groups, C) groups
    g3 = build_channel_groups(22, ch_names=None, max_groups=6)
    assert len(g3) == 6 and sorted(c for g in g3 for c in g) == list(range(22))
    # tiny C with singleton groups must not crash
    g4 = build_channel_groups(3, ch_names=None, max_groups=6)
    assert sorted(c for g in g4 for c in g) == [0, 1, 2]


def test_fblgg_erm_tiny_run_finite():
    from cmi.train.trainer import train_model
    X, y, d = _synth(C=22, T=128)
    C, T, n_cls = X.shape[1], X.shape[2], 2
    torch.manual_seed(0)
    bb = build_backbone("FBLGGGraph", C, T, n_cls, device="cpu")
    bb, post, out = train_model(bb, X, y, d, n_cls, method="erm", lam=0.0,
                                epochs=2, bs=16, n_inner=1, warmup=1, device="cpu", seed=0)
    assert math.isfinite(out["inloop_reg"])          # training completed (ERM diagnostic present)
    bb.eval()
    with torch.no_grad():
        logits, fused_z = bb(torch.from_numpy(X[:16]))
    assert torch.isfinite(logits).all() and logits.shape == (16, n_cls)
    assert torch.isfinite(fused_z).all()             # whole pipeline (stem->graph->temporal->fuse) finite
