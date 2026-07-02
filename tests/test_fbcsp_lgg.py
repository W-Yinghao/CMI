"""CIGL_49 P5 — FBCSPLGGGraph (FBLGG + FBCSP spatial branch + 3-way gated fusion) tests (CPU only).

Verifies the spatial-spectral branch added to fix the P4/CIGL_48 4-class CSP gap: distinct fused_z
5-tuple, a finite spatial_z, a zero_spatial ablation, 3-way softmax gate instrumentation, central_strip_v1
grouping, graphdualpc head-split compatibility, and the source-only firewall.
"""
import math

import numpy as np
import pytest
import torch

from cmi.models.backbones import build_backbone
from cmi.models.fb_lgg_dualcmi import FBCSPLGGGraph, central_strip_groups
from cmi.run_loso import _DATASET_CH_NAMES
from cmi.train.trainer import train_model, predict


def _synth(n_per_cell=8, C=22, T=128, n_cls=2, n_dom=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            X.append(rng.standard_normal((n_per_cell, C, T)).astype("float32"))
            y += [yi] * n_per_cell
            d += [di] * n_per_cell
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def test_fbcsp_forward_graph_5tuple_and_spatial_z():          # (1)+(2)
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu")
    assert isinstance(bb, FBCSPLGGGraph)
    bb.eval()
    out = bb.forward_graph(torch.randn(4, 22, 128))
    assert len(out) == 5 and out[0].shape == (4, 2) and out[3] is None
    logits, gz, nz, el, fz = out
    assert fz is not gz and fz.shape == (4, bb.fused_z_dim)
    sz = bb.last_aux["spatial_z"]
    assert sz.shape == (4, bb.spatial_z_dim) and torch.isfinite(sz).all()   # spatial_z exists + finite
    assert torch.isfinite(fz).all()


def test_fbcsp_meta_declares_zero_spatial():                  # ablation contract
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu")
    assert set(bb.meta["ablation_modes"]) == {"zero_graph", "zero_temporal", "zero_spatial", "permute_nodes"}
    assert bb.meta.get("has_spatial_branch") is True and bb.meta.get("distinct_fused_z") is True


def test_fbcsp_all_ablations_change_logits_eval():            # (3)+(4)
    torch.manual_seed(1)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu"); bb.eval()
    x = torch.randn(16, 22, 128)
    full = bb.forward_graph(x)[0]
    for mode in ("zero_graph", "zero_temporal", "zero_spatial", "permute_nodes"):
        torch.manual_seed(5)
        ab = bb.ablate(x, mode)
        assert ab.shape == (16, 2), f"ablate({mode}) shape"
        assert not torch.allclose(full, ab, atol=1e-4), f"ablate({mode}) did not change logits"


def test_fbcsp_gate_summary_softmax():                        # gate instrumentation (P5-D)
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu")
    g = bb.gate_summary(torch.randn(8, 22, 128))
    for nm in ("graph", "temporal", "spatial"):
        assert f"gate_{nm}_mean" in g and f"gate_{nm}_std" in g
        assert 0.0 <= g[f"gate_{nm}_mean"] <= 1.0
    # softmax over 3 branches -> means sum ~1
    assert math.isclose(sum(g[f"gate_{nm}_mean"] for nm in ("graph", "temporal", "spatial")), 1.0, abs_tol=1e-4)


@pytest.mark.parametrize("ds,C,ng", [("BNCI2014_001", 22, 9), ("BNCI2015_001", 13, 5)])
def test_fbcsp_central_strip_presets(ds, C, ng):              # (5)+(6)
    ch = _DATASET_CH_NAMES[ds]
    idx, named, warn = central_strip_groups(ds, ch)
    assert warn is None and len(idx) == ng
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", C, 128, 2, device="cpu",
                        ch_names=ch, groups=idx, group_names=named, grouping_scheme="central_strip_v1")
    assert bb.grouping_scheme == "central_strip_v1" and bb.n_groups == ng
    bb.eval()
    assert len(bb.forward_graph(torch.randn(3, C, 128))) == 5


def test_fbcsp_graphdualpc_head_split_runs():                 # (7)
    X, y, d = _synth()
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu")
    bb, post, out = train_model(bb, X, y, d, 2, method="graphdualpc", lam=0.01, beta=0.01, lam_edge=0.0,
                                gamma=0.1, dec_scale=300.0, epochs=2, bs=16, n_inner=1, warmup=1,
                                device="cpu", seed=0)
    assert math.isfinite(out["dec_js_res"]) and math.isfinite(out["reg_graph_gls"])


def test_fbcsp_erm_tiny_run_and_diagnostics():                # (9) train + branch ablation + gate diagnostics
    X, y, d = _synth()
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu")
    bb, post, out = train_model(bb, X, y, d, 2, method="erm", lam=0.0,
                                epochs=2, bs=16, n_inner=1, warmup=1, device="cpu", seed=0)
    assert math.isfinite(out["inloop_reg"])
    bb.eval()
    with torch.no_grad():
        for mode in bb.meta["ablation_modes"]:
            assert bb.ablate(torch.from_numpy(X[:16]), mode).shape == (16, 2)
    g = bb.gate_summary(torch.from_numpy(X[:16]))
    assert all(math.isfinite(v) for v in g.values())


def test_fbcsp_firewall_determinism_with_early_stop():        # (8)
    X, y, d = _synth(seed=2)
    Xte = np.random.default_rng(7).standard_normal((16, 22, 128)).astype("float32")
    preds = []
    for _ in range(2):
        torch.manual_seed(0); np.random.seed(0)
        bb = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu")
        bb, _, out = train_model(bb, X, y, d, 2, method="graphdualpc", lam=0.01, beta=0.01, lam_edge=0.0,
                                 gamma=0.1, dec_scale=300.0, epochs=3, bs=16, n_inner=1, warmup=1,
                                 device="cpu", seed=0, early_stop=True, source_val_domains=[2])
        assert out["source_val_subjects"] == [2]
        preds.append(predict(bb, Xte, "cpu"))
    assert np.allclose(preds[0], preds[1]), "FBCSPLGG + early stopping not deterministic in (source, seed)"
