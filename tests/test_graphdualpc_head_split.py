"""CIGL_47 P3-C — graphdualpc encoder/decoder head split (CPU only).

CIGL_46 refused a distinct fused_z (NotImplementedError). P3-C allocates a SEPARATE decoder posterior
(post_dec) on the fused_z dim when the backbone DECLARES meta['distinct_fused_z']; the encoder graph-CMI
head stays on graph_z. For DGCNNGraph (z_dec == graph_z) the heads still share (post_dec := post), so the
CIGL_46 path is byte-identical (covered by tests/test_graph_dualcmi_scaffold.py).

The decisive structural check: build FBLGG with fused_z_dim != z_dim and run graphdualpc. If post_dec were
still sized to z_dim, dec_js_residual(fused_z) would raise a shape error — a clean run proves post_dec is
independently sized to the fused_z dim.
"""
import math

import numpy as np
import pytest
import torch

from cmi.models.backbones import build_backbone
from cmi.models.fb_lgg_dualcmi import FBLGGDualCMIBackbone
from cmi.train.trainer import train_model


def _synth(n_per_cell=8, C=22, T=128, n_cls=2, n_dom=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            X.append(rng.standard_normal((n_per_cell, C, T)).astype("float32"))
            y += [yi] * n_per_cell
            d += [di] * n_per_cell
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def test_fblgg_graphdualpc_runs_with_distinct_fused_z():
    # CIGL_46 raised NotImplementedError here; P3-C must run and produce finite dual-CMI diagnostics.
    X, y, d = _synth()
    C, T, n_cls = X.shape[1], X.shape[2], 2
    torch.manual_seed(0)
    bb = build_backbone("FBLGGGraph", C, T, n_cls, device="cpu")
    bb, post, out = train_model(bb, X, y, d, n_cls, method="graphdualpc",
                                lam=0.01, beta=0.01, lam_edge=0.0, gamma=0.1,
                                epochs=2, bs=16, n_inner=1, warmup=1, device="cpu", seed=0)
    for k in ("reg_graph_gls", "reg_node_gls", "dec_js_res", "dec_ce_res", "loss_ce",
              "stepA_graph_dom_acc_gls", "stepA_node_dom_acc_gls"):
        assert k in out and math.isfinite(out[k]), f"{k} missing/non-finite: {out.get(k)}"
    assert out["dec_js_res"] >= 0.0                      # JS residual is non-negative


def test_graphdualpc_sizes_decoder_head_to_fused_z_dim():
    # z_dim != fused_z_dim: a shared/mis-sized post_dec would shape-error in dec_js_residual(fused_z).
    X, y, d = _synth()
    C, T, n_cls = X.shape[1], X.shape[2], 2
    torch.manual_seed(0)
    bb = FBLGGDualCMIBackbone(C, T, n_cls, z_dim=32, fused_z_dim=24)   # deliberately distinct dims
    assert bb.z_dim == 32 and bb.fused_z_dim == 24
    # sanity: forward_graph really returns graph_z[.,32] and fused_z[.,24]
    with torch.no_grad():
        _, gz, _, _, fz = bb.forward_graph(torch.randn(4, C, T))
    assert gz.shape[1] == 32 and fz.shape[1] == 24
    bb, post, out = train_model(bb, X, y, d, n_cls, method="graphdualpc",
                                lam=0.01, beta=0.01, lam_edge=0.0, gamma=0.1,
                                epochs=2, bs=16, n_inner=1, warmup=1, device="cpu", seed=0)
    assert math.isfinite(out["dec_js_res"]) and math.isfinite(out["reg_graph_gls"])


def test_undeclared_distinct_fused_z_still_fails_closed():
    # A backbone that returns a distinct fused_z but does NOT declare meta['distinct_fused_z'] must still
    # fail closed (post_dec would wrongly alias the encoder post) — the P3-C guard preserves fail-closed.
    class _UndeclaredFused(torch.nn.Module):
        z_dim, node_z_dim, fused_z_dim = 8, 16, 8
        meta = dict(graph_compatible=True)                # NOTE: no distinct_fused_z key
        def __init__(self):
            super().__init__()
            self.g = torch.nn.Linear(16, 8); self.head = torch.nn.Linear(8, 2); self.f = torch.nn.Linear(8, 8)
        def forward_graph(self, x):
            B, C, T = x.shape
            nz = x.mean(-1, keepdim=True).expand(B, C, 16).contiguous()
            gz = torch.relu(self.g(nz.mean(1)))
            return self.head(gz), gz, nz, None, self.f(gz)   # distinct fused_z, undeclared
        def forward(self, x):
            o = self.forward_graph(x); return o[0], o[1]

    X, y, d = _synth(C=8, T=64)
    with pytest.raises(NotImplementedError, match="fused_z"):
        train_model(_UndeclaredFused(), X, y, d, 2, method="graphdualpc",
                    lam=0.01, beta=0.01, lam_edge=0.0, gamma=0.1, epochs=1, bs=16, n_inner=1, warmup=1, device="cpu")


def test_fblgg_graphdualpc_deterministic_for_source_seed():
    # (source, seed) -> identical diagnostics (no hidden target-label / nondeterministic state)
    X, y, d = _synth()
    C, T, n_cls = X.shape[1], X.shape[2], 2
    outs = []
    for _ in range(2):
        torch.manual_seed(0)                              # seed BEFORE build_backbone (weight init)
        bb = build_backbone("FBLGGGraph", C, T, n_cls, device="cpu")
        _, _, out = train_model(bb, X, y, d, n_cls, method="graphdualpc",
                                lam=0.01, beta=0.01, lam_edge=0.0, gamma=0.1,
                                epochs=2, bs=16, n_inner=1, warmup=1, device="cpu", seed=0)
        outs.append(out["dec_js_res"])
    assert outs[0] == outs[1], "graphdualpc on FBLGG must be deterministic for a fixed (source, seed)"
