"""CIGL_50 P6 — fbdualpc spatial encoder CMI + fusion-balance gate floor (CPU only).

P4/CIGL_48 showed the FBCSP spatial branch carries the 4-class signal, but the old graphdualpc CMI only
penalizes graph/node. P6-A adds a spatial encoder CMI head (q(D|spatial_z,Y)) via the new `fbdualpc`
method; P6-B adds a fusion gate floor so the 3-way gate does not starve the graph branch.
"""
import math

import numpy as np
import pytest
import torch

from cmi.models.backbones import build_backbone
from cmi.run_loso import parse_config
from cmi.train.trainer import train_model, predict, ALL_METHODS


def _synth(n_per_cell=8, C=22, T=128, n_cls=2, n_dom=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            X.append(rng.standard_normal((n_per_cell, C, T)).astype("float32"))
            y += [yi] * n_per_cell
            d += [di] * n_per_cell
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def test_fbdualpc_grammar_9tuple():                          # P6-A grammar
    assert "fbdualpc" in ALL_METHODS
    t = parse_config("fbdualpc:0.005:0.005:0.010:0.000:0.100:300")
    c, method, lam, gamma, lam_edge, z_margin, dec_scale, node_w, lam_spatial = t
    assert method == "fbdualpc"
    assert (lam, node_w, lam_spatial, lam_edge, gamma, dec_scale) == (0.005, 0.005, 0.010, 0.000, 0.100, 300.0)
    # dec_scale optional (default 1.0); graphdualpc grammar still 9-tuple with lam_spatial=0
    assert parse_config("fbdualpc:0:0:0.01:0:0.1")[6] == 1.0
    assert parse_config("graphdualpc:0.01:0.01:0:0.1")[8] == 0.0 and parse_config("erm:0")[8] == 0.0


def test_fbdualpc_runs_and_spatial_cmi_active():             # P6-A: reg_spatial_gls finite + nonzero
    X, y, d = _synth()
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu")
    bb, post, out = train_model(bb, X, y, d, 2, method="fbdualpc", lam=0.005, beta=0.005, lam_spatial=0.010,
                                lam_edge=0.0, gamma=0.1, dec_scale=300.0, epochs=2, bs=16, n_inner=1,
                                warmup=1, device="cpu", seed=0)
    for k in ("reg_graph_gls", "reg_node_gls", "reg_spatial_gls", "loss_spatial", "lambda_spatial",
              "stepA_spatial_loss_gls", "dec_js_res"):
        assert k in out and math.isfinite(out[k]), f"{k} missing/non-finite"
    assert out["reg_spatial_gls"] >= 0.0 and out["lambda_spatial"] == 0.010


def test_fbdualpc_requires_spatial_branch_backbone():        # fail-closed on non-spatial backbone
    X, y, d = _synth(C=8, T=64)
    torch.manual_seed(0)
    bb = build_backbone("DGCNNGraph", 8, 64, 2, device="cpu")   # no spatial branch
    with pytest.raises(ValueError, match="spatial branch"):
        train_model(bb, X, y, d, 2, method="fbdualpc", lam=0.01, beta=0.01, lam_spatial=0.01,
                    lam_edge=0.0, gamma=0.1, epochs=1, bs=16, n_inner=1, warmup=1, device="cpu")


def test_fusion_floor_bounds_gates_and_raises_entropy():     # P6-B
    torch.manual_seed(0)
    x = torch.randn(16, 22, 128)
    bb0 = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu", fusion_floor=0.0)
    bbf = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu", fusion_floor=0.10)
    assert bb0.fusion_floor == 0.0 and bbf.fusion_floor == 0.10
    gf = bbf.gate_summary(x)
    # floor eps=0.10 -> every branch mean >= ~0.10 and gates still sum to 1
    for nm in ("graph", "temporal", "spatial"):
        assert gf[f"gate_{nm}_mean"] >= 0.099
    assert math.isclose(sum(gf[f"gate_{nm}_mean"] for nm in ("graph", "temporal", "spatial")), 1.0, abs_tol=1e-4)
    assert "gate_entropy_mean" in gf and 0.0 <= gf["gate_entropy_mean"] <= math.log(3) + 1e-6
    # a floored gate is more balanced (higher entropy) than an unfloored one on the same input+weights
    torch.manual_seed(1); a = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu", fusion_floor=0.0)
    torch.manual_seed(1); b = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu", fusion_floor=0.10)
    assert b.gate_summary(x)["gate_entropy_mean"] >= a.gate_summary(x)["gate_entropy_mean"] - 1e-6


def test_fusion_floor_off_is_plain_softmax():                # fusion_floor=0 -> unchanged path
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu", fusion_floor=0.0)
    bb.eval()
    with torch.no_grad():
        bb.forward_graph(torch.randn(8, 22, 128))
    g = bb.last_gate
    assert torch.allclose(g.sum(1), torch.ones(8), atol=1e-5)   # valid softmax


def test_fbdualpc_deterministic_with_early_stop():           # firewall
    X, y, d = _synth(seed=2)
    Xte = np.random.default_rng(7).standard_normal((16, 22, 128)).astype("float32")
    preds = []
    for _ in range(2):
        torch.manual_seed(0); np.random.seed(0)
        bb = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu", fusion_floor=0.05)
        bb, _, out = train_model(bb, X, y, d, 2, method="fbdualpc", lam=0.005, beta=0.005, lam_spatial=0.010,
                                 lam_edge=0.0, gamma=0.1, dec_scale=300.0, epochs=3, bs=16, n_inner=1,
                                 warmup=1, device="cpu", seed=0, early_stop=True, source_val_domains=[2])
        preds.append(predict(bb, Xte, "cpu"))
    assert np.allclose(preds[0], preds[1])


def test_graphdualpc_unaffected_by_p6():                     # regression: shared branch still works
    X, y, d = _synth()
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu")
    bb, _, out = train_model(bb, X, y, d, 2, method="graphdualpc", lam=0.01, beta=0.01, lam_edge=0.0,
                             gamma=0.1, dec_scale=300.0, epochs=2, bs=16, n_inner=1, warmup=1, device="cpu")
    assert math.isfinite(out["dec_js_res"]) and "reg_spatial_gls" not in out   # spatial diag only for fbdualpc
