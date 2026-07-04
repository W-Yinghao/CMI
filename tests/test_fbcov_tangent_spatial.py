"""CIGL_51 P7a — FBCov-Tangent spatial branch (CPU only, no GPU/training-of-scale).

P6 showed the spatial-CMI penalty damages the CSP-decodable subjects because the log-var branch keeps only
per-filter variance (the diagonal after a learned point spatial filter). P7a swaps the spatial feature to a
covariance-tangent representation `vech(logm(S_band))` that exposes the full second-order geometry to a linear
head, gated ERM-first on full-LOSO. spatial_mode="logvar" MUST stay the exact P6 path (byte/state-dict
identical); "cov_tangent" is the new hypothesis. These tests fix that contract before any GPU spend.
"""
import math

import numpy as np
import pytest
import torch

from cmi.models.backbones import build_backbone
from cmi.models.fb_lgg_dualcmi import FBCSPLGGGraph, central_strip_groups, _vech, _spd_logm
from cmi.train.trainer import train_model
from test_fbcsp_lgg import _DATASET_CH_NAMES

_COV_KEYS = ("spatial_mode", "cov_shrinkage", "cov_eps", "cov_eig_min_mean", "cov_eig_min_p05",
             "cov_log_feature_norm_mean", "cov_log_feature_norm_p95")


def _synth(n_per_cell=8, C=22, T=128, n_cls=2, n_dom=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            X.append(rng.standard_normal((n_per_cell, C, T)).astype("float32"))
            y += [yi] * n_per_cell
            d += [di] * n_per_cell
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def test_logvar_is_default_and_byte_identical():                     # (1)
    # default spatial_mode == logvar; state_dict keys identical to a plain build; forward deterministic.
    torch.manual_seed(0); a = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu")
    torch.manual_seed(0); b = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu", spatial_mode="logvar")
    assert a.meta["spatial_mode"] == "logvar" and a.spatial_mode == "logvar"
    assert list(a.state_dict().keys()) == list(b.state_dict().keys())   # no new/renamed params in logvar path
    a.eval(); b.eval()
    x = torch.randn(6, 22, 128)
    la, *_ , za = a.forward_graph(x); lb, *_ , zb = b.forward_graph(x)
    assert torch.allclose(la, lb) and torch.allclose(za, zb)            # identical logvar computation
    assert za.shape == (6, 32)
    assert a.cov_summary(x) == {"spatial_mode": "logvar"}               # no cov diag emitted in logvar


def test_cov_tangent_forward_5tuple():                               # (2)
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu", spatial_mode="cov_tangent")
    assert bb.meta["spatial_mode"] == "cov_tangent" and bb.meta["has_spatial_branch"]
    logits, gz, nz, el, fz = bb.forward_graph(torch.randn(8, 22, 128))
    assert logits.shape == (8, 4) and fz.shape == (8, 32) and el is None
    # per-band feature dim is the tangent-vech size n_filt * C(C+1)/2 (22 -> 253), not n_filt*K
    assert [b.out_dim for b in bb.spatial.bands] == [8 * 253, 8 * 253, 8 * 253]


def test_cov_tangent_spatial_z_finite_and_backward():               # (3)
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu", spatial_mode="cov_tangent")
    logits, *_ = bb.forward_graph(torch.randn(8, 22, 128))
    assert torch.isfinite(bb.last_spatial_z).all()
    (logits.sum() + bb.last_spatial_z.sum()).backward()               # gradient must flow through eigh
    grads = [p.grad for p in bb.spatial.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_cov_tangent_no_nan_on_near_singular():                     # (4) eigenvalue clamp / shrinkage
    # rank-deficient input: all channels identical -> band covariance is rank 1 (near-singular).
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu", spatial_mode="cov_tangent")
    base = torch.randn(8, 1, 128)
    x = base.expand(8, 22, 128).contiguous().requires_grad_(True)     # every channel identical -> singular cov
    logits, *_ = bb.forward_graph(x)
    assert torch.isfinite(logits).all() and torch.isfinite(bb.last_spatial_z).all()
    logits.sum().backward()
    assert torch.isfinite(x.grad).all()
    cs = bb.cov_summary(x.detach())
    assert cs["cov_eig_min_mean"] >= 0.0 and math.isfinite(cs["cov_log_feature_norm_p95"])


def test_cov_tangent_zero_spatial_ablation():                       # (5)
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu", spatial_mode="cov_tangent")
    x = torch.randn(8, 22, 128)
    out = bb.ablate(x, "zero_spatial")
    assert out.shape == (8, 4) and torch.isfinite(out).all()
    for m in ("zero_graph", "zero_temporal", "permute_nodes"):        # all declared ablations run
        assert bb.ablate(x, m).shape == (8, 4)


def test_cov_tangent_gate_summary_sums_to_one():                    # (6)
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu", spatial_mode="cov_tangent")
    g = bb.gate_summary(torch.randn(16, 22, 128))
    assert math.isclose(sum(g[f"gate_{n}_mean"] for n in ("graph", "temporal", "spatial")), 1.0, abs_tol=1e-4)
    assert 0.0 <= g["gate_entropy_mean"] <= math.log(3) + 1e-6


@pytest.mark.parametrize("ds,C,ng", [("BNCI2014_001", 22, 9), ("BNCI2015_001", 13, 5)])
def test_cov_tangent_central_strip_both_datasets(ds, C, ng):        # (7) C-agnostic + grouping intact
    ch = _DATASET_CH_NAMES[ds]
    idx, named, warn = central_strip_groups(ds, ch)
    assert warn is None and len(idx) == ng
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", C, 128, 2, device="cpu", ch_names=ch, groups=idx,
                        group_names=named, grouping_scheme="central_strip_v1", spatial_mode="cov_tangent")
    assert bb.grouping_scheme == "central_strip_v1" and bb.n_groups == ng
    bb.eval()
    logits, *_ , fz = bb.forward_graph(torch.randn(4, C, 128))
    assert logits.shape == (4, 2) and fz.shape == (4, 32)
    # covariance dim tracks the actual channel count (13 -> 91, 22 -> 253)
    assert bb.spatial.bands[0].out_dim == 8 * (C * (C + 1) // 2)


def test_cpu_tiny_erm_writes_cov_diagnostics():                     # (8)
    X, y, d = _synth()
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", 22, 128, 2, device="cpu", spatial_mode="cov_tangent")
    bb, _, _ = train_model(bb, X, y, d, 2, method="erm", epochs=2, bs=16, n_inner=1, warmup=1,
                           device="cpu", seed=0)
    cs = bb.cov_summary(torch.as_tensor(X[:64], dtype=torch.float32))
    for k in _COV_KEYS:
        assert k in cs, f"missing cov diagnostic {k}"
    assert cs["spatial_mode"] == "cov_tangent" and cs["cov_shrinkage"] == 0.05
    assert all(math.isfinite(cs[k]) for k in _COV_KEYS if isinstance(cs[k], float))
    assert cs["cov_eig_min_mean"] >= 0.05 / 22 - 1e-9                 # post-shrinkage floor a/C


def test_cli_accepts_spatial_mode_flags():                          # runner wiring
    from cmi.run_loso import build_parser
    args = build_parser().parse_args(
        ["--dataset", "BNCI2014_001", "--backbone", "FBCSPLGGGraph", "--configs", "erm:0",
         "--spatial_mode", "cov_tangent", "--cov_shrinkage", "0.05", "--cov_eps", "1e-4", "--device", "cpu"])
    assert args.spatial_mode == "cov_tangent" and args.cov_shrinkage == 0.05 and args.cov_eps == 1e-4
    assert build_parser().parse_args(
        ["--dataset", "BNCI2014_001", "--configs", "erm:0"]).spatial_mode == "logvar"   # default


def test_vech_is_frobenius_isometry():                              # tangent-map sanity
    torch.manual_seed(0)
    A = torch.randn(4, 5, 5); S = A + A.transpose(-1, -2)             # symmetric batch
    v = _vech(S, math.sqrt(2.0))
    assert v.shape == (4, 15)                                         # 5*6/2
    assert torch.allclose(v.norm(dim=-1), S.reshape(4, -1).norm(dim=-1), atol=1e-4)   # ||vech||=||S||_F
    # _spd_logm round-trips an SPD matrix's eigenvalues
    spd = torch.eye(5) + 0.1 * (A[0] @ A[0].t())
    logS, ev = _spd_logm(spd.unsqueeze(0), 1e-6)
    assert torch.isfinite(logS).all() and (ev > 0).all()
