"""CIGL_52 P8 — source-CSP-initialized spatial branch + spatial auxiliary head (CPU only).

P7a killed the covariance-tangent feature (underfit on near-rank-deficient 2a). P8 keeps the load-bearing
logvar spatial branch but (A) initializes its per-band spatial filters from SOURCE-only CSP filters (then
trains them), and (B) adds a source-only auxiliary classifier over spatial_z. FIREWALL: CSP is fit on
source-train trials only — the LOSO caller removes the held-out target subject, and the source-val subject
is removed before fitting. spatial_init='random' (default) must reproduce the P6 path.
"""
import math

import numpy as np
import pytest
import torch

from cmi.models.backbones import build_backbone
from cmi.models.fb_lgg_dualcmi import FBCSPLGGGraph
from cmi.models.csp_init import source_csp_filters
from cmi.train.trainer import train_model, predict
from test_fbcsp_lgg import _DATASET_CH_NAMES
from cmi.models.fb_lgg_dualcmi import central_strip_groups


def _synth(n_cls=4, C=22, T=128, n_per=10, n_dom=4, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            # class-dependent spatial pattern so CSP has real structure to find
            pat = np.zeros((1, C, 1), "float32"); pat[0, yi % C, 0] = 1.0
            X.append((rng.standard_normal((n_per, C, T)).astype("float32") + 0.6 * yi * pat))
            y += [yi] * n_per; d += [di] * n_per
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def test_random_is_default_and_unchanged():                         # (7)+baseline
    torch.manual_seed(0); a = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu")
    torch.manual_seed(0); b = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu", spatial_init="random")
    assert a.meta["spatial_init"] == "random" and a.spatial_init == "random"
    a.eval(); b.eval()
    x = torch.randn(5, 22, 128)
    la, *_, za = a.forward_graph(x); lb, *_, zb = b.forward_graph(x)
    assert torch.allclose(la, lb) and torch.allclose(za, zb) and za.shape == (5, 32)
    assert len(a.forward_graph(x)) == 5 and a.forward_graph(x)[3] is None   # 5-tuple, edge None


def test_source_csp_filters_finite_normalized_shapes():             # (3)+(4)+(5)
    X4, y4, _ = _synth(4)
    W4, disc4, pres4 = source_csp_filters(X4, y4, 4, m=4)
    assert W4.shape == (16, 22) and pres4 == [0, 1, 2, 3]            # one-vs-rest: 4 classes x 4
    assert np.isfinite(W4).all() and np.isfinite(disc4).all()
    assert np.allclose(np.linalg.norm(W4, axis=1), 1.0, atol=1e-4)  # unit-norm filters
    X2, y2, _ = _synth(2)
    W2, _, pres2 = source_csp_filters(X2, y2, 2, m=4)
    assert W2.shape == (8, 22) and pres2 == [0, 1]                  # binary: 2 x m


@pytest.mark.parametrize("n_cls", [4, 2])
def test_csp_init_changes_spatial_filters(n_cls):                   # (4)+(5) applied to the backbone
    X, y, d = _synth(n_cls)
    torch.manual_seed(0); bb = build_backbone("FBCSPLGGGraph", 22, 128, n_cls, device="cpu", spatial_init="source_csp")
    before = bb.spatial.bands[0].spatial.weight.data.clone()
    meta = bb.init_spatial_from_csp(X, y, n_cls, source_domains=d)
    after = bb.spatial.bands[0].spatial.weight.data
    assert not torch.allclose(before, after)                       # CSP filters written in
    assert meta["csp_n_filters_used"] == 4 and meta["csp_excluded_target"] is True
    assert torch.isfinite(after).all()
    assert bb.forward_graph(torch.randn(4, 22, n_cls * 0 + 128))[0].shape == (4, n_cls)


def test_source_val_and_target_excluded_from_csp_fit():            # (1)+(2) FIREWALL
    X, y, d = _synth(4, n_dom=4)                                    # source domains 0,1,2,3
    torch.manual_seed(0); bb = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu", spatial_init="source_csp")
    bb, _, out = train_model(bb, X, y, d, 4, method="erm", epochs=1, bs=16, warmup=1, device="cpu",
                             seed=0, early_stop=True, source_val_domains=[3])
    m = out["csp_meta"]
    assert m["csp_excluded_source_val"] == [3]                     # source-val subject not in CSP
    assert 3 not in m["csp_fit_subjects"] and set(m["csp_fit_subjects"]) == {0, 1, 2}
    assert m["csp_excluded_target"] is True                        # target never enters train_model in LOSO


def test_csp_fit_is_deterministic_source_only():                  # (1) no hidden randomness / target leak
    X, y, _ = _synth(4)
    W1, _, _ = source_csp_filters(X, y, 4, m=4)
    W2, _, _ = source_csp_filters(X, y, 4, m=4)
    assert np.allclose(W1, W2)                                     # pure function of source (X,y)


def test_spatial_aux_loss_finite_and_trains():                    # (6)
    X, y, d = _synth(4)
    torch.manual_seed(0); bb = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu")
    w0 = bb.spatial_aux_head.weight.data.clone()
    bb, _, out = train_model(bb, X, y, d, 4, method="erm", spatial_aux_weight=0.2, epochs=3, bs=16,
                             warmup=1, device="cpu", seed=0)
    assert "loss_spatial_aux" in out and math.isfinite(out["loss_spatial_aux"])
    assert out["spatial_aux_weight"] == 0.2
    assert not torch.allclose(w0, bb.spatial_aux_head.weight.data)  # aux head actually trained
    # aux OFF (weight 0) -> no aux loss key, head untrained
    torch.manual_seed(0); b2 = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu")
    _, _, o2 = train_model(b2, X, y, d, 4, method="erm", spatial_aux_weight=0.0, epochs=2, bs=16,
                           warmup=1, device="cpu", seed=0)
    assert "loss_spatial_aux" not in o2


def test_all_ablations_emitted_both_inits():                      # (8)
    X, y, d = _synth(4)
    for init in ("random", "source_csp"):
        torch.manual_seed(0); bb = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu", spatial_init=init)
        if init == "source_csp":
            bb.init_spatial_from_csp(X, y, 4, source_domains=d)
        x = torch.randn(6, 22, 128)
        for mode in ("zero_graph", "zero_temporal", "zero_spatial", "permute_nodes"):
            assert bb.ablate(x, mode).shape == (6, 4)


def test_csp_init_deterministic_end_to_end():                     # (9) firewall/determinism
    X, y, d = _synth(4, seed=2)
    Xte = np.random.default_rng(7).standard_normal((12, 22, 128)).astype("float32")
    preds = []
    for _ in range(2):
        torch.manual_seed(0); np.random.seed(0)
        bb = build_backbone("FBCSPLGGGraph", 22, 128, 4, device="cpu", spatial_init="source_csp")
        bb, _, _ = train_model(bb, X, y, d, 4, method="erm", spatial_aux_weight=0.2, epochs=3, bs=16,
                               warmup=1, device="cpu", seed=0, early_stop=True, source_val_domains=[3])
        preds.append(predict(bb, Xte, "cpu"))
    assert np.allclose(preds[0], preds[1])


@pytest.mark.parametrize("ds,C,n_cls", [("BNCI2014_001", 22, 4), ("BNCI2015_001", 13, 2)])
def test_csp_init_central_strip_datasets(ds, C, n_cls):           # dataset wiring (1-vs-rest 2a / binary 2015)
    ch = _DATASET_CH_NAMES[ds]; idx, named, warn = central_strip_groups(ds, ch)
    assert warn is None
    X = np.random.default_rng(0).standard_normal((n_cls * 12, C, 128)).astype("float32")
    y = np.tile(np.arange(n_cls), 12).astype("int64"); d = np.repeat(np.arange(4), (n_cls * 12) // 4).astype("int64")
    torch.manual_seed(0)
    bb = build_backbone("FBCSPLGGGraph", C, 128, n_cls, device="cpu", ch_names=ch, groups=idx,
                        group_names=named, grouping_scheme="central_strip_v1", spatial_init="source_csp")
    meta = bb.init_spatial_from_csp(X, y, n_cls, source_domains=d)
    assert meta["csp_n_filters_pool"] == (n_cls * meta["csp_m_per_contrast"] if n_cls > 2 else 2 * meta["csp_m_per_contrast"])
    assert bb.forward_graph(torch.randn(4, C, 128))[0].shape == (4, n_cls)


def test_cli_accepts_p8_flags():                                  # runner wiring
    from cmi.run_loso import build_parser
    a = build_parser().parse_args(["--dataset", "BNCI2014_001", "--backbone", "FBCSPLGGGraph",
                                   "--configs", "erm:0", "--spatial_init", "source_csp",
                                   "--spatial_aux_weight", "0.2", "--device", "cpu"])
    assert a.spatial_init == "source_csp" and a.spatial_aux_weight == 0.2
    assert build_parser().parse_args(["--dataset", "BNCI2014_001", "--configs", "erm:0"]).spatial_init == "random"
