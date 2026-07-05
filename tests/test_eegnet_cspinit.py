"""CIGL_56 P10 — EEGNetMiniCSPInit: does source-CSP initialization help the strongest compact decoder?
CSP-inits EEGNetMini's depthwise SPATIAL conv (block1[2]) from source-only CSP filters, then trains normally.
EEGNetMini itself is the frozen P9 baseline and MUST be unchanged. Firewall: target + source-val excluded."""
import sys
import numpy as np
import pytest
import torch

from cmi.models.backbones import build_backbone
from cmi.models.sanity_backbones import EEGNetMini, EEGNetMiniCSPInit
from cmi.train.trainer import train_model, predict
from cmi.run_loso import build_parser


def _synth(n_cls=4, C=22, T=128, n_per=8, n_dom=4, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            pat = np.zeros((1, C, 1), "float32"); pat[0, yi % C, 0] = 1.0
            X.append(rng.standard_normal((n_per, C, T)).astype("float32") + 0.6 * yi * pat)
            y += [yi] * n_per; d += [di] * n_per
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def test_cli_and_build_no_braindecode_no_graph():
    a = build_parser().parse_args(["--dataset", "BNCI2014_001", "--backbone", "EEGNetMiniCSPInit",
                                   "--configs", "erm:0", "--device", "cpu"])
    assert a.backbone == "EEGNetMiniCSPInit"
    bb = build_backbone("EEGNetMiniCSPInit", 22, 128, 4, device="cpu")
    assert isinstance(bb, EEGNetMiniCSPInit) and bb.spatial_init == "source_csp"
    assert "braindecode" not in sys.modules
    assert not callable(getattr(bb, "forward_graph", None))
    logits, z = bb(torch.randn(5, 22, 128))
    assert logits.shape == (5, 4) and z.shape[1] == bb.z_dim


@pytest.mark.parametrize("n_cls", [4, 2])
def test_csp_init_changes_depthwise_spatial(n_cls):
    X, y, d = _synth(n_cls)
    torch.manual_seed(0); bb = build_backbone("EEGNetMiniCSPInit", 22, 128, n_cls, device="cpu")
    before = bb.block1[2].weight.data.clone()             # depthwise spatial conv
    meta = bb.init_spatial_from_csp(X, y, n_cls, source_domains=d)
    assert not torch.allclose(before, bb.block1[2].weight.data)
    assert meta["csp_n_filters_used"] == bb.block1[2].weight.shape[0]   # all F1*D slots filled (pool>=n_out)
    assert meta["csp_excluded_target"] is True and torch.isfinite(bb.block1[2].weight.data).all()


def test_firewall_target_and_source_val_excluded():
    X, y, d = _synth(4, n_dom=4)
    torch.manual_seed(0); bb = build_backbone("EEGNetMiniCSPInit", 22, 128, 4, device="cpu")
    bb, _, out = train_model(bb, X, y, d, 4, method="erm", epochs=1, bs=16, warmup=1, device="cpu",
                             seed=0, early_stop=True, source_val_domains=[3])
    m = out["csp_meta"]
    assert m["csp_excluded_source_val"] == [3] and 3 not in m["csp_fit_subjects"]
    assert set(m["csp_fit_subjects"]) == {0, 1, 2} and m["csp_excluded_target"] is True


def test_eegnetmini_baseline_unchanged():
    # EEGNetMini (frozen baseline) must NOT be CSP-initialized and must be forward-identical to a plain build
    torch.manual_seed(0); base = build_backbone("EEGNetMini", 22, 128, 4, device="cpu")
    assert getattr(base, "spatial_init", "random") != "source_csp"
    assert not callable(getattr(base, "init_spatial_from_csp", None))
    torch.manual_seed(0); ref = EEGNetMini(22, 128, 4)
    base.eval(); ref.eval()
    x = torch.randn(4, 22, 128)
    assert torch.allclose(base(x)[0], ref(x)[0])          # same weights/forward -> baseline untouched


def test_csp_init_trains_and_deterministic():
    X, y, d = _synth(4, seed=2)
    Xte = np.random.default_rng(7).standard_normal((12, 22, 128)).astype("float32")
    preds = []
    for _ in range(2):
        torch.manual_seed(0); np.random.seed(0)
        bb = build_backbone("EEGNetMiniCSPInit", 22, 128, 4, device="cpu")
        bb, _, out = train_model(bb, X, y, d, 4, method="erm", epochs=3, bs=16, warmup=1, device="cpu",
                                 seed=0, early_stop=True, source_val_domains=[3])
        assert np.isfinite(predict(bb, Xte, "cpu")).all()
        preds.append(predict(bb, Xte, "cpu"))
    assert np.allclose(preds[0], preds[1])                 # firewall/determinism
