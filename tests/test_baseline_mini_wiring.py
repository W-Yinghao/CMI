"""P9-F — baseline sidecar wiring: EEGNetMini/ShallowConvNetMini/DeepConvNetMini reachable via run_loso
(build_backbone + --backbone choices), train through plain ERM, and carry NO braindecode dependency and NO
graph objects. Distinct names from the braindecode EEGNet/ShallowConvNet/Deep4Net (which need braindecode)."""
import sys
import numpy as np
import pytest
import torch

from cmi.models.backbones import build_backbone
from cmi.models.sanity_backbones import EEGNetMini, ShallowConvNetMini, DeepConvNetMini
from cmi.train.trainer import train_model, predict
from cmi.run_loso import build_parser

MINIS = {"EEGNetMini": EEGNetMini, "ShallowConvNetMini": ShallowConvNetMini, "DeepConvNetMini": DeepConvNetMini}


def _synth(n_cls=4, C=22, T=128, n_per=8, n_dom=4, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            X.append(rng.standard_normal((n_per, C, T)).astype("float32")); y += [yi] * n_per; d += [di] * n_per
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def test_cli_accepts_mini_backbones():
    for name in MINIS:
        a = build_parser().parse_args(["--dataset", "BNCI2014_001", "--backbone", name,
                                       "--configs", "erm:0", "--device", "cpu"])
        assert a.backbone == name


@pytest.mark.parametrize("name", list(MINIS))
def test_build_backbone_returns_mini_no_braindecode_no_graph(name):
    # must build without braindecode installed, and be the pure-torch mini class
    bb = build_backbone(name, 22, 128, 4, device="cpu")
    assert isinstance(bb, MINIS[name])
    assert "braindecode" not in sys.modules       # the mini path must NOT import braindecode
    assert hasattr(bb, "z_dim") and bb.z_dim > 0
    assert not callable(getattr(bb, "forward_graph", None))   # non-graph CNN: no fabricated graph leakage
    logits, z = bb(torch.randn(5, 22, 128))
    assert logits.shape == (5, 4) and z.shape[1] == bb.z_dim


@pytest.mark.parametrize("name", list(MINIS))
def test_mini_trains_erm_with_source_val_early_stop(name):
    X, y, d = _synth()
    torch.manual_seed(0)
    bb = build_backbone(name, 22, 128, 4, device="cpu")
    bb, _, out = train_model(bb, X, y, d, 4, method="erm", epochs=2, bs=16, warmup=1, device="cpu",
                             seed=0, early_stop=True, source_val_domains=[3])
    p = predict(bb, X[:12], "cpu")
    assert np.isfinite(p).all() and p.shape == (12, 4)
    for k in ("best_source_val_bacc", "final_val_source_bacc", "source_val_subjects"):
        assert k in out                          # source-only early-stop metadata present
    assert out["source_val_subjects"] == [3]


def test_mini_names_distinct_from_braindecode_names():
    # the braindecode names must NOT be silently rerouted to the mini classes
    p = build_parser().parse_args(["--dataset", "BNCI2014_001", "--backbone", "EEGNet",
                                   "--configs", "erm:0", "--device", "cpu"])
    assert p.backbone == "EEGNet"                # stays the braindecode name (HookedBackbone path)
    # and EEGNetMini is a different, valid choice
    assert "EEGNetMini" in build_parser()._option_string_actions["--backbone"].choices
