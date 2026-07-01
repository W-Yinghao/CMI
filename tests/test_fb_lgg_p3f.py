"""CIGL_47 P3-F — hardening patch tests (CPU only).

Covers the PI-flagged GPU blockers and hardening items:
  F.1 graphcmi must not crash on FBLGG's 5-tuple forward_graph;
  F.2 backbones declare ablation_modes (FBLGG adds zero_temporal; DGCNN does not);
  F.3 dataset ch_names presets: exact match -> preset, count mismatch / no preset -> loud index fallback;
  F.5 FBLGG dropout present + eval-deterministic (train-stochastic).
"""
import math

import numpy as np
import torch

from cmi.models.backbones import build_backbone
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


def test_fblgg_graphcmi_runs_no_5tuple_crash():
    # P3-F.1: the graphcmi branch used a raw 4-tuple unpack -> "too many values to unpack" on FBLGG.
    X, y, d = _synth()
    C, T = X.shape[1], X.shape[2]
    torch.manual_seed(0)
    bb = build_backbone("FBLGGGraph", C, T, 2, device="cpu")
    bb, post, out = train_model(bb, X, y, d, 2, method="graphcmi", lam=0.01, gamma=0.01, lam_edge=0.0,
                                epochs=2, bs=16, n_inner=1, warmup=1, device="cpu", seed=0)
    assert math.isfinite(out["reg_graph"]) and out["lambda_g"] == 0.01


def test_backbones_declare_ablation_modes():
    # P3-F.2: FBLGG exposes zero_temporal; static DGCNN does not.
    fblgg = build_backbone("FBLGGGraph", 22, 128, 2, device="cpu")
    dgcnn = build_backbone("DGCNNGraph", 22, 128, 2, device="cpu")
    fm = fblgg.meta["ablation_modes"]
    dm = dgcnn.meta["ablation_modes"]
    assert "zero_temporal" in fm and "zero_graph" in fm and "permute_nodes" in fm
    assert "zero_temporal" not in dm and "zero_graph" in dm


def test_infer_ch_names_preset_and_loud_fallback():
    # P3-F.3: exact preset match; count mismatch and unknown dataset -> index fallback (no silent misuse).
    from cmi.run_loso import _infer_ch_names, _DATASET_CH_NAMES
    names, src = _infer_ch_names("BNCI2014_001", 22)
    assert names is not None and len(names) == 22 and src == "preset:BNCI2014_001"
    names2, src2 = _infer_ch_names("BNCI2015_001", 13)
    assert names2 is not None and len(names2) == 13 and src2 == "preset:BNCI2015_001"
    # channel-count mismatch -> do NOT reuse the preset
    n3, s3 = _infer_ch_names("BNCI2014_001", 20)
    assert n3 is None and "mismatch" in s3
    # unknown dataset -> no preset
    n4, s4 = _infer_ch_names("SomeOtherDataset", 30)
    assert n4 is None and s4 == "index_fallback(no_preset)"
    # presets actually engage name-aware grouping on the real montage (>=2 groups on 2014)
    from cmi.models.fb_lgg_dualcmi import build_channel_groups
    g = build_channel_groups(22, ch_names=_DATASET_CH_NAMES["BNCI2014_001"], max_groups=6)
    assert len(g) >= 2 and sorted(c for grp in g for c in grp) == list(range(22))


def test_fblgg_dropout_present_eval_deterministic_train_stochastic():
    # P3-F.5: dropout active in train mode, off in eval mode.
    torch.manual_seed(0)
    bb = build_backbone("FBLGGGraph", 22, 128, 2, device="cpu")
    assert isinstance(bb.drop, torch.nn.Dropout) and bb.drop.p > 0.0
    x = torch.randn(8, 22, 128)
    bb.eval()
    with torch.no_grad():
        a = bb.forward_graph(x)[0]; b = bb.forward_graph(x)[0]
    assert torch.allclose(a, b), "eval-mode dropout must be OFF (deterministic)"
    bb.train()
    torch.manual_seed(1); p1 = bb.forward_graph(x)[0]
    torch.manual_seed(2); p2 = bb.forward_graph(x)[0]
    assert not torch.allclose(p1, p2), "train-mode dropout should make forward stochastic"
