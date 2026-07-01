"""CIGL_47 — scaffold-review follow-up tests (CPU only).

Added after an adversarial review of the FB-LGG scaffold (firewall/backward-compat/fail-closed/numerical
all clean). These close the *robust* test-coverage gaps the review identified. Two review items are
deliberately NOT unit-tested here because they are real-data GPU-gate properties, not CPU-checkable:
  * zero_graph MATERIAL accuracy drop — on random synth the model collapses to a near-constant predictor
    (verified: 0/60 decision flips under ablation), so accuracy-level drop is meaningless on CPU. The
    logit-level ablation tests (test_fb_lgg_backbone.py) rule out a graph branch that is dormant BY
    CONSTRUCTION; a branch dormant BY TRAINING is checked on real data at the first GPU gate (CIGL_47 §7).
  * run_loso ch_names passthrough — name-aware grouping needs ch_names wired from the dataset into
    build_backbone; the runner currently uses the (roadmap-sanctioned) index-partition fallback. Tracked
    as a handoff note, not a bug.
"""
import math

import numpy as np
import pytest
import torch

from cmi.models.backbones import build_backbone
from cmi.models.fb_lgg_dualcmi import build_channel_groups, _region_of
from cmi.train.trainer import train_model, predict


def _synth(n_per_cell=10, C=22, T=128, n_cls=2, n_dom=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            X.append(rng.standard_normal((n_per_cell, C, T)).astype("float32"))
            y += [yi] * n_per_cell
            d += [di] * n_per_cell
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def test_fblgg_graphdualpc_lam_edge_fail_closed():
    # FBLGG has static adjacency (edge_logits=None); graphdualpc with lambda_edge>0 must fail closed.
    # (Previously only covered for DGCNNGraph.)
    X, y, d = _synth()
    C, T = X.shape[1], X.shape[2]
    torch.manual_seed(0)
    bb = build_backbone("FBLGGGraph", C, T, 2, device="cpu")
    with pytest.raises(ValueError, match="edge_logits"):
        train_model(bb, X, y, d, 2, method="graphdualpc",
                    lam=0.01, beta=0.01, lam_edge=0.01, gamma=0.1,   # lam_edge>0 with no per-sample edge object
                    epochs=1, bs=16, n_inner=1, warmup=1, device="cpu")


def test_channel_groups_real_bnci_montages():
    # Real 10-20 montages: BNCI2014_001 (22ch, multi-region -> name-aware) and BNCI2015_001 (13ch, all
    # central -> index-partition fallback). Both must fully cover channels with no overlap and not crash.
    bnci2014 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
    bnci2015 = ["FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CPz", "CP4"]
    for names in (bnci2014, bnci2015):
        g = build_channel_groups(len(names), ch_names=names, max_groups=6)
        flat = [c for grp in g for c in grp]
        assert sorted(flat) == list(range(len(names)))          # full coverage
        assert len(flat) == len(set(flat))                       # no overlap
        assert 1 <= len(g) <= 6 and all(len(grp) > 0 for grp in g)
    # 2014 must engage the NAME-AWARE path (>=2 functional regions present)
    assert len({_region_of(n) for n in bnci2014} - {None}) >= 2
    g14 = build_channel_groups(len(bnci2014), ch_names=bnci2014, max_groups=6)
    assert len(g14) >= 2


def test_dgcnngraph_uses_shared_decoder_head_path():
    # DGCNNGraph must expose a 4-tuple forward_graph (no distinct fused_z) -> P3-C routes it through the
    # SHARED decoder head (post_dec is post), which is the byte-identical CIGL_46 path. FBLGG is 5-tuple.
    torch.manual_seed(0)
    dgcnn = build_backbone("DGCNNGraph", 22, 128, 2, device="cpu")
    fblgg = build_backbone("FBLGGGraph", 22, 128, 2, device="cpu")
    x = torch.randn(2, 22, 128)
    assert len(dgcnn.forward_graph(x)) == 4, "DGCNNGraph must stay a 4-tuple (shared decoder head)"
    assert len(fblgg.forward_graph(x)) == 5, "FBLGGGraph must be a 5-tuple (distinct fused_z)"
    assert not bool(getattr(dgcnn, "meta", {}).get("distinct_fused_z", False))
    assert bool(getattr(fblgg, "meta", {}).get("distinct_fused_z", False))


def test_fblgg_determinism_with_early_stop():
    # FB-LGG (gated fusion + distinct fused_z + separate post_dec) must stay deterministic in (source,seed)
    # WITH source-only early stopping ON -> no hidden target dependence in the extra state.
    X, y, d = _synth(seed=2)
    C, T = X.shape[1], X.shape[2]
    Xte = np.random.default_rng(7).standard_normal((16, C, T)).astype("float32")   # target FEATURES only
    preds = []
    for _ in range(2):
        torch.manual_seed(0); np.random.seed(0)
        bb = build_backbone("FBLGGGraph", C, T, 2, device="cpu")
        bb, _, out = train_model(bb, X, y, d, 2, method="graphdualpc", lam=0.01, beta=0.01, lam_edge=0.0,
                                 gamma=0.1, epochs=3, bs=16, n_inner=1, warmup=1, device="cpu", seed=0,
                                 early_stop=True, source_val_domains=[2])
        assert out["source_val_subjects"] == [2]
        preds.append(predict(bb, Xte, "cpu"))
    assert np.allclose(preds[0], preds[1]), "FBLGG + early stopping is not deterministic in (source, seed)"
