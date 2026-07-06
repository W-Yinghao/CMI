"""CIGL_69A preflight — EEGNetMini + EEGConformerMini feature_z audit/R3/head-replay (CPU, engineering only)."""
import numpy as np
import pytest
import torch

from cmi.models.sanity_backbones import build_sanity_backbone, EEGConformerMini, EEGNetMini
from cmi.eval.head_export import forward_feature_capture, save_fold_audit, extract_task_head
from cmi.eval.audit_npz import load_audit_npz, validate_audit_npz, head_replay_ok
from cmi.eval.leakage_removal import evaluate_reliance

DIMS = [("2a", 22, 128, 4), ("2015", 13, 384, 2)]


@pytest.mark.parametrize("bb", ["eegnet", "conformer"])
@pytest.mark.parametrize("tag,C,T,ncls", DIMS)
def test_build_forward_and_exact_head_replay(bb, tag, C, T, ncls):
    torch.manual_seed(0)
    net = build_sanity_backbone(bb, C, T, ncls).eval()
    x = torch.randn(5, C, T)
    lo, z = net(x)
    assert lo.shape == (5, ncls) and z.shape[1] == net.z_dim
    W = net.head.weight.detach(); b = net.head.bias.detach()
    assert float((lo - (z @ W.t() + b)).abs().max()) < 1e-5          # feature_z head-replay is exact


def test_conformer_is_faithful_mini_not_official():
    net = build_sanity_backbone("conformer", 22, 128, 4)
    assert isinstance(net, EEGConformerMini) and hasattr(net, "transformer")   # internal transformer, not braindecode
    W, b, kind, inp = extract_task_head(net)
    assert kind == "linear" and inp == "graph_z"                     # feature_z stored in the graph_z slot


def test_forward_feature_capture_shapes_and_replay():
    net = build_sanity_backbone("eegnet", 22, 128, 4).eval()
    X = np.random.default_rng(0).standard_normal((30, 22, 128)).astype("float32")
    logits, feat, node = forward_feature_capture(net, X, "cpu")
    assert logits.shape == (30, 4) and feat.shape[1] == net.z_dim and node.shape == (30, 1, 1)
    W = net.head.weight.detach().cpu().numpy(); b = net.head.bias.detach().cpu().numpy()
    assert np.abs(logits - (feat @ W.T + b)).max() < 1e-4            # exact linear head


@pytest.mark.parametrize("bb", ["eegnet", "conformer"])
def test_feature_audit_is_r3_consumable(bb, tmp_path):
    torch.manual_seed(0)
    net = build_sanity_backbone(bb, 8, 64, 2).eval()
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((40, 8, 64)).astype("float32"); ys = rng.integers(0, 2, 40); ds = rng.integers(0, 3, 40)
    Xt = rng.standard_normal((12, 8, 64)).astype("float32"); yt = rng.integers(0, 2, 12)
    p, ok, mad = save_fold_audit(str(tmp_path / f"{bb}_fold"), model=net, X_source=Xs, y_source=ys, d_source=ds,
                                 device="cpu", fold=0, seed=0, target_subject="9", method=f"{bb}:erm", dataset="syn",
                                 X_target=Xt, y_target=yt, target_domain=3,
                                 source_indices=np.arange(40), target_indices=np.arange(40, 52))
    assert ok and mad < 1e-4
    d = load_audit_npz(p)
    assert validate_audit_npz(d) == [] and head_replay_ok(d)
    assert d["graph_z"].shape[1] == net.z_dim                        # feature_z stored in graph_z slot
    row = evaluate_reliance(d, target_domain=3, k=2)
    assert row["removal_mode"] == "head_replay" and row["firewall_passed"]


def test_no_target_label_leakage_in_r3(tmp_path):
    """Target y is stored eval-only; corrupting it must not change the source-fit R3 subspace / row."""
    torch.manual_seed(0)
    net = build_sanity_backbone("conformer", 8, 64, 2).eval()
    rng = np.random.default_rng(1)
    Xs = rng.standard_normal((40, 8, 64)).astype("float32"); ys = rng.integers(0, 2, 40); ds = rng.integers(0, 3, 40)
    Xt = rng.standard_normal((12, 8, 64)).astype("float32")
    p, _, _ = save_fold_audit(str(tmp_path / "c"), model=net, X_source=Xs, y_source=ys, d_source=ds, device="cpu",
                              fold=0, seed=0, target_subject="9", X_target=Xt, y_target=np.zeros(12, int),
                              target_domain=3, source_indices=np.arange(40), target_indices=np.arange(40, 52))
    d = load_audit_npz(p)
    r0 = evaluate_reliance(d, target_domain=3, k=2)
    d2 = dict(d); d2["y"] = d["y"].copy(); d2["y"][d2["d"] == 3] = 1 - d2["y"][d2["d"] == 3]   # flip TARGET labels
    r1 = evaluate_reliance(d2, target_domain=3, k=2)
    assert np.isclose(r0["source_task_bacc_after"], r1["source_task_bacc_after"])   # source-fit unaffected by target y


def test_no_dependency_breakage():
    import importlib
    m = importlib.import_module("cmi.models.sanity_backbones")
    for bb in ("eegnet", "conformer", "shallow_convnet", "deep_convnet"):
        assert build_sanity_backbone(bb, 8, 64, 2) is not None       # all build in eeg2025 (no braindecode)
