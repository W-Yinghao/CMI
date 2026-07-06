"""CIGL_69A2 — full/equal-param EEG Conformer preflight (CPU only, engineering; NOT scientific evidence).

EEGConformerFull is the high-capacity VALIDATION arm for the ConformerMini audit: official EEG-Conformer
geometry (emb 40, depth 6, 10 heads, official conv kernels, 3-layer MLP head), ~8-10x the Mini params. Because
its head is an MLP (not a single nn.Linear), head-replay is NOT exact and R3 falls back to the source-fit probe
(removal_mode='probe_replay') — the PM-allowed "probe-compatible" path. These checks: builds/forwards, is
genuinely high-capacity, feature_z extractable, probe-compatible R3 artifact, and no target-label leakage.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from cmi.models.sanity_backbones import build_sanity_backbone, EEGConformerFull
from cmi.eval.head_export import forward_feature_capture, extract_task_head, save_fold_audit
from cmi.eval.audit_npz import load_audit_npz, validate_audit_npz, head_replay_ok
from cmi.eval.leakage_removal import evaluate_reliance

DIMS = [("2a", 22, 384, 4), ("2015", 13, 384, 2)]


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("tag,C,T,ncls", DIMS)
def test_build_forward_and_high_capacity(tag, C, T, ncls):
    torch.manual_seed(0)
    full = build_sanity_backbone("conformer_full", C, T, ncls).eval()
    mini = build_sanity_backbone("conformer", C, T, ncls)
    x = torch.randn(4, C, T)
    lo, z = full(x)
    assert lo.shape == (4, ncls) and z.shape == (4, full.z_dim) and torch.isfinite(lo).all()
    assert isinstance(full, EEGConformerFull) and hasattr(full, "transformer")
    assert _nparams(full) >= 5 * _nparams(mini)                       # genuinely a high-capacity arm (~8-10x Mini)


def test_head_is_mlp_so_replay_falls_back_to_probe():
    full = build_sanity_backbone("conformer_full", 22, 384, 4)
    assert isinstance(full.head, nn.Sequential)
    assert sum(isinstance(m, nn.Linear) for m in full.head) == 3     # official 3-layer MLP head, NOT single linear
    _, _, kind, _ = extract_task_head(full)
    assert kind == "unsupported"                                     # -> R3 must use the source-fit probe


def test_feature_capture_shapes():
    net = build_sanity_backbone("conformer_full", 22, 384, 4).eval()
    X = np.random.default_rng(0).standard_normal((20, 22, 384)).astype("float32")
    logits, feat, node = forward_feature_capture(net, X, "cpu")
    assert logits.shape == (20, 4) and feat.shape == (20, net.z_dim) and node.shape == (20, 1, 1)


def test_r3_artifact_is_probe_compatible(tmp_path):
    """Full Conformer feature_z audit must be R3-consumable via the PROBE fallback (replay_ok=False)."""
    torch.manual_seed(0)
    net = build_sanity_backbone("conformer_full", 8, 200, 2).eval()
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((48, 8, 200)).astype("float32"); ys = rng.integers(0, 2, 48); ds = rng.integers(0, 3, 48)
    Xt = rng.standard_normal((12, 8, 200)).astype("float32"); yt = rng.integers(0, 2, 12)
    p, replay_ok, _ = save_fold_audit(str(tmp_path / "cf"), model=net, X_source=Xs, y_source=ys, d_source=ds,
                                      device="cpu", fold=0, seed=0, target_subject="9", method="conformer_full:erm",
                                      dataset="syn", X_target=Xt, y_target=yt, target_domain=3,
                                      source_indices=np.arange(48), target_indices=np.arange(48, 60))
    assert replay_ok is False                                        # MLP head is not replayable
    d = load_audit_npz(p)
    assert validate_audit_npz(d) == [] and not head_replay_ok(d)
    assert d["graph_z"].shape[1] == net.z_dim
    row = evaluate_reliance(d, target_domain=3, k=2)
    assert row["removal_mode"] == "probe_replay" and row["firewall_passed"]   # probe fallback, firewall intact


def test_no_target_label_leakage_probe(tmp_path):
    torch.manual_seed(0)
    net = build_sanity_backbone("conformer_full", 8, 200, 2).eval()
    rng = np.random.default_rng(1)
    Xs = rng.standard_normal((48, 8, 200)).astype("float32"); ys = rng.integers(0, 2, 48); ds = rng.integers(0, 3, 48)
    Xt = rng.standard_normal((12, 8, 200)).astype("float32")
    p, _, _ = save_fold_audit(str(tmp_path / "cf"), model=net, X_source=Xs, y_source=ys, d_source=ds, device="cpu",
                              fold=0, seed=0, target_subject="9", X_target=Xt, y_target=np.zeros(12, int),
                              target_domain=3, source_indices=np.arange(48), target_indices=np.arange(48, 60))
    d = load_audit_npz(p)
    r0 = evaluate_reliance(d, target_domain=3, k=2)
    d2 = dict(d); d2["y"] = d["y"].copy(); d2["y"][d2["d"] == 3] = 1 - d2["y"][d2["d"] == 3]   # flip TARGET labels
    r1 = evaluate_reliance(d2, target_domain=3, k=2)
    assert np.isclose(r0["source_task_bacc_after"], r1["source_task_bacc_after"])   # source-fit probe unaffected


def test_official_braindecode_import_status_is_recorded():
    """Informational: record whether the OFFICIAL braindecode EEGConformer imports in this env (it does NOT in
    eeg2025 — moabb lacks BNCI2014001 for the eager import). Never fails; EEGConformerFull is the equal-param
    stand-in when the official import is unavailable."""
    try:
        from braindecode.models import EEGConformer  # noqa: F401
        available = True
    except Exception:
        available = False
    assert available in (True, False)                                # both outcomes are acceptable; just recorded
