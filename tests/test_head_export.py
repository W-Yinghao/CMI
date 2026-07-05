"""CIGL R2.5 — verified head-replay export. The 6 required guarantees plus a real DGCNN-adapter roundtrip and
an end-to-end save_fold_audit (source + eval-only target) export.

Required (PM):
  1 linear head export roundtrip: saved graph_z + task_head_* exactly reconstruct model_logits
  2 replay mismatch fails closed: replayed logits off by > tol -> task_head_replay_ok=False
  3 unsupported head fails closed: nonlinear/unknown head is not marked replayable
  4 existing .audit.npz without task_head_* still validates
  5 R3 uses head_replay when task_head_replay_ok=True
  6 R3 falls back to source-fit probe when task_head_replay_ok=False or fields absent
"""
import numpy as np
import torch

from cmi.eval.audit_npz import (
    save_audit_npz, load_audit_npz, validate_audit_npz, pack_task_head_fields,
    has_task_head, head_replay_ok, DEFAULT_REPLAY_TOL,
)
from cmi.eval.head_export import extract_task_head, forward_graph_capture, save_fold_audit
from cmi.eval.leakage_removal import evaluate_reliance
from cmi.models.graph_task_backbones import build_graph_task_backbone


def _build_adapter(n_chans=8, n_times=128, n_cls=2):
    torch.manual_seed(0)
    return build_graph_task_backbone("dgcnn_forward_graph_adapter", n_chans, n_times, n_cls).eval()


def _real_capture(N=24, n_chans=8, n_times=128, n_cls=2):
    net = _build_adapter(n_chans, n_times, n_cls)
    X = np.random.default_rng(0).standard_normal((N, n_chans, n_times)).astype("float32")
    logits, gz, nz = forward_graph_capture(net, X, "cpu")
    W, b, kind, inp = extract_task_head(net)
    return net, X, logits, gz, nz, W, b, kind, inp


# ---- 1: linear head export roundtrip (real DGCNN adapter) --------------------------------------------------
def test_1_linear_head_roundtrip_reconstructs_logits(tmp_path):
    net, X, logits, gz, nz, W, b, kind, inp = _real_capture()
    assert kind == "linear" and inp == "graph_z" and W.shape == (2, net.z_dim)
    replay = gz @ W.T + b
    assert np.abs(logits - replay).max() <= DEFAULT_REPLAY_TOL          # head IS the logits
    y = np.random.default_rng(1).integers(0, 2, len(X)); d = np.zeros(len(X), dtype=int)
    p = save_audit_npz(tmp_path / "f", graph_z=gz, node_z=nz, y=y, d=d, model_logits=logits,
                       fold=0, seed=0, target_subject="9", method="erm", dataset="2a",
                       task_head_weight=W, task_head_bias=b)
    data = load_audit_npz(p)
    assert data["task_head_replay_ok"] is True
    assert data["task_head_replay_max_abs_diff"] <= DEFAULT_REPLAY_TOL
    assert np.allclose(data["graph_z"] @ data["task_head_weight"].T + data["task_head_bias"],
                       data["model_logits"], atol=1e-4)


# ---- 2: replay mismatch fails closed -----------------------------------------------------------------------
def test_2_replay_mismatch_fails_closed(tmp_path):
    net, X, logits, gz, nz, W, b, kind, inp = _real_capture()
    bad_logits = logits + 1.0                                           # deliberately inconsistent
    y = np.zeros(len(X), dtype=int); d = np.zeros(len(X), dtype=int)
    p = save_audit_npz(tmp_path / "bad", graph_z=gz, node_z=nz, y=y, d=d, model_logits=bad_logits,
                       fold=0, seed=0, target_subject="9", task_head_weight=W, task_head_bias=b)
    data = load_audit_npz(p)
    assert data["task_head_replay_ok"] is False
    assert data["task_head_replay_max_abs_diff"] > DEFAULT_REPLAY_TOL


# ---- 3: unsupported head fails closed ----------------------------------------------------------------------
def test_3_unsupported_head_fails_closed():
    W = np.random.default_rng(0).standard_normal((2, 8)); b = np.zeros(2)
    gz = np.random.default_rng(1).standard_normal((10, 8)); logits = gz @ W.T + b
    fields = pack_task_head_fields(W, b, logits, gz, kind="mlp")        # nonlinear/unknown kind
    assert fields["task_head_replay_ok"] is False
    assert np.isnan(fields["task_head_replay_max_abs_diff"])
    # a model with no nn.Linear head -> extract returns unsupported
    W2, b2, kind2, _ = extract_task_head(torch.nn.Sequential(torch.nn.ReLU()))
    assert W2 is None and kind2 == "unsupported"


# ---- 4: back-compat — no task_head_* still validates -------------------------------------------------------
def test_4_no_task_head_still_validates(tmp_path):
    rng = np.random.default_rng(0); N = 12
    p = save_audit_npz(tmp_path / "plain", graph_z=rng.standard_normal((N, 64)),
                       node_z=rng.standard_normal((N, 8, 16)), y=rng.integers(0, 2, N),
                       d=rng.integers(0, 3, N), model_logits=rng.standard_normal((N, 2)),
                       fold=0, seed=0, target_subject="1")
    data = load_audit_npz(p)
    assert validate_audit_npz(data) == []
    assert not has_task_head(data) and not head_replay_ok(data)


# ---- 5 & 6: R3 consumes the gate -----------------------------------------------------------------------------
def _reliance_data(replay_ok):
    rng = np.random.default_rng(0); N, Z = 80, 6
    d = np.repeat(np.arange(4), N // 4); y = rng.integers(0, 2, N)
    z = rng.standard_normal((N, Z)); z[:, 1] += (d - 1.5)
    W = rng.standard_normal((2, Z)); b = np.zeros(2)
    data = {"graph_z": z, "y": y, "d": d, "model_logits": z @ W.T + b,
            "task_head_weight": W, "task_head_bias": b, "task_head_kind": "linear",
            "task_head_input": "graph_z", "task_head_replay_ok": replay_ok}
    return data


def test_5_R3_uses_head_replay_when_ok():
    row = evaluate_reliance(_reliance_data(True), target_domain=3)
    assert row["removal_mode"] == "head_replay" and row["head_replay_available"] and not row["probe_replay_used"]


def test_6_R3_falls_back_to_probe_when_not_ok_or_absent():
    row = evaluate_reliance(_reliance_data(False), target_domain=3)      # replay_ok False
    assert row["removal_mode"] == "probe_replay" and row["probe_replay_used"]
    d2 = _reliance_data(True); del d2["task_head_replay_ok"]              # fields absent
    row2 = evaluate_reliance(d2, target_domain=3)
    assert row2["removal_mode"] == "probe_replay" and row2["probe_replay_used"]


# ---- end-to-end save_fold_audit (source + eval-only target) ------------------------------------------------
def test_save_fold_audit_end_to_end(tmp_path):
    net = _build_adapter()
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((30, 8, 128)).astype("float32"); ys = rng.integers(0, 2, 30)
    ds = rng.integers(0, 3, 30)                                          # 3 source subjects
    Xt = rng.standard_normal((10, 8, 128)).astype("float32"); yt = rng.integers(0, 2, 10)
    p, ok, mad = save_fold_audit(tmp_path / "fold9", model=net, X_source=Xs, y_source=ys, d_source=ds,
                                 device="cpu", fold=0, seed=0, target_subject="9", method="erm", dataset="2a",
                                 X_target=Xt, y_target=yt, target_domain=3,
                                 source_indices=np.arange(30), target_indices=np.arange(30, 40),
                                 source_val_indices=np.arange(20, 30))
    assert ok and mad <= DEFAULT_REPLAY_TOL
    data = load_audit_npz(p)
    assert validate_audit_npz(data) == []
    assert len(data["y"]) == 40 and set(np.unique(data["d"])) == {0, 1, 2, 3}   # target tagged domain 3
    assert 3 not in np.unique(data["d"][data["source_indices"]])        # firewall: target id absent from source
    assert data["task_head_replay_ok"] is True
    # R3 reliance consumes it: fit on source (d != 3), target eval-only, head-replay used
    row = evaluate_reliance(data, target_domain=3)
    assert row["removal_mode"] == "head_replay" and row["firewall_passed"]


def test_target_domain_must_be_distinct(tmp_path):
    net = _build_adapter()
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((12, 8, 128)).astype("float32"); ys = rng.integers(0, 2, 12)
    ds = rng.integers(0, 3, 12); Xt = rng.standard_normal((4, 8, 128)).astype("float32")
    try:
        save_fold_audit(tmp_path / "bad", model=net, X_source=Xs, y_source=ys, d_source=ds, device="cpu",
                        fold=0, seed=0, target_subject="1", X_target=Xt, y_target=np.zeros(4, int),
                        target_domain=1)                                 # 1 collides with a source id
        assert False, "expected ValueError for colliding target_domain"
    except ValueError:
        pass
