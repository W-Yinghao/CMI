"""CIGL_67 functional CMI — CPU engineering + firewall tests (synthetic = engineering only, no science claim)."""
import numpy as np
import pytest
import torch

from cmi.train.trainer import train_model, predict, FCIGL_METHODS, ALL_METHODS, _head_module
from cmi.models.graph_task_backbones import build_graph_task_backbone
from cmi.eval.gap_diagnostic import subject_offset_matrix, subject_subspace, task_head_alignment
from cmi.eval.head_export import forward_graph_capture


def _syn(n_dom=4, n_cls=2, n_per=44, C=8, T=128, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for dd in range(n_dom):
        for cc in range(n_cls):
            b = rng.standard_normal((n_per, C, T)).astype("float32") * 0.5
            b[:, 0, :] += 1.5 * cc                              # class signal
            b[:, 1, :] += (dd - 1.5) * 1.5                      # subject signal
            X.append(b); y += [cc] * n_per; d += [dd] * n_per
    return np.concatenate(X), np.array(y), np.array(d)


def _train(method, strength, epochs=18, seed=0, **kw):
    X, y, d = _syn(seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    net = build_graph_task_backbone("dgcnn_forward_graph_adapter", 8, 128, 2)
    net, _, out = train_model(net, X, y, d, 2, method=method, lam=0.010, gamma=0.010, lam_edge=0.0,
                              fcigl_strength=strength, fcigl_k=2, fcigl_update_every=3,
                              epochs=epochs, bs=64, warmup=3, device="cpu", seed=seed, **kw)
    return net, X, y, d, out


def _alignment(net, X, y, d):
    _, gz, _ = forward_graph_capture(net, X, "cpu")
    S = subject_subspace(subject_offset_matrix(gz, y, d), 2)
    W = _head_module(net).weight.detach().cpu().numpy()
    return task_head_alignment(W, S)


def test_registered_and_train_finite():
    assert FCIGL_METHODS <= ALL_METHODS
    for m, s in (("fcigl_align", 0.05), ("fcigl_removal_aug", 1.0)):
        net, X, y, d, out = _train(m, s, epochs=6)
        assert np.isfinite(predict(net, X, "cpu")).all()
        assert out.get("inloop_fcigl") is not None            # functional term applied + logged


def test_align_reduces_task_head_alignment():
    net0, X, y, d, _ = _train("fcigl_align", 0.0)              # penalty off
    net1, X1, y1, d1, _ = _train("fcigl_align", 0.5)          # penalty on (same seed/data)
    a0, a1 = _alignment(net0, X, y, d), _alignment(net1, X1, y1, d1)
    assert a1 < a0 + 1e-6                                      # penalty must not INCREASE alignment
    assert a1 <= a0 * 0.9 or a1 < 0.05                        # and should measurably reduce it (or drive it small)


def test_removal_aug_trains_and_logs():
    net, X, y, d, out = _train("fcigl_removal_aug", 1.0)
    assert out.get("inloop_fcigl") is not None and out["inloop_fcigl"] >= 0.0   # it's a CE -> non-negative
    assert np.isfinite(predict(net, X, "cpu")).all()


def test_projector_is_source_only_by_construction():
    # train_model only ever receives source (Xtr,ytr,dtr); corrupting a held-out target set is impossible here
    # because it never enters train_model. Assert the API surface: no target argument exists.
    import inspect
    params = inspect.signature(train_model).parameters
    assert "Xtr" in params and not any("target" in p for p in params)


def test_projector_deterministic_under_seed():
    from cmi.train.trainer import _fcigl_subject_projector
    X, y, d = _syn()
    torch.manual_seed(0)
    net = build_graph_task_backbone("dgcnn_forward_graph_adapter", 8, 128, 2)
    S1, P1 = _fcigl_subject_projector(net, X, y, d, 2, "cpu")
    S2, P2 = _fcigl_subject_projector(net, X, y, d, 2, "cpu")
    assert torch.allclose(S1, S2) and torch.allclose(P1, P2)


def test_fails_closed_on_non_graph_backbone():
    import torch.nn as nn
    bad = nn.Linear(4, 2); bad.z_dim = 4                       # no forward_graph, no head module
    with pytest.raises(ValueError):
        train_model(bad, np.random.randn(20, 4).astype("float32"), np.zeros(20, int), np.zeros(20, int),
                    2, method="fcigl_align", fcigl_strength=0.05, epochs=2, device="cpu")


def test_removal_projector_removes_subject_subspace():
    from cmi.train.trainer import _fcigl_subject_projector
    X, y, d = _syn()
    torch.manual_seed(0)
    net = build_graph_task_backbone("dgcnn_forward_graph_adapter", 8, 128, 2)
    S, P = _fcigl_subject_projector(net, X, y, d, 2, "cpu")
    # P = I - SᵀS is a projector: P@P ≈ P, and it kills the subject subspace rows S
    Pn = P.cpu().numpy(); Sn = S.cpu().numpy()
    assert np.allclose(Pn @ Pn, Pn, atol=1e-4)
    assert np.allclose(Sn @ Pn, 0.0, atol=1e-4)               # removed subspace projects to ~0
