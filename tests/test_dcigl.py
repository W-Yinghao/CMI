"""CIGL_68 direct-reliance CMI — CPU engineering + firewall tests (synthetic = engineering only, no science)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cmi.train.trainer import (train_model, predict, DCIGL_METHODS, ALL_METHODS, PROJ_METHODS,
                               _head_module, _fcigl_subject_projector)
from cmi.models.graph_task_backbones import build_graph_task_backbone


def _entangled(n_dom=4, n_per=60, seed=0):
    """Subject SPURIOUSLY correlated with class -> the model relies on subject directions, so removing the
    subject subspace changes the prediction (SymKL > 0). This is the regime dcigl is meant to fix."""
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for dd in range(n_dom):
        p1 = 0.15 if dd < 2 else 0.85                          # subjects 0,1 -> mostly class 0; 2,3 -> class 1
        for _ in range(n_per):
            cc = int(rng.random() < p1)
            b = rng.standard_normal((8, 128)).astype("float32") * 0.4
            b[0, :] += 0.4 * cc                                # weak genuine class signal
            b[1, :] += (dd - 1.5) * 2.0                        # strong subject signal (spuriously carries class)
            X.append(b); y.append(cc); d.append(dd)
    return np.stack(X), np.array(y), np.array(d)


def _symkl_removed(net, X, y, d):
    net.eval()
    with torch.no_grad():
        _, gz, _, _ = net.forward_graph(torch.tensor(X, dtype=torch.float32))
    S, P = _fcigl_subject_projector(net, X, y, d, 2, "cpu"); h = _head_module(net)
    with torch.no_grad():
        lo, lo_rm = h(gz), h(gz @ P.t())
        lp, lq = F.log_softmax(lo, 1), F.log_softmax(lo_rm, 1)
        return 0.5 * (F.kl_div(lq, lp.exp(), reduction="batchmean") + F.kl_div(lp, lq.exp(), reduction="batchmean")).item()


def _train(method, strength, epochs=20, seed=0, **kw):
    X, y, d = _entangled(seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    net = build_graph_task_backbone("dgcnn_forward_graph_adapter", 8, 128, 2)
    net, _, out = train_model(net, X, y, d, 2, method=method, lam=0.010, gamma=0.010, lam_edge=0.0,
                              fcigl_strength=strength, fcigl_k=2, fcigl_update_every=3, epochs=epochs, bs=64,
                              warmup=3, device="cpu", seed=seed, **kw)
    return net, X, y, d, out


def test_registered_and_trains():
    assert DCIGL_METHODS <= ALL_METHODS and DCIGL_METHODS <= PROJ_METHODS
    net, X, y, d, out = _train("dcigl_consistency", 0.5, epochs=6, dcigl_gamma=0.5)
    assert np.isfinite(predict(net, X, "cpu")).all()
    assert out.get("inloop_fcigl") is not None                # functional (consistency) term applied + logged


def test_symkl_math():
    lo = torch.tensor([[2.0, -1.0], [0.0, 0.0]])
    lp = F.log_softmax(lo, 1)
    sk_same = 0.5 * (F.kl_div(lp, lp.exp(), reduction="batchmean") + F.kl_div(lp, lp.exp(), reduction="batchmean"))
    assert abs(sk_same.item()) < 1e-6                         # identical -> 0
    lo2 = torch.tensor([[-2.0, 1.0], [1.0, -1.0]]); lq = F.log_softmax(lo2, 1)
    sk = 0.5 * (F.kl_div(lq, lp.exp(), reduction="batchmean") + F.kl_div(lp, lq.exp(), reduction="batchmean"))
    sk_rev = 0.5 * (F.kl_div(lp, lq.exp(), reduction="batchmean") + F.kl_div(lq, lp.exp(), reduction="batchmean"))
    assert sk.item() > 0 and abs(sk.item() - sk_rev.item()) < 1e-6   # positive + symmetric


def test_consistency_reduces_prediction_change_under_removal():
    """The defining behavior: dcigl (beta>0) should make the prediction MORE invariant to subject-subspace
    removal than beta=0 (on entangled data where removal otherwise changes the prediction)."""
    net0, X0, y0, d0, _ = _train("dcigl_consistency", 0.0, dcigl_gamma=0.5)     # consistency off
    net1, X1, y1, d1, _ = _train("dcigl_consistency", 2.0, dcigl_gamma=0.5)     # consistency on (strong)
    s0, s1 = _symkl_removed(net0, X0, y0, d0), _symkl_removed(net1, X1, y1, d1)
    assert s0 > 1e-4                                           # entangled data: removal DOES change prediction at beta=0
    assert s1 < s0                                             # dcigl reduces the prediction change under removal


def test_differs_from_removal_aug():
    """dcigl penalizes the BEFORE-vs-AFTER prediction change; removal_aug only requires the removed rep to
    classify (does not constrain the original prediction). dcigl should yield lower SymKL under removal."""
    nd, Xd, yd, dd, _ = _train("dcigl_consistency", 2.0, dcigl_gamma=0.5)
    nr, Xr, yr, dr, _ = _train("fcigl_removal_aug", 2.0)
    assert _symkl_removed(nd, Xd, yd, dd) <= _symkl_removed(nr, Xr, yr, dr) + 1e-6


def test_projector_source_only_and_deterministic():
    X, y, d = _entangled()
    torch.manual_seed(0)
    net = build_graph_task_backbone("dgcnn_forward_graph_adapter", 8, 128, 2)
    S1, P1 = _fcigl_subject_projector(net, X, y, d, 2, "cpu")
    S2, P2 = _fcigl_subject_projector(net, X, y, d, 2, "cpu")
    assert torch.allclose(S1, S2) and torch.allclose(P1, P2)
    import inspect
    assert not any("target" in p for p in inspect.signature(train_model).parameters)   # no target arg


def test_fails_closed_on_non_graph_backbone():
    import torch.nn as nn
    bad = nn.Linear(4, 2); bad.z_dim = 4
    with pytest.raises(ValueError):
        train_model(bad, np.random.randn(20, 4).astype("float32"), np.zeros(20, int), np.zeros(20, int),
                    2, method="dcigl_consistency", fcigl_strength=0.5, epochs=2, device="cpu")
