"""Tests for Version 1 Train-Through-Erasure — erasure-before-head math + lower-encoder freeze + projectors."""
import numpy as np
import torch
import pytest

from cmi.models.graph_task_backbones import build_graph_task_backbone
from cmi.train import train_through_erasure as TTE


def _adapter():
    torch.manual_seed(0); np.random.seed(0)
    return build_graph_task_backbone("dgcnn_forward_graph_adapter", 8, 64, 3)


def test_erasure_applied_before_head():
    adap = _adapter()
    x = torch.randn(5, 8, 64)
    d = adap.z_dim
    P0 = np.zeros((d, d))
    Pt0 = torch.tensor(P0, dtype=torch.float32)
    logits_id, gz, gz_e = TTE._erased_logits(adap, x, Pt0)
    # with P=0, erased == full
    assert torch.allclose(gz, gz_e, atol=1e-6)
    # with a nonzero projector, the erased graph_z drops that subspace
    rng = np.random.default_rng(0); Q, _ = np.linalg.qr(rng.standard_normal((d, 3)))
    Pt = torch.tensor((Q @ Q.T), dtype=torch.float32)
    _, gz2, gz2e = TTE._erased_logits(adap, x, Pt)
    removed = (gz2 @ Pt.t())
    assert torch.allclose(gz2 - gz2e, removed, atol=1e-5)   # gz_e = (I-P) gz


def test_lower_top_split_and_freeze():
    adap = _adapter()
    lower, top = TTE._lower_top_params(adap)
    assert len(lower) > 0 and len(top) > 0
    # top must include the head; lower must not
    top_ids = {id(p) for p in top}
    assert id(adap.net.head.weight) in top_ids
    # after a short TTE fit with freeze_lower, lower params are unchanged
    X = np.random.default_rng(1).standard_normal((60, 8, 64)).astype("float32")
    y = np.random.default_rng(2).integers(0, 3, 60)
    before = [p.detach().clone() for p in lower]
    P0 = np.zeros((adap.z_dim, adap.z_dim))
    TTE.train_through_erasure(adap, P0, X, y, freeze_lower=True, reinit_head=True, epochs=2, bs=32, device="cpu")
    lower2, _ = TTE._lower_top_params(adap)
    for b, a in zip(before, lower2):
        assert torch.allclose(b, a, atol=1e-6)              # lower encoder frozen (unchanged)


def test_projectors_rank_and_shape():
    d = 64
    Pr = TTE.random_projector(d, 7, seed=0)
    assert Pr.shape == (d, d)
    assert np.allclose(Pr, Pr.T, atol=1e-8) and np.allclose(Pr @ Pr, Pr, atol=1e-6)
    assert abs(np.trace(Pr) - 7) < 1e-6                     # rank-7 projector -> trace 7
    Z = np.random.default_rng(0).standard_normal((200, d))
    y = np.random.default_rng(1).integers(0, 3, 200); dsub = np.random.default_rng(2).integers(0, 5, 200)
    Ps = TTE.subject_projector(Z, y, dsub, 4)
    assert Ps.shape == (d, d) and abs(np.trace(Ps) - 4) < 0.5


def test_predict_erased_runs():
    adap = _adapter()
    X = np.random.default_rng(3).standard_normal((40, 8, 64)).astype("float32")
    P0 = np.zeros((adap.z_dim, adap.z_dim))
    adap = TTE.train_through_erasure(adap, P0, X, np.random.default_rng(4).integers(0, 3, 40),
                                     freeze_lower=True, epochs=1, bs=20, device="cpu")
    prob = TTE.predict_erased(adap, X, P0, device="cpu")
    assert prob.shape == (40, 3) and np.allclose(prob.sum(1), 1.0, atol=1e-5)
