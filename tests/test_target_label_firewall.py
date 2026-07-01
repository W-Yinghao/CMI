"""Target-label firewall: training and inference must be a pure function of the SOURCE data (+ seed).

The strict-DG contract is that target labels may enter ONLY the final evaluation metric, never training,
selection, early stopping, or model construction. We enforce this structurally:
  (1) train_model / predict take NO target-label argument (signature firewall);
  (2) training is deterministic in (source, seed) — two runs give byte-identical predictions — so the model
      cannot depend on any target labeling.
Covers both graph methods (graphdualpc and graphcmi).
"""
import inspect

import numpy as np
import torch

from cmi.models.backbones import build_backbone
from cmi.train.trainer import train_model, predict


def _source(seed=0, n_per_cell=8, C=8, T=64, n_cls=2, n_dom=3):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            X.append(rng.standard_normal((n_per_cell, C, T)).astype("float32"))
            y += [yi] * n_per_cell
            d += [di] * n_per_cell
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def test_train_and_predict_take_no_target_labels():
    tm = set(inspect.signature(train_model).parameters)
    forbidden = {"yte", "y_te", "y_target", "target_y", "Xte", "target", "d_te", "d_target"}
    assert not (tm & forbidden), f"train_model must not accept target labels/data; got {tm & forbidden}"
    pr = set(inspect.signature(predict).parameters)
    assert not (pr & {"y", "yte", "labels", "target"}), "predict must not accept labels"


def test_training_is_target_label_independent_and_deterministic():
    Xtr, ytr, dtr = _source(seed=2)
    C, T = Xtr.shape[1], Xtr.shape[2]
    Xte = np.random.default_rng(7).standard_normal((16, C, T)).astype("float32")   # target FEATURES only
    for method, kw in (("graphdualpc", dict(lam=0.01, beta=0.01, gamma=0.1)),
                       ("graphcmi", dict(lam=0.01, gamma=0.01))):
        preds = []
        for _ in range(2):   # identical (source, seed) -> identical model -> identical target predictions
            torch.manual_seed(0); np.random.seed(0)   # seed BEFORE build (weight init precedes train_model's reseed)
            bb = build_backbone("DGCNNGraph", C, T, 2, device="cpu")
            bb, _, _ = train_model(bb, Xtr, ytr, dtr, 2, method=method, lam_edge=0.0,
                                   epochs=2, bs=16, n_inner=1, warmup=1, device="cpu", seed=0, **kw)
            preds.append(predict(bb, Xte, "cpu"))
        assert np.allclose(preds[0], preds[1]), (
            f"{method}: training/inference not deterministic in (source, seed) -> possible hidden state; "
            f"any target-label dependence would break the source-only firewall")
