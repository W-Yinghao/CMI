"""CIGL_47 P3-E — source-only early stopping (CPU only).

CIGL_46 trained 300 fixed epochs with source bAcc = 1.000 (catastrophic source memorization). P3-E adds
OPT-IN early stopping on a held-out SOURCE subject (domain), restoring the best-source-val-bAcc epoch.
Target labels are NEVER used for selection — the firewall (train_model/predict take no target labels,
determinism in (source, seed)) must still hold with early stopping ON.
"""
import inspect
import math

import numpy as np
import torch

from cmi.models.backbones import build_backbone
from cmi.train.trainer import train_model, predict


def _synth(n_per_cell=10, C=8, T=64, n_cls=2, n_dom=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            X.append(rng.standard_normal((n_per_cell, C, T)).astype("float32"))
            y += [yi] * n_per_cell
            d += [di] * n_per_cell
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def test_early_stop_off_by_default_records_nothing():
    X, y, d = _synth()
    C, T = X.shape[1], X.shape[2]
    torch.manual_seed(0)
    bb = build_backbone("DGCNNGraph", C, T, 2, device="cpu")
    _, _, out = train_model(bb, X, y, d, 2, method="graphcmi", lam=0.01, gamma=0.01,
                            epochs=3, bs=16, n_inner=1, warmup=1, device="cpu", seed=0)
    for k in ("source_val_subjects", "best_epoch", "best_source_val_bacc"):
        assert k not in out, f"early stopping must be OFF by default; leaked {k}"


def test_early_stop_records_metadata_and_holds_out_source_domain():
    X, y, d = _synth()
    C, T = X.shape[1], X.shape[2]
    torch.manual_seed(0)
    bb = build_backbone("FBLGGGraph", C, T, 2, device="cpu")
    _, _, out = train_model(bb, X, y, d, 2, method="graphdualpc", lam=0.01, beta=0.01, lam_edge=0.0,
                            gamma=0.1, epochs=4, bs=16, n_inner=1, warmup=1, device="cpu", seed=0,
                            early_stop=True, source_val_domains=[2])
    assert out["source_val_subjects"] == [2]
    assert 0 <= out["best_epoch"] <= 3
    for k in ("best_source_val_bacc", "final_val_source_bacc", "final_train_source_bacc"):
        assert k in out and math.isfinite(out[k]) and 0.0 <= out[k] <= 1.0


def test_early_stop_restores_best_epoch_weights():
    # The RETURNED model must be the best-source-val-bAcc epoch: re-scoring the held-out domain with the
    # returned weights must reproduce best_source_val_bacc exactly (proves the restore happened).
    X, y, d = _synth()
    C, T = X.shape[1], X.shape[2]
    torch.manual_seed(0)
    bb = build_backbone("FBLGGGraph", C, T, 2, device="cpu")
    bb, _, out = train_model(bb, X, y, d, 2, method="graphdualpc", lam=0.01, beta=0.01, lam_edge=0.0,
                             gamma=0.1, epochs=5, bs=16, n_inner=1, warmup=1, device="cpu", seed=0,
                             early_stop=True, source_val_domains=[2])
    Xval, yval = X[d == 2], y[d == 2]
    preds = predict(bb, Xval, "cpu").argmax(1)
    recs = [float((preds[yval == c] == c).mean()) for c in range(2) if (yval == c).any()]
    recomputed = float(np.mean(recs))
    assert math.isclose(recomputed, out["best_source_val_bacc"], abs_tol=1e-6), \
        f"returned model ({recomputed}) != best-epoch val bAcc ({out['best_source_val_bacc']}) -> not restored"


def test_early_stop_invalid_split_is_safe_noop():
    # holding out a non-existent domain -> empty val -> early stopping disabled, run still completes
    X, y, d = _synth()
    C, T = X.shape[1], X.shape[2]
    torch.manual_seed(0)
    bb = build_backbone("DGCNNGraph", C, T, 2, device="cpu")
    _, _, out = train_model(bb, X, y, d, 2, method="graphcmi", lam=0.01, gamma=0.01,
                            epochs=2, bs=16, n_inner=1, warmup=1, device="cpu", seed=0,
                            early_stop=True, source_val_domains=[99])   # no such domain
    assert "source_val_subjects" not in out          # no split -> no early-stop metadata, no crash


def test_firewall_signature_and_determinism_with_early_stop():
    # (1) new params are not target-label params
    params = set(inspect.signature(train_model).parameters)
    assert not (params & {"yte", "y_te", "y_target", "target_y", "Xte", "target", "d_te", "d_target"})
    assert "source_val_domains" in params and "early_stop" in params
    # (2) determinism in (source, seed) STILL holds with early stopping ON (source-only selection)
    X, y, d = _synth(seed=2)
    C, T = X.shape[1], X.shape[2]
    Xte = np.random.default_rng(7).standard_normal((16, C, T)).astype("float32")   # target FEATURES only
    preds = []
    for _ in range(2):
        torch.manual_seed(0); np.random.seed(0)
        bb = build_backbone("DGCNNGraph", C, T, 2, device="cpu")
        bb, _, _ = train_model(bb, X, y, d, 2, method="graphdualpc", lam=0.01, beta=0.01, lam_edge=0.0,
                               gamma=0.1, epochs=3, bs=16, n_inner=1, warmup=1, device="cpu", seed=0,
                               early_stop=True, source_val_domains=[2])
        preds.append(predict(bb, Xte, "cpu"))
    assert np.allclose(preds[0], preds[1]), "early stopping introduced nondeterminism / target dependence"
