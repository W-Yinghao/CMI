"""Pinned tests for Risk-Weighted MCC: source-LOSO excess pairwise risk weights, weight-permutation control, and
the weighted MCC loss. Synthetic data checks the implementation only; source-only (no target)."""
import inspect
import numpy as np
import pytest
import torch

from tos_cmi.train.risk_weighted_mcc import (source_loso_excess_risk_weights, permute_weights, rw_mcc_loss,
                                             weight_hash)
from tos_cmi.train.mechanism_consistency import class_pairs


def _src(m=5, C=4, per=40, p=12, outlier=None, seed=0):
    """m subjects x C classes; if outlier is set, that subject's class means are ROTATED so its contrast transfers
    poorly (high LOSO excess risk) while the rest share a consistent, well-transferring rule."""
    rng = np.random.default_rng(seed); shared = rng.standard_normal((C, p)) * 3.0
    Z, y, d = [], [], []
    for s in range(m):
        base = shared.copy()
        if s == outlier:
            base = base[::-1].copy()                         # reversed class means -> anti-transfer
        for c in range(C):
            Z.append(base[c] + 0.4 * rng.standard_normal((per, p))); y += [c] * per; d += [s] * per
    return np.vstack(Z), np.array(y), np.array(d)


def test_no_target_in_signature():
    p = set(inspect.signature(source_loso_excess_risk_weights).parameters)
    assert not ({"Z_target", "y_target", "target", "Xte", "yte"} & p)      # weights are source-only


def test_excess_risk_nonnegative_and_isolates_outlier_subject():
    Z, y, d = _src(outlier=2)
    out = source_loso_excess_risk_weights(Z, y, d)
    assert out["status"] == "ok"
    assert all(v >= 0 for v in out["r"].values())                          # positive part
    # the outlier subject (2) should carry more total weight than the average subject
    subs, pairs = out["subs"], out["pairs"]; w = out["weights"]
    tot = {s: sum(w[(s, p)] for p in pairs) for s in subs}
    assert tot[2] == max(tot.values()) and tot[2] > np.mean([tot[s] for s in subs if s != 2])


def test_subtraction_removes_pairwise_baseline_difficulty():
    # all subjects consistent -> classifier transfers well to held-out subject -> hold ~ ref -> excess risk ~ 0
    # (only tiny noise-driven positives). Key property: no cell DOMINATES (weights stay spread) and the outlier
    # case concentrates far more than the no-outlier case.
    o_none = source_loso_excess_risk_weights(*_src(outlier=None))
    o_out = source_loso_excess_risk_weights(*_src(outlier=2))
    # no-outlier: only tiny noise-driven excess (systematic pair difficulty is subtracted out)
    assert o_none["status"] == "NO_POSITIVE_SOURCE_TRANSFER_GAP" or max(o_none["r"].values()) < 0.05
    # a real outlier concentrates the weight MUCH more (far lower effective support) than the no-outlier case
    assert o_out["effective_weight_support"] < 0.6 * o_none["effective_weight_support"]


def test_all_zero_is_no_op_not_uniform_fallback():
    Z, y, d = _src(outlier=None, seed=3)
    out = source_loso_excess_risk_weights(Z, y, d)
    if out["status"] == "NO_POSITIVE_SOURCE_TRANSFER_GAP":
        assert all(v == 0.0 for v in out["weights"].values())              # NOT uniform 1.0
        assert out["effective_weight_support"] == 0.0


def test_weights_winsorized_and_clipped_no_remean():
    Z, y, d = _src(outlier=2)
    out = source_loso_excess_risk_weights(Z, y, d, clip=4.0)
    wv = np.array(list(out["weights"].values()))
    assert wv.max() <= 4.0 + 1e-9                                           # clip at 4
    # after clip the mean is NOT re-forced to 1 (few unstable cells -> mean below 1)
    assert abs(wv.mean() - 1.0) > 1e-6 or out["max_weight"] < 4.0


def test_weight_permutation_preserves_multiset_and_perpair_total():
    Z, y, d = _src(outlier=2)
    out = source_loso_excess_risk_weights(Z, y, d); subs, pairs, w = out["subs"], out["pairs"], out["weights"]
    wp = permute_weights(w, subs, pairs, seed=1)
    assert sorted(w.values()) == sorted(wp.values())                        # same multiset
    for p in pairs:                                                         # same per-pair total
        assert abs(sum(w[(s, p)] for s in subs) - sum(wp[(s, p)] for s in subs)) < 1e-9
    assert w != wp                                                          # assignment differs


def test_rw_mcc_loss_weighted_and_permute_differs():
    Z, y, d = _src(outlier=2, p=10)
    out = source_loso_excess_risk_weights(Z, y, d); subs, pairs, w = out["subs"], out["pairs"], out["weights"]
    Zt = torch.tensor(Z, dtype=torch.float32, requires_grad=True)
    L, info = rw_mcc_loss(Zt, y, d, w); L.backward()
    assert np.isfinite(float(L)) and float(Zt.grad.abs().sum()) > 0        # differentiable
    wp = permute_weights(w, subs, pairs, seed=1)
    Lp, _ = rw_mcc_loss(torch.tensor(Z, dtype=torch.float32), y, d, wp)
    assert abs(float(L) - float(Lp)) > 1e-6                                 # true weights != permuted


def test_weight_hash_deterministic():
    Z, y, d = _src(outlier=2); out = source_loso_excess_risk_weights(Z, y, d)
    h1 = weight_hash(out["weights"], out["subs"], out["pairs"]); h2 = weight_hash(out["weights"], out["subs"], out["pairs"])
    assert h1 == h2 and weight_hash(permute_weights(out["weights"], out["subs"], out["pairs"], 1), out["subs"], out["pairs"]) != h1
