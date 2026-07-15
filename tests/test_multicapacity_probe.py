"""CMI-Trace P1.3 tests — multi-capacity probe family + valid familywise-max null."""
import numpy as np
import pytest

from cmi.eval import multicapacity_probe as mcp
from cmi.eval.conditional_subject_leakage import three_way_support_split


def _make(subject_strength=3.0, n_per_cell=60, n_cls=2, n_dom=3, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    cm = rng.standard_normal((n_cls, dim)) * 2
    sd = rng.standard_normal((n_dom, dim)); sd /= np.linalg.norm(sd, axis=1, keepdims=True)
    Z, y, d = [], [], []
    for c in range(n_cls):
        for g in range(n_dom):
            Z.append(cm[c] + subject_strength * sd[g] + 0.5 * rng.standard_normal((n_per_cell, dim)))
            y += [c] * n_per_cell; d += [g] * n_per_cell
    return np.vstack(Z), np.array(y), np.array(d)


def test_reports_each_capacity_separately():
    Z, y, d = _make(seed=1)
    er, pt, pe, _ = three_way_support_split(y, d, seed=1)
    out = mcp.multicapacity_cmi(Z, y, d, 2, 3, pt, pe, n_perm=12, seed=1, epochs=60)
    for cap in ("linear", "mlp_small", "mlp_large"):
        assert cap in out["per_capacity"]
        r = out["per_capacity"][cap]
        for f in ("kl", "excess_over_null", "perm_p", "train_loss", "domain_acc", "arch"):
            assert f in r
    assert out["primary_capacity"] == "mlp_small"
    assert out["primary_kl"] == pytest.approx(out["per_capacity"]["mlp_small"]["kl"])


def test_familywise_null_is_max_over_capacities():
    Z, y, d = _make(seed=2)
    er, pt, pe, _ = three_way_support_split(y, d, seed=2)
    out = mcp.multicapacity_cmi(Z, y, d, 2, 3, pt, pe, n_perm=12, seed=2, epochs=60)
    # familywise max kl == max over capacity kl
    assert out["familywise_max_kl"] == pytest.approx(max(out["per_capacity"][c]["kl"]
                                                         for c in out["per_capacity"]))
    # the familywise p is a valid probability in (0, 1]
    assert 0.0 < out["familywise_max_perm_p"] <= 1.0
    # familywise (max-over-capacities) null is at least as conservative as the winning capacity's own null:
    # its p must be >= that capacity's single-null p for the SAME observed max (never anti-conservative).
    win = out["familywise_max_capacity"]
    assert out["familywise_max_perm_p"] >= out["per_capacity"][win]["perm_p"] - 1e-9


def test_leaky_detected_clean_null():
    Zl, yl, dl = _make(subject_strength=3.0, seed=3)
    erl, ptl, pel, _ = three_way_support_split(yl, dl, seed=3)
    leaky = mcp.multicapacity_cmi(Zl, yl, dl, 2, 3, ptl, pel, n_perm=15, seed=3, epochs=60)
    assert leaky["per_capacity"]["mlp_small"]["excess_over_null"] > 0
    assert leaky["per_capacity"]["mlp_small"]["perm_p"] <= 0.15

    Zc, yc, dc = _make(subject_strength=0.0, seed=4)
    erc, ptc, pec, _ = three_way_support_split(yc, dc, seed=4)
    clean = mcp.multicapacity_cmi(Zc, yc, dc, 2, 3, ptc, pec, n_perm=15, seed=4, epochs=60)
    assert clean["familywise_max_kl"] < leaky["familywise_max_kl"]


def test_linear_capacity_is_actually_linear():
    from cmi.eval.graph_leakage import _build_probe
    import torch.nn as nn
    lin = _build_probe(10, 3, [], 0.0)
    assert sum(isinstance(m, nn.Linear) for m in lin) == 1     # linear = a single Linear, no hidden ReLU
    large = _build_probe(10, 3, [128, 128], 0.1)
    assert sum(isinstance(m, nn.Dropout) for m in large) == 2  # mlp_large carries dropout


def test_probe_split_disjoint_no_row_crossing():
    Z, y, d = _make(seed=5)
    er, pt, pe, diag = three_way_support_split(y, d, seed=5)
    assert diag["disjoint"] is True
    assert len(np.intersect1d(pt, pe)) == 0                     # probe train/eval never share a trial
