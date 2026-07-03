"""CIGL_47 P3-D — decoder-residual activation: dec_scale grammar + activation diagnostics (CPU only).

CIGL_46 left the decoder residual DORMANT (dec_js_res ~ 3e-4, its loss contribution << 1% of CE), so
I(Y;D|Z) was never exercised. P3-D exposes dec_scale via the config grammar (backward-compatible) and
emits activation diagnostics so we can SEE whether the [dec_scale*JS - dec_margin]_+ term is live and
tune it into the ~1-10%-of-CE target BEFORE any GPU run.
"""
import math

import numpy as np
import torch

from cmi.models.backbones import build_backbone
from cmi.run_loso import parse_config
from cmi.train.trainer import train_model


def _synth(n_per_cell=8, C=22, T=128, n_cls=2, n_dom=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            X.append(rng.standard_normal((n_per_cell, C, T)).astype("float32"))
            y += [yi] * n_per_cell
            d += [di] * n_per_cell
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def test_grammar_dec_scale_optional_backward_compat():
    # 4-field config (CIGL_46 form) -> dec_scale defaults to 1.0
    lbl, method, lam, gamma, lam_edge, z_margin, dec_scale, node_w, lam_spatial = parse_config(
        "graphdualpc:0.010:0.010:0.000:0.100", default_beta=0.0)
    assert method == "graphdualpc" and lam_spatial == 0.0
    assert (lam, node_w, lam_edge, gamma, dec_scale) == (0.010, 0.010, 0.000, 0.100, 1.0)
    # 5-field config -> dec_scale parsed
    p = parse_config("graphdualpc:0.010:0.010:0.000:0.100:50", default_beta=0.0)
    assert p[1] == "graphdualpc" and p[6] == 50.0 and (p[2], p[7], p[4], p[3]) == (0.010, 0.010, 0.000, 0.100)
    # tuple ORDER unchanged (dec_scale at index 6)
    assert parse_config("graphdualpc:0:0:0:0.1:7")[6] == 7.0


def test_activation_diagnostics_present_and_finite():
    X, y, d = _synth()
    C, T, n_cls = X.shape[1], X.shape[2], 2
    torch.manual_seed(0)
    bb = build_backbone("FBLGGGraph", C, T, n_cls, device="cpu")
    bb, post, out = train_model(bb, X, y, d, n_cls, method="graphdualpc",
                                lam=0.01, beta=0.01, lam_edge=0.0, gamma=0.1, dec_scale=10.0,
                                epochs=2, bs=16, n_inner=1, warmup=1, device="cpu", seed=0)
    for k in ("dec_js_res_raw", "dec_js_res_scaled", "loss_dec", "loss_dec_over_ce", "dec_gate_active_frac"):
        assert k in out and math.isfinite(out[k]), f"{k} missing/non-finite: {out.get(k)}"
    assert out["dec_gate_active_frac"] >= 0.0 and out["dec_gate_active_frac"] <= 1.0
    # scaled residual == dec_scale * raw residual (both averaged over the same batches)
    assert math.isclose(out["dec_js_res_scaled"], 10.0 * out["dec_js_res_raw"], rel_tol=1e-5, abs_tol=1e-9)
    assert out["loss_dec"] >= 0.0


def test_dec_scale_increases_decoder_loss_share():
    # Raising dec_scale must materially increase the decoder loss share (activation control works).
    X, y, d = _synth()
    C, T, n_cls = X.shape[1], X.shape[2], 2

    def run(scale):
        torch.manual_seed(0)                              # identical init/data -> only dec_scale differs
        bb = build_backbone("FBLGGGraph", C, T, n_cls, device="cpu")
        _, _, out = train_model(bb, X, y, d, n_cls, method="graphdualpc",
                                lam=0.01, beta=0.01, lam_edge=0.0, gamma=0.1, dec_scale=scale,
                                epochs=2, bs=16, n_inner=1, warmup=1, device="cpu", seed=0)
        return out

    lo, hi = run(1.0), run(200.0)
    assert hi["loss_dec"] > lo["loss_dec"], "dec_scale did not increase the decoder loss contribution"
    assert hi["loss_dec_over_ce"] > lo["loss_dec_over_ce"]
    # at dec_scale=1.0 the decoder term is ~dormant (CIGL_46 regime); the diagnostic must reveal it
    assert lo["loss_dec_over_ce"] < 0.05
