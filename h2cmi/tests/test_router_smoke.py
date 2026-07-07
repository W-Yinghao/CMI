"""End-to-end smoke test for the Project B router integration (Step-2E).

Samples a small synthetic EEG dataset, trains a tiny H2 model, and runs the router harness,
asserting the report is well-formed and that the conservative default router does not select
OFFLINE_TTA when the source ACAR-harm calibration is degenerate/unavailable.
"""
from __future__ import annotations

import numpy as np

from h2cmi.config import H2Config
from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
from h2cmi.train.trainer import train_h2, reference_prior
from h2cmi.eval.router_harness import (
    evaluate_router_offline_tta, make_support_calibrated_feature_config,
)
from h2cmi.router.router import RefusalFirstRouter, RouterConfig


def run():
    shift = ShiftSpec(cov=1.0, prior=0.4, concept=0.0, concept_site_frac=0.0, montage=0.2, noise=0.3)
    sim = EEGSimulator(3, 16, 128, 128.0, shift=shift, seed=0).sample(5, 3, 2, 40)
    src_idx, tgt_idx = train_target_split(sim, n_target_sites=1, seed=0)

    cfg = H2Config(n_classes=3).small()
    cfg.encoder.n_chans = 16
    cfg.encoder.n_times = 128
    cfg.encoder.fs = 128.0

    Xs, ys = sim.X[src_idx], sim.y[src_idx]
    src_dom = sim.domains.subset(src_idx)
    model, *_ = train_h2(Xs, ys, src_dom, sim.dag, cfg, align_factor="site", verbose=False)
    pi_star = reference_prior(ys, 3, cfg.align.reference_prior)

    Xt, yt = sim.X[tgt_idx], sim.y[tgt_idx]
    tgt_unit = sim.domains.subset(tgt_idx).factor("subject")
    src_unit = src_dom.factor("subject")

    rep = evaluate_router_offline_tta(
        model, Xt, yt, tgt_unit, cfg, pi_star,
        X_src=Xs, y_src=ys, source_pseudo_levels=src_unit, device=cfg.train.device)

    # structure
    for key in ("identity", "offline_tta_raw", "router_selected", "router_summary", "per_domain"):
        assert key in rep, f"missing report key {key}"
    s = rep["router_summary"]

    # finite raw metrics
    assert np.isfinite(rep["identity"]["balanced_acc"]), "identity bAcc not finite"
    assert np.isfinite(rep["offline_tta_raw"]["balanced_acc"]), "offline-TTA raw bAcc not finite"

    # action bookkeeping
    assert sum(s["action_counts"].values()) == s["n_domains"], "action_counts sum != n_domains"
    assert 0.0 <= s["coverage"] <= 1.0, "coverage out of [0,1]"

    # every per-domain decision carries both action scores + finite diagnostic vectors
    for did, dv in rep["per_domain"].items():
        assert "identity" in dv["action_scores"] and "offline_tta" in dv["action_scores"], did
        for which in ("diagnostics_identity", "diagnostics_offline_tta"):
            rc = dv[which]["reason_codes"]
            assert isinstance(rc, list) and all(isinstance(c, str) for c in rc), (did, which)

    # conservative default: no OFFLINE_TTA when source harm calibration is degenerate/unavailable
    if s["source_acar_harm_calibration_state"] in ("degenerate", "unavailable"):
        assert s["action_counts"].get("offline_tta", 0) == 0, s["action_counts"]

    # no NaN/inf in the per-domain support diagnostics (feature vectors are finite by construction)
    for dv in rep["per_domain"].values():
        for k, v in dv["support"].items():
            assert np.isfinite(v), (k, v)

    # a caller-supplied support threshold flows through router -> harness -> report (Step-2F path)
    thr = 99.0                                            # deliberately loose: identity support-admissible
    fcfg = make_support_calibrated_feature_config(
        max_density_nll_target_prior=thr, min_target_n=max(20, cfg.tta.min_target))
    router = RefusalFirstRouter(RouterConfig(feature_config=fcfg))
    rep2 = evaluate_router_offline_tta(
        model, Xt, yt, tgt_unit, cfg, pi_star, router=router,
        X_src=Xs, y_src=ys, source_pseudo_levels=src_unit, device=cfg.train.device,
        calibrate_source_support=False, support_calibration_mode="custom_test")
    s2 = rep2["router_summary"]
    assert abs(s2["source_support_threshold_nll_target_prior"] - thr) < 1e-9, s2
    assert s2["source_support_calibration_mode"] == "custom_test"
    assert sum(s2["action_counts"].values()) == s2["n_domains"]
    if s2["source_acar_harm_calibration_state"] in ("degenerate", "unavailable"):
        assert s2["action_counts"].get("offline_tta", 0) == 0, s2["action_counts"]

    print("router  actions=%s  coverage=%.2f  acar_harm=%s  missed_benefit=%.3f"
          % (dict(s["action_counts"]), s["coverage"],
             s["source_acar_harm_calibration_state"], s["missed_benefit"]))
    print("custom-threshold pass-through: thr=%.1f mode=%s actions=%s"
          % (s2["source_support_threshold_nll_target_prior"],
             s2["source_support_calibration_mode"], dict(s2["action_counts"])))
    print("\nROUTER SMOKE TEST PASSED")
    return rep


def test_router_smoke():
    run()


if __name__ == "__main__":
    run()
