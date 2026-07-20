"""End-to-end smoke test: exercises every H2-CMI component on the EEG simulator.

Run:  python -m h2cmi.tests.test_smoke      (or: pytest h2cmi/tests/test_smoke.py)

It asserts the pipeline runs, produces finite metrics, and that the corrected
hierarchical-CMI estimator, reference-prior alignment, class-conditional density, selective
TTA, safety gate and three-setting harness all return well-formed outputs.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

# This is the SLOW end-to-end test (trains a model); keep it out of the fast unit suite:
#   pytest -m "not integration"     # fast unit tests only
#   pytest -m integration           # this end-to-end smoke
try:
    import pytest
    pytestmark = pytest.mark.integration
except ImportError:
    pass


def _tiny_config(n_classes, chans, times):
    from h2cmi.config import H2Config
    cfg = H2Config(n_classes=n_classes)
    cfg.encoder.n_chans = chans
    cfg.encoder.n_times = times
    cfg.encoder.z_c_dim = 16
    cfg.encoder.z_n_dim = 8
    cfg.train.epochs = 8
    cfg.train.batch_size = 48
    cfg.tta.em_iters = 6
    cfg.tta.min_target = 12
    return cfg


def run():
    from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
    from h2cmi.train.trainer import train_h2, reference_prior
    from h2cmi.eval.harness import run_three_settings
    from h2cmi.eval.leakage import crossfit_conditional_leakage

    # Recoverable (covariance/montage/prior) shift so the learning assertion is reliable.
    # The concept-shift + label-mechanism generation paths are covered by the simulator's
    # own self-test (python -m h2cmi.data.eeg_simulator).
    n_classes, chans, times = 3, 12, 128
    sim = EEGSimulator(n_classes, chans, times,
                       shift=ShiftSpec(cov=1.0, prior=0.4, concept=0.0, concept_site_frac=0.0,
                                       montage=0.2, noise=0.3, label_mechanism_rho=0.0),
                       seed=0).sample(n_sites=4, subjects_per_site=3, sessions_per_subject=2,
                                      trials_per_session=24)
    assert np.isfinite(sim.X).all(), "simulator produced non-finite X"

    src_idx, tgt_idx = train_target_split(sim, 1, seed=0)
    cfg = _tiny_config(n_classes, chans, times)
    Xs, ys = sim.X[src_idx], sim.y[src_idx]
    src_domains = sim.domains.subset(src_idx)

    model, hcmi, dual, hist = train_h2(Xs, ys, src_domains, sim.dag, cfg, align_factor="site")
    assert len(hist) == cfg.train.epochs
    assert all(np.isfinite(h["hybrid"]) for h in hist), "non-finite training loss"

    pi_star = reference_prior(ys, n_classes, cfg.align.reference_prior)
    Xt, yt = sim.X[tgt_idx], sim.y[tgt_idx]
    tgt_unit = sim.domains.subset(tgt_idx).factor("subject")
    src_unit = src_domains.factor("subject")

    res = run_three_settings(model, Xt, yt, tgt_unit, cfg, pi_star,
                             X_src=Xs, y_src=ys, gate_pseudo_levels=src_unit)

    # strict DG panel well-formed + above chance (the synthetic task is learnable)
    sdg = res["strict_dg"]
    for k in ("balanced_acc", "macro_f1", "nll", "brier", "ece", "worst_domain_bacc"):
        assert np.isfinite(sdg[k]), f"strict_dg {k} not finite"
    assert sdg["balanced_acc"] >= 0.40, f"model failed to learn (bAcc={sdg['balanced_acc']:.3f})"

    # offline TTA: panels + bootstrap + selective risk
    off = res["offline_tta"]
    assert "delta_adapt" in off and "gain_bootstrap" in off
    assert np.isfinite(off["gain_bootstrap"]["mean"])
    assert 0.0 <= off["selective_risk"]["coverage"] <= 1.0

    # online TTA panel
    on = res["online_tta"]
    assert np.isfinite(on["balanced_acc"])

    # safety gate trained on inner pseudo-targets
    assert res["gate_info"]["n_pseudo"] >= 1

    # cross-fitted leakage (signed, with null) per factor
    Zs = model.embed(Xs)
    leak = crossfit_conditional_leakage(Zs, ys, src_domains, sim.dag, n_classes, n_perm=10, seed=0)
    for f in ("site", "subject", "session"):
        assert f in leak and np.isfinite(leak[f]["I_hat"]), f"leakage[{f}] missing/non-finite"

    print("strict-DG  bAcc=%.3f  worst-dom=%.3f  ece=%.3f"
          % (sdg["balanced_acc"], sdg["worst_domain_bacc"], sdg["ece"]))
    print("offline-TTA  d_bAcc(adapt)=%+.3f  d_bAcc(selective)=%+.3f  coverage=%.2f"
          % (off["delta_adapt"]["d_balanced_acc"], off["delta_selective"]["d_balanced_acc"],
             off["selective_risk"]["coverage"]))
    print("online-TTA  bAcc=%.3f" % on["balanced_acc"])
    print("leakage I_hat:", {f: round(leak[f]["I_hat"], 3) for f in ("site", "subject", "session")})
    print("\nSMOKE TEST PASSED")
    return res


def test_smoke():
    run()


if __name__ == "__main__":
    run()
