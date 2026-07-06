"""Project A — contract test: h2cmi/eval/harness.py output -> audited eval bridge.

Confirms (1) the bridge turns a harness `per_domain_pi_T` into a target-prior claim that is
identifiable only under C1∧C2∧C3, and (2) the REAL harness (evaluate_offline_tta) actually
exports `per_domain_pi_T` + `per_domain_tta_diagnostics` on a tiny simulator run (so the bridge
has the evidence it needs). Run:

    python -m h2cmi.tests.test_eval_bridge_harness_contract
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from h2cmi.observability import (ContractID as C, Estimand, check_claim_allowed,  # noqa: E402
                                 claims_for_offline_tta)


def test_bridge_consumes_per_domain_pi_T():
    # a harness-shaped offline_tta dict carrying a prior estimate
    offline = {"delta_adapt": {"d_balanced_acc": 0.05},
               "per_domain_pi_T": {"0": [0.5, 0.5], "1": [0.4, 0.6]},
               "per_domain_tta_diagnostics": {"0": {"delta_density_nll": 0.1}}}
    # with C1∧C2∧C3 -> identifiable prior (TU-1), payload carried through
    with_c = claims_for_offline_tta(offline, prior_contracts={C.C1, C.C2, C.C3})
    prior = [c for c in with_c if c.estimand == Estimand.TARGET_PRIOR][0]
    assert prior.metric_payload and "per_domain_pi_T" in prior.metric_payload
    assert check_claim_allowed(prior).identifiable
    # without contracts -> rejected
    without_c = claims_for_offline_tta(offline, prior_contracts=set())
    prior2 = [c for c in without_c if c.estimand == Estimand.TARGET_PRIOR][0]
    assert check_claim_allowed(prior2).rejected


def test_harness_offline_tta_exports_pi_T():
    """Tiny REAL run: evaluate_offline_tta must export per_domain_pi_T + diagnostics."""
    from h2cmi.config import H2Config
    from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
    from h2cmi.eval.harness import evaluate_offline_tta
    from h2cmi.train.trainer import reference_prior, train_h2

    n_classes, chans, times = 2, 8, 64
    sim = EEGSimulator(n_classes, chans, times,
                       shift=ShiftSpec(cov=1.0, prior=0.3, montage=0.2, noise=0.3),
                       seed=0).sample(n_sites=3, subjects_per_site=2, sessions_per_subject=1,
                                      trials_per_session=24)
    src_idx, tgt_idx = train_target_split(sim, 1, seed=0)
    cfg = H2Config(n_classes=n_classes)
    cfg.encoder.n_chans = chans; cfg.encoder.n_times = times
    cfg.encoder.z_c_dim = 12; cfg.encoder.z_n_dim = 6
    cfg.train.epochs = 2; cfg.tta.em_iters = 4; cfg.tta.min_target = 12

    Xs, ys = sim.X[src_idx], sim.y[src_idx]
    src_domains = sim.domains.subset(src_idx)
    model, _h, _d, _hist = train_h2(Xs, ys, src_domains, sim.dag, cfg, align_factor="site")
    pi_star = reference_prior(ys, n_classes, cfg.align.reference_prior)
    Xt, yt = sim.X[tgt_idx], sim.y[tgt_idx]
    tgt_unit = sim.domains.subset(tgt_idx).factor("subject")

    off = evaluate_offline_tta(model, Xt, yt, tgt_unit, cfg, pi_star)
    assert "per_domain_pi_T" in off and len(off["per_domain_pi_T"]) >= 1
    assert "per_domain_tta_diagnostics" in off
    # every exported prior vector is a proper distribution
    for pi in off["per_domain_pi_T"].values():
        assert abs(sum(pi) - 1.0) < 1e-4

    # and the bridge turns it into a (correctly gated) target-prior claim
    claims = claims_for_offline_tta(off, prior_contracts={C.C1, C.C2, C.C3})
    assert any(c.estimand == Estimand.TARGET_PRIOR for c in claims)


ALL_TESTS = [test_bridge_consumes_per_domain_pi_T, test_harness_offline_tta_exports_pi_T]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} HARNESS-CONTRACT TESTS PASSED")


if __name__ == "__main__":
    run()
