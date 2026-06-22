"""
Power (CSC-P0) — the certificate must FIRE on the shifts it CAN see:

  * covariate shift        -> COVARIATE_COMPATIBLE
  * visible concept shift  -> CONCEPT_SUSPECT  (the positive control)
  * source concept evidence is detected when genuinely present (direction-linked).

No power -> nothing to certify -> the direction is dead (PREREGISTRATION termination).
Fast smoke bounds; precise rates are in run_synthetic.py.
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from csc.sim.shift_simulator import SimConfig, make_source, make_target
from csc.certificate import analyze_source, certify, COVARIATE_COMPATIBLE, CONCEPT_SUSPECT

NB, NDB = 20, 150     # concept evidence is direction-linked -> spend budget on n_dir_boot


def _analyze(seed):
    cfg = SimConfig(seed=seed)
    src = make_source(cfg, n_domains=8, concept_domains=3, seed=seed)
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=NB, n_dir_boot=NDB, seed=seed)
    return cfg, src, sa


def test_residual_detects_concept():
    sigs, n = 0, 6
    for s in range(n):
        _, _, sa = _analyze(500 + s)
        sigs += int(sa.concept_evidenced)
    assert sigs / n >= 0.66, f"concept evidence power too low: {sigs/n}"
    print(f"OK source concept evidence detected {sigs}/{n} (>=0.66)")


def test_covariate_compatible():
    # NOTE: coverage is intentionally CONSERVATIVE now -- COVARIATE_COMPATIBLE requires the
    # positive cov_stable equivalence evidence (CSC-P1.1 #5), so borderline-stable cases
    # abstain rather than certify. This is a safe (non-forbidden) miss; raising coverage is a
    # freeze-sweep tuning target (alpha / eps / bootstrap budget), not a gate to weaken.
    # The smoke bound only checks the gate FIRES sometimes and is never a false certification.
    ok, forbidden, n = 0, 0, 10
    for s in range(n):
        cfg, src, sa = _analyze(600 + s)
        tb = make_target("covariate", cfg, geom=src.geom, seed=6000 + s)
        st = certify(sa, tb.Z).state
        ok += int(st == COVARIATE_COMPATIBLE)
        forbidden += int(st == CONCEPT_SUSPECT)        # the forbidden outcome for covariate
    assert forbidden == 0, f"covariate FALSE-certified as concept {forbidden}/{n}"
    assert ok / n >= 0.4, f"covariate compatible coverage implausibly low: {ok/n}"
    print(f"OK covariate -> COVARIATE_COMPATIBLE {ok}/{n} (>=0.4 smoke; 0 false concept)")


def test_visible_concept_suspect():
    ok, n = 0, 8
    for s in range(n):
        cfg, src, sa = _analyze(700 + s)
        tb = make_target("boundary_coupled", cfg, geom=src.geom, seed=7000 + s)
        ok += int(certify(sa, tb.Z).state == CONCEPT_SUSPECT)
    assert ok / n >= 0.75, f"visible-concept power too low: {ok/n}"
    print(f"OK visible concept -> CONCEPT_SUSPECT {ok}/{n} (>=0.75)")


if __name__ == "__main__":
    test_residual_detects_concept()
    test_covariate_compatible()
    test_visible_concept_suspect()
    print("\nall power tests passed")
