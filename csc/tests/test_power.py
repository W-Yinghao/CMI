"""
Power — the certificate must actually FIRE on the shifts it can see:

  * covariate shift        -> COVARIATE_ADAPTABLE
  * visible concept shift  -> CONCEPT_SUSPECT  (the positive control)

and the source residual test must detect genuine concept variation when it is present.
If these fail, the direction is dead (no power -> nothing to certify); see the
PREREGISTRATION termination clause.
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from csc.sim.shift_simulator import SimConfig, make_source, make_target
from csc.certificate import (
    residual_decoder_test, build_atlas, certify,
    COVARIATE_ADAPTABLE, CONCEPT_SUSPECT,
)


def test_residual_test_detects_concept():
    sigs = 0
    n = 8
    for s in range(n):
        cfg = SimConfig(seed=400 + s)
        src = make_source(cfg, n_domains=8, concept_domains=3, seed=400 + s)
        rt = residual_decoder_test(src.Z, src.Y, src.D, n_perm=60, seed=400 + s)
        sigs += int(rt.significant)
    rate = sigs / n
    assert rate >= 0.75, f"residual test power too low: {rate}"
    print(f"OK residual test detected concept structure {sigs}/{n} (>=0.75)")


def test_covariate_certified_adaptable():
    ok = 0
    n = 10
    for s in range(n):
        cfg = SimConfig(seed=500 + s)
        src = make_source(cfg, n_domains=8, concept_domains=3, seed=500 + s)
        atlas = build_atlas(src.Z, src.Y, src.D)
        rt = residual_decoder_test(src.Z, src.Y, src.D, n_perm=40, seed=500 + s)
        tb = make_target("covariate", cfg, geom=src.geom, seed=5000 + s)
        cert = certify(atlas, rt, tb.Z)
        ok += int(cert.state == COVARIATE_ADAPTABLE)
    assert ok / n >= 0.8, f"covariate adaptability power too low: {ok/n}"
    print(f"OK covariate -> COVARIATE_ADAPTABLE {ok}/{n} (>=0.80)")


def test_visible_concept_flagged_suspect():
    ok = 0
    n = 10
    for s in range(n):
        cfg = SimConfig(seed=600 + s)
        src = make_source(cfg, n_domains=8, concept_domains=3, seed=600 + s)
        atlas = build_atlas(src.Z, src.Y, src.D)
        rt = residual_decoder_test(src.Z, src.Y, src.D, n_perm=40, seed=600 + s)
        tb = make_target("boundary_coupled", cfg, geom=src.geom, seed=6000 + s)
        cert = certify(atlas, rt, tb.Z)
        ok += int(cert.state == CONCEPT_SUSPECT)
    assert ok / n >= 0.8, f"visible-concept power too low: {ok/n}"
    print(f"OK visible concept -> CONCEPT_SUSPECT {ok}/{n} (>=0.80)")


if __name__ == "__main__":
    test_residual_test_detects_concept()
    test_covariate_certified_adaptable()
    test_visible_concept_flagged_suspect()
    print("\nall power tests passed")
