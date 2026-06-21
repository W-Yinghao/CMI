"""
Null calibration — the two ways the certificate must NOT cry wolf:

  1. residual test FALSE-POSITIVE rate: a source with NO concept domains (covariate-only)
     must produce significant T only ~alpha of the time (the permutation null is honest).
  2. INVISIBLE-shift FALSE-CERTIFICATION: a pure conditional shift (Z identical to clean)
     must NEVER be certified "safe" / "suspect"; it must abstain (UNIDENTIFIABLE).

These are the cheap, deterministic guards. The full pre-registered rates live in
run_synthetic.py.
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from csc.sim.shift_simulator import SimConfig, make_source, make_target
from csc.certificate import (
    residual_decoder_test, build_atlas, certify, FORBIDDEN, UNIDENTIFIABLE,
)


def test_residual_test_null_not_significant():
    """No concept domains -> T should usually be non-significant."""
    sigs = 0
    n = 8
    for s in range(n):
        cfg = SimConfig(seed=100 + s)
        src = make_source(cfg, n_domains=6, concept_domains=0, seed=100 + s)
        rt = residual_decoder_test(src.Z, src.Y, src.D, n_perm=60, alpha=0.05,
                                   seed=100 + s)
        sigs += int(rt.significant)
    rate = sigs / n
    # with only 8 reps this is a smoke bound, not the calibrated rate (see run_synthetic)
    assert rate <= 0.30, f"covariate-only source false-positive rate too high: {rate}"
    print(f"OK residual-test null false-positive rate = {rate:.2f} over {n} seeds (<=0.30)")


def test_invisible_shift_never_falsely_certified():
    """Pure conditional shift (Z byte-identical to clean) must always abstain."""
    bad = 0
    n = 12
    for s in range(n):
        cfg = SimConfig(seed=200 + s)
        src = make_source(cfg, n_domains=8, concept_domains=3, seed=200 + s)
        atlas = build_atlas(src.Z, src.Y, src.D)
        rt = residual_decoder_test(src.Z, src.Y, src.D, n_perm=40, seed=200 + s)
        tb = make_target("pure_conditional", cfg, geom=src.geom, seed=2000 + s)
        cert = certify(atlas, rt, tb.Z)
        if cert.state != UNIDENTIFIABLE:
            bad += 1
        assert cert.state not in FORBIDDEN["CONCEPT_INVISIBLE"], \
            f"FORBIDDEN false certification on invisible shift: {cert.state}"
    print(f"OK invisible shift abstained {n-bad}/{n} (0 forbidden certifications)")


def test_certifier_never_alarms_on_clean():
    """Clean target must never be CONCEPT_SUSPECT."""
    for s in range(8):
        cfg = SimConfig(seed=300 + s)
        src = make_source(cfg, n_domains=8, concept_domains=3, seed=300 + s)
        atlas = build_atlas(src.Z, src.Y, src.D)
        rt = residual_decoder_test(src.Z, src.Y, src.D, n_perm=40, seed=300 + s)
        tb = make_target("clean", cfg, geom=src.geom, seed=3000 + s)
        cert = certify(atlas, rt, tb.Z)
        assert cert.state not in FORBIDDEN["NONE"], \
            f"false concept alarm on clean: {cert.state}"
    print("OK clean target never raised a concept alarm (8 seeds)")


if __name__ == "__main__":
    test_residual_test_null_not_significant()
    test_invisible_shift_never_falsely_certified()
    test_certifier_never_alarms_on_clean()
    print("\nall null-calibration tests passed")
