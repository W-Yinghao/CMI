"""
Null calibration (CSC-P0) — the certificate must never cry wolf. Each guard targets a
specific FORBIDDEN false-certification (certifier.FORBIDDEN):

  1. CLEAN target               -> must ABSTAIN (UNIDENTIFIABLE), not "compatible".
  2. PURE CONDITIONAL (invisible) -> must ABSTAIN.
  3. LABEL SHIFT                 -> must ABSTAIN (the review's counterexample: the v0 code
                                   fired CONCEPT_SUSPECT ~31-53% of the time here).
  4. residual test NULL          -> a covariate-only source rarely shows concept evidence.

These are fast smoke bounds; the calibrated rates + Rule-of-Three upper bounds are in
run_synthetic.py.
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from csc.sim.shift_simulator import SimConfig, make_source, make_target
from csc.certificate import analyze_source, certify, FORBIDDEN, UNIDENTIFIABLE

NB, NDB = 25, 60       # bootstrap sizes for the source analysis (smoke)


def _analyze(seed, n_domains=8, concept_domains=3):
    cfg = SimConfig(seed=seed)
    src = make_source(cfg, n_domains=n_domains, concept_domains=concept_domains, seed=seed)
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=NB, n_dir_boot=NDB, seed=seed)
    return cfg, src, sa


def test_clean_must_abstain():
    for s in range(6):
        cfg, src, sa = _analyze(100 + s)
        tb = make_target("clean", cfg, geom=src.geom, seed=1000 + s)
        c = certify(sa, tb.Z)
        assert c.state == UNIDENTIFIABLE, f"clean must abstain, got {c.state}"
        assert c.state not in FORBIDDEN["NONE"]
    print("OK clean target -> UNIDENTIFIABLE (6 seeds)")


def test_invisible_must_abstain():
    for s in range(6):
        cfg, src, sa = _analyze(200 + s)
        tb = make_target("pure_conditional", cfg, geom=src.geom, seed=2000 + s)
        c = certify(sa, tb.Z)
        assert c.state == UNIDENTIFIABLE, f"invisible must abstain, got {c.state}"
        assert c.state not in FORBIDDEN["CONCEPT_INVISIBLE"]
    print("OK pure-conditional (invisible) -> UNIDENTIFIABLE (6 seeds)")


def test_label_shift_must_abstain():
    """The review's direct counterexample. Both moderate and extreme skews must abstain."""
    bad = 0
    for s in range(6):
        cfg, src, sa = _analyze(300 + s)
        for peak in (0.8, 0.95):
            tb = make_target("label_shift", cfg, geom=src.geom, label_peak=peak,
                             seed=3000 + s)
            c = certify(sa, tb.Z)
            assert c.state not in FORBIDDEN["LABEL_SHIFT"], \
                f"label shift (peak={peak}) FALSE-certified as {c.state}"
            bad += int(c.state != UNIDENTIFIABLE)
    print(f"OK label-shift abstained {12-bad}/12 (0 forbidden false certifications)")


def test_residual_null_rarely_significant():
    # NULL CALIBRATION is only valid at the CLUSTER (subject) unit: subjects carry a random
    # effect and each subject has ONE label, so the random effect is confounded with class ->
    # at the epoch level it manufactures spurious class-conditional domain structure that the
    # epoch null cannot remove. Passing group_ids uses the subject-coherent null (the P1.4.1
    # inference unit), under which a covariate-only source rarely shows false concept evidence.
    sigs, n = 0, 6
    for s in range(n):
        cfg = SimConfig(seed=400 + s)
        src = make_source(cfg, n_domains=6, concept_domains=0, seed=400 + s)
        sa = analyze_source(src.Z, src.Y, src.D, n_boot=NB, n_dir_boot=NDB,
                            group_ids=src.group_ids, seed=400 + s)
        sigs += int(sa.concept_evidenced)
    rate = sigs / n
    assert rate <= 0.34, f"covariate-only source false concept-evidence rate {rate}"
    print(f"OK covariate-only source (subject-level null): concept evidence {sigs}/{n} (<=0.34)")


if __name__ == "__main__":
    test_clean_must_abstain()
    test_invisible_must_abstain()
    test_label_shift_must_abstain()
    test_residual_null_rarely_significant()
    print("\nall null-calibration tests passed")
