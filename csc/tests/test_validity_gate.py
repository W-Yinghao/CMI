"""
The support-graph validity gate must REJECT degenerate domain/label graphs, and the
certifier must then ABSTAIN (UNIDENTIFIABLE) rather than emit a concept reading.

This is the "single-class subject-domain" case from the proposal: when a domain carries
only one class, I(Y;D|Z) collapses onto label predictability and the residual decoder is
confounded -> no honest concept evidence is possible.
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from csc.sim.shift_simulator import SimConfig, make_source, make_target
from csc.certificate import (
    check_support_graph, residual_decoder_test, build_atlas, certify,
    UNIDENTIFIABLE,
)


def test_single_class_domain_is_invalid():
    cfg = SimConfig(seed=7)
    src = make_source(cfg, n_domains=6, concept_domains=2, seed=7)
    # corrupt one domain into a single-class domain
    d0 = src.D == 0
    Y = src.Y.copy()
    Y[d0] = Y[d0][0]                       # collapse domain 0 to one class
    sg = check_support_graph(Y, src.D)
    assert not sg.valid, "single-class domain must fail the support graph"
    assert sg.min_classes_per_domain == 1
    print("OK single-class domain -> support graph INVALID:", sg.reasons[0])


def test_valid_source_passes():
    cfg = SimConfig(seed=8)
    src = make_source(cfg, n_domains=8, concept_domains=3, seed=8)
    sg = check_support_graph(src.Y, src.D)
    assert sg.valid, "class-spanning source must pass the support graph"
    print("OK class-spanning source -> support graph VALID "
          f"(min classes/domain={sg.min_classes_per_domain}, "
          f"min domains/class={sg.min_domains_per_class})")


def test_invalid_source_forces_abstention():
    cfg = SimConfig(seed=9)
    src = make_source(cfg, n_domains=6, concept_domains=2, seed=9)
    Y = src.Y.copy()
    Y[src.D == 0] = Y[src.D == 0][0]      # single-class domain
    rt = residual_decoder_test(src.Z, Y, src.D, n_perm=10, seed=9)
    assert rt.status == "INVALID"
    atlas = build_atlas(src.Z, Y, src.D)
    # even a large covariate target must come back UNIDENTIFIABLE (no valid concept atlas)
    tb = make_target("covariate", cfg, geom=src.geom, seed=99)
    cert = certify(atlas, rt, tb.Z)
    assert cert.state == UNIDENTIFIABLE, f"expected abstention, got {cert.state}"
    print("OK invalid support graph -> certifier ABSTAINS:", cert.reason[:70])


if __name__ == "__main__":
    test_single_class_domain_is_invalid()
    test_valid_source_passes()
    test_invalid_source_forces_abstention()
    print("\nall validity-gate tests passed")
