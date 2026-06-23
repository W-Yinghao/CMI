"""
Support-graph validity gate (CSC-P0): it must reject NOT ONLY single-class domains but
also DISCONNECTED domain-class graphs and ill-posed designs, and the certifier must then
ABSTAIN. (The v0 gate only checked degree and passed a disconnected graph.)
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from csc.sim.shift_simulator import SimConfig, make_source, make_target
from csc.certificate import (
    check_support_graph, analyze_source, certify, UNIDENTIFIABLE,
)


def test_single_class_domain_is_invalid():
    src = make_source(SimConfig(seed=7), n_domains=6, concept_domains=2, seed=7)
    Y = src.Y.copy()
    Y[src.D == 0] = Y[src.D == 0][0]                 # collapse a domain to one class
    sg = check_support_graph(Y, src.D, Z=src.Z)
    assert not sg.valid and sg.min_classes_per_domain == 1
    print("OK single-class domain -> INVALID:", sg.reasons[0][:60])


def test_disconnected_graph_is_invalid():
    """Each domain has >=2 classes and each class is in >=2 domains, yet the bipartite
    (domain,class) graph splits into two components -> NOT jointly identifiable."""
    # domains 0,1 carry classes {0,1}; domains 2,3 carry classes {2,3}
    Y, D = [], []
    for d, cls in [(0, (0, 1)), (1, (0, 1)), (2, (2, 3)), (3, (2, 3))]:
        for c in cls:
            Y += [c] * 30; D += [d] * 30
    Y = np.array(Y); D = np.array(D)
    sg = check_support_graph(Y, D)
    assert sg.min_classes_per_domain >= 2 and sg.min_domains_per_class >= 2, "degree ok"
    assert not sg.valid and not sg.connected and sg.n_components == 2, \
        f"disconnected graph must be INVALID (components={sg.n_components})"
    print(f"OK disconnected graph -> INVALID (degree passes, {sg.n_components} components)")


def test_valid_source_passes():
    src = make_source(SimConfig(seed=8), n_domains=8, concept_domains=3, seed=8)
    sg = check_support_graph(src.Y, src.D, Z=src.Z)
    assert sg.valid and sg.connected
    print(f"OK class-spanning connected source -> VALID "
          f"(min_cell={sg.min_cell_count}, cond={sg.design_condition:.1e})")


def test_invalid_source_forces_abstention():
    cfg = SimConfig(seed=9)
    src = make_source(cfg, n_domains=6, concept_domains=2, seed=9)
    Y = src.Y.copy()
    Y[src.D == 0] = Y[src.D == 0][0]
    sa = analyze_source(src.Z, Y, src.D, n_boot=8, n_dir_boot=8, seed=9)
    assert sa.test.status == "INVALID_SUPPORT"           # CSC-P1.4.3 #2 specific status
    assert sa.source_status == "INVALID_SUPPORT"
    tb = make_target("covariate", cfg, geom=src.geom, seed=99)
    cert = certify(sa, tb.Z)
    assert cert.state == UNIDENTIFIABLE
    print("OK invalid support -> certifier ABSTAINS:", cert.reason[:55])


if __name__ == "__main__":
    test_single_class_domain_is_invalid()
    test_disconnected_graph_is_invalid()
    test_valid_source_passes()
    test_invalid_source_forces_abstention()
    print("\nall validity-gate tests passed")
