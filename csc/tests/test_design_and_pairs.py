"""
Two structural guarantees from the CSC-P0 rewrite:

1. REFERENCE CODING removes the deterministic rank deficiency the review found (h0: 1,
   h: 1+d under full one-hot). The reference-coded designs must be FULL COLUMN RANK.

2. The IMPOSSIBILITY RESULT, operationalised: clean and pure-conditional targets that share
   BYTE-IDENTICAL Z must receive the IDENTICAL certificate (the certifier sees Z only).
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from csc.sim.shift_simulator import SimConfig, make_source, make_paired_clean_pure
from csc.certificate import analyze_source, certify
from csc.certificate.residual_test import _features, _standardise


def test_reference_coding_is_full_rank():
    src = make_source(SimConfig(seed=11), n_domains=4, concept_domains=2, seed=11)
    Zs = _standardise(src.Z)
    domains = list(np.unique(src.D))
    for interaction in (False, True):
        X = _features(Zs, src.D, domains, interaction)
        # add the implicit intercept column the LR fits, then check full rank
        Xi = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        r = np.linalg.matrix_rank(Xi)
        assert r == Xi.shape[1], \
            f"design (interaction={interaction}) rank {r} != cols {Xi.shape[1]}"
        print(f"OK reference-coded design (interaction={interaction}): "
              f"full rank {r}/{Xi.shape[1]}")


def test_paired_clean_pure_identical_certificate():
    cfg = SimConfig(seed=12)
    src = make_source(cfg, n_domains=8, concept_domains=3, seed=12)
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=20, n_dir_boot=50, seed=12)
    clean, pure = make_paired_clean_pure(cfg, geom=src.geom, seed=120)
    assert np.array_equal(clean.Z, pure.Z), "paired targets must share identical Z"
    c1 = certify(sa, clean.Z)
    c2 = certify(sa, pure.Z)
    assert c1.state == c2.state, \
        f"Z-only certifier must agree on identical Z: {c1.state} vs {c2.state}"
    print(f"OK identical Z -> identical certificate ({c1.state}) for clean & pure-conditional")


if __name__ == "__main__":
    test_reference_coding_is_full_rank()
    test_paired_clean_pure_identical_certificate()
    print("\nall design/pairs tests passed")
