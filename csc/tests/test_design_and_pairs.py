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
from csc.certificate import analyze_source, certify, check_support_graph
from csc.certificate.residual_test import _features, _standardise
from csc.certificate.atlas import _orthonormal_complement


def test_orthonormal_complement_of_self_is_empty():
    """The CSC-P1.1 bug fix: complement(A, A) must be ZERO columns (the v0 QR-norm test kept
    spurious unit-norm columns), and complement(A, B) must be orthogonal to B."""
    rng = np.random.default_rng(0)
    d = 10
    A, _ = np.linalg.qr(rng.standard_normal((d, 3)))
    A = A[:, :3]
    comp_self = _orthonormal_complement(A, A)
    assert comp_self.shape == (d, 0), f"complement(A,A) must be empty, got {comp_self.shape}"
    B, _ = np.linalg.qr(rng.standard_normal((d, 4)))
    B = B[:, :4]
    comp = _orthonormal_complement(B, A)
    assert comp.shape[1] >= 1
    leak = float(np.abs(A.T @ comp).max())
    assert leak < 1e-8, f"complement must be orthogonal to `against`, leak={leak}"
    print(f"OK complement(A,A)->0 cols; complement(B,A)_|_A (max leak {leak:.1e})")


def test_rank_gate_rejects_duplicate_feature():
    """A duplicated Z feature makes the interaction design rank-deficient; the gate must
    reject it even though connectivity + cell counts pass (the v0 condition-number gate
    dropped exact-zero singular values and passed it)."""
    src = make_source(SimConfig(seed=13), n_domains=4, concept_domains=2, seed=13)
    Zdup = np.concatenate([src.Z, src.Z[:, :1]], axis=1)   # exact duplicate column
    sg = check_support_graph(src.Y, src.D, Z=Zdup)
    assert sg.connected and sg.min_classes_per_domain >= 2, "graph degree/connectivity ok"
    assert not sg.full_rank and not sg.valid, \
        f"duplicate feature must fail the rank gate (rank {sg.design_rank}/{sg.design_ncols})"
    print(f"OK duplicate feature -> rank gate INVALID "
          f"(rank {sg.design_rank} < {sg.design_ncols} cols)")


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
    test_orthonormal_complement_of_self_is_empty()
    test_rank_gate_rejects_duplicate_feature()
    test_reference_coding_is_full_rank()
    test_paired_clean_pure_identical_certificate()
    print("\nall design/pairs tests passed")
