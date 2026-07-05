"""Guard (Stage-2B2): the conditioning is bounded so CORAL's (inv)sqrt cannot overflow on rank-deficient batches. TWO mechanisms:
(a) the shrink alone caps cond(Σ_shrunk) ≤ (1-rho)·D/rho + 1 (the PRIMARY conditioner at rho=0.1 / D=256), and (b) the eigenvalue
FLOOR caps cond ≤ CONDITION_NUMBER_CAP (a redundant safety net, tested directly since the shrink keeps it inert at D=256).
Torch-free. Synthetic. Doubles as the regression fixture vs the old near-singular path."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_stable_coral as SC
from acar.v5.tests._util import ok, stage2b_synthetic_source_state, stage2b_rank_deficient_batch


def test_shrink_bounds_condition_and_output_finite():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=256, seed=1))
    Z = stage2b_rank_deficient_batch(n=32, D=256, rank=5, seed=0)             # rank 5 << 256 -> raw cov singular
    raw_cond = SC._cond(np.cov(Z, rowvar=False))
    op = SC.transport_operator(lda, Z)
    assert raw_cond > SC.CONDITION_NUMBER_CAP                                 # OLD path: raw target cov near-singular
    # the shrink alone bounds cond(Σ_shrunk) at ~(1-rho)*D/rho + 1 (≈2305 at rho=0.1, D=256) — well under the cap
    shrink_bound = (1.0 - SC.RHO) * 256 / SC.RHO + 1.0
    assert op["cond_T_after_floor"] <= shrink_bound + 1e-6
    assert op["cond_T_after_floor"] <= SC.CONDITION_NUMBER_CAP
    pa, zp = SC.stable_matched_coral_v1(lda, Z)
    assert np.isfinite(pa).all() and np.isfinite(zp).all()                    # frozen path would be overflow-prone here
    ok("raw rank-deficient cov near-singular (cond>cap); shrink caps cond ≤ %.0f; stable output finite" % shrink_bound)


def test_eigenvalue_floor_actually_fires():
    # bypass the shrink: a spectrum with cond >> cap and a min eigenvalue below lam_max/cap, so the FLOOR must raise it
    C = np.diag([1e12, 1e3]).astype(float)
    w_out, _, lam_floor = SC._conditioned_eig(C)
    assert lam_floor == max(SC.EPS, 1e12 / SC.CONDITION_NUMBER_CAP)           # = 1e6
    assert float(w_out.min()) == lam_floor and float(w_out.min()) > 1e3       # floor RAISED the min from 1e3 to 1e6
    assert (float(w_out.max()) / float(w_out.min())) <= SC.CONDITION_NUMBER_CAP * (1 + 1e-9)   # cond capped at the cap
    # a spectrum already within the cap is left untouched (floor is a no-op)
    C2 = np.diag([10.0, 1.0]).astype(float)
    w2, _, floor2 = SC._conditioned_eig(C2)
    assert np.allclose(w2, np.array([1.0, 10.0])) and floor2 == SC.EPS
    ok("the eigenvalue floor fires when cond>>cap (min raised to lam_max/cap); no-op when already within the cap")


def main():
    print("ACAR v5 Stage-2B2 guard: condition number capped (shrink bound + floor mechanism)")
    test_shrink_bounds_condition_and_output_finite()
    test_eigenvalue_floor_actually_fires()
    print("ALL V5 STAGE2B2-STABLE-CORAL-COND-CAP GUARDS PASS")


if __name__ == "__main__":
    main()
