"""Guard (Stage-2B3): stable_matched_coral_v1 / transport_operator FAIL CLOSED on n<2 (target covariance undefined) — they raise
Stage2StableCoralError and NEVER return an identity-equivalent output (which would hide a forced-tail misuse). n>=2 still works.
Torch-free. Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_stable_coral as SC
from acar.v5 import stage2_action_records as AR
from acar.v5.tests._util import ok, expect_raises, stage2b_synthetic_source_state


def test_n1_direct_call_raises_and_n2_ok():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=8, seed=7))
    Z1 = np.random.RandomState(0).randn(1, 8)                  # single window -> covariance undefined
    expect_raises(SC.Stage2StableCoralError, lambda: SC.stable_matched_coral_v1(lda, Z1))
    expect_raises(SC.Stage2StableCoralError, lambda: SC.transport_operator(lda, Z1))
    # n=2 is the smallest valid batch: must NOT raise (guard is n<2, not n<something-larger) and returns a valid contract
    Z2 = np.random.RandomState(1).randn(2, 8)
    pa, zp = SC.stable_matched_coral_v1(lda, Z2)
    assert pa.shape == (2, 2) and np.isfinite(pa).all() and zp.shape == (2, 8) and np.isfinite(zp).all()
    ok("stable_matched_coral_v1 / transport_operator raise Stage2StableCoralError at n<2 (no identity fallback); n>=2 valid")


def main():
    print("ACAR v5 Stage-2B3 guard: stable_matched_coral n<2 direct call fails closed")
    test_n1_direct_call_raises_and_n2_ok()
    print("ALL V5 STAGE2B3-STABLE-CORAL-N1-FAILCLOSED GUARDS PASS")


if __name__ == "__main__":
    main()
