"""Guard (Stage-2B2): stable_matched_coral_v1 has NO silent identity fallback — if the bounded operator still yields a non-finite
output, it raises Stage2StableCoralError (rather than returning identity or a NaN). Torch-free. Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_stable_coral as SC
from acar.v5.tests._util import expect_raises, ok, stage2b_synthetic_source_state, stage2b_rank_deficient_batch


def test_non_finite_readout_fails_closed():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=64, seed=1))
    Z = stage2b_rank_deficient_batch(n=32, D=64, rank=5, seed=0)
    p_ref, z_ref = SC.stable_matched_coral_v1(lda, Z)                          # normally finite
    assert np.isfinite(p_ref).all()
    # force a non-finite readout: the fail-closed path must RAISE, not silently return identity / NaN
    lda.predict_proba = lambda z: np.full((np.asarray(z).shape[0], 2), np.nan)
    expect_raises(SC.Stage2StableCoralError, lambda: SC.stable_matched_coral_v1(lda, Z))
    ok("a non-finite f_0 readout → Stage2StableCoralError (no silent identity fallback)")


def main():
    print("ACAR v5 Stage-2B2 guard: no silent identity fallback")
    test_non_finite_readout_fails_closed()
    print("ALL V5 STAGE2B2-STABLE-CORAL-NO-FALLBACK GUARDS PASS")


if __name__ == "__main__":
    main()
