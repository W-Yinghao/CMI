"""Guard (Stage-2B2): stable_matched_coral_v1 stays finite on rank-deficient 32x256 batches (the regime that made the frozen CORAL
produce non-finite p_a). Torch-free. Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_stable_coral as SC
from acar.v5.tests._util import ok, stage2b_synthetic_source_state, stage2b_rank_deficient_batch


def test_finite_on_rank_deficient_batches():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=256, seed=1))
    for rank, seed in ((1, 0), (2, 3), (5, 7), (16, 9), (31, 11)):
        Z = stage2b_rank_deficient_batch(n=32, D=256, rank=rank, seed=seed)
        pa, zp = SC.stable_matched_coral_v1(lda, Z)
        assert pa.shape == (32, 2) and np.isfinite(pa).all() and np.allclose(pa.sum(1), 1.0)
        assert zp.shape == (32, 256) and np.isfinite(zp).all()
    # exactly-duplicated rows (rank 1)
    Z = np.tile(np.random.RandomState(2).randn(1, 256), (32, 1))
    pa, zp = SC.stable_matched_coral_v1(lda, Z)
    assert np.isfinite(pa).all() and np.isfinite(zp).all()
    ok("stable_matched_coral_v1 finite on rank-{1,2,5,16,31} and fully-duplicated 32x256 batches")


def main():
    print("ACAR v5 Stage-2B2 guard: stable CORAL finite on rank-deficient 32x256")
    test_finite_on_rank_deficient_batches()
    print("ALL V5 STAGE2B2-STABLE-CORAL-RANK-DEFICIENT GUARDS PASS")


if __name__ == "__main__":
    main()
