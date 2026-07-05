"""Guard (Stage-2B2): stable_matched_coral_v1 satisfies the action-output contract — p_a [n,2] finite in [0,1] with rows summing
to 1, and a finite [n,D] z_post (matched_coral is geometric, not probability-only). Torch-free. Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_stable_coral as SC
from acar.v5 import stage2_action_provider_validation as V
from acar.v5 import stage2_real_action_provider as RAP
from acar.v5.tests._util import ok, stage2b_synthetic_source_state, stage2b_rank_deficient_batch


def test_probability_and_shape_contract():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=256, seed=2))
    for rank, seed in ((3, 0), (16, 5), (31, 9)):
        Z = stage2b_rank_deficient_batch(n=32, D=256, rank=rank, seed=seed)
        pa, zp = SC.stable_matched_coral_v1(lda, Z)
        assert pa.shape == (32, 2)
        assert np.isfinite(pa).all() and (pa >= -1e-9).all() and (pa <= 1 + 1e-9).all()
        assert np.allclose(pa.sum(1), 1.0, atol=1e-6)
        assert zp is not None and zp.shape == (32, 256) and np.isfinite(zp).all()
        V.validate_action_output("matched_coral", pa, zp, 32)                 # passes the shared output validator
        # and via the Stage-2 provider
        pa2, zp2 = RAP.real_action_provider("matched_coral", lda, Z)
        assert np.array_equal(pa, pa2) and np.array_equal(zp, zp2)
    ok("stable matched_coral: p_a [32,2] finite in [0,1] row-sum 1; finite [32,256] z_post; validate_action_output OK")


def main():
    print("ACAR v5 Stage-2B2 guard: stable CORAL shape + probability contract")
    test_probability_and_shape_contract()
    print("ALL V5 STAGE2B2-STABLE-CORAL-CONTRACT GUARDS PASS")


if __name__ == "__main__":
    main()
