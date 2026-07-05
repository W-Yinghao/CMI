"""Guard (Stage-2B2): stable_matched_coral_v1 is deterministic — repeated calls on the same input are byte-identical (no
randomness, unlike the intermittent frozen path). Torch-free. Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_stable_coral as SC
from acar.v5 import stage2_real_action_provider as RAP
from acar.v5.tests._util import ok, stage2b_synthetic_source_state, stage2b_rank_deficient_batch


def test_repeated_calls_bit_identical():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=256, seed=5))
    Z = stage2b_rank_deficient_batch(n=32, D=256, rank=5, seed=2)
    pa1, zp1 = SC.stable_matched_coral_v1(lda, Z)
    for _ in range(5):
        pa, zp = SC.stable_matched_coral_v1(lda, Z)
        assert np.array_equal(pa, pa1) and np.array_equal(zp, zp1)
    # via the provider too
    pa2, zp2 = RAP.real_action_provider("matched_coral", lda, Z)
    assert np.array_equal(pa1, pa2) and np.array_equal(zp1, zp2)
    # source references no randomness
    import inspect
    src = inspect.getsource(SC)
    for tok in ("random", "randn", "RandomState", "np.random", "seed("):
        assert tok not in src, f"stable CORAL references randomness: {tok!r}"
    ok("stable_matched_coral_v1 is byte-deterministic across repeated calls; no randomness in the source")


def main():
    print("ACAR v5 Stage-2B2 guard: stable CORAL deterministic")
    test_repeated_calls_bit_identical()
    print("ALL V5 STAGE2B2-STABLE-CORAL-DETERMINISTIC GUARDS PASS")


if __name__ == "__main__":
    main()
