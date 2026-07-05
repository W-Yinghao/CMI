"""Guard (Stage-2B2): the whiten-color transport operator gain is SVD-capped at TRANSPORT_OPERATOR_SMAX, and the cap is
demonstrably active when the raw operator would exceed it. Torch-free. Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_stable_coral as SC
from acar.v5.tests._util import ok, stage2b_synthetic_source_state, stage2b_scaled_source_state, stage2b_rank_deficient_batch


def test_operator_gain_capped():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=256, seed=1))
    for rank, seed in ((1, 0), (5, 3), (31, 5)):
        op = SC.transport_operator(lda, stage2b_rank_deficient_batch(n=32, D=256, rank=rank, seed=seed))
        assert op["M_smax"] <= SC.TRANSPORT_OPERATOR_SMAX + 1e-9
    ok(f"capped operator gain M_smax ≤ {SC.TRANSPORT_OPERATOR_SMAX} on rank-deficient batches")


def test_cap_is_active_when_raw_exceeds():
    # large source covariance -> raw operator gain > smax -> the cap actively reduces it to exactly smax
    lda = AR.SourceLDA(stage2b_scaled_source_state(D=64, scale=1e6, seed=2))
    op = SC.transport_operator(lda, stage2b_rank_deficient_batch(n=32, D=64, rank=5, seed=4))
    assert op["M_raw_smax"] > SC.TRANSPORT_OPERATOR_SMAX
    assert abs(op["M_smax"] - SC.TRANSPORT_OPERATOR_SMAX) <= 1e-6
    ok("when M_raw_smax > smax, the SVD cap clamps M_smax to exactly smax")


def main():
    print("ACAR v5 Stage-2B2 guard: transport operator norm capped")
    test_operator_gain_capped()
    test_cap_is_active_when_raw_exceeds()
    print("ALL V5 STAGE2B2-STABLE-CORAL-OPERATOR-CAP GUARDS PASS")


if __name__ == "__main__":
    main()
