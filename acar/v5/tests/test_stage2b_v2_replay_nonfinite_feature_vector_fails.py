"""Guard (Stage-2B1): the v2-replay comparator fails closed (V2ReplayNotEvaluable) if a feature_vector is non-finite. Torch-free
(synthetic action provider; feature_vector monkeypatched to inject a NaN). Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_v2_replay as VR
from acar.v5.tests._util import expect_raises, ok, stage2b_disease_inputs


def test_nonfinite_feature_vector_fails():
    import acar.features as AF
    di = stage2b_disease_inputs(seed=2)
    orig = AF.feature_vector
    AF.feature_vector = lambda phi, ctx: np.array([np.nan] * 11)   # inject a non-finite 11-D vector
    try:
        expect_raises(VR.V2ReplayNotEvaluable,
                      lambda: VR.v2_replay_red_by_disease("PD", di["PD"]["folds"],
                                                          action_provider=AR.synthetic_action_provider))
    finally:
        AF.feature_vector = orig
    # sanity: restored → runs
    red = VR.v2_replay_red_by_disease("PD", di["PD"]["folds"], action_provider=AR.synthetic_action_provider)
    assert np.isfinite(red)
    ok("a non-finite v2 feature_vector → V2ReplayNotEvaluable (fail-closed)")


def main():
    print("ACAR v5 Stage-2B1 guard: v2-replay non-finite feature_vector fails closed")
    test_nonfinite_feature_vector_fails()
    print("ALL V5 STAGE2B1-V2REPLAY-NONFINITE GUARDS PASS")


if __name__ == "__main__":
    main()
