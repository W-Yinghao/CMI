"""Guard (Stage-2B1): the v5→old source-state adapter is validated fail-closed — every required old_state field, correct shapes,
invertible covariance. Torch-free. Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_action_provider_validation as V
from acar.v5.tests._util import expect_raises, ok, stage2b_synthetic_source_state


class _Adapter:
    def __init__(self, D, old_state):
        self.D = D
        self.old_state = old_state


def test_adapter_valid_then_fail_closed():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=5, seed=1))
    assert V.validate_source_state_adapter(lda) is True
    # each missing required field fails
    for f in V.REQUIRED_OLD_STATE_FIELDS:
        st = dict(lda.old_state)
        del st[f]
        expect_raises(V.Stage2ActionValidationError, lambda st=st: V.validate_source_state_adapter(_Adapter(lda.D, st)))
    # wrong mu_y shape
    st = dict(lda.old_state); st["mu_y"] = np.zeros((3, lda.D))
    expect_raises(V.Stage2ActionValidationError, lambda: V.validate_source_state_adapter(_Adapter(lda.D, st)))
    # non-invertible covariance
    st = dict(lda.old_state); st["Sig_pool0"] = np.zeros((lda.D, lda.D))
    expect_raises(V.Stage2ActionValidationError, lambda: V.validate_source_state_adapter(_Adapter(lda.D, st)))
    # Sig_y0 not a length-2 list
    st = dict(lda.old_state); st["Sig_y0"] = lda.old_state["Sig_pool0"]
    expect_raises(V.Stage2ActionValidationError, lambda: V.validate_source_state_adapter(_Adapter(lda.D, st)))
    ok("adapter validates; a missing field / wrong shape / singular cov / bad Sig_y0 → Stage2ActionValidationError")


def main():
    print("ACAR v5 Stage-2B1 guard: source-state adapter required fields")
    test_adapter_valid_then_fail_closed()
    print("ALL V5 STAGE2B1-ADAPTER-FIELDS GUARDS PASS")


if __name__ == "__main__":
    main()
