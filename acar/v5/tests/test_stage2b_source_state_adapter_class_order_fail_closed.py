"""Guard (Stage-2B1): class order must be [0,1] = {control,case}. SourceLDA construction rejects any other order, and the adapter
validator rejects a clf whose classes_ != [0,1]. Torch-free. Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_action_provider_validation as V
from acar.v5.tests._util import expect_raises, ok, stage2b_synthetic_source_state


class _BadClf:
    classes_ = np.array([1, 0])                                   # wrong order
    coef_ = np.zeros((1, 4))
    intercept_ = np.zeros(1)

    def predict_proba(self, z):
        return z

    def predict(self, z):
        return z


class _Adapter:
    def __init__(self, D, old_state):
        self.D = D
        self.old_state = old_state


def test_class_order_fail_closed():
    ss = stage2b_synthetic_source_state(D=4, seed=0)
    ss["classes"] = np.array([1, 0])
    expect_raises(AR.Stage2ActionError, lambda: AR.SourceLDA(ss))     # SourceLDA rejects at construction
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=4, seed=0))
    st = dict(lda.old_state)
    st["clf"] = _BadClf()
    expect_raises(V.Stage2ActionValidationError, lambda: V.validate_source_state_adapter(_Adapter(4, st)))
    ok("class order != [0,1] → SourceLDA and adapter validator both fail closed")


def main():
    print("ACAR v5 Stage-2B1 guard: source-state adapter class-order fail-closed")
    test_class_order_fail_closed()
    print("ALL V5 STAGE2B1-ADAPTER-CLASS-ORDER GUARDS PASS")


if __name__ == "__main__":
    main()
