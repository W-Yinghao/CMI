"""Guard (Stage-2B1): the real action provider routes matched_coral/spdim/t3a through the FROZEN acar.actions.apply_action (with
the v5→old source-state adapter), while identity is the source-state LDA. Torch-free (apply_action is monkeypatched). Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import protocol as P
from acar.v5 import stage2_action_records as AR
from acar.v5.tests._util import ok, stage2b_synthetic_source_state


def test_non_identity_actions_call_apply_action():
    import acar.actions as A
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=6, seed=1))
    Z = np.random.RandomState(0).randn(10, 6)
    calls = []
    orig = A.apply_action

    def _rec(name, state, z):
        calls.append((name, state is lda.old_state))
        n = np.asarray(z).shape[0]
        return np.full((n, 2), 0.5), (None if name == "t3a" else np.asarray(z, float))

    A.apply_action = _rec
    try:
        for a in P.ACTIONS:                                       # matched_coral, spdim, t3a — all go through apply_action
            AR.production_action_provider(a, lda, Z)
        pid, zid = AR.production_action_provider("identity", lda, Z)   # identity is the LDA, NOT apply_action
    finally:
        A.apply_action = orig
    assert [c[0] for c in calls] == list(P.ACTIONS)               # exactly the 3 non-identity actions, in order
    assert all(c[1] for c in calls)                              # each called with the v5→old adapter state
    assert np.allclose(pid, lda.predict_proba(Z)) and np.allclose(zid, Z)   # identity bypassed apply_action
    ok("matched_coral/spdim/t3a route through frozen acar.actions.apply_action(adapter_state, z); identity = LDA f_0")


def main():
    print("ACAR v5 Stage-2B1 guard: real actions call frozen acar.actions")
    test_non_identity_actions_call_apply_action()
    print("ALL V5 STAGE2B1-REAL-ACTIONS-CALL-ACAR-ACTIONS GUARDS PASS")


if __name__ == "__main__":
    main()
