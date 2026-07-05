"""Guard (Stage-2B1): every real action returns p_a of shape [n,2], finite, in [0,1], rows summing to 1, with class order [0,1].
identity/matched_coral/t3a run on both Pythons; spdim's real transform needs torch (validated on py3.13, skipped on py3.9)."""
from __future__ import annotations
import numpy as np
from acar.v5 import protocol as P
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_action_provider_validation as V
from acar.v5 import stage2_real_action_provider as RAP
from acar.v5.tests._util import ok, has_torch, stage2b_synthetic_source_state


def test_real_action_probability_contract():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=6, seed=4))
    Z = np.random.RandomState(2).randn(25, 6)
    actions = ("identity", "matched_coral", "t3a") + (("spdim",) if has_torch() else ())
    rep = RAP.probe_real_actions(lda, Z, actions=actions)          # validates output contract for each probed action
    for a in actions:
        pa, z_post = RAP.validated_real_action(a, lda, Z)
        assert pa.shape == (25, 2) and np.isfinite(pa).all() and np.allclose(pa.sum(1), 1.0)
        V.validate_action_output(a, pa, z_post, 25)
    if not has_torch():
        print("  [skip:no-torch] spdim real-transform functional check (validated on py3.13)")
    ok(f"real actions {actions} → p_a [n,2] finite in [0,1] row-sum 1 (class order [0,1])")


def main():
    print("ACAR v5 Stage-2B1 guard: real action probability shape + finiteness")
    test_real_action_probability_contract()
    print("ALL V5 STAGE2B1-REAL-ACTION-PROB-SHAPE GUARDS PASS")


if __name__ == "__main__":
    main()
