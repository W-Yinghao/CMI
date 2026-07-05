"""Guard (Stage-2B2): the Stage-2B2 amendment did not touch the authorization gate — the selection engine still fails closed
without a valid Stage-2B auth bound to the admitted package, and the amended provider routes matched_coral through the stable
operator. Synthetic (torch-free). Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_stable_coral as SC
from acar.v5 import stage2_real_action_provider as RAP
from acar.v5 import stage2b_authorization as AUTH
from acar.v5 import stage2_selection_engine as ENG
from acar.v5.tests._util import expect_raises, ok, stage2b_auth, stage2b_disease_inputs

SYN = AR.synthetic_action_provider


def test_engine_gate_intact_after_amendment():
    a = stage2b_auth()
    di = stage2b_disease_inputs(seed=4)
    expect_raises(AUTH.Stage2bAuthorizationError, lambda: ENG.run_selection(
        stage2b_auth(statement="nope"), stage1b_run_id=a["stage1b_run_id"],
        stage1b_registry_sha256=a["stage1b_registry_sha256"], disease_inputs=di,
        action_provider=SYN, v2_replay_provider=lambda d, c: -0.5))
    expect_raises(AUTH.Stage2bAuthorizationError, lambda: ENG.run_selection(
        a, stage1b_run_id=a["stage1b_run_id"], stage1b_registry_sha256="b" * 64, disease_inputs=di,
        action_provider=SYN, v2_replay_provider=lambda d, c: -0.5))
    rep = ENG.run_selection(a, stage1b_run_id=a["stage1b_run_id"], stage1b_registry_sha256=a["stage1b_registry_sha256"],
                            disease_inputs=di, action_provider=SYN, v2_replay_provider=lambda d, c: -0.5)
    assert rep["outcome"] in ("SELECTED", "DEV_STOP")
    ok("engine still fails closed on invalid/unbound Stage-2B auth after the stable-CORAL amendment")


def test_amended_provider_routes_matched_coral_to_stable():
    lda = AR.SourceLDA({"means": np.random.RandomState(1).randn(2, 32) * 0.5, "cov": np.eye(32) + 0.02,
                        "priors": np.array([0.5, 0.5]), "classes": np.array([0, 1])})
    Z = np.random.RandomState(0).randn(16, 32)
    pa_prov, zp_prov = RAP.real_action_provider("matched_coral", lda, Z)
    pa_stable, zp_stable = SC.stable_matched_coral_v1(lda, Z)
    assert np.array_equal(pa_prov, pa_stable) and np.array_equal(zp_prov, zp_stable)
    ok("real_action_provider('matched_coral') == stable_matched_coral_v1 (amended routing wired)")


def main():
    print("ACAR v5 Stage-2B2 guard: real runner still requires Stage-2B auth")
    test_engine_gate_intact_after_amendment()
    test_amended_provider_routes_matched_coral_to_stable()
    print("ALL V5 STAGE2B2-REAL-RUNNER-REQUIRES-AUTH GUARDS PASS")


if __name__ == "__main__":
    main()
