"""Guard (Stage-2B0): the selection engine cannot run without a valid Stage-2B authorization bound to the admitted package, and
it fails closed (DEV_STOP) when the v2-replay comparator is not evaluable. With the default (real, unwired) v2-replay seam it
stops; no real selection is produced. Synthetic only (torch/sklearn-free)."""
from __future__ import annotations
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2b_authorization as AUTH
from acar.v5 import stage2_selection_engine as ENG
from acar.v5.tests._util import expect_raises, ok, stage2b_auth, stage2b_disease_inputs


def test_invalid_or_unbound_auth_raises():
    a = stage2b_auth()
    di = stage2b_disease_inputs()
    # invalid auth (bad statement)
    expect_raises(AUTH.Stage2bAuthorizationError, lambda: ENG.run_selection(
        stage2b_auth(statement="nope"), stage1b_run_id=a["stage1b_run_id"],
        stage1b_registry_sha256=a["stage1b_registry_sha256"], disease_inputs=di,
        action_provider=AR.synthetic_action_provider, v2_replay_provider=lambda d, ctx: -0.5))
    # package-binding mismatch (registry sha)
    expect_raises(AUTH.Stage2bAuthorizationError, lambda: ENG.run_selection(
        a, stage1b_run_id=a["stage1b_run_id"], stage1b_registry_sha256="b" * 64, disease_inputs=di,
        action_provider=AR.synthetic_action_provider, v2_replay_provider=lambda d, ctx: -0.5))
    ok("engine raises Stage2bAuthorizationError on an invalid auth or a package-binding mismatch")


def test_valid_auth_runs():
    a = stage2b_auth()
    rep = ENG.run_selection(a, stage1b_run_id=a["stage1b_run_id"], stage1b_registry_sha256=a["stage1b_registry_sha256"],
                            disease_inputs=stage2b_disease_inputs(), action_provider=AR.synthetic_action_provider,
                            v2_replay_provider=lambda d, ctx: -0.5)
    assert rep["outcome"] in ("SELECTED", "DEV_STOP")
    ok("a valid, bound Stage-2B auth runs the engine (synthetic → a valid report)")


def test_v2_replay_fail_closed_by_default():
    a = stage2b_auth()
    # omit v2_replay_provider → the DEFAULT real (unwired) seam raises V2ReplayNotEvaluable → DEV_STOP
    rep = ENG.run_selection(a, stage1b_run_id=a["stage1b_run_id"], stage1b_registry_sha256=a["stage1b_registry_sha256"],
                            disease_inputs=stage2b_disease_inputs(), action_provider=AR.synthetic_action_provider)
    assert rep["outcome"] == "DEV_STOP"
    assert "v2_replay" in rep["notes"]["dev_stop_reason"]
    ok("the default (unwired) v2-replay comparator → DEV_STOP 'v2_replay not evaluable' (real selection cannot run)")


def main():
    print("ACAR v5 Stage-2B0 guard: binding real run requires a valid auth; v2-replay fail-closed")
    test_invalid_or_unbound_auth_raises()
    test_valid_auth_runs()
    test_v2_replay_fail_closed_by_default()
    print("ALL V5 STAGE2B0-BINDING-REQUIRES-AUTH GUARDS PASS")


if __name__ == "__main__":
    main()
