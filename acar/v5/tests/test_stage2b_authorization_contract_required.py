"""Guard (Stage-2B0): the structured Stage-2B authorization is required + fail-closed, and must bind the admitted package
(run_id + registry_sha256). Synthetic only (no real authorization issued)."""
from __future__ import annotations
from acar.v5 import stage2b_authorization as AUTH
from acar.v5.tests._util import expect_raises, ok, stage2b_auth


def test_valid_auth_passes():
    assert AUTH.validate_stage2b_authorization(stage2b_auth()) is True
    ok("a fully-formed Stage-2B authorization validates")


def test_field_and_value_violations_fail():
    a = stage2b_auth()
    del a["statement"]
    expect_raises(AUTH.Stage2bAuthorizationError, lambda: AUTH.validate_stage2b_authorization(a))          # missing field
    expect_raises(AUTH.Stage2bAuthorizationError, lambda: AUTH.validate_stage2b_authorization(stage2b_auth(extra="x")))  # extra
    for over in ({"stage": "Stage-1B"}, {"protocol_tag": "x"}, {"protocol_tag_target_sha": "4278435"},
                 {"statement": "nope"}, {"implementation_base_sha": "z" * 40}, {"stage1b_registry_sha256": "x" * 63}):
        expect_raises(AUTH.Stage2bAuthorizationError, lambda o=over: AUTH.validate_stage2b_authorization(stage2b_auth(**o)))
    for f in ("forbid_s1_refs_for_selection", "forbid_external_read", "forbid_lockbox"):
        expect_raises(AUTH.Stage2bAuthorizationError, lambda f=f: AUTH.validate_stage2b_authorization(stage2b_auth(**{f: False})))
    ok("missing/extra field, wrong stage/tag/target/statement/sha, or a forbid-flag != True → Stage2bAuthorizationError")


def test_require_ready_binds_package():
    a = stage2b_auth()
    assert AUTH.require_stage2b_ready(a, stage1b_run_id=a["stage1b_run_id"],
                                     stage1b_registry_sha256=a["stage1b_registry_sha256"]) is True
    expect_raises(AUTH.Stage2bAuthorizationError, lambda: AUTH.require_stage2b_ready(
        a, stage1b_run_id="other-run", stage1b_registry_sha256=a["stage1b_registry_sha256"]))
    expect_raises(AUTH.Stage2bAuthorizationError, lambda: AUTH.require_stage2b_ready(
        a, stage1b_run_id=a["stage1b_run_id"], stage1b_registry_sha256="b" * 64))
    ok("require_stage2b_ready binds the auth to the admitted run_id + registry_sha256 (fail-closed on mismatch)")


def main():
    print("ACAR v5 Stage-2B0 guard: authorization contract required")
    test_valid_auth_passes()
    test_field_and_value_violations_fail()
    test_require_ready_binds_package()
    print("ALL V5 STAGE2B0-AUTH-CONTRACT GUARDS PASS")


if __name__ == "__main__":
    main()
