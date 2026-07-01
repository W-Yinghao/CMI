"""Guard (Stage-1B1): the runtime lock must include AND cross-bind implementation_base_sha to the authorization. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import stage1_runtime_lock as RL
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def test_impl_sha_in_lock_fields():
    assert "implementation_base_sha" in RL.RUNTIME_LOCK_FIELDS
    ok("implementation_base_sha is a required runtime-lock field")


def test_matching_impl_ok():
    RL.validate_runtime_lock(stage1b_lock(implementation_base_sha="a" * 40), stage1b_auth(implementation_base_sha="a" * 40))
    ok("lock.implementation_base_sha == auth.implementation_base_sha → validates")


def test_mismatched_impl_rejected():
    expect_raises(RL.Stage1RuntimeLockError,
                  lambda: RL.validate_runtime_lock(stage1b_lock(implementation_base_sha="b" * 40), stage1b_auth(implementation_base_sha="a" * 40)))
    ok("lock.implementation_base_sha != auth.implementation_base_sha → Stage1RuntimeLockError")


def test_missing_impl_in_lock_rejected():
    lk = stage1b_lock()
    del lk["implementation_base_sha"]
    expect_raises(RL.Stage1RuntimeLockError, lambda: RL.validate_runtime_lock(lk, stage1b_auth()))
    ok("a runtime lock missing implementation_base_sha → rejected")


def test_full_build_gate_enforces_impl_binding():
    pl = stage1b_full_plan()
    a = stage1b_auth(protocol_tag_target_sha=FULL, implementation_base_sha="c" * 40)
    bad_lock = stage1b_lock(protocol_tag_target_sha=FULL, implementation_base_sha="d" * 40)
    expect_raises(RL.Stage1RuntimeLockError, lambda: RL.require_stage1b_full_build_ready(pl, a, bad_lock))
    good_lock = stage1b_lock(protocol_tag_target_sha=FULL, implementation_base_sha="c" * 40)
    assert RL.require_stage1b_full_build_ready(pl, a, good_lock)["status"] == "STAGE1B_FULL_BUILD_READY"
    ok("full-build gate: lock must cross-bind implementation_base_sha to the authorization")


def main():
    print("ACAR v5 Stage-1B1 guard: runtime lock binds implementation sha")
    test_impl_sha_in_lock_fields()
    test_matching_impl_ok()
    test_mismatched_impl_rejected()
    test_missing_impl_in_lock_rejected()
    test_full_build_gate_enforces_impl_binding()
    print("ALL V5 STAGE1B-RUNTIME-LOCK-IMPL-SHA GUARDS PASS")


if __name__ == "__main__":
    main()
