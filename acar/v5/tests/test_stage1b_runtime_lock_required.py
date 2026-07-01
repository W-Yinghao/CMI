"""Guard (Stage-1B0): real Stage-1B readiness requires BOTH the structured authorization AND a matching runtime lock (plus the
DEV whitelist). Synthetic only — validates contracts/strings; opens nothing."""
from __future__ import annotations
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import stage1_runtime_lock as RL
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock


def _plan_with_dev_path():
    pl = PLAN.build_substrate_plan()
    r0 = pl["fold_contained_refs"][0]                          # a PD fold ref (sorted → PD first)
    pl["fold_contained_refs"][0] = dict(r0, source_path=f"/projects/datalake/{PLAN.build_substrate_plan()['dev_cohorts'][r0['disease']][0]}/sub-1")
    return pl


def test_ready_requires_auth_and_lock():
    pl = _plan_with_dev_path()
    rep = RL.require_stage1b_ready(pl, stage1b_auth(), stage1b_lock())
    assert rep["status"] == "STAGE1B_READY" and rep["admitted_fold_refs"] == 1
    ok("valid auth + valid lock + whitelisted DEV path → STAGE1B_READY (1 admitted fold ref; no data read)")


def test_missing_or_bad_lock_rejected():
    pl = _plan_with_dev_path()
    expect_raises(RL.Stage1RuntimeLockError, lambda: RL.require_stage1b_ready(pl, stage1b_auth(), None), "no lock")
    expect_raises(RL.Stage1RuntimeLockError, lambda: RL.require_stage1b_ready(pl, stage1b_auth(), stage1b_lock(status="PENDING")), "unverified lock")
    expect_raises(RL.Stage1RuntimeLockError, lambda: RL.require_stage1b_ready(pl, stage1b_auth(), stage1b_lock(stage="Stage-1A")), "wrong stage")
    ok("missing / unverified / wrong-stage runtime lock → Stage1RuntimeLockError")


def test_lock_must_match_auth():
    pl = _plan_with_dev_path()
    # run_id mismatch between auth and lock
    expect_raises(RL.Stage1RuntimeLockError,
                  lambda: RL.require_stage1b_ready(pl, stage1b_auth(run_id="A"), stage1b_lock(run_id="B")))
    # target sha mismatch
    expect_raises(RL.Stage1RuntimeLockError,
                  lambda: RL.require_stage1b_ready(pl, stage1b_auth(), stage1b_lock(protocol_tag_target_sha="deadbeef")))
    ok("runtime lock must be cross-bound to the authorization (run_id + target sha)")


def test_bad_auth_rejected_before_lock():
    pl = _plan_with_dev_path()
    expect_raises(SA.Stage1BuildNotAuthorizedError,
                  lambda: RL.require_stage1b_ready(pl, stage1b_auth(stage="Stage-1A"), stage1b_lock()))
    ok("a malformed authorization is rejected (before/independent of the lock)")


def main():
    print("ACAR v5 Stage-1B0 guard: runtime lock required")
    test_ready_requires_auth_and_lock()
    test_missing_or_bad_lock_rejected()
    test_lock_must_match_auth()
    test_bad_auth_rejected_before_lock()
    print("ALL V5 STAGE1B-RUNTIME-LOCK-REQUIRED GUARDS PASS")


if __name__ == "__main__":
    main()
