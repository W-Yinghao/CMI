"""Guard (Stage-1B1): a REAL Stage-1B full build must construct ALL 30 fold substrates — the gate rejects default/partial plans
and requires exactly 30 admitted refs. Synthetic only; opens nothing."""
from __future__ import annotations
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import stage1_runtime_lock as RL
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_full_build_manifest as FBM
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def _auth():
    return stage1b_auth(protocol_tag_target_sha=FULL)


def _lock():
    return stage1b_lock(protocol_tag_target_sha=FULL)


def test_full_build_ready_all_30():
    rep = RL.require_stage1b_full_build_ready(stage1b_full_plan(), _auth(), _lock())
    assert rep["status"] == "STAGE1B_FULL_BUILD_READY" and rep["built_fold_substrates"] == 30
    ok("full plan + full-40 auth + matching lock → STAGE1B_FULL_BUILD_READY, 30 substrates (no data read)")


def test_prefix_target_sha_rejected_for_full_build():
    pl = stage1b_full_plan()
    expect_raises(SA.Stage1BuildNotAuthorizedError,
                  lambda: RL.require_stage1b_full_build_ready(pl, stage1b_auth(protocol_tag_target_sha="4278435"), stage1b_lock(protocol_tag_target_sha="4278435")))
    ok("a full build requires the FULL 40-hex target sha (prefix rejected)")


def test_default_plan_rejected_for_full_build():
    # default plan has no source_paths_by_cohort → not a full build
    expect_raises(FBM.Stage1bFullBuildError, lambda: RL.require_stage1b_full_build_ready(PLAN.build_substrate_plan(), _auth(), _lock()))
    ok("default (plan-only) spec → Stage1bFullBuildError (0-real-ref plan is not a full build)")


def test_partial_plan_rejected():
    pl = stage1b_full_plan()
    pl["fold_contained_refs"] = pl["fold_contained_refs"][:-1]   # drop one → 29
    # rejected fail-closed: schema count-check (ValueError) fires first; else the full-build present-set check (Stage1bFullBuildError)
    expect_raises((ValueError, FBM.Stage1bFullBuildError), lambda: RL.require_stage1b_full_build_ready(pl, _auth(), _lock()))
    ok("a partial plan (29 of 30 refs) → rejected (missing canonical fold ref)")


def test_scalar_source_path_rejected_in_full_build():
    pl = stage1b_full_plan()
    pl["fold_contained_refs"][0] = dict(pl["fold_contained_refs"][0], source_path="/projects/dl/ds002778/sub-1")
    expect_raises(FBM.Stage1bFullBuildError, lambda: RL.require_stage1b_full_build_ready(pl, _auth(), _lock()))
    ok("a fold ref carrying a scalar source_path (not source_paths_by_cohort) → rejected in full build")


def main():
    print("ACAR v5 Stage-1B1 guard: full build requires 30 refs")
    test_full_build_ready_all_30()
    test_prefix_target_sha_rejected_for_full_build()
    test_default_plan_rejected_for_full_build()
    test_partial_plan_rejected()
    test_scalar_source_path_rejected_in_full_build()
    print("ALL V5 STAGE1B-FULL-BUILD-30-REFS GUARDS PASS")


if __name__ == "__main__":
    main()
