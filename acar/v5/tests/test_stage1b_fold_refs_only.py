"""Guard (Stage-1B0): Stage-1B admits ONLY fold-contained refs from the authorized set; a fold ref not in allowed_refs, or a
final-external ref, cannot be built. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import stage1b_manifest as MAN
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1_runtime_lock as RL
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock


def _pd_cohort():
    return PLAN.build_substrate_plan()["dev_cohorts"]["PD"][0]


def test_underspecified_allowed_refs_rejected():
    pl = PLAN.build_substrate_plan()
    pl["fold_contained_refs"][0] = dict(pl["fold_contained_refs"][0], source_path=f"/projects/dl/{_pd_cohort()}/sub-1")
    short = stage1b_auth(allowed_refs=[r["ref"] for r in PLAN.fold_refs()][1:])   # 29 (missing one) → not the exact 30
    expect_raises(SA.Stage1BuildNotAuthorizedError, lambda: RL.require_stage1b_ready(pl, short, stage1b_lock()))
    ok("an authorization whose allowed_refs != the exact 30 → rejected (can't under-specify the ref set)")


def test_manifest_rejects_ref_outside_allowed():
    # a valid auth (exact 30) but a plan fold ref whose ref string was tampered to one not in allowed set
    pl = PLAN.build_substrate_plan()
    tampered = dict(pl["fold_contained_refs"][0], ref="PD/fold9/seed20260711", source_path=f"/projects/dl/{_pd_cohort()}/sub-1")
    pl["fold_contained_refs"][0] = tampered
    # is_fold_ref rejects fold9 (out of range) at schema validation OR whitelist; require_stage1b_ready must raise
    expect_raises(Exception, lambda: RL.require_stage1b_ready(pl, stage1b_auth(), stage1b_lock()))
    ok("a fold ref outside the authorized/valid set cannot be admitted for build")


def test_final_external_cannot_build():
    pl = PLAN.build_substrate_plan()
    pl["final_external_refs"][0] = dict(pl["final_external_refs"][0], source_path=f"/projects/dl/{_pd_cohort()}/sub-1")
    expect_raises(MAN.Stage1bWhitelistError, lambda: RL.require_stage1b_ready(pl, stage1b_auth(), stage1b_lock()))
    ok("a final-external ref can never be built (schema-only), even with valid auth + lock")


def test_default_plan_admits_zero_real_refs():
    # default plan has no source_paths → require_stage1b_ready admits 0 real fold refs (pure validation, no read)
    rep = RL.require_stage1b_ready(PLAN.build_substrate_plan(), stage1b_auth(), stage1b_lock())
    assert rep["status"] == "STAGE1B_READY" and rep["admitted_fold_refs"] == 0
    ok("default (plan-only) build manifest → STAGE1B_READY with 0 admitted real refs (validation only)")


def main():
    print("ACAR v5 Stage-1B0 guard: fold refs only")
    test_underspecified_allowed_refs_rejected()
    test_manifest_rejects_ref_outside_allowed()
    test_final_external_cannot_build()
    test_default_plan_admits_zero_real_refs()
    print("ALL V5 STAGE1B-FOLD-REFS-ONLY GUARDS PASS")


if __name__ == "__main__":
    main()
