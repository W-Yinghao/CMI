"""Guard (Stage-1B3): within a disease, all fold/seed refs must declare the IDENTICAL source_paths_by_cohort. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import stage1b_full_build_manifest as FBM
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def test_consistent_plan_ok():
    FBM.validate_source_paths_consistent(stage1b_full_plan())
    ok("a full plan with identical per-disease source_paths_by_cohort passes")


def _inconsistent_plan():
    pl = stage1b_full_plan()
    # change the PATH for one PD fold ref's first cohort → inconsistent within PD
    for e in pl["fold_contained_refs"]:
        if e["disease"] == "PD":
            c0 = sorted(e["source_paths_by_cohort"])[0]
            e2 = dict(e, source_paths_by_cohort=dict(e["source_paths_by_cohort"], **{c0: f"/projects/OTHER/{c0}/sub-Z"}))
            pl["fold_contained_refs"][pl["fold_contained_refs"].index(e)] = e2
            break
    return pl


def test_inconsistent_rejected_by_validator():
    expect_raises(FBM.Stage1bFullBuildError, lambda: FBM.validate_source_paths_consistent(_inconsistent_plan()))
    ok("a differing per-disease source_paths_by_cohort → Stage1bFullBuildError (validator)")


def test_inconsistent_rejected_by_build_gate():
    expect_raises(FBM.Stage1bFullBuildError,
                  lambda: B.run_stage1b_build(_inconsistent_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                                              dev_reader=FakeDevReader(), trainer=FakeTrainer()))
    ok("inconsistent per-disease paths → rejected by the full-build gate (before any read)")


def main():
    print("ACAR v5 Stage-1B3 guard: source paths consistent across refs")
    test_consistent_plan_ok()
    test_inconsistent_rejected_by_validator()
    test_inconsistent_rejected_by_build_gate()
    print("ALL V5 STAGE1B-SOURCE-PATHS-CONSISTENT GUARDS PASS")


if __name__ == "__main__":
    main()
