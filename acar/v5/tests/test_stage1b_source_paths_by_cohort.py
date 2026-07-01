"""Guard (Stage-1B1): full-build real DEV inputs are source_paths_by_cohort — keys == the disease's frozen DEV cohorts, each path
cohort-exact. Synthetic; string checks only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import stage1b_full_build_manifest as FBM
from acar.v5.substrate import stage1b_manifest as MAN
from acar.v5.tests._util import expect_raises, ok


def _pd_paths():
    return {c: f"/projects/dl/{c}/sub-1" for c in P.DEV_COHORTS["PD"]}


def test_valid_mapping_ok():
    assert FBM.validate_source_paths_by_cohort("PD", _pd_paths())
    assert FBM.validate_source_paths_by_cohort("SCZ", {c: f"/projects/dl/{c}/sub-1" for c in P.DEV_COHORTS["SCZ"]})
    ok("source_paths_by_cohort with all disease cohorts, cohort-exact paths → valid")


def test_missing_or_extra_keys_rejected():
    m = _pd_paths()
    m.pop(P.DEV_COHORTS["PD"][0])
    expect_raises(FBM.Stage1bFullBuildError, lambda: FBM.validate_source_paths_by_cohort("PD", m), "missing cohort key")
    m2 = _pd_paths()
    m2["ds003944"] = "/projects/dl/ds003944/sub-1"             # SCZ cohort key in a PD mapping
    expect_raises(FBM.Stage1bFullBuildError, lambda: FBM.validate_source_paths_by_cohort("PD", m2), "foreign cohort key")
    ok("keys must equal the disease's frozen DEV cohorts (missing/extra → rejected)")


def test_cohort_exact_path():
    # a path filed under ds002778 that actually points at ds003490 → rejected
    m = _pd_paths()
    m["ds002778"] = "/projects/dl/ds003490/sub-1"
    expect_raises(FBM.Stage1bFullBuildError, lambda: FBM.validate_source_paths_by_cohort("PD", m), "wrong cohort in path")
    # a path that references two cohorts at once → rejected
    expect_raises(FBM.Stage1bFullBuildError, lambda: FBM.validate_cohort_source_path("PD", "ds002778", "/x/ds002778/ds004584/y"), "two cohorts")
    ok("each cohort path must reference EXACTLY its cohort (wrong/multiple cohorts → rejected)")


def test_site_and_artifact_paths_rejected():
    for bad in ("/data/ds007526/sub-1", "/data/zenodo14808296/sub-1",
                "/x/scps/cache/PD.npz", "/x/feat_dump_v4/audit.npz", "/home/acar_v4_regen_outputs/e.pt"):
        # fail-closed via the DEV whitelist (Stage1bWhitelistError) or the cohort-exact check (Stage1bFullBuildError)
        expect_raises((FBM.Stage1bFullBuildError, MAN.Stage1bWhitelistError),
                      lambda bad=bad: FBM.validate_cohort_source_path("PD", "ds002778", bad), bad)
    ok("external-site / scps-cache / v4-artifact paths → rejected even under a valid cohort key")


def main():
    print("ACAR v5 Stage-1B1 guard: source_paths_by_cohort")
    test_valid_mapping_ok()
    test_missing_or_extra_keys_rejected()
    test_cohort_exact_path()
    test_site_and_artifact_paths_rejected()
    print("ALL V5 STAGE1B-SOURCE-PATHS-BY-COHORT GUARDS PASS")


if __name__ == "__main__":
    main()
