"""Guard (Stage-1A): no external primary / provisional / excluded site may enter the Stage-1A read/build plan. Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import build_manifest_schema as SCH
from acar.v5.substrate import stage1_preflight as PF
from acar.v5.tests._util import expect_raises, ok

GOOD_AUTH = {"protocol_tag": PF.PROTOCOL_TAG, "statement": PF.REQUIRED_STAGE1B_STATEMENT}
ALL_SITES = tuple(P.EXTERNAL_PRIMARY.values()) + tuple(P.EXTERNAL_PROVISIONAL_NOT_ADMITTED) + tuple(P.EXTERNAL_EXCLUDED)


def test_no_site_token_in_plan_refs():
    pl = PLAN.build_substrate_plan()
    blob = " ".join(r["ref"] for r in pl["fold_contained_refs"]) + " ".join(r["ref"] for r in pl["final_external_refs"])
    for site in ALL_SITES:
        assert site not in blob, f"external site {site} must not appear in any plan ref"
    ok(f"no external site token ({', '.join(ALL_SITES)}) appears in any plan ref")


def test_site_paths_are_forbidden_targets():
    for site in ALL_SITES:
        assert SCH.path_is_forbidden(f"/data/{site}/sub-001/eeg.edf"), site
        pl = PLAN.build_substrate_plan()
        pl["fold_contained_refs"][0] = dict(pl["fold_contained_refs"][0], source_path=f"/data/{site}/x")
        expect_raises(PF.Stage1ForbiddenTargetError, lambda pl=pl: PF.run_preflight(pl, stage1b_authorization=GOOD_AUTH))
    ok("any source_path containing an external site (primary/provisional/excluded) → Stage1ForbiddenTargetError")


def test_provisional_and_excluded_classified():
    assert P.EXTERNAL_PROVISIONAL_NOT_ADMITTED == ("zenodo14178398",)
    assert P.EXTERNAL_EXCLUDED == ("ds007020",)
    assert P.EXTERNAL_PRIMARY == {"SCZ": "zenodo14808296", "PD": "ds007526"}
    ok("external sites classified: primary SCZ zenodo14808296 + PD ds007526; ASZED provisional; ds007020 excluded")


def main():
    print("ACAR v5 Stage-1A guard: external sites forbidden")
    test_no_site_token_in_plan_refs()
    test_site_paths_are_forbidden_targets()
    test_provisional_and_excluded_classified()
    print("ALL V5 STAGE1-EXTERNAL-SITES-FORBIDDEN GUARDS PASS")


if __name__ == "__main__":
    main()
