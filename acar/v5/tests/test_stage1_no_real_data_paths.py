"""Guard (Stage-1A): the preflight is fail-closed on real-data paths — a declared source_path needs Stage-1B authorization (never
issued in Stage-1A), and forbidden targets (real DEV / v4 artifacts / caches / external) are refused regardless. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import stage1_preflight as PF
from acar.v5.tests._util import expect_raises, ok

GOOD_AUTH = {"protocol_tag": PF.PROTOCOL_TAG, "statement": PF.REQUIRED_STAGE1B_STATEMENT}


def _plan_with_path(path):
    pl = PLAN.build_substrate_plan()
    pl["fold_contained_refs"][0] = dict(pl["fold_contained_refs"][0], source_path=path)
    return pl


def test_default_plan_reads_nothing():
    rep = PF.run_preflight()
    assert rep["real_data_entries"] == 0
    ok("default plan-only preflight declares 0 real-data source_paths")


def test_source_path_without_auth_rejected():
    pl = _plan_with_path("/some/synthetic/dir/sub-001")       # benign-looking, but still a real read → needs Stage-1B auth
    expect_raises(PF.Stage1BuildNotAuthorizedError, lambda: PF.run_preflight(pl))
    ok("a declared source_path with NO Stage-1B authorization → Stage1BuildNotAuthorizedError")


def test_source_path_with_bad_auth_rejected():
    pl = _plan_with_path("/some/synthetic/dir/sub-001")
    expect_raises(PF.Stage1BuildNotAuthorizedError,
                  lambda: PF.run_preflight(pl, stage1b_authorization={"protocol_tag": "wrong", "statement": "x"}))
    ok("a declared source_path with a badly-bound authorization → still rejected")


def test_forbidden_targets_refused_even_with_auth():
    for bad in ("/projects/EEG-foundation-model/datalake/raw/scps/cache/PD.npz",
                "/home/infres/yinwang/acar_v4_regen_outputs/PD_all_dev_substrate_b99fa4f",
                "/home/infres/yinwang/CMI_AAAI/archive/lpc-cmi-failed/results/feat_dump_v4/audit_PD_ds002778_erm_0.npz"):
        pl = _plan_with_path(bad)
        expect_raises(PF.Stage1ForbiddenTargetError, lambda pl=pl: PF.run_preflight(pl, stage1b_authorization=GOOD_AUTH))
    ok("real DEV cache / v4 artifact / feat_dump paths → Stage1ForbiddenTargetError even WITH a valid Stage-1B auth")


def main():
    print("ACAR v5 Stage-1A guard: no real-data paths")
    test_default_plan_reads_nothing()
    test_source_path_without_auth_rejected()
    test_source_path_with_bad_auth_rejected()
    test_forbidden_targets_refused_even_with_auth()
    print("ALL V5 STAGE1-NO-REAL-DATA GUARDS PASS")


if __name__ == "__main__":
    main()
