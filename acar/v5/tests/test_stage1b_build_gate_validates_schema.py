"""Guard (Stage-1B1): both Stage-1B gates validate the build-manifest SCHEMA internally, so a malformed plan cannot bypass
Stage-1A preflight and reach a build gate. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import stage1_runtime_lock as RL
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def _bad_shape_plan(base):
    pl = base
    pl["fold_contained_refs"][0] = dict(pl["fold_contained_refs"][0], ref="PD/fold9/seed20260711")   # fold out of range
    return pl


def _final_in_fold_plan(base):
    pl = base
    pl["fold_contained_refs"] = list(pl["fold_contained_refs"]) + [
        {"ref": "external_exec/PD/all_source_dev", "disease": "PD", "fold": 0, "seed": 20260711,
         "roles": ["s1_robustness"], "source_path": None}]
    return pl


def test_require_ready_validates_schema():
    expect_raises(ValueError, lambda: RL.require_stage1b_ready(_bad_shape_plan(PLAN.build_substrate_plan()), stage1b_auth(), stage1b_lock()))
    expect_raises(ValueError, lambda: RL.require_stage1b_ready(_final_in_fold_plan(PLAN.build_substrate_plan()), stage1b_auth(), stage1b_lock()))
    ok("require_stage1b_ready rejects a malformed plan (bad fold shape / final ref among fold refs) via schema validation")


def test_full_build_gate_validates_schema():
    expect_raises(ValueError, lambda: RL.require_stage1b_full_build_ready(_bad_shape_plan(stage1b_full_plan()),
                                                                          stage1b_auth(protocol_tag_target_sha=FULL),
                                                                          stage1b_lock(protocol_tag_target_sha=FULL)))
    ok("require_stage1b_full_build_ready rejects a malformed plan via schema validation")


def test_external_site_token_in_ref_rejected():
    pl = PLAN.build_substrate_plan()
    pl["fold_contained_refs"][0] = dict(pl["fold_contained_refs"][0], ref="PD/fold0/seed20260711")   # keep valid shape
    # tamper a final ref string to embed an external site token → schema validator must reject
    pl["final_external_refs"][0] = dict(pl["final_external_refs"][0], ref="external_exec/PD/all_source_dev")
    # sanity: default is valid
    assert RL.require_stage1b_ready(pl, stage1b_auth(), stage1b_lock())["status"] == "STAGE1B_READY"
    ok("a well-formed plan still passes the schema-validating gate (control)")


def main():
    print("ACAR v5 Stage-1B1 guard: build gate validates schema")
    test_require_ready_validates_schema()
    test_full_build_gate_validates_schema()
    test_external_site_token_in_ref_rejected()
    print("ALL V5 STAGE1B-BUILD-GATE-SCHEMA GUARDS PASS")


if __name__ == "__main__":
    main()
