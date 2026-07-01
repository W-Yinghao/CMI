"""Guard (Stage-1B0): a final all-source external-execution ref can NEVER carry a source_path — schema-only ALWAYS, even with a
fully-valid Stage-1B authorization + runtime lock. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import stage1b_manifest as MAN
from acar.v5.substrate import stage1_preflight as PF
from acar.v5.substrate import stage1_runtime_lock as RL
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock


def _plan_with_final_source(path="/dev/PD/ds002778/sub-001"):
    pl = PLAN.build_substrate_plan()
    pl["final_external_refs"][0] = dict(pl["final_external_refs"][0], source_path=path)
    return pl


def test_assert_helper_rejects():
    pl = _plan_with_final_source()
    expect_raises(MAN.Stage1bWhitelistError, lambda: MAN.assert_final_external_schema_only(pl["final_external_refs"]))
    ok("assert_final_external_schema_only rejects any final-external source_path")


def test_preflight_rejects_even_with_valid_auth():
    pl = _plan_with_final_source()
    expect_raises(MAN.Stage1bWhitelistError, lambda: PF.run_preflight(pl, stage1b_authorization=stage1b_auth()))
    ok("run_preflight rejects a final-external source_path even with a valid Stage-1B auth")


def test_build_gate_rejects_even_with_auth_and_lock():
    pl = _plan_with_final_source()
    expect_raises(MAN.Stage1bWhitelistError,
                  lambda: RL.require_stage1b_ready(pl, stage1b_auth(), stage1b_lock()))
    ok("require_stage1b_ready rejects a final-external source_path even with valid auth + runtime lock")


def test_default_plan_final_refs_ok():
    MAN.assert_final_external_schema_only(PLAN.build_substrate_plan()["final_external_refs"])
    ok("default plan final-external refs are schema-only (source_path=None) → accepted")


def main():
    print("ACAR v5 Stage-1B0 guard: final external still forbidden")
    test_assert_helper_rejects()
    test_preflight_rejects_even_with_valid_auth()
    test_build_gate_rejects_even_with_auth_and_lock()
    test_default_plan_final_refs_ok()
    print("ALL V5 STAGE1B-FINAL-EXTERNAL-FORBIDDEN GUARDS PASS")


if __name__ == "__main__":
    main()
