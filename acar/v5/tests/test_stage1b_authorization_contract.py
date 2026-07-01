"""Guard (Stage-1B0): the Stage-1B authorization is a STRUCTURED contract, not a magic statement. Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.tests._util import expect_raises, ok, stage1b_auth


def test_valid_contract_passes():
    assert SA.validate_stage1b_authorization(stage1b_auth())["stage"] == "Stage-1B"
    ok("a fully-valid structured Stage-1B contract validates")


def test_missing_and_extra_fields_rejected():
    a = stage1b_auth()
    del a["allowed_refs"]
    expect_raises(SA.Stage1BuildNotAuthorizedError, lambda: SA.validate_stage1b_authorization(a), "missing field")
    b = stage1b_auth()
    b["surprise"] = 1
    expect_raises(SA.Stage1BuildNotAuthorizedError, lambda: SA.validate_stage1b_authorization(b), "extra field")
    expect_raises(SA.Stage1BuildNotAuthorizedError, lambda: SA.validate_stage1b_authorization(None), "None")
    ok("missing / extra field / None → Stage1BuildNotAuthorizedError")


def test_field_value_bindings():
    cases = [
        ("stage", "Stage-1A"), ("protocol_tag", "wrong"), ("protocol_tag_target_sha", "deadbeef"),
        ("implementation_base_sha", "short"), ("allowed_ref_type", "anything"),
        ("selection_seed", 20260712), ("run_id", ""), ("statement", "nope"),
        ("forbid_final_external_refs", False), ("forbid_external_read", "yes"),
    ]
    for k, v in cases:
        expect_raises(SA.Stage1BuildNotAuthorizedError, lambda k=k, v=v: SA.validate_stage1b_authorization(stage1b_auth(**{k: v})), f"{k}={v}")
    ok("each contract field is bound (stage/tag/target-sha/impl-sha/ref-type/selection-seed/run_id/statement/forbid-flags)")


def test_allowed_refs_must_be_exactly_30():
    short = stage1b_auth(allowed_refs=sorted(SA.CANONICAL_FOLD_REFS)[:-1])
    expect_raises(SA.Stage1BuildNotAuthorizedError, lambda: SA.validate_stage1b_authorization(short), "29 refs")
    extra = stage1b_auth(allowed_refs=sorted(SA.CANONICAL_FOLD_REFS) + ["external_exec/PD/all_source_dev"])
    expect_raises(SA.Stage1BuildNotAuthorizedError, lambda: SA.validate_stage1b_authorization(extra), "final ref sneaked in")
    assert len(SA.CANONICAL_FOLD_REFS) == 30
    ok("allowed_refs must be EXACTLY the 30 fold refs (no missing, no final-external sneak-in)")


def test_allowed_seeds_and_target_prefix():
    expect_raises(SA.Stage1BuildNotAuthorizedError, lambda: SA.validate_stage1b_authorization(stage1b_auth(allowed_seeds=[P.SELECTION_SEED])))
    assert SA.validate_stage1b_authorization(stage1b_auth(protocol_tag_target_sha="4278435975a72b1127803dd2cffab420c083e430"))
    expect_raises(SA.Stage1BuildNotAuthorizedError, lambda: SA.validate_stage1b_authorization(stage1b_auth(protocol_tag_target_sha="1234567")))
    ok("allowed_seeds must be the pinned 3; target sha must be a hex prefix of 4278435…")


def main():
    print("ACAR v5 Stage-1B0 guard: authorization contract")
    test_valid_contract_passes()
    test_missing_and_extra_fields_rejected()
    test_field_value_bindings()
    test_allowed_refs_must_be_exactly_30()
    test_allowed_seeds_and_target_prefix()
    print("ALL V5 STAGE1B-AUTH-CONTRACT GUARDS PASS")


if __name__ == "__main__":
    main()
