"""Guard: no held-out/external read is possible in the Step-3 scaffold — the external gate is fail-closed on every path
(no Stage-4 pass, no/badly-bound authorization). Synthetic only; NO external data is ever touched."""
from __future__ import annotations
from acar.v5 import deploy
from acar.v5 import protocol as P
from acar.v5.tests._util import expect_raises, ok


def test_no_auth_no_stage4_raises():
    expect_raises(deploy.ExternalNotAuthorizedError, lambda: deploy.external_read_gate())
    ok("no authorization + no Stage-4 pass → ExternalNotAuthorizedError")


def test_stage4_only_still_raises():
    expect_raises(deploy.ExternalNotAuthorizedError, lambda: deploy.external_read_gate(None, stage4_passed=True))
    ok("Stage-4 passed but no authorization → still fail-closed")


def test_malformed_authorizations_raise():
    bad = [
        {"protocol_tag": "wrong", "statement": deploy.REQUIRED_EXTERNAL_STATEMENT, "site": "zenodo14808296"},
        {"protocol_tag": deploy.PROTOCOL_TAG, "statement": "nope", "site": "zenodo14808296"},
        {"protocol_tag": deploy.PROTOCOL_TAG, "statement": deploy.REQUIRED_EXTERNAL_STATEMENT, "site": "ds007020"},   # excluded
        {"protocol_tag": deploy.PROTOCOL_TAG, "statement": deploy.REQUIRED_EXTERNAL_STATEMENT, "site": "zenodo14178398"},  # ASZED provisional
    ]
    for a in bad:
        expect_raises(deploy.ExternalNotAuthorizedError, lambda a=a: deploy.external_read_gate(a, stage4_passed=True))
    ok("every malformed / non-admitted-site authorization → fail-closed (incl. ds007020 excluded, ASZED provisional)")


def test_gate_validation_logic_is_reachable_but_reads_nothing():
    # A well-formed synthetic authorization exercises ONLY the gate's validation branch (returns True); it performs NO read.
    # Step-3 never constructs such an authorization for a real run — this only proves the gate is not a constant `raise`.
    a = {"protocol_tag": deploy.PROTOCOL_TAG, "statement": deploy.REQUIRED_EXTERNAL_STATEMENT,
         "site": P.EXTERNAL_PRIMARY["SCZ"]}
    assert deploy.external_read_gate(a, stage4_passed=True) is True
    ok("gate returns True only for a fully tag-bound primary-site auth (validation only; reads no data)")


def main():
    print("ACAR v5 guard: no external before tag/authorization")
    test_no_auth_no_stage4_raises()
    test_stage4_only_still_raises()
    test_malformed_authorizations_raise()
    test_gate_validation_logic_is_reachable_but_reads_nothing()
    print("ALL V5 NO-EXTERNAL-BEFORE-TAG GUARDS PASS")


if __name__ == "__main__":
    main()
