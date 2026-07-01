"""Guard: once the jointly-selected candidate_id is fixed (Stage-2), Stage-4/Stage-5 cannot switch to a different candidate.
Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.deploy import FixedCandidate, ReselectionError
from acar.v5.tests._util import expect_raises, ok


def test_fix_then_same_ok():
    fc = FixedCandidate()
    fc.fix("V5-P1-001")
    fc.fix("V5-P1-001")                                        # idempotent re-fix to the SAME id is fine
    assert fc.candidate_id == "V5-P1-001"
    assert fc.assert_candidate("V5-P1-001") is True
    ok("fix a candidate + re-affirm the same id → ok")


def test_reselection_blocked():
    fc = FixedCandidate()
    fc.fix("V5-P3-003")
    expect_raises(ReselectionError, lambda: fc.fix("V5-P5-001"), "re-fix to a different id")
    expect_raises(ReselectionError, lambda: fc.assert_candidate("V5-P5-001"), "assert a different id")
    ok("re-fixing / asserting a DIFFERENT candidate_id → ReselectionError (no reselection across stages)")


def test_invalid_id_rejected():
    fc = FixedCandidate()
    expect_raises(ValueError, lambda: fc.fix("V5-P9-999"), "not in manifest")
    ok("fixing an id outside the pinned 22-row manifest → rejected")


def test_use_before_fix_errors():
    fc = FixedCandidate()
    expect_raises(ReselectionError, lambda: fc.candidate_id, "no candidate fixed")
    expect_raises(ReselectionError, lambda: fc.assert_candidate("V5-P1-001"), "no candidate fixed")
    ok("using the fixed candidate before any fix → error")


def test_all_manifest_ids_are_fixable():
    for cid in P.CANDIDATE_IDS:
        FixedCandidate().fix(cid)
    ok(f"all {len(P.CANDIDATE_IDS)} manifest ids are valid fix targets")


def main():
    print("ACAR v5 guard: fixed candidate / no reselection")
    test_fix_then_same_ok()
    test_reselection_blocked()
    test_invalid_id_rejected()
    test_use_before_fix_errors()
    test_all_manifest_ids_are_fixable()
    print("ALL V5 FIXED-CANDIDATE-NO-RESELECTION GUARDS PASS")


if __name__ == "__main__":
    main()
