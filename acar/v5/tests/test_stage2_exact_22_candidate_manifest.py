"""Guard (Stage-2A, PROP 5): the Stage-2 selection manifest binding is EXACTLY the frozen 22 rows (P1=4,P2=4,P3=6,P4=2,P5=6),
ids identical to protocol.CANDIDATE_IDS, and any drift fails closed. Synthetic (data-free)."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_selection_manifest as MANIFEST
from acar.v5.tests._util import expect_raises, ok


def test_exact_22_and_counts():
    m = MANIFEST.selection_manifest()
    assert len(m) == 22
    assert MANIFEST.family_counts() == {"P1": 4, "P2": 4, "P3": 6, "P4": 2, "P5": 6}
    assert MANIFEST.selection_candidate_ids() == tuple(P.CANDIDATE_IDS)
    assert len(set(MANIFEST.selection_candidate_ids())) == 22
    ok("selection manifest = exactly 22 rows, counts {P1:4,P2:4,P3:6,P4:2,P5:6}, ids == protocol.CANDIDATE_IDS (PROP 5)")


def test_drift_fails_closed():
    orig = P.CANDIDATE_MANIFEST
    try:
        P.CANDIDATE_MANIFEST = orig[:21]                            # 21 rows → len check fires
        expect_raises(MANIFEST.Stage2ManifestError, MANIFEST.selection_manifest)
        altered = tuple(dict(c) for c in orig)                      # 22 rows but wrong family mix
        altered[0]["family"] = "P5"                                 # P1=3, P5=7
        P.CANDIDATE_MANIFEST = altered
        expect_raises(MANIFEST.Stage2ManifestError, MANIFEST.selection_manifest)
    finally:
        P.CANDIDATE_MANIFEST = orig
    # restored
    assert len(MANIFEST.selection_manifest()) == 22
    ok("a truncated or family-count-drifted manifest → Stage2ManifestError; restore leaves the frozen 22 intact (PROP 5)")


def main():
    print("ACAR v5 Stage-2A guard: exact 22-candidate manifest binding (PROP 5)")
    test_exact_22_and_counts()
    test_drift_fails_closed()
    print("ALL V5 STAGE2A-EXACT-22-MANIFEST GUARDS PASS")


if __name__ == "__main__":
    main()
