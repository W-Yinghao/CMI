"""Guard (Stage-2B0): the Stage-2B authorization's candidate universe must be EXACTLY the 22 frozen ids, and the engine binds the
same 22-row manifest. Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2b_authorization as AUTH
from acar.v5 import stage2_selection_manifest as MANIFEST
from acar.v5.tests._util import expect_raises, ok, stage2b_auth


def test_exact_22_candidate_universe():
    assert AUTH.validate_stage2b_authorization(stage2b_auth()) is True
    # 21 candidates -> reject
    expect_raises(AUTH.Stage2bAuthorizationError,
                  lambda: AUTH.validate_stage2b_authorization(stage2b_auth(allowed_candidate_ids=list(P.CANDIDATE_IDS)[:21])))
    # 23 (an out-of-space id added) -> reject
    expect_raises(AUTH.Stage2bAuthorizationError,
                  lambda: AUTH.validate_stage2b_authorization(stage2b_auth(allowed_candidate_ids=list(P.CANDIDATE_IDS) + ["V5-P9-999"])))
    # the engine binds exactly the 22-row manifest
    assert len(MANIFEST.selection_manifest()) == 22
    assert MANIFEST.selection_candidate_ids() == tuple(P.CANDIDATE_IDS)
    ok("allowed_candidate_ids must be exactly the 22 frozen ids; the engine binds the same 22-row manifest")


def main():
    print("ACAR v5 Stage-2B0 guard: exact 22-candidate universe")
    test_exact_22_candidate_universe()
    print("ALL V5 STAGE2B0-EXACT-22-UNIVERSE GUARDS PASS")


if __name__ == "__main__":
    main()
