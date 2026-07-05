"""Guard (Stage-2B0): a Stage-2B authorization whose allowed_selection_refs include (or substitute in) a seed-20260712/20260713
(S1-robustness) ref is rejected — only the exact 10 seed-20260711 selection refs are admissible. Synthetic only."""
from __future__ import annotations
from acar.v5 import stage2b_authorization as AUTH
from acar.v5.tests._util import expect_raises, ok, stage2b_auth


def test_s1_seed_refs_rejected_in_auth():
    ten = list(AUTH.CANONICAL_SELECTION_REFS)
    # add an S1-seed ref (11 refs) -> reject
    expect_raises(AUTH.Stage2bAuthorizationError,
                  lambda: AUTH.validate_stage2b_authorization(stage2b_auth(allowed_selection_refs=ten + ["PD/fold0/seed20260712"])))
    # substitute an S1-seed ref for a selection ref (still 10, wrong set) -> reject
    swapped = sorted(ten[1:] + ["SCZ/fold2/seed20260713"])
    expect_raises(AUTH.Stage2bAuthorizationError,
                  lambda: AUTH.validate_stage2b_authorization(stage2b_auth(allowed_selection_refs=swapped)))
    # every canonical selection ref carries the selection seed only
    assert all(r.endswith("seed20260711") for r in AUTH.CANONICAL_SELECTION_REFS)
    assert len(AUTH.CANONICAL_SELECTION_REFS) == 10
    # the exact 10 pass
    assert AUTH.validate_stage2b_authorization(stage2b_auth()) is True
    ok("an S1-seed (12/13) ref added to or swapped into allowed_selection_refs → rejected; exactly the 10 seed-711 refs pass")


def main():
    print("ACAR v5 Stage-2B0 guard: S1-seed refs rejected for selection authorization")
    test_s1_seed_refs_rejected_in_auth()
    print("ALL V5 STAGE2B0-REJECTS-S1-SEED GUARDS PASS")


if __name__ == "__main__":
    main()
