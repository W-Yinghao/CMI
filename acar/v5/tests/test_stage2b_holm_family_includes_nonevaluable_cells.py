"""Guard (Stage-2B0b): the Holm certification family INCLUDES non-evaluable candidate×disease cells (they are not skipped).
Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_selection_engine as ENG
from acar.v5.tests._util import ok, stage2b_holm_per


def test_family_includes_nonevaluable_cells():
    per = stage2b_holm_per(nonevaluable_ids=["V5-P4-001", "V5-P4-002"])   # 2 candidates non-evaluable (12 cells)
    keys, pvals = ENG.holm_family_keys_pvalues(per, P.CANDIDATE_MANIFEST)
    keyset = set(keys)
    for cid in ("V5-P4-001", "V5-P4-002"):
        for d in ENG.DISEASES:
            for h in ("H1", "H2", "H3"):
                assert (cid, d, h) in keyset                     # the non-evaluable cells ARE in the family
    assert len(keys) == len(P.CANDIDATE_IDS) * len(ENG.DISEASES) * 3
    ok("non-evaluable candidate×disease cells are part of the Holm family (never skipped)")


def main():
    print("ACAR v5 Stage-2B0b guard: Holm family includes non-evaluable cells")
    test_family_includes_nonevaluable_cells()
    print("ALL V5 STAGE2B0B-HOLM-INCLUDES-NONEVALUABLE GUARDS PASS")


if __name__ == "__main__":
    main()
