"""Guard (Stage-2B0b): the Holm family size is FIXED at 22 candidates × 2 diseases × 3 hypotheses = 132, regardless of how many
candidate×disease cells are non-evaluable. Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_selection_engine as ENG
from acar.v5.tests._util import ok, stage2b_holm_per

EXPECTED = 22 * 2 * 3


def test_family_size_fixed_at_132():
    assert EXPECTED == 132
    for non in ([], ["V5-P1-001"], ["V5-P3-001", "V5-P3-002", "V5-P5-006"], list(P.CANDIDATE_IDS)):
        per = stage2b_holm_per(nonevaluable_ids=non)
        keys, pvals = ENG.holm_family_keys_pvalues(per, P.CANDIDATE_MANIFEST)
        assert len(keys) == EXPECTED and len(pvals) == EXPECTED
        assert len(ENG.holm_adjusted_map(per, P.CANDIDATE_MANIFEST)) == EXPECTED
    ok("Holm family size is fixed at 132 for 0, 1, several, or ALL non-evaluable candidates")


def main():
    print("ACAR v5 Stage-2B0b guard: Holm family size fixed at 132")
    test_family_size_fixed_at_132()
    print("ALL V5 STAGE2B0B-HOLM-SIZE-FIXED GUARDS PASS")


if __name__ == "__main__":
    main()
