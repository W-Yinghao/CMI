"""Guard (Stage-2B0b): a non-evaluable candidate×disease cell contributes raw p = 1.0 for H1/H2/H3 to the Holm family, while
evaluable cells keep their raw p. Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_selection_engine as ENG
from acar.v5.tests._util import ok, stage2b_holm_per


def test_nonevaluable_cell_pvalues_are_one():
    per = stage2b_holm_per(evaluable_p=0.001, nonevaluable_ids=["V5-P2-003"])
    keys, pvals = ENG.holm_family_keys_pvalues(per, P.CANDIDATE_MANIFEST)
    pv = dict(zip(keys, pvals))
    for d in ENG.DISEASES:
        for h in ("H1", "H2", "H3"):
            assert pv[("V5-P2-003", d, h)] == 1.0                # non-evaluable → raw p = 1
    assert pv[(P.CANDIDATE_IDS[0], "PD", "H1")] == 0.001         # evaluable cells keep their raw p
    ok("non-evaluable cell raw p-values are exactly 1.0 for H1/H2/H3; evaluable cells keep their raw p")


def main():
    print("ACAR v5 Stage-2B0b guard: non-evaluable cell p-values are 1.0")
    test_nonevaluable_cell_pvalues_are_one()
    print("ALL V5 STAGE2B0B-NONEVALUABLE-PVALUES-ONE GUARDS PASS")


if __name__ == "__main__":
    main()
