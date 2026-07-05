"""Guard (Stage-2B0): the selection winner tie-break is deterministic — max min-margin, then lower macro harm_among_adapted,
then higher macro coverage, then more conservative family (P3 ≺ P1/P2/P4 ≺ P5), then id. Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_selection_engine as ENG
from acar.v5.tests._util import ok

MANIFEST = P.CANDIDATE_MANIFEST


def _ev(cov, harm):
    return {"eval_gate": {"coverage_lcb": cov, "harm_among_adapted_ucb": harm}}


def _per(pairs):
    per = {}
    for cid, cov, harm in pairs:
        per[(cid, "PD")] = _ev(cov, harm)
        per[(cid, "SCZ")] = _ev(cov, harm)
    return per


def test_conservative_family_breaks_full_tie():
    p3, p5 = "V5-P3-001", "V5-P5-001"
    per = _per([(p3, 0.5, 0.1), (p5, 0.5, 0.1)])                  # identical margin/harm/coverage
    assert ENG._select_winner([(p3, 0.1), (p5, 0.1)], per, MANIFEST) == p3   # P3 more conservative
    ok("full tie → more conservative family wins (P3 ≺ P5)")


def test_lower_harm_then_higher_coverage_precede_family():
    p3, p5 = "V5-P3-001", "V5-P5-001"
    per = _per([(p3, 0.5, 0.30), (p5, 0.5, 0.10)])                # p5 has lower harm
    assert ENG._select_winner([(p3, 0.1), (p5, 0.1)], per, MANIFEST) == p5   # lower harm beats conservatism
    per2 = _per([(p3, 0.20, 0.10), (p5, 0.60, 0.10)])            # equal harm; p5 higher coverage
    assert ENG._select_winner([(p3, 0.1), (p5, 0.1)], per2, MANIFEST) == p5
    ok("tie-break order: lower harm → higher coverage → conservative family (deterministic)")


def test_max_margin_wins_before_tiebreak():
    p3, p1 = "V5-P3-001", "V5-P1-001"
    per = _per([(p3, 0.9, 0.0), (p1, 0.5, 0.5)])
    assert ENG._select_winner([(p3, 0.05), (p1, 0.20)], per, MANIFEST) == p1  # larger min-margin wins outright
    ok("the objective (max min-margin) precedes the tie-break")


def main():
    print("ACAR v5 Stage-2B0 guard: deterministic tie-break")
    test_conservative_family_breaks_full_tie()
    test_lower_harm_then_higher_coverage_precede_family()
    test_max_margin_wins_before_tiebreak()
    print("ALL V5 STAGE2B0-TIEBREAK GUARDS PASS")


if __name__ == "__main__":
    main()
