"""Guard (Stage-2B0): candidate thresholds are computed over FIT proposed-action records ONLY; zero FIT records ⇒ NON-EVALUABLE
(the candidate fails); a non-FIT (None) input is rejected. Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_thresholds as TH
from acar.v5.tests._util import batch, expect_raises, ok


def test_fit_only_thresholds():
    p1 = P.CANDIDATE_MANIFEST[0]                                 # V5-P1-001 (benefit d_margin + veto)
    fit = [batch("b0", matched_coral={"d_margin": 0.5, "flip_rate": 0.1, "JS": 0.1},
                 spdim={"d_margin": 0.2, "flip_rate": 0.2, "JS": 0.2},
                 t3a={"d_margin": 0.1, "flip_rate": 0.3, "JS": 0.3})]
    thr = TH.fit_thresholds(p1, fit)
    assert "benefit" in thr and "veto_flip" in thr and "veto_js" in thr
    # zero FIT proposed-action records -> NonEvaluableCandidate
    expect_raises(TH.NonEvaluableCandidate, lambda: TH.fit_thresholds(p1, []))
    # a non-FIT (None) argument is refused
    expect_raises(TH.Stage2ThresholdError, lambda: TH.fit_thresholds(p1, None))
    ok("thresholds fit over FIT proposed-action records only; empty FIT → NonEvaluableCandidate; None → Stage2ThresholdError")


def main():
    print("ACAR v5 Stage-2B0 guard: FIT-only threshold fitting")
    test_fit_only_thresholds()
    print("ALL V5 STAGE2B0-FIT-ONLY GUARDS PASS")


if __name__ == "__main__":
    main()
