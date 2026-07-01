"""Guard: action-record scalarization + FIT quantile universe (CANDIDATE_SPACE §1.6/§1.7) is bit-executable and matches the tag.
Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import scalarization as S
from acar.v5.tests._util import expect_raises, ok, batch


def _cand(cid):
    return next(c for c in P.CANDIDATE_MANIFEST if c["id"] == cid)


def test_manifest_structure():
    assert P._self_check() is True
    counts = {f: sum(1 for c in P.CANDIDATE_MANIFEST if c["family"] == f) for f in P.FAMILIES}
    assert counts == {"P1": 4, "P2": 4, "P3": 6, "P4": 2, "P5": 6} and len(P.CANDIDATE_MANIFEST) == 22
    assert all(("q75" not in str(c["params"].values())) for c in P.CANDIDATE_MANIFEST)
    ok("22-row manifest structure + no q75 + families P1:4 P2:4 P3:6 P4:2 P5:6")


def test_selectors_and_tiebreak():
    c1 = _cand("V5-P1-001")
    b = batch("b", matched_coral={"d_margin": 0.9}, spdim={"d_margin": 0.1}, t3a={"d_margin": 0.2})
    assert S.proposed_action(c1, b) == "matched_coral"                      # argmax d_margin
    b2 = batch("b", matched_coral={"d_margin": 0.5}, spdim={"d_margin": 0.5}, t3a={"d_margin": 0.1})
    assert S.proposed_action(c1, b2) == "matched_coral"                     # tie → action order (matched_coral ≺ spdim)
    b3 = batch("b", matched_coral={"d_margin": 0.1}, spdim={"d_margin": 0.5}, t3a={"d_margin": 0.5})
    assert S.proposed_action(c1, b3) == "spdim"                             # tie between spdim,t3a → spdim
    p3 = _cand("V5-P3-003")                                                 # fixed action = spdim
    assert S.proposed_action(p3, b) == "spdim"
    ok("P1/P2/P5 argmax d_margin + fixed tie-break; P3 fixed action")


def test_p4_agreement():
    c2 = _cand("V5-P4-001")  # k=2
    c3 = _cand("V5-P4-002")  # k=3
    # margin-best=matched_coral, post_sep-best=matched_coral, JS-min=t3a → matched_coral has 2 votes
    b = batch("b", matched_coral={"d_margin": 0.9, "post_sep": 0.9, "JS": 0.9},
              spdim={"d_margin": 0.1, "post_sep": 0.1, "JS": 0.5},
              t3a={"d_margin": 0.2, "post_sep": 0.2, "JS": 0.05})
    assert S.proposed_action(c2, b) == "matched_coral"                     # 2-of-3 fires
    assert S.proposed_action(c3, b) is None                                # 3-of-3 does NOT agree → no proposal
    ok("P4 agreement: 2-of-3 fires, 3-of-3 abstains (None)")


def test_zero_fit_records_non_evaluable():
    c3 = _cand("V5-P4-002")  # 3-of-3
    # no batch ever gets 3-way agreement → zero FIT proposed-action records → NonEvaluable
    fit = [batch(f"b{i}", matched_coral={"d_margin": 0.9, "post_sep": 0.1, "JS": 0.9},
                 spdim={"d_margin": 0.1, "post_sep": 0.9, "JS": 0.5},
                 t3a={"d_margin": 0.2, "post_sep": 0.2, "JS": 0.05}) for i in range(10)]
    expect_raises(S.NonEvaluableCandidate, lambda: S.fit_quantiles(c3, fit), "3-of-3 never agrees")
    ok("zero FIT proposed-action records → NonEvaluableCandidate (fail)")


def test_decide_thresholds_and_veto():
    c1 = _cand("V5-P1-001")  # benefit_q=q60, veto_q=q80
    # FIT: a* = matched_coral for all; d_margin spread; flip/JS low
    fit = [batch(f"b{i}", matched_coral={"d_margin": float(i) / 10.0, "flip_rate": 0.1, "JS": 0.1})
           for i in range(11)]
    th = S.fit_quantiles(c1, fit)
    # a high-margin, low-violence batch adapts; a low-margin one abstains; a high-violence one abstains
    hi = batch("hi", matched_coral={"d_margin": 1.0, "flip_rate": 0.05, "JS": 0.05})
    lo = batch("lo", matched_coral={"d_margin": 0.0, "flip_rate": 0.05, "JS": 0.05})
    violent = batch("v", matched_coral={"d_margin": 1.0, "flip_rate": 0.99, "JS": 0.99})
    assert S.decide(c1, hi, th) == "matched_coral"
    assert S.decide(c1, lo, th) == P.IDENTITY
    assert S.decide(c1, violent, th) == P.IDENTITY                         # harm veto blocks a high-benefit but disruptive adapt
    ok("decide: benefit threshold + harm veto (flip_rate/JS) gate the adapt/abstain")


def test_p2_entropy_veto():
    c2 = _cand("V5-P2-001")
    fit = [batch(f"b{i}", matched_coral={"d_margin": float(i) / 10.0, "flip_rate": 0.1, "JS": 0.1, "d_entropy": -0.1})
           for i in range(11)]
    th = S.fit_quantiles(c2, fit)
    good = batch("g", matched_coral={"d_margin": 1.0, "flip_rate": 0.05, "JS": 0.05, "d_entropy": -0.2})
    ent_up = batch("e", matched_coral={"d_margin": 1.0, "flip_rate": 0.05, "JS": 0.05, "d_entropy": 0.2})
    assert S.decide(c2, good, th) == "matched_coral"
    assert S.decide(c2, ent_up, th) == P.IDENTITY                          # d_entropy>0 blocks P2 even with high margin
    ok("P2 requires d_entropy ≤ 0 (no confidence gain → abstain)")


def main():
    print("ACAR v5 guard: action scalarization / quantile universe")
    test_manifest_structure()
    test_selectors_and_tiebreak()
    test_p4_agreement()
    test_zero_fit_records_non_evaluable()
    test_decide_thresholds_and_veto()
    test_p2_entropy_veto()
    print("ALL V5 SCALARIZATION GUARDS PASS")


if __name__ == "__main__":
    main()
