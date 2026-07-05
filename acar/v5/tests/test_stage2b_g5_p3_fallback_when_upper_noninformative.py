"""Guard (Stage-2B0): G5 benefit retention = red ≥ 0.25·red_upper OR red ≥ best-eligible-P3; and when red_upper ≤ 0 the upper
arm is NON-INFORMATIVE, so G5 falls back to the P3 comparator (which must exist). Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_gates as G
from acar.v5.tests._util import ok


def test_g5_upper_arm_and_p3_arm():
    assert P.BENEFIT_RETENTION_FRAC == 0.25
    assert G.g5_pass(red=0.30, red_upper=1.0, red_p3_best=None) is True       # 0.30 ≥ 0.25·1.0
    assert G.g5_pass(red=0.10, red_upper=1.0, red_p3_best=None) is False      # 0.10 < 0.25, no P3
    assert G.g5_pass(red=0.10, red_upper=1.0, red_p3_best=0.05) is True       # red ≥ P3-best
    ok("G5 passes via red ≥ 0.25·red_upper OR red ≥ best-eligible-P3")


def test_g5_p3_fallback_when_upper_noninformative():
    # red_upper ≤ 0 → upper arm non-informative → P3 comparator ONLY
    assert G.g5_pass(red=0.30, red_upper=0.0, red_p3_best=None) is False      # no P3 comparator → fail
    assert G.g5_pass(red=0.30, red_upper=-0.10, red_p3_best=0.20) is True     # red ≥ P3-best
    assert G.g5_pass(red=0.10, red_upper=-0.10, red_p3_best=0.20) is False    # red < P3-best
    ok("red_upper ≤ 0 → G5 falls back to the P3 comparator (upper arm ignored)")


def main():
    print("ACAR v5 Stage-2B0 guard: G5 P3 fallback when red_upper non-informative")
    test_g5_upper_arm_and_p3_arm()
    test_g5_p3_fallback_when_upper_noninformative()
    print("ALL V5 STAGE2B0-G5-P3-FALLBACK GUARDS PASS")


if __name__ == "__main__":
    main()
