"""Guard (Step 3b): the FIT-only quantile is the PINNED explicit linear (Type-7) method, bit-stable + permutation-independent,
with no dependency on any library's default interpolation. Synthetic only."""
from __future__ import annotations
import random
from acar.v5 import scalarization as S
from acar.v5.tests._util import ok


def _approx(a, b, eps=1e-12):
    return abs(a - b) <= eps


def test_pinned_values():
    assert _approx(S._Q([0, 10], "q50"), 5.0)
    assert _approx(S._Q([0, 10], "q80"), 8.0)
    assert _approx(S._Q([0, 10, 20, 30], "q50"), 15.0)
    assert _approx(S._Q([0, 10, 20, 30], "q90"), 27.0)
    assert _approx(S._Q([5.0], "q90"), 5.0)                    # n=1 → the single value
    ok("Type-7 quantile: [0,10]→q50=5/q80=8; [0,10,20,30]→q50=15/q90=27; n=1→value")


def test_permutation_independent():
    xs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    base = {lvl: S._Q(xs, lvl) for lvl in ("q50", "q70", "q90")}
    sh = list(xs)
    random.Random(11).shuffle(sh)
    for lvl in base:
        assert _approx(S._Q(sh, lvl), base[lvl]), lvl
    ok("quantile is permutation-independent (input sorted internally)")


def test_empty_raises():
    try:
        S._Q([], "q50")
    except S.NonEvaluableCandidate:
        ok("empty quantile input → NonEvaluableCandidate")
        return
    raise AssertionError("expected NonEvaluableCandidate on empty input")


def main():
    print("ACAR v5 guard: quantile method pinned (Step 3b)")
    test_pinned_values()
    test_permutation_independent()
    test_empty_raises()
    print("ALL V5 QUANTILE-METHOD-PINNED GUARDS PASS")


if __name__ == "__main__":
    main()
