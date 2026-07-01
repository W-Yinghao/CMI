"""Guard (Step 3b): the DEV split has EXACT nearest-integer cardinalities (not approximate hash thresholds): non-EVAL FIT/CAL =
round(0.70·n)/rest; FIT TRAIN/VAL = round(0.80·|FIT|)/rest. Still subject-disjoint + permutation-independent. Synthetic ids only."""
from __future__ import annotations
import math
import random
from acar.v5 import protocol as P
from acar.v5 import splits
from acar.v5.tests._util import ok

SUBJECTS = [f"ds00{d}/sub-{i:03d}" for d in range(3) for i in range(121)]   # 363 synthetic subjects


def _round_half_up(frac, n):
    n_a = int(math.floor(frac * n + 0.5))
    if 0.0 < frac < 1.0 and n >= 2:
        n_a = min(max(n_a, 1), n - 1)
    return n_a


def test_exact_cardinalities_each_fold():
    for k in range(P.OUTER_K):
        s = splits.make_fold(SUBJECTS, k)
        non_eval = len(SUBJECTS) - len(s["eval"])
        assert len(s["fit"]) + len(s["cal"]) == non_eval
        assert len(s["fit"]) == _round_half_up(P.FIT_FRAC, non_eval), (k, "FIT", len(s["fit"]), non_eval)
        assert len(s["cal"]) == non_eval - len(s["fit"]), (k, "CAL")
        assert len(s["train"]) == _round_half_up(P.TRAIN_FRAC, len(s["fit"])), (k, "TRAIN")
        assert len(s["val"]) == len(s["fit"]) - len(s["train"]), (k, "VAL")
    ok("each fold: |FIT|=round(0.70·non_eval), |CAL|=rest, |TRAIN|=round(0.80·|FIT|), |VAL|=rest (exact)")


def test_permutation_independent_cardinality():
    a = splits.make_fold(SUBJECTS, 2)
    shuffled = list(SUBJECTS)
    random.Random(7).shuffle(shuffled)
    b = splits.make_fold(shuffled, 2)
    assert a == b, "exact-count split must remain permutation-independent"
    ok("exact-count split is permutation-independent")


def test_no_empty_side_for_real_split():
    small = [f"s{i}" for i in range(10)]
    fit, cal = splits._rank_split(small, P.FIT_FRAC, "fitcal")
    assert len(fit) >= 1 and len(cal) >= 1 and len(fit) + len(cal) == 10
    ok("nontrivial split keeps both sides non-empty")


def main():
    print("ACAR v5 guard: split ratio cardinality (Step 3b)")
    test_exact_cardinalities_each_fold()
    test_permutation_independent_cardinality()
    test_no_empty_side_for_real_split()
    print("ALL V5 SPLIT-RATIO-CARDINALITY GUARDS PASS")


if __name__ == "__main__":
    main()
