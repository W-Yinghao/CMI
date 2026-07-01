"""Guard: the DEV split (SPLITS §5) is subject-disjoint, each subject EVAL exactly once, and deterministic/permutation-independent.
Synthetic subject ids only."""
from __future__ import annotations
import random
from acar.v5 import protocol as P
from acar.v5 import splits
from acar.v5.tests._util import expect_raises, ok


SUBJECTS = [f"ds00{d}/sub-{i:03d}" for d in range(3) for i in range(120)]   # 360 synthetic subjects


def test_fold_disjoint_and_partition():
    seen_eval = set()
    for k in range(P.OUTER_K):
        s = splits.make_fold(SUBJECTS, k)
        e, c, f = set(s["eval"]), set(s["cal"]), set(s["fit"])
        assert not (e & c) and not (e & f) and not (c & f), "FIT/CAL/EVAL not disjoint"
        assert set(s["train"]) | set(s["val"]) == f and not (set(s["train"]) & set(s["val"]))
        assert e & seen_eval == set(), "a subject appears in EVAL of >1 fold"
        seen_eval |= e
    assert seen_eval == set(SUBJECTS), "union of EVAL over folds must equal all subjects"
    ok("K=5: FIT/CAL/EVAL + TRAIN/VAL disjoint; each subject EVAL exactly once; EVAL union == all subjects")


def test_deterministic_and_permutation_independent():
    a = splits.make_fold(SUBJECTS, 0)
    shuffled = list(SUBJECTS)
    random.Random(123).shuffle(shuffled)
    b = splits.make_fold(shuffled, 0)
    assert a == b, "split must be permutation-independent (hash-based, not RNG)"
    ok("split is deterministic + permutation-independent (identical under a shuffled input)")


def test_ratios_reasonable():
    s = splits.make_fold(SUBJECTS, 0)
    n = len(SUBJECTS)
    assert 0.10 * n <= len(s["eval"]) <= 0.30 * n, ("~1/K in EVAL", len(s["eval"]))
    non_eval = n - len(s["eval"])
    assert 0.55 * non_eval <= len(s["fit"]) <= 0.85 * non_eval, ("~70% FIT", len(s["fit"]))
    ok(f"ratios: EVAL≈1/K ({len(s['eval'])}/{n}), FIT≈70% of non-EVAL ({len(s['fit'])}/{non_eval})")


def test_duplicate_subjects_rejected():
    expect_raises(ValueError, lambda: splits.assign_outer_folds(SUBJECTS + [SUBJECTS[0]]), "dup subject")
    ok("duplicate subjects rejected")


def main():
    print("ACAR v5 guard: subject-disjoint split")
    test_fold_disjoint_and_partition()
    test_deterministic_and_permutation_independent()
    test_ratios_reasonable()
    test_duplicate_subjects_rejected()
    print("ALL V5 SUBJECT-DISJOINT GUARDS PASS")


if __name__ == "__main__":
    main()
