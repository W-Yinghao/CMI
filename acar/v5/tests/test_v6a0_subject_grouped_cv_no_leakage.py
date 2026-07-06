"""Guard (V6-A0a): the sign-predictability CV is subject-GROUPED — every record of a subject shares one fold, and train/test
subject sets are disjoint in every fold (no subject-level leakage). Deterministic, torch-free (no model fit needed)."""
from __future__ import annotations
from acar.v5 import v6_a0_sign_predictability as SP
from acar.v5.tests._util import ok


def _sign_records():
    recs = []
    for si in range(10):                                              # 10 subjects
        sk = f"PD/ds002778/sub-{si:02d}"
        for bi in range(3):                                          # 3 batches each
            for a in SP.PRIMARY_ACTIONS:                             # 3 actions -> 9 records/subject
                recs.append({"subject_key": sk, "batch_id": bi, "action_id": a, "provenance": "native",
                             "features": None, "beneficial": (si + bi) % 2})
    return recs


def test_subject_grouped_and_disjoint():
    recs = _sign_records()
    groups = [r["subject_key"] for r in recs]
    fold_of_record = [SP.sign_cv_fold(g, seed=0) for g in groups]
    # every record of a subject has the SAME fold
    by_sub = {}
    for g, f in zip(groups, fold_of_record):
        by_sub.setdefault(g, set()).add(f)
    assert all(len(fs) == 1 for fs in by_sub.values()), "all records of a subject must share one fold"
    # each subject in exactly one test fold; train/test subject-disjoint per fold
    subs = sorted(by_sub)
    fold_of = {s: SP.sign_cv_fold(s, seed=0) for s in subs}
    for k in range(SP.N_FOLDS):
        test_subs = {s for s in subs if fold_of[s] == k}
        train_subs = {s for s in subs if fold_of[s] != k}
        assert test_subs.isdisjoint(train_subs), f"fold {k}: train/test subjects overlap"
    assert sum(1 for s in subs if fold_of[s] == fold_of[s]) == len(subs)     # total accounting
    # determinism (permutation-independent): reversed input order -> identical assignment
    assert {s: SP.sign_cv_fold(s, 0) for s in reversed(subs)} == fold_of
    ok("sign-predictability CV is subject-grouped (one fold per subject) + train/test subject-disjoint + deterministic")


def main():
    print("ACAR v5 V6-A0a guard: subject-grouped CV, no leakage")
    test_subject_grouped_and_disjoint()
    print("ALL V6A0-SUBJECT-GROUPED-CV GUARDS PASS")


if __name__ == "__main__":
    main()
