"""Guard (Stage-1B3): the subject index fails closed on duplicate (cohort, raw), wrong cohort keys, or empty raw ids. Synthetic."""
from __future__ import annotations
from acar.v5.substrate import subject_index as SI
from acar.v5.tests._util import expect_raises, ok


def test_duplicate_cohort_raw_rejected():
    expect_raises(SI.SubjectIndexError,
                  lambda: SI.build_subject_index("PD", {"ds002778": ["sub-1", "sub-1"], "ds003490": ["sub-2"], "ds004584": ["sub-3"]}))
    ok("duplicate (cohort, raw) within a cohort → SubjectIndexError")


def test_wrong_cohort_keys_rejected():
    expect_raises(SI.SubjectIndexError, lambda: SI.build_subject_index("PD", {"ds002778": ["sub-1"]}), "missing cohorts")
    expect_raises(SI.SubjectIndexError,
                  lambda: SI.build_subject_index("PD", {"ds002778": ["sub-1"], "ds003490": ["sub-1"], "ds004584": ["sub-1"], "ds003944": ["sub-1"]}),
                  "foreign cohort key")
    ok("per_cohort_raw keys must equal the disease's frozen DEV cohorts (missing/foreign → rejected)")


def test_empty_raw_and_unknown_disease():
    expect_raises(SI.SubjectIndexError,
                  lambda: SI.build_subject_index("PD", {"ds002778": [""], "ds003490": ["sub-1"], "ds004584": ["sub-2"]}), "empty raw")
    expect_raises(SI.SubjectIndexError, lambda: SI.build_subject_index("NOPE", {}), "unknown disease")
    ok("empty raw id / unknown disease → SubjectIndexError")


def main():
    print("ACAR v5 Stage-1B3 guard: duplicate subject keys rejected")
    test_duplicate_cohort_raw_rejected()
    test_wrong_cohort_keys_rejected()
    test_empty_raw_and_unknown_disease()
    print("ALL V5 STAGE1B-DUP-SUBJECT-KEYS GUARDS PASS")


if __name__ == "__main__":
    main()
