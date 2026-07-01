"""Guard (Stage-1B3): the subject index builds canonical SubjectKeys "{disease}/{cohort}/{raw}" — a raw id repeated across cohorts
does NOT collapse. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import subject_index as SI
from acar.v5.tests._util import ok


def test_canonical_keys_and_no_collapse():
    idx = SI.build_subject_index("PD", {"ds002778": ["sub-000", "sub-001"], "ds003490": ["sub-000"], "ds004584": ["sub-000"]})
    assert set(idx.subject_keys) == {"PD/ds002778/sub-000", "PD/ds002778/sub-001", "PD/ds003490/sub-000", "PD/ds004584/sub-000"}
    assert len(idx) == 4                                       # sub-000 appears in 3 cohorts → 3 DISTINCT keys (no collapse)
    assert idx.cohort_of("PD/ds003490/sub-000") == "ds003490" and idx.raw_of("PD/ds003490/sub-000") == "sub-000"
    ok("canonical SubjectKeys include cohort; a raw id shared across cohorts yields distinct keys (no collapse)")


def test_key_helper():
    assert SI.canonical_subject_key("SCZ", "ds003944", "sub-7") == "SCZ/ds003944/sub-7"
    ok("canonical_subject_key = <disease>/<cohort>/<raw>")


def main():
    print("ACAR v5 Stage-1B3 guard: subject key canonicalization")
    test_canonical_keys_and_no_collapse()
    test_key_helper()
    print("ALL V5 STAGE1B-SUBJECT-KEY-CANON GUARDS PASS")


if __name__ == "__main__":
    main()
