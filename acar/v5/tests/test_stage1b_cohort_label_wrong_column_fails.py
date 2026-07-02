"""Guard (Stage-1B10): if a cohort's participants.tsv lacks the pinned label column, it FAILS closed. Column matching is
case-insensitive + whitespace-stripped (so 'Group'/'GROUP'/' group ' all match the pinned name), but a genuinely absent column
fails. Synthetic only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import cohort_label_spec as CLS
from acar.v5.tests._util import expect_raises, ok


def _tsv(d, header, rows):
    p = os.path.join(d, "participants.tsv")
    with open(p, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    return p


def test_absent_column_fails():
    with tempfile.TemporaryDirectory() as d:
        p = _tsv(d, ["participant_id", "diagnosis"], [("sub-001", "PD")])   # ds003490 wants 'Group', not 'diagnosis'
        expect_raises(CLS.CohortLabelSpecError, lambda: CLS.resolve_label("PD", "ds003490", "sub-001", p))
    ok("a participants.tsv without the pinned label column → CohortLabelSpecError")


def test_case_insensitive_column_match_ok():
    for header_col in ("group", "GROUP", " Group "):          # all casefold/strip-match the pinned 'Group' for ds003490
        with tempfile.TemporaryDirectory() as d:
            p = _tsv(d, ["participant_id", header_col], [("sub-001", "CTL"), ("sub-002", "PD")])
            assert CLS.resolve_label("PD", "ds003490", "sub-001", p) == 0
            assert CLS.resolve_label("PD", "ds003490", "sub-002", p) == 1
    ok("the pinned column is matched case-insensitively + whitespace-stripped (group/GROUP/' Group ')")


def main():
    print("ACAR v5 Stage-1B10 guard: cohort label wrong column fails")
    test_absent_column_fails()
    test_case_insensitive_column_match_ok()
    print("ALL V5 STAGE1B-COHORT-LABEL-WRONG-COLUMN GUARDS PASS")


if __name__ == "__main__":
    main()
