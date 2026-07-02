"""Guard (Stage-1B10): a participants.tsv value that is neither the cohort's pinned control nor case value FAILS closed (no
disease-wide fallback, no guessing). Synthetic only."""
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


def test_unknown_value_fails():
    with tempfile.TemporaryDirectory() as d:
        p = _tsv(d, ["participant_id", "type"], [("sub-x", "Other"), ("sub-y", "Unknown")])   # ds003944 expects Control/Psychosis
        expect_raises(CLS.CohortLabelSpecError, lambda: CLS.resolve_label("SCZ", "ds003944", "sub-x", p))
        assert CLS.label_resolvable("SCZ", "ds003944", "sub-x", p) is False
    with tempfile.TemporaryDirectory() as d:
        p = _tsv(d, ["participant_id", "Group"], [("sub-z", "MSA")])   # ds003490 expects CTL/PD
        expect_raises(CLS.CohortLabelSpecError, lambda: CLS.resolve_label("PD", "ds003490", "sub-z", p))
    ok("a value not in {pinned control, pinned case} → CohortLabelSpecError; label_resolvable → False (fail-closed)")


def test_missing_subject_fails():
    with tempfile.TemporaryDirectory() as d:
        p = _tsv(d, ["participant_id", "type"], [("sub-1448", "Control")])
        expect_raises(CLS.CohortLabelSpecError, lambda: CLS.resolve_label("SCZ", "ds003944", "sub-9999", p))
    ok("a subject absent from participants.tsv → CohortLabelSpecError")


def main():
    print("ACAR v5 Stage-1B10 guard: cohort label unknown value fails")
    test_unknown_value_fails()
    test_missing_subject_fails()
    print("ALL V5 STAGE1B-COHORT-LABEL-UNKNOWN GUARDS PASS")


if __name__ == "__main__":
    main()
