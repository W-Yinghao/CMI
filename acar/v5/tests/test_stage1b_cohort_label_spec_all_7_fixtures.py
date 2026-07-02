"""Guard (Stage-1B10): the frozen per-cohort label spec resolves control→0 / case→1 for ALL 7 DEV cohorts, using each cohort's exact
column (case-insensitive) + pinned values, or the subject-id prefix for ds002778. Synthetic participants.tsv fixtures only (no real
DEV read)."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import cohort_label_spec as CLS
from acar.v5.tests._util import ok


def _tsv(d, header, rows):
    p = os.path.join(d, "participants.tsv")
    with open(p, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    return p


def test_all_seven_cohorts_resolve():
    # ds002778 = id-prefix (no participants.tsv column needed)
    assert CLS.resolve_label("PD", "ds002778", "sub-hc1", None) == 0
    assert CLS.resolve_label("PD", "ds002778", "sub-pd3", None) == 1
    # column cohorts: (disease, cohort, column-header-as-in-real-data, control_value, case_value)
    cases = [
        ("PD", "ds003490", "Group", "CTL", "PD"),
        ("PD", "ds004584", "GROUP", "Control", "PD"),
        ("SCZ", "ds003944", "type", "Control", "Psychosis"),
        ("SCZ", "ds003947", "type", "Control", "Psychosis"),
        ("SCZ", "ds004000", "group", "HC", "P"),
        ("SCZ", "ds004367", "Group", "Control", "Patient"),
    ]
    for disease, cohort, col, ctrl_v, case_v in cases:
        with tempfile.TemporaryDirectory() as d:
            p = _tsv(d, ["participant_id", col], [("sub-c", ctrl_v), ("sub-p", case_v)])
            assert CLS.resolve_label(disease, cohort, "sub-c", p) == 0, (cohort, "control")
            assert CLS.resolve_label(disease, cohort, "sub-p", p) == 1, (cohort, "case")
    ok("all 7 DEV cohorts resolve control→0 / case→1 via the frozen per-cohort spec (column or id-prefix)")


def test_spec_covers_exactly_the_seven_dev_cohorts():
    from acar.v5 import protocol as P
    dev = {(d, c) for d in P.DEV_COHORTS for c in P.DEV_COHORTS[d]}
    assert set(CLS.COHORT_LABEL_SPEC) == dev
    ok("COHORT_LABEL_SPEC covers EXACTLY the 7 frozen DEV cohorts (3 PD + 4 SCZ)")


def main():
    print("ACAR v5 Stage-1B10 guard: cohort label spec all 7 fixtures")
    test_all_seven_cohorts_resolve()
    test_spec_covers_exactly_the_seven_dev_cohorts()
    print("ALL V5 STAGE1B-COHORT-LABEL-ALL7 GUARDS PASS")


if __name__ == "__main__":
    main()
