"""Guard (Stage-1B12): the broken_internal_pointer_rewrite repair is pinned to ds004000/sub-042's TWO exact recordings — a broken
pointer in any other cohort / subject / recording gets NO repair plan (fail-closed, never a general BrainVision repair)."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet


def _broken_pointer(stem):
    raw_dir = tempfile.mkdtemp()
    return make_brainvision_triplet(raw_dir, stem, ("Fp1", "Fp2"), with_marker=True,
                                    data_file="019_P1.dat", marker_file="019_P1.vmrk")


def test_other_recording_of_sub042_not_repaired():
    vhdr = _broken_pointer("sub-042_task-other_run-1_eeg")            # not one of the two pinned recordings
    assert BR.plan_repair("SCZ", "ds004000", "sub-042", vhdr) is None
    ok("a broken-pointer recording of sub-042 NOT in the pinned set is not rewritten")


def test_other_subject_not_repaired():
    vhdr = _broken_pointer("sub-042_task-proposer_run-1_eeg")        # pinned recording name but a different subject
    assert BR.plan_repair("SCZ", "ds004000", "sub-999", vhdr) is None
    ok("a broken-pointer recording under a subject != sub-042 is not rewritten")


def test_other_cohort_not_repaired():
    vhdr = _broken_pointer("sub-042_task-proposer_run-1_eeg")
    for disease, cohort in (("SCZ", "ds004367"), ("SCZ", "ds003944"), ("PD", "ds004584")):
        assert BR.plan_repair(disease, cohort, "sub-042", vhdr) is None
    ok("a broken-pointer recording outside ds004000 is not rewritten (no general BrainVision repair)")


def main():
    print("ACAR v5 Stage-1B12 guard: broken-pointer repair forbidden outside the pinned sub-042 recordings")
    test_other_recording_of_sub042_not_repaired()
    test_other_subject_not_repaired()
    test_other_cohort_not_repaired()
    print("ALL V5 STAGE1B12-BV-POINTER-REWRITE-FORBIDDEN GUARDS PASS")


if __name__ == "__main__":
    main()
