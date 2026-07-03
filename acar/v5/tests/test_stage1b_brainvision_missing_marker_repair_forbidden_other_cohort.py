"""Guard (Stage-1B12): the missing_markerfile_minimal_vmrk repair is NARROW — a marker-less BrainVision header in a NON-whitelisted
cohort gets NO repair plan (so it fails closed at mne, not silently repaired). Only ds003944/ds003947 are eligible."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet


def _marker_less(stem="sub-x_task-Rest_eeg"):
    raw_dir = tempfile.mkdtemp()
    return make_brainvision_triplet(raw_dir, stem, ("Fp1", "Fp2"), with_marker=False)


def test_non_whitelisted_cohorts_get_no_missing_marker_repair():
    vhdr = _marker_less()
    for disease, cohort in (("PD", "ds002778"), ("PD", "ds003490"), ("PD", "ds004584"),
                            ("SCZ", "ds004000"), ("SCZ", "ds004367")):
        assert BR.plan_repair(disease, cohort, "sub-x", vhdr) is None, f"{cohort} must NOT get a missing-marker repair"
    ok("a marker-less header outside ds003944/ds003947 gets NO repair plan (fail-closed, not silently repaired)")


def test_pointer_rewrite_cohort_does_not_borrow_missing_marker_mode():
    # ds004000 is the pointer-rewrite cohort; a marker-less header there is NOT the pointer defect and is not whitelisted for mode A
    vhdr = _marker_less("sub-042_task-proposer_run-1_eeg")
    assert BR.plan_repair("SCZ", "ds004000", "sub-042", vhdr) is None
    ok("ds004000 (pointer-rewrite cohort) does not fall through to the missing-marker mode for a genuinely marker-less header")


def main():
    print("ACAR v5 Stage-1B12 guard: missing-marker BrainVision repair forbidden outside the whitelisted cohorts")
    test_non_whitelisted_cohorts_get_no_missing_marker_repair()
    test_pointer_rewrite_cohort_does_not_borrow_missing_marker_mode()
    print("ALL V5 STAGE1B12-BV-MISSING-MARKER-FORBIDDEN GUARDS PASS")


if __name__ == "__main__":
    main()
