"""Guard (Stage-1B12): a marker-less BrainVision header in a WHITELISTED cohort (ds003944 / ds003947) gets the
missing_markerfile_minimal_vmrk repair — a synthesized minimal marker + a repaired header (pointing at the ORIGINAL .eeg) under the
staging dir — and the repaired header opens. SYNTHETIC triplet; real mne read of the repaired header."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet


def _marker_less(stem="sub-1448_task-Rest_eeg"):
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, stem, ("Fp1", "Fp2", "Cz"), n_points=300, with_marker=False)
    return raw_dir, vhdr


def test_missing_marker_repair_allowed_both_cohorts():
    import mne
    for cohort in ("ds003944", "ds003947"):
        raw_dir, vhdr = _marker_less()
        staging = tempfile.mkdtemp()
        plan = BR.plan_repair("SCZ", cohort, "sub-1448", vhdr)
        assert plan is not None and plan.mode == BR.MODE_MISSING_MARKER
        assert plan.marker_target == "" and plan.data_target == os.path.abspath(os.path.join(raw_dir, "sub-1448_task-Rest_eeg.eeg"))
        repaired, man = BR.apply_repair(plan, staging)
        assert os.path.realpath(repaired).startswith(os.path.realpath(staging) + os.sep)
        assert man["repair_mode"] == BR.MODE_MISSING_MARKER and man["generated_marker_sha256"] is not None
        r = mne.io.read_raw_brainvision(repaired, preload=False, verbose="ERROR")
        assert list(r.ch_names) == ["Fp1", "Fp2", "Cz"]
        # the synthesized marker lives in staging, NOT in the raw tree
        assert os.path.realpath(man["marker_file_target"]).startswith(os.path.realpath(staging) + os.sep)
    ok("marker-less BrainVision in ds003944/ds003947 → synthesized minimal marker + repaired header (in staging) that mne opens")


def test_whitelisted_cohort_but_marker_present_is_not_repaired():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-1448_task-Rest_eeg", ("Fp1", "Fp2"), with_marker=True)   # has a MarkerFile
    assert BR.plan_repair("SCZ", "ds003944", "sub-1448", vhdr) is None
    ok("a whitelisted-cohort recording that already has a MarkerFile is NOT repaired (repair only fixes the exact defect)")


def main():
    print("ACAR v5 Stage-1B12 guard: missing-marker BrainVision repair allowed (SCZ ds003944/ds003947)")
    test_missing_marker_repair_allowed_both_cohorts()
    test_whitelisted_cohort_but_marker_present_is_not_repaired()
    print("ALL V5 STAGE1B12-BV-MISSING-MARKER-ALLOWED GUARDS PASS")


if __name__ == "__main__":
    main()
