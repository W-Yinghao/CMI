"""Guard (Stage-1B12): the broken_internal_pointer_rewrite repair fixes ds004000/sub-042's two exact recordings whose .vhdr declares
DataFile/MarkerFile pointers to files that were renamed at BIDS-ification. The repaired header repoints at the EXISTING BIDS sibling
data/marker files (no marker synthesized) and opens. SYNTHETIC triplet with stale internal pointers; real mne read."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet

_RECORDINGS = ("sub-042_task-proposer_run-1_eeg", "sub-042_task-responder_run-1_eeg")


def _broken_pointer(stem):
    """A BrainVision triplet whose header points at bogus (pre-rename) DataFile/MarkerFile, but whose real BIDS siblings exist."""
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, stem, ("Fp1", "Fp2", "Cz"), n_points=300, with_marker=True,
                                    data_file="019_P1.dat", marker_file="019_P1.vmrk")
    return raw_dir, vhdr


def test_pointer_rewrite_repairs_both_sub042_recordings():
    import mne
    for stem in _RECORDINGS:
        raw_dir, vhdr = _broken_pointer(stem)
        staging = tempfile.mkdtemp()
        plan = BR.plan_repair("SCZ", "ds004000", "sub-042", vhdr)
        assert plan is not None and plan.mode == BR.MODE_POINTER_REWRITE
        assert plan.data_target == os.path.abspath(os.path.join(raw_dir, stem + ".eeg"))
        assert plan.marker_target == os.path.abspath(os.path.join(raw_dir, stem + ".vmrk"))
        repaired, man = BR.apply_repair(plan, staging)
        assert man["generated_marker_sha256"] is None                     # no marker synthesized (the real one exists)
        assert BR.assert_manifest_consistent(man) is True
        r = mne.io.read_raw_brainvision(repaired, preload=True, verbose="ERROR")
        assert list(r.ch_names) == ["Fp1", "Fp2", "Cz"] and r.n_times == 300
    ok("ds004000/sub-042's two recordings with stale DataFile/MarkerFile pointers → repaired header repoints at the BIDS siblings")


def test_pointer_rewrite_requires_pointers_actually_broken():
    # a sub-042 recording whose declared pointers already resolve is NOT rewritten (nothing to repair)
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, _RECORDINGS[0], ("Fp1", "Fp2"), with_marker=True)   # correct internal pointers
    assert BR.plan_repair("SCZ", "ds004000", "sub-042", vhdr) is None
    ok("a sub-042 recording whose internal pointers already resolve is not rewritten (repair only fixes the actual defect)")


def main():
    print("ACAR v5 Stage-1B12 guard: broken-pointer BrainVision repair (ds004000 sub-042)")
    test_pointer_rewrite_repairs_both_sub042_recordings()
    test_pointer_rewrite_requires_pointers_actually_broken()
    print("ALL V5 STAGE1B12-BV-POINTER-REWRITE-SUB042 GUARDS PASS")


if __name__ == "__main__":
    main()
