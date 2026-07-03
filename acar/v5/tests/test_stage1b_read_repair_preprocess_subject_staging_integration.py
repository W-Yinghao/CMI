"""Guard (Stage-1B12): end-to-end — preprocess_subject(..., staging_dir=...) opens a marker-less BrainVision recording via the
reviewed read-repair (repaired header materialized in staging), runs the pinned DSP, and returns a validated SubjectWindows whose
provenance + read_repair record the repair. The raw tree is untouched. SYNTHETIC 19-channel BIDS subject; real mne."""
from __future__ import annotations
import hashlib
import os
import tempfile
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _bids_subject(stem):
    root = tempfile.mkdtemp()
    eeg = os.path.join(root, "sub-1448", "eeg")
    vhdr = make_brainvision_triplet(eeg, stem, modern_channel_names(), n_points=3000, sfreq=256.0, with_marker=False)
    return os.path.join(root, "sub-1448"), eeg, vhdr


def test_preprocess_subject_repairs_and_records():
    import mne  # noqa: F401  (ensures the real numeric env is present)
    stem = "sub-1448_task-Rest_eeg"
    subject_dir, eeg_dir, vhdr = _bids_subject(stem)
    before = {f: _sha(os.path.join(eeg_dir, f)) for f in sorted(os.listdir(eeg_dir))}
    staging = tempfile.mkdtemp()
    sw = RMR.preprocess_subject("SCZ", "ds003944", "sub-1448", subject_dir, staging_dir=staging)
    # validated canonical SubjectWindows
    assert tuple(sw.channels) == PC.CHANNELS_19 and sw.n_channels == 19 and sw.sfreq == 128 and sw.n_samples == 512
    assert sw.n_windows >= 1
    # read-repair recorded (subject-level + provenance)
    assert sw.read_repair["repaired"] == [stem + ".vhdr"]
    assert len(sw.read_repair["by_recording"]) == 1
    assert sw.read_repair["by_recording"][0]["repair_mode"] == "missing_markerfile_minimal_vmrk"
    assert "n_read_repaired=1" in sw.provenance
    assert f"brainvision_read_repair_policy_sha256={PC.brainvision_read_repair_policy_sha256()}" in sw.provenance
    # native 19 → no interpolation
    assert sw.montage_completion["n_interpolated"] == 0
    # raw tree untouched (no .vmrk written into eeg/)
    after = {f: _sha(os.path.join(eeg_dir, f)) for f in sorted(os.listdir(eeg_dir))}
    assert before == after and (stem + ".vmrk") not in after
    ok("preprocess_subject(staging_dir=...) read-repairs a marker-less recording, runs DSP, records read_repair, leaves raw untouched")


def main():
    print("ACAR v5 Stage-1B12 guard: preprocess_subject read-repair staging integration")
    test_preprocess_subject_repairs_and_records()
    print("ALL V5 STAGE1B12-READ-REPAIR-PREPROCESS-INTEGRATION GUARDS PASS")


if __name__ == "__main__":
    main()
