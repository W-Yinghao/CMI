"""Guard (Stage-1B15): END-TO-END through the PRODUCTION reader — RealBidsDevReader.read_subject_windows, with a real repair staging
root, repairs a generic-ordinal ds003944 BrainVision recording (the exact class that crashed the pre-wiring real build) and returns a
validated canonical SubjectWindows with the channels.tsv rename recorded; the raw tree is left untouched. SYNTHETIC BIDS fixture; real
mne (no real DEV data)."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import real_dev_reader as RDR
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.tests._util import ok, stage1b_reader_ctx, make_brainvision_triplet, modern_channel_names


def test_production_reader_repairs_generic_ordinal_scz_recording():
    import mne  # noqa: F401  (ensures the real numeric env is present)
    cohort_root = tempfile.mkdtemp()                            # the approved raw cohort source path
    eeg = os.path.join(cohort_root, "sub-1448", "eeg")
    make_brainvision_triplet(eeg, "sub-1448_task-Rest_eeg", modern_channel_names(), n_points=3000, sfreq=256.0,
                             with_marker=False, generic_header=True, write_channels_tsv=True)   # marker-less + EEG001..EEG019
    before = sorted(os.listdir(eeg))
    staging = tempfile.mkdtemp()                                # a real, disjoint repair staging root
    reader = RDR.make_real_dev_reader(stage1b_reader_ctx("SCZ", "ds003944", cohort_root, staging))
    sw = reader.read_subject_windows("SCZ", "ds003944", "sub-1448", cohort_root)
    # validated canonical SubjectWindows produced via the reviewed repair
    assert tuple(sw.channels) == PC.CHANNELS_19 and sw.n_channels == 19 and sw.sfreq == 128 and sw.n_samples == 512
    assert sw.n_windows >= 1
    assert sw.read_repair["channel_name_repaired"] == ["sub-1448_task-Rest_eeg.vhdr"]
    assert "n_channel_name_repaired=1" in sw.provenance
    # raw tree untouched; per-call staging subdir cleaned; staging root empty again
    assert sorted(os.listdir(eeg)) == before and "sub-1448_task-Rest_eeg.vmrk" not in before
    assert os.path.isdir(staging) and os.listdir(staging) == []
    ok("production RealBidsDevReader repairs a generic-ordinal ds003944 recording → canonical SubjectWindows (rename recorded); raw untouched")


def main():
    print("ACAR v5 Stage-1B15 guard: production reader uses the repair branch (SCZ fixture)")
    test_production_reader_repairs_generic_ordinal_scz_recording()
    print("ALL V5 STAGE1B15-REAL-READER-REPAIR-BRANCH-SCZ GUARDS PASS")


if __name__ == "__main__":
    main()
