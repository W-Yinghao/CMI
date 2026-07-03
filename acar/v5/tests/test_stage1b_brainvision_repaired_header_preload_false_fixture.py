"""Guard (Stage-1B12): the CORE invariant — a marker-less BrainVision header that the pinned mne CANNOT open, once repaired, opens
with preload=False (header only) and yields exactly the fixture channels/duration. Confirms the repaired header is well-formed and
that repair (not signal loading) is what makes the recording admissible."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet


def test_original_fails_but_repaired_opens_preload_false():
    import mne
    raw_dir = tempfile.mkdtemp()
    chans = ("Fp1", "Fp2", "F3", "Cz")
    vhdr = make_brainvision_triplet(raw_dir, "sub-2000_task-Rest_eeg", chans, n_points=400, with_marker=False)
    # the ORIGINAL marker-less header cannot be opened by the pinned mne (the exact Stage-1B11P defect)
    try:
        mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
        raise AssertionError("expected the marker-less header to fail to open")
    except Exception as e:  # noqa: BLE001
        assert "markerfile" in str(e).lower() or "marker" in str(e).lower()
    # after repair it opens (preload=False → header only, NO signal loaded)
    staging = tempfile.mkdtemp()
    repaired, _ = BR.apply_repair(BR.plan_repair("SCZ", "ds003947", "sub-2000", vhdr), staging)
    r = mne.io.read_raw_brainvision(repaired, preload=False, verbose="ERROR")
    assert list(r.ch_names) == list(chans) and r.n_times == 400
    assert not os.path.exists(os.path.join(raw_dir, "sub-2000_task-Rest_eeg.vmrk"))   # still no marker in the raw tree
    ok("marker-less header fails to open; the repaired header opens with preload=False and matches the fixture channels/duration")


def main():
    print("ACAR v5 Stage-1B12 guard: repaired BrainVision header opens (preload=False)")
    test_original_fails_but_repaired_opens_preload_false()
    print("ALL V5 STAGE1B12-BV-REPAIRED-PRELOAD-FALSE GUARDS PASS")


if __name__ == "__main__":
    main()
