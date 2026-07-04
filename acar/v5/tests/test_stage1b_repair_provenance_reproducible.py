"""Guard (Stage-1B15 review fix): the SubjectWindows.provenance raw_manifest_sha256 is REPRODUCIBLE for a repaired recording across
distinct per-call staging dirs — the ephemeral (random) synthesized-marker PATH must NOT be folded into the audit hash (only its stable
CONTENT hash). SYNTHETIC BIDS fixture; real mne."""
from __future__ import annotations
import os
import re
import tempfile
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names


def _raw_manifest_sha256(sw):
    m = re.search(r"raw_manifest_sha256=([0-9a-f]{64})", sw.provenance)
    return m.group(1) if m else None


def test_marker_synth_recording_manifest_hash_reproducible():
    import mne  # noqa: F401
    root = tempfile.mkdtemp()
    make_brainvision_triplet(os.path.join(root, "sub-1448", "eeg"), "sub-1448_task-Rest_eeg", modern_channel_names(),
                             n_points=3000, sfreq=256.0, with_marker=False, generic_header=True, write_channels_tsv=True)
    subject_dir = os.path.join(root, "sub-1448")
    sw1 = RMR.preprocess_subject("SCZ", "ds003944", "sub-1448", subject_dir, staging_dir=tempfile.mkdtemp())
    sw2 = RMR.preprocess_subject("SCZ", "ds003944", "sub-1448", subject_dir, staging_dir=tempfile.mkdtemp())
    h1, h2 = _raw_manifest_sha256(sw1), _raw_manifest_sha256(sw2)
    assert h1 and h1 == h2, (h1, h2)                            # identical despite different ephemeral staging dirs
    ok("a marker-synth recording's raw_manifest_sha256 is reproducible across staging dirs (no ephemeral path in the audit)")


def main():
    print("ACAR v5 Stage-1B15 guard: repair provenance reproducible across staging dirs")
    test_marker_synth_recording_manifest_hash_reproducible()
    print("ALL V5 STAGE1B15-REPAIR-PROVENANCE-REPRODUCIBLE GUARDS PASS")


if __name__ == "__main__":
    main()
