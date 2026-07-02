"""Guard (Stage-1B7): raw recording discovery is RAW-BIDS-ONLY — it looks only in <sub>/eeg and <sub>/ses-*/eeg, ignores files
elsewhere (subject root, derivatives/, sourcedata/), rejects symlinked recordings, and produces a deterministic hashed manifest.
Synthetic temp files only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import raw_recording_manifest as RM
from acar.v5.tests._util import expect_raises, ok


def _touch(*parts):
    p = os.path.join(*parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(b"EEGDATA")
    return p


def test_only_eeg_dirs_discovered():
    with tempfile.TemporaryDirectory() as sub:
        good1 = _touch(sub, "eeg", "sub-001_task-rest_eeg.edf")
        good2 = _touch(sub, "ses-01", "eeg", "sub-001_ses-01_eeg.edf")
        _touch(sub, "sub-001_stray_eeg.edf")                          # in subject ROOT (ignored)
        _touch(sub, "derivatives", "eeg", "sub-001_desc-clean_eeg.edf")   # derivatives (ignored — not <sub>/eeg)
        _touch(sub, "sourcedata", "eeg", "sub-001_eeg.edf")          # sourcedata (ignored)
        found = RM.discover_raw_recordings(sub)
        assert found == sorted([good1, good2]), found
    ok("discovery returns ONLY <sub>/eeg and <sub>/ses-*/eeg recordings (root/derivatives/sourcedata ignored)")


def test_no_eeg_dir_fails_closed():
    with tempfile.TemporaryDirectory() as sub:
        _touch(sub, "anat", "sub-001_T1w.nii")                       # no eeg/ at all
        expect_raises(RM.RawManifestError, lambda: RM.discover_raw_recordings(sub))
    with tempfile.TemporaryDirectory() as sub:
        os.makedirs(os.path.join(sub, "eeg"))                        # eeg/ exists but empty of recordings
        expect_raises(RM.RawManifestError, lambda: RM.discover_raw_recordings(sub))
    ok("no eeg/ directory or an eeg/ with no raw recording → RawManifestError")


def test_symlinked_recording_rejected():
    with tempfile.TemporaryDirectory() as sub, tempfile.TemporaryDirectory() as outside:
        target = _touch(outside, "real.edf")
        link = os.path.join(sub, "eeg", "sub-001_eeg.edf")
        os.makedirs(os.path.dirname(link))
        os.symlink(target, link)
        expect_raises(RM.RawManifestError, lambda: RM.discover_raw_recordings(sub))
    ok("a symlinked recording under eeg/ → RawManifestError")


def test_symlinked_eeg_or_session_dir_rejected():
    # a symlinked eeg/ directory (could point into derivatives) → rejected
    with tempfile.TemporaryDirectory() as sub, tempfile.TemporaryDirectory() as other:
        _touch(other, "eeg", "x_eeg.edf")
        os.symlink(os.path.join(other, "eeg"), os.path.join(sub, "eeg"))
        expect_raises(RM.RawManifestError, lambda: RM.discover_raw_recordings(sub))
    # a symlinked session directory (ses-01 -> derivatives) → rejected before any file is admitted
    with tempfile.TemporaryDirectory() as sub, tempfile.TemporaryDirectory() as deriv:
        _touch(deriv, "eeg", "leak_eeg.edf")
        os.symlink(deriv, os.path.join(sub, "ses-01"))
        expect_raises(RM.RawManifestError, lambda: RM.discover_raw_recordings(sub))
    ok("a symlinked eeg/ or ses-*/ directory (pointing into derivatives) → RawManifestError (no normpath bypass)")


def test_manifest_deterministic_hashed():
    with tempfile.TemporaryDirectory() as sub:
        _touch(sub, "eeg", "sub-001_run-1_eeg.edf")
        _touch(sub, "eeg", "sub-001_run-2_eeg.edf")
        m1, m2 = RM.build_manifest(sub), RM.build_manifest(sub)
        assert m1["manifest_sha256"] == m2["manifest_sha256"] and len(m1["files"]) == 2
        assert all(len(e["sha256"]) == 64 and e["n_bytes"] > 0 for e in m1["files"])
    ok("build_manifest is deterministic (stable manifest_sha256) with per-file sha256 + sizes")


def main():
    print("ACAR v5 Stage-1B7 guard: raw BIDS discovery excludes derivatives")
    test_only_eeg_dirs_discovered()
    test_no_eeg_dir_fails_closed()
    test_symlinked_recording_rejected()
    test_symlinked_eeg_or_session_dir_rejected()
    test_manifest_deterministic_hashed()
    print("ALL V5 STAGE1B-RAW-BIDS-DISCOVERY GUARDS PASS")


if __name__ == "__main__":
    main()
