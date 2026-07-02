"""Guard (Stage-1B8): the raw-recording manifest resolves + hashes FORMAT SIDECARS (BrainVision .eeg/.vmrk via the .vhdr header;
EEGLAB .fdt), so the exact bytes mne consumes are audited — not just the header path. Missing/symlinked/escaping sidecars fail
closed. Synthetic temp files only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import raw_recording_manifest as RM
from acar.v5.tests._util import expect_raises, ok


def _mk(path, data=b"DATA"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _vhdr(eeg_dir, base, datafile, markerfile):
    p = os.path.join(eeg_dir, base + ".vhdr")
    with open(p, "w") as f:
        f.write("Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\n"
                f"DataFile={datafile}\nMarkerFile={markerfile}\n")
    return p


def test_brainvision_sidecars_hashed():
    with tempfile.TemporaryDirectory() as sub:
        eeg = os.path.join(sub, "eeg")
        os.makedirs(eeg)
        _vhdr(eeg, "sub-001_eeg", "sub-001_eeg.eeg", "sub-001_eeg.vmrk")
        _mk(os.path.join(eeg, "sub-001_eeg.eeg"))
        _mk(os.path.join(eeg, "sub-001_eeg.vmrk"))
        man = RM.build_manifest(sub)
        roles = {os.path.basename(e["path"]): e["role"] for e in man["files"]}
        assert roles["sub-001_eeg.vhdr"] == "primary"
        assert roles["sub-001_eeg.eeg"] == "sidecar" and roles["sub-001_eeg.vmrk"] == "sidecar"
        assert all(len(e["sha256"]) == 64 for e in man["files"]) and len(man["files"]) == 3
    ok("BrainVision .vhdr → .eeg + .vmrk resolved as sidecars and hashed (3 audited files)")


def test_eeglab_fdt_sidecar_hashed():
    with tempfile.TemporaryDirectory() as sub:
        eeg = os.path.join(sub, "eeg")
        os.makedirs(eeg)
        _mk(os.path.join(eeg, "sub-002_eeg.set"))
        _mk(os.path.join(eeg, "sub-002_eeg.fdt"))
        man = RM.build_manifest(sub)
        roles = {os.path.basename(e["path"]): e["role"] for e in man["files"]}
        assert roles["sub-002_eeg.set"] == "primary" and roles["sub-002_eeg.fdt"] == "sidecar"
    ok("EEGLAB .set → .fdt resolved as a sidecar and hashed")


def test_missing_symlink_escape_sidecars_fail():
    # missing declared DataFile
    with tempfile.TemporaryDirectory() as sub:
        eeg = os.path.join(sub, "eeg")
        os.makedirs(eeg)
        _vhdr(eeg, "sub-001_eeg", "sub-001_eeg.eeg", "sub-001_eeg.vmrk")
        _mk(os.path.join(eeg, "sub-001_eeg.vmrk"))            # .eeg missing
        expect_raises(RM.RawManifestError, lambda: RM.build_manifest(sub))
    # escaping DataFile (path, not bare basename)
    with tempfile.TemporaryDirectory() as sub:
        eeg = os.path.join(sub, "eeg")
        os.makedirs(eeg)
        _vhdr(eeg, "sub-001_eeg", "../../secret.eeg", "sub-001_eeg.vmrk")
        _mk(os.path.join(eeg, "sub-001_eeg.vmrk"))
        expect_raises(RM.RawManifestError, lambda: RM.build_manifest(sub))
    # symlinked sidecar
    with tempfile.TemporaryDirectory() as sub, tempfile.TemporaryDirectory() as outside:
        eeg = os.path.join(sub, "eeg")
        os.makedirs(eeg)
        _vhdr(eeg, "sub-001_eeg", "sub-001_eeg.eeg", "sub-001_eeg.vmrk")
        _mk(os.path.join(eeg, "sub-001_eeg.vmrk"))
        os.symlink(_mk(os.path.join(outside, "real.eeg")), os.path.join(eeg, "sub-001_eeg.eeg"))
        expect_raises(RM.RawManifestError, lambda: RM.build_manifest(sub))
    ok("missing / escaping / symlinked sidecar → RawManifestError (fail-closed)")


def main():
    print("ACAR v5 Stage-1B8 guard: raw sidecar manifest (BrainVision/EEGLAB)")
    test_brainvision_sidecars_hashed()
    test_eeglab_fdt_sidecar_hashed()
    test_missing_symlink_escape_sidecars_fail()
    print("ALL V5 STAGE1B-RAW-SIDECAR-MANIFEST GUARDS PASS")


if __name__ == "__main__":
    main()
