"""Guard (Stage-1B11): interpolation is AUDITED — SubjectWindows.provenance records the channel-alias + montage-completion policy
hashes and which channels were interpolated; SubjectWindows.montage_completion carries the structured record. Synthetic mne RawArray."""
from __future__ import annotations
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.tests._util import ok, make_mne_raw, modern_channel_names


def test_provenance_records_interpolation():
    present = [c for c in modern_channel_names() if c != "Pz"] + ["AF3", "FC1"]
    sw = RMR.raw_to_windows(make_mne_raw(present, 2048, 256.0), "PD", "ds004584", "sub-001")
    assert "interpolated=['Pz']" in sw.provenance
    assert f"channel_alias_policy_sha256={PC.channel_alias_policy_sha256()}" in sw.provenance
    assert f"montage_completion_policy_sha256={PC.montage_completion_policy_sha256()}" in sw.provenance
    mc = sw.montage_completion
    assert mc["interpolated"] == ["Pz"] and mc["n_interpolated"] == 1 and mc["donor_count"] > 0
    assert mc["by_recording"] and mc["by_recording"][0]["interpolated"] == ["Pz"]
    ok("SubjectWindows.provenance + .montage_completion record the interpolated channel(s), donor count, and policy hashes")


def test_native_recording_records_no_interpolation():
    sw = RMR.raw_to_windows(make_mne_raw(modern_channel_names(), 2048, 256.0), "PD", "ds002778", "sub-hc1")
    assert "interpolated=[]" in sw.provenance and sw.montage_completion["n_interpolated"] == 0
    ok("a native full-montage recording records interpolated=[] / n_interpolated=0 (Stage-2 can tell native vs completed)")


def main():
    print("ACAR v5 Stage-1B11 guard: montage completion provenance recorded")
    test_provenance_records_interpolation()
    test_native_recording_records_no_interpolation()
    print("ALL V5 STAGE1B-MONTAGE-COMPLETION-PROVENANCE GUARDS PASS")


if __name__ == "__main__":
    main()
