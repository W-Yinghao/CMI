"""Guard (Stage-1B13): the rename manifest records channels_tsv_sha256 + channel_name_mapping_sha256 + original/repaired header-name
sha256; assert_manifest_consistent re-verifies them and rejects a tampered repaired header OR a tampered channels.tsv source."""
from __future__ import annotations
import hashlib
import os
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, expect_raises, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]


def _mk():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True)
    return BR.apply_repair(BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr), tempfile.mkdtemp())


def test_manifest_channel_name_hashes_match_and_reverify():
    repaired, man = _mk()
    for k in ("channel_name_source", "channels_tsv_path", "channels_tsv_sha256", "channel_name_mapping_sha256",
              "original_header_channel_names_sha256", "repaired_header_channel_names_sha256",
              "channel_name_repair_policy_sha256"):
        assert k in man, f"manifest missing {k}"
    assert man["channels_tsv_sha256"] == hashlib.sha256(open(man["channels_tsv_path"], "rb").read()).hexdigest()
    assert man["original_header_channel_names_sha256"] != man["repaired_header_channel_names_sha256"]
    assert BR.assert_manifest_consistent(man) is True
    ok("rename manifest records + re-verifies channels_tsv / mapping / original+repaired header-name hashes")


def test_tampered_repaired_header_rejected():
    repaired, man = _mk()
    with open(repaired, "ab") as f:
        f.write(b"\nCh999=BOGUS,,\n")
    expect_raises(BR.BrainvisionReadRepairError, lambda: BR.assert_manifest_consistent(man),
                  "a tampered repaired header must fail re-verification")
    ok("assert_manifest_consistent rejects a tampered repaired header")


def test_tampered_channels_tsv_source_rejected():
    repaired, man = _mk()
    with open(man["channels_tsv_path"], "a") as f:              # mutate the raw-tree channels.tsv AFTER the repair
        f.write("EXTRA\tEEG\n")
    expect_raises(BR.BrainvisionReadRepairError, lambda: BR.assert_manifest_consistent(man),
                  "a channels.tsv whose bytes changed must fail re-verification")
    ok("assert_manifest_consistent rejects a channels.tsv source whose bytes no longer match channels_tsv_sha256")


def main():
    print("ACAR v5 Stage-1B13 guard: channels.tsv rename manifest hashes")
    test_manifest_channel_name_hashes_match_and_reverify()
    test_tampered_repaired_header_rejected()
    test_tampered_channels_tsv_source_rejected()
    print("ALL V5 STAGE1B13-CHANNELS-TSV-RENAME-MANIFEST-HASHES GUARDS PASS")


if __name__ == "__main__":
    main()
