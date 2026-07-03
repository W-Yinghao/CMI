"""Guard (Stage-1B12): every repair emits an audited manifest — original/repaired header + synthesized-marker sha256 match the bytes
on disk; assert_manifest_consistent re-verifies before the reader consumes the repaired header, and rejects a tampered header."""
from __future__ import annotations
import hashlib
import os
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, expect_raises, make_brainvision_triplet


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_manifest_hashes_match_disk_and_reverify():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-1448_task-Rest_eeg", ("Fp1", "Fp2"), with_marker=False)
    staging = tempfile.mkdtemp()
    repaired, man = BR.apply_repair(BR.plan_repair("SCZ", "ds003944", "sub-1448", vhdr), staging)
    assert man["original_header_sha256"] == _sha(vhdr)
    assert man["repaired_header_sha256"] == _sha(repaired)
    assert man["generated_marker_sha256"] == _sha(man["marker_file_target"])
    assert man["brainvision_read_repair_policy_sha256"] and len(man["brainvision_read_repair_policy_sha256"]) == 64
    assert BR.assert_manifest_consistent(man) is True
    ok("manifest original/repaired/generated-marker sha256 match the on-disk bytes and re-verify")


def test_tampered_repaired_header_is_rejected():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-1448_task-Rest_eeg", ("Fp1", "Fp2"), with_marker=False)
    staging = tempfile.mkdtemp()
    repaired, man = BR.apply_repair(BR.plan_repair("SCZ", "ds003944", "sub-1448", vhdr), staging)
    with open(repaired, "ab") as f:
        f.write(b"\n; tampered\n")
    expect_raises(BR.BrainvisionReadRepairError, lambda: BR.assert_manifest_consistent(man),
                  "a tampered repaired header must fail re-verification")
    ok("assert_manifest_consistent rejects a repaired header whose bytes no longer match the manifest hash")


def test_empty_manifest_set_hash_is_deterministic_hex64():
    h = BR.manifest_set_sha256([])
    assert h == BR.EMPTY_MANIFEST_SET_SHA256 and len(h) == 64
    assert BR.manifest_set_sha256([]) == h
    ok("manifest_set_sha256([]) is a deterministic hex64 sentinel (the no-repair feature-dump default)")


def main():
    print("ACAR v5 Stage-1B12 guard: read-repair manifest hashes")
    test_manifest_hashes_match_disk_and_reverify()
    test_tampered_repaired_header_is_rejected()
    test_empty_manifest_set_hash_is_deterministic_hex64()
    print("ALL V5 STAGE1B12-BV-MANIFEST-HASHES GUARDS PASS")


if __name__ == "__main__":
    main()
