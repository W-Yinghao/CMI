"""Guard (Stage-1B14): the type-prefixed ordinal rename, like the pure-EEG one, never modifies the raw tree — the original
.vhdr/.eeg/channels.tsv are byte-identical after the repair and the repaired header lives only under the staging dir; the manifest
re-verifies (including subtype) and rejects a tampered repaired header."""
from __future__ import annotations
import hashlib
import os
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, expect_raises, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]
_TYPED = {20: "EOG", 21: "ECG"}


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_raw_tree_byte_identical_and_manifest_reverifies():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True, ordinal_prefix_overrides=_TYPED)
    before = {f: _sha(os.path.join(raw_dir, f)) for f in sorted(os.listdir(raw_dir))}
    staging = tempfile.mkdtemp()
    repaired, man = BR.apply_repair(BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr), staging)
    after = {f: _sha(os.path.join(raw_dir, f)) for f in sorted(os.listdir(raw_dir))}
    assert before == after and os.path.dirname(os.path.realpath(repaired)) == os.path.realpath(staging)
    assert man["channel_name_repair_subtype"] == "type_prefixed_ordinal"
    assert BR.assert_manifest_consistent(man) is True
    with open(repaired, "ab") as f:
        f.write(b"\nCh999=BOGUS,,\n")
    expect_raises(BR.BrainvisionReadRepairError, lambda: BR.assert_manifest_consistent(man),
                  "a tampered repaired header must fail re-verification")
    ok("type-prefixed rename leaves the raw tree byte-identical; manifest re-verifies (subtype) and rejects a tampered header")


def main():
    print("ACAR v5 Stage-1B14 guard: type-prefixed ordinal rename preserves the raw tree")
    test_raw_tree_byte_identical_and_manifest_reverifies()
    print("ALL V5 STAGE1B14-TYPE-PREFIXED-ORDINAL-PRESERVES-RAW GUARDS PASS")


if __name__ == "__main__":
    main()
