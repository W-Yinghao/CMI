"""Guard (Stage-1B13): the channels.tsv rename never modifies the raw tree — the original .vhdr, .eeg, and channels.tsv are
byte-identical after the repair, and the repaired header lives only under the staging dir."""
from __future__ import annotations
import hashlib
import os
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_raw_tree_byte_identical_after_rename():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True)
    before = {f: _sha(os.path.join(raw_dir, f)) for f in sorted(os.listdir(raw_dir))}
    staging = tempfile.mkdtemp()
    repaired, man = BR.apply_repair(BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr), staging)
    after = {f: _sha(os.path.join(raw_dir, f)) for f in sorted(os.listdir(raw_dir))}
    assert before == after, "raw tree changed during the rename repair"
    assert set(before) == {"sub-x_task-Rest_eeg.eeg", "sub-x_task-Rest_eeg.vhdr", "sub-x_task-Rest_channels.tsv"}
    assert os.path.dirname(os.path.realpath(repaired)) == os.path.realpath(staging)
    # the channels.tsv used as the rename SOURCE is the untouched raw-tree file
    assert os.path.realpath(man["channels_tsv_path"]) == os.path.realpath(os.path.join(raw_dir, "sub-x_task-Rest_channels.tsv"))
    ok("the rename leaves the original .vhdr/.eeg/channels.tsv byte-identical; the repaired header is only in staging")


def main():
    print("ACAR v5 Stage-1B13 guard: channels.tsv rename preserves the raw tree")
    test_raw_tree_byte_identical_after_rename()
    print("ALL V5 STAGE1B13-CHANNELS-TSV-RENAME-PRESERVES-RAW GUARDS PASS")


if __name__ == "__main__":
    main()
