"""Guard (Stage-1B13): the channels.tsv rename requires channels.tsv names to be unique after strip+casefold. A duplicate name → no
rename (fail-closed)."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]


def test_duplicate_names_block_rename():
    dup = list(_REAL)
    dup[1] = dup[0]                                             # channels.tsv now has a duplicate logical name
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True, channels_tsv_names=dup)
    plan = BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr)
    assert plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV
    ok("a channels.tsv with duplicate names (after strip+casefold) → no rename (fail-closed)")


def test_casefold_duplicate_names_block_rename():
    dup = list(_REAL)
    dup[1] = dup[0].upper()                                     # same name differing only in case → still a duplicate
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-y_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True, channels_tsv_names=dup)
    plan = BR.plan_repair("SCZ", "ds003944", "sub-y", vhdr)
    assert plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV
    ok("a channels.tsv duplicate that differs only by case → no rename (uniqueness is casefolded)")


def main():
    print("ACAR v5 Stage-1B13 guard: channels.tsv rename requires unique names")
    test_duplicate_names_block_rename()
    test_casefold_duplicate_names_block_rename()
    print("ALL V5 STAGE1B13-CHANNELS-TSV-RENAME-UNIQUE GUARDS PASS")


if __name__ == "__main__":
    main()
