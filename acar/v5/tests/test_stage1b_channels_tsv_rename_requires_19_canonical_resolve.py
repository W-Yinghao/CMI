"""Guard (Stage-1B13): the channels.tsv rename requires ALL 19 canonical channels to resolve (via the Stage-1B10 aliases) with no
logical duplicate. A channels.tsv that does not cover the 19 canonical → no rename (fail-closed)."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]


def test_missing_canonical_blocks_rename():
    no_fz = ["XYZ" if n == "Fz" else n for n in _REAL]         # drop canonical Fz (replaced by a non-canonical name)
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True, channels_tsv_names=no_fz)
    plan = BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr)
    assert plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV
    ok("a channels.tsv missing a canonical channel (Fz) → no rename (fail-closed)")


def test_logical_duplicate_canonical_blocks_rename():
    # two names collapse to the SAME canonical (T7→T3 and T3 both present) → logical duplicate → resolve fails → no rename
    dup_logical = ["T3" if n == "VEOG" else n for n in _REAL]  # _REAL already has T7 (→T3); adding T3 makes a logical dup
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-y_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True, channels_tsv_names=dup_logical)
    plan = BR.plan_repair("SCZ", "ds003944", "sub-y", vhdr)
    assert plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV
    ok("a channels.tsv where two names map to the same canonical channel → no rename (fail-closed)")


def main():
    print("ACAR v5 Stage-1B13 guard: channels.tsv rename requires all 19 canonical to resolve")
    test_missing_canonical_blocks_rename()
    test_logical_duplicate_canonical_blocks_rename()
    print("ALL V5 STAGE1B13-CHANNELS-TSV-RENAME-19-CANONICAL GUARDS PASS")


if __name__ == "__main__":
    main()
