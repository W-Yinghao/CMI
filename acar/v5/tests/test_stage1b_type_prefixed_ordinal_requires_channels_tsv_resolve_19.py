"""Guard (Stage-1B14): the widened ordinal detector does not relax the channels.tsv requirements. A type-prefixed ordinal header whose
channels.tsv does NOT resolve all 19 canonical (or has a row-count/uniqueness/latin-1 problem) → NO rename (fail-closed)."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]
_TYPED = {20: "EOG", 21: "ECG"}


def _typed(raw_dir, stem, tsv_names):
    return make_brainvision_triplet(raw_dir, stem, _REAL, with_marker=False, generic_header=True,
                                    write_channels_tsv=True, channels_tsv_names=tsv_names, ordinal_prefix_overrides=_TYPED)


def test_missing_canonical_blocks_type_prefixed_rename():
    no_fz = ["XYZ" if n == "Fz" else n for n in _REAL]
    plan = BR.plan_repair("SCZ", "ds003944", "sub-x", _typed(tempfile.mkdtemp(), "sub-x_task-Rest_eeg", no_fz))
    assert plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV
    ok("type-prefixed header + channels.tsv missing a canonical (Fz) → no rename (fail-closed)")


def test_row_count_mismatch_blocks_type_prefixed_rename():
    plan = BR.plan_repair("SCZ", "ds003947", "sub-y", _typed(tempfile.mkdtemp(), "sub-y_task-Rest_eeg", _REAL[:-1]))
    assert plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV
    ok("type-prefixed header + channels.tsv row count != header count → no rename (fail-closed)")


def main():
    print("ACAR v5 Stage-1B14 guard: type-prefixed rename still requires channels.tsv to resolve 19 (+ row/unique)")
    test_missing_canonical_blocks_type_prefixed_rename()
    test_row_count_mismatch_blocks_type_prefixed_rename()
    print("ALL V5 STAGE1B14-TYPE-PREFIXED-ORDINAL-CHANNELS-TSV-19 GUARDS PASS")


if __name__ == "__main__":
    main()
