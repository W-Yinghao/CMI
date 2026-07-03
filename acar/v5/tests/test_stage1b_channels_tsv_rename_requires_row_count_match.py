"""Guard (Stage-1B13): the channels.tsv rename requires channels.tsv row count == raw header channel count. A mismatch → no rename
(fail-closed; falls back to the marker-only fix so the generic-name defect is surfaced downstream, never silently renamed)."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]                # 21


def test_row_count_mismatch_blocks_rename():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True, channels_tsv_names=_REAL[:-1])  # 20 rows vs 21
    plan = BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr)
    assert plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV and plan.mode == BR.MODE_MISSING_MARKER
    ok("channels.tsv row count != header channel count → no rename (fail-closed; marker-only fallback)")


def main():
    print("ACAR v5 Stage-1B13 guard: channels.tsv rename requires row-count match")
    test_row_count_mismatch_blocks_rename()
    print("ALL V5 STAGE1B13-CHANNELS-TSV-RENAME-ROW-COUNT GUARDS PASS")


if __name__ == "__main__":
    main()
