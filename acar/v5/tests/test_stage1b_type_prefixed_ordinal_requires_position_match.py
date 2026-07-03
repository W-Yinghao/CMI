"""Guard (Stage-1B14): the ordinal in a placeholder name must equal the 1-based data-column position for EVERY channel. A name whose
integer does not match its position (e.g. EOG099 at position 20) → NO rename (fail-closed → marker-only fallback)."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]


def test_ordinal_must_equal_position():
    # a valid prefix (EOG) but the integer 99 != position 20 → not an ordinal placeholder → no rename
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True, ordinal_prefix_overrides={20: "EOG"})
    # rewrite the header so position 20's number is 99 instead of 20 (position mismatch)
    import re
    txt = open(vhdr, encoding="latin-1").read().replace("Ch20=EOG020", "Ch20=EOG099")
    open(vhdr, "w", encoding="latin-1").write(txt)
    plan = BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr)
    assert plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV
    ok("a placeholder whose integer != its 1-based position (EOG099@20) → no rename (fail-closed)")


def main():
    print("ACAR v5 Stage-1B14 guard: ordinal placeholder requires integer == position")
    test_ordinal_must_equal_position()
    print("ALL V5 STAGE1B14-TYPE-PREFIXED-ORDINAL-POSITION-MATCH GUARDS PASS")


if __name__ == "__main__":
    main()
