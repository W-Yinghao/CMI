"""Guard (Stage-1B14): EVERY header channel must be an ordinal placeholder. A single real-named channel among the ordinals (partial
pattern) → NO rename (fail-closed; no partial rename)."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]


def test_one_real_named_channel_blocks_rename():
    # header: EEG001.. ordinals except position 5 which carries a REAL electrode name ("Fz")
    header = [f"EEG{i+1:03d}" for i in range(len(_REAL))]
    header[4] = "Fz"
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", header, with_marker=False,
                                    generic_header=False, write_channels_tsv=True, channels_tsv_names=_REAL)
    plan = BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr)
    assert plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV
    ok("a header with one real-named channel among the ordinals → no rename (all channels must be ordinal placeholders)")


def main():
    print("ACAR v5 Stage-1B14 guard: ordinal rename requires ALL channels ordinal")
    test_one_real_named_channel_blocks_rename()
    print("ALL V5 STAGE1B14-TYPE-PREFIXED-ORDINAL-ALL-CHANNELS GUARDS PASS")


if __name__ == "__main__":
    main()
