"""Guard (Stage-1B14): only the PINNED ordinal prefixes {EEG, EOG, ECG} are accepted. A different prefix (e.g. GSR, MISC, REF) — even
with a correct position ordinal — is NOT an accepted ordinal placeholder → NO rename (fail-closed). No arbitrary alpha prefixes."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]


def test_arbitrary_prefix_blocks_rename():
    for bad_prefix in ("GSR", "MISC", "REF", "STA"):
        raw_dir = tempfile.mkdtemp()
        vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, with_marker=False,
                                        generic_header=True, write_channels_tsv=True, ordinal_prefix_overrides={20: bad_prefix})
        plan = BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr)
        assert plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV, f"prefix {bad_prefix} must NOT be renamed"
    ok("an ordinal name with a non-{EEG,EOG,ECG} prefix (GSR/MISC/REF/STA) → no rename (only the pinned prefix set)")


def test_pinned_prefix_set_is_exactly_eeg_eog_ecg():
    assert PC.PREPROCESSING_CONFIG["channel_name_repair_allowed_ordinal_prefixes"] == ["EEG", "EOG", "ECG"]
    ok("the pinned ordinal prefix set is exactly {EEG, EOG, ECG}")


def main():
    print("ACAR v5 Stage-1B14 guard: ordinal rename rejects arbitrary prefixes")
    test_arbitrary_prefix_blocks_rename()
    test_pinned_prefix_set_is_exactly_eeg_eog_ecg()
    print("ALL V5 STAGE1B14-TYPE-PREFIXED-ORDINAL-REJECTS-ARBITRARY GUARDS PASS")


if __name__ == "__main__":
    main()
