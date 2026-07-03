"""Guard (Stage-1B14): widening the ordinal prefix set does NOT weaken raw-header decisiveness. A header carrying REAL electrode names
(including EEG/EOG/ECG-adjacent real names like Fp1, F7, O2 that are NOT <prefix><ordinal>) is never renamed from channels.tsv."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]


def test_real_named_header_never_renamed():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=False, write_channels_tsv=True)   # header ALREADY has the real names
    plan = BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr)
    assert plan is not None and plan.mode == BR.MODE_MISSING_MARKER   # marker-only fix; channels.tsv does NOT override real names
    ok("a header with real electrode names is never renamed under the widened detector (raw header decisive)")


def test_real_name_that_starts_like_a_prefix_is_not_ordinal():
    # 'ECG' or 'EOG' as a BARE real channel name (no ordinal) must NOT be treated as an ordinal placeholder
    header = [f"EEG{i+1:03d}" for i in range(len(_REAL))]
    header[20] = "ECG"        # bare 'ECG' (no ordinal digits) at position 21
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-y_task-Rest_eeg", header, with_marker=False,
                                    generic_header=False, write_channels_tsv=True, channels_tsv_names=_REAL)
    plan = BR.plan_repair("SCZ", "ds003944", "sub-y", vhdr)
    assert plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV   # 'ECG' with no ordinal is not <prefix><position>
    ok("a bare 'ECG'/'EOG' real channel name (no ordinal digits) is not treated as an ordinal placeholder")


def main():
    print("ACAR v5 Stage-1B14 guard: ordinal rename never overrides real header names")
    test_real_named_header_never_renamed()
    test_real_name_that_starts_like_a_prefix_is_not_ordinal()
    print("ALL V5 STAGE1B14-TYPE-PREFIXED-ORDINAL-NO-OVERRIDE GUARDS PASS")


if __name__ == "__main__":
    main()
