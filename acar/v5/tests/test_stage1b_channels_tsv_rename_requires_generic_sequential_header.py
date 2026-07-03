"""Guard (Stage-1B13): the channels.tsv rename fires ONLY when the raw header is exactly generic-sequential (EEG001..EEG0NN). A header
that already carries REAL names, or a header that is only PARTIALLY generic, is never renamed — the raw header stays decisive."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]


def test_real_named_header_is_not_renamed():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=False, write_channels_tsv=True)   # header ALREADY has real names
    plan = BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr)
    assert plan is not None and plan.mode == BR.MODE_MISSING_MARKER   # marker fix only; channels.tsv does NOT override real names
    ok("a header carrying real electrode names is never renamed from channels.tsv (raw header decisive)")


def test_partially_generic_header_is_not_renamed():
    mixed = [f"EEG{i+1:03d}" for i in range(len(_REAL) - 1)] + ["FOO"]   # generic except the last channel
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-y_task-Rest_eeg", mixed, with_marker=False,
                                    generic_header=False, write_channels_tsv=True, channels_tsv_names=_REAL)
    plan = BR.plan_repair("SCZ", "ds003944", "sub-y", vhdr)
    assert plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV   # not exactly generic-sequential → no rename
    ok("a header that is only partially generic (EEG001..,FOO) is not renamed (must be exactly generic-sequential)")


def main():
    print("ACAR v5 Stage-1B13 guard: channels.tsv rename requires an exactly generic-sequential header")
    test_real_named_header_is_not_renamed()
    test_partially_generic_header_is_not_renamed()
    print("ALL V5 STAGE1B13-CHANNELS-TSV-RENAME-REQUIRES-GENERIC GUARDS PASS")


if __name__ == "__main__":
    main()
