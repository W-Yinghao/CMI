"""Guard (Stage-1B14): a TYPE-PREFIXED ordinal BrainVision header (EOG/ECG on the eye/cardiac channels, EEG elsewhere; every integer ==
its 1-based position) in ds003944/ds003947 with a valid channels.tsv gets the channels.tsv row-order rename, recorded with
subtype='type_prefixed_ordinal'; a pure-EEG ordinal header still records subtype='pure_eeg_ordinal'."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.substrate import channel_aliases as CA
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]                    # 21 rows: 19 canonical (modern) + 2 non-canonical extras
_TYPED = {20: "EOG", 21: "ECG"}                                    # positions 20/21 → EOG020 / ECG021 in the header


def test_type_prefixed_ordinal_renamed_both_cohorts():
    import mne
    for cohort in ("ds003944", "ds003947"):
        raw_dir = tempfile.mkdtemp()
        vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, with_marker=False,
                                        generic_header=True, write_channels_tsv=True, ordinal_prefix_overrides=_TYPED)
        plan = BR.plan_repair("SCZ", cohort, "sub-x", vhdr)
        assert plan is not None and plan.mode == BR.MODE_CHANNEL_NAMES_FROM_TSV
        assert plan.channel_name_repair_subtype == "type_prefixed_ordinal"
        assert tuple(plan.original_header_ordinal_prefixes[19:21]) == ("EOG", "ECG")
        repaired, man = BR.apply_repair(plan, tempfile.mkdtemp())
        assert man["channel_name_repair_subtype"] == "type_prefixed_ordinal"
        r = mne.io.read_raw_brainvision(repaired, preload=False, verbose="ERROR")
        assert list(r.ch_names) == _REAL and sum(1 for n in r.ch_names if CA.normalize_channel(n)) == 19
    ok("type-prefixed ordinal header (EOG/ECG) in ds003944/ds003947 → row-order rename, subtype=type_prefixed_ordinal, resolves 19")


def test_pure_eeg_ordinal_records_pure_subtype():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-y_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True)   # all EEG00i
    plan = BR.plan_repair("SCZ", "ds003944", "sub-y", vhdr)
    _, man = BR.apply_repair(plan, tempfile.mkdtemp())
    assert plan.channel_name_repair_subtype == "pure_eeg_ordinal" and man["channel_name_repair_subtype"] == "pure_eeg_ordinal"
    ok("a pure-EEG ordinal header records subtype=pure_eeg_ordinal (Stage-2 can distinguish the two)")


def main():
    print("ACAR v5 Stage-1B14 guard: type-prefixed ordinal rename allowed (ds003944/ds003947)")
    test_type_prefixed_ordinal_renamed_both_cohorts()
    test_pure_eeg_ordinal_records_pure_subtype()
    print("ALL V5 STAGE1B14-TYPE-PREFIXED-ORDINAL-ALLOWED GUARDS PASS")


if __name__ == "__main__":
    main()
