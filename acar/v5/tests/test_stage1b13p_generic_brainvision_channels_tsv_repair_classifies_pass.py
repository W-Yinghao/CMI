"""Guard (Stage-1B13P): the preflight classifier turns a generic-headered ds003944/ds003947 BrainVision recording (marker-less +
EEG001..EEG0NN header + valid channels.tsv) into read_repair_plus_channel_name_repair_required (it materializes the repaired header,
opens it at preload=False, and the renamed names resolve all 19 canonical). A generic header whose channels.tsv is INVALID falls back
to the marker-only fix and is surfaced as header_channel_names_non_canonical. SYNTHETIC fixtures; no real data."""
from __future__ import annotations
import tempfile
from acar.v5 import stage1b13p_preflight as P13
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]


def _state():
    st = {"mod": None, "std": None}
    return st, P13._std_positions_casefold(st)


def test_generic_plus_valid_tsv_classifies_repair_plus_name():
    st, std = _state()
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-1448_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True)
    r = P13.classify("SCZ", "ds003944", "sub-1448", vhdr, tempfile.mkdtemp(), st, std, {})
    assert r["status"] == "read_repair_plus_channel_name_repair_required", r
    assert r["repair_mode"] == "channel_names_from_channels_tsv_for_generic_brainvision"
    ok("generic marker-less header + valid channels.tsv → read_repair_plus_channel_name_repair_required (renamed 19 resolve)")


def test_generic_plus_invalid_tsv_falls_back_and_fails_non_canonical():
    st, std = _state()
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-99_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True, channels_tsv_names=_REAL[:-1])  # row mismatch
    r = P13.classify("SCZ", "ds003944", "sub-99", vhdr, tempfile.mkdtemp(), st, std, {})
    assert r["status"] == "fail" and r["failure_type"] == "header_channel_names_non_canonical", r
    ok("generic header + INVALID channels.tsv → marker-only fallback → header_channel_names_non_canonical (surfaced for review)")


def main():
    print("ACAR v5 Stage-1B13P guard: generic BrainVision + channels.tsv repair classification")
    test_generic_plus_valid_tsv_classifies_repair_plus_name()
    test_generic_plus_invalid_tsv_falls_back_and_fails_non_canonical()
    print("ALL V5 STAGE1B13P-GENERIC-CHANNELS-TSV-REPAIR-CLASSIFY GUARDS PASS")


if __name__ == "__main__":
    main()
