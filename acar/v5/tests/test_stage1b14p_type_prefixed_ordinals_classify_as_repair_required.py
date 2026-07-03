"""Guard (Stage-1B14P): the preflight classifier turns a TYPE-PREFIXED ordinal ds003944/ds003947 recording (marker-less + EOG/ECG
placeholders + valid channels.tsv) into read_repair_plus_channel_name_repair_required (the widened detector renames it, the header
opens at preload=False, and the renamed names resolve all 19 canonical). SYNTHETIC fixtures; no real data."""
from __future__ import annotations
import tempfile
from acar.v5 import stage1b14p_preflight as P14
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]
_TYPED = {20: "EOG", 21: "ECG"}


def _state():
    st = {"mod": None, "std": None}
    return st, P14._std_positions_casefold(st)


def test_type_prefixed_ordinal_classifies_repair_plus_name():
    st, std = _state()
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-2235A_task-rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True, ordinal_prefix_overrides=_TYPED)
    r = P14.classify("SCZ", "ds003947", "sub-2235A", vhdr, tempfile.mkdtemp(), st, std, {})
    assert r["status"] == "read_repair_plus_channel_name_repair_required", r
    assert r["repair_mode"] == "channel_names_from_channels_tsv_for_generic_brainvision"
    ok("type-prefixed ordinal marker-less header + valid channels.tsv → read_repair_plus_channel_name_repair_required")


def test_arbitrary_prefix_still_fails_non_canonical():
    st, std = _state()
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-9_task-rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True, ordinal_prefix_overrides={20: "GSR"})
    r = P14.classify("SCZ", "ds003947", "sub-9", vhdr, tempfile.mkdtemp(), st, std, {})
    assert r["status"] == "fail" and r["failure_type"] == "header_channel_names_non_canonical", r
    ok("an arbitrary-prefix (GSR) header still falls back to marker-only → header_channel_names_non_canonical (surfaced for review)")


def main():
    print("ACAR v5 Stage-1B14P guard: type-prefixed ordinal repair classification")
    test_type_prefixed_ordinal_classifies_repair_plus_name()
    test_arbitrary_prefix_still_fails_non_canonical()
    print("ALL V5 STAGE1B14P-TYPE-PREFIXED-ORDINAL-CLASSIFY GUARDS PASS")


if __name__ == "__main__":
    main()
