"""Guard (Stage-1B11): a channels.tsv duplicate (like ds004367's F7) is a metadata inconsistency, NOT decisive — if the RAW HEADER
resolves cleanly the verdict is a non-fatal WARN (the DSP consumes the raw header, so the recording is usable). No mne needed."""
from __future__ import annotations
from acar.v5.substrate import channel_aliases as CA
from acar.v5.tests._util import ok, modern_channel_names


def test_tsv_dup_with_clean_raw_header_is_warn_not_fail():
    tsv_names = modern_channel_names() + ["F7"]               # ds004367-style: channels.tsv lists F7 twice
    raw_header = modern_channel_names()                       # raw header has one F7 (clean)
    r = CA.adjudicate_channel_source(tsv_names, raw_header)
    assert r["verdict"] == "WARN_TSV_DUPLICATE"               # non-fatal
    assert r["raw_duplicates"] == [] and "F7" in str(r["tsv_duplicates"])
    ok("channels.tsv duplicate F7 + clean raw header → WARN_TSV_DUPLICATE (non-fatal; raw header is decisive)")


def main():
    print("ACAR v5 Stage-1B11 guard: channels.tsv duplicate warns not decisive if raw header clean")
    test_tsv_dup_with_clean_raw_header_is_warn_not_fail()
    print("ALL V5 STAGE1B-TSV-DUP-WARN GUARDS PASS")


if __name__ == "__main__":
    main()
