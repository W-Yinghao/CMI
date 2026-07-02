"""Guard (Stage-1B11): channel-source adjudication is RAW-HEADER decisive — a duplicate logical channel in the raw header FAILS; a
duplicate only in channels.tsv (raw header clean) is a non-fatal WARN; both clean → PASS. No 'keep-first' / no silent de-dup."""
from __future__ import annotations
from acar.v5.substrate import channel_aliases as CA
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.tests._util import ok, modern_channel_names


def test_adjudication_matrix():
    clean = list(PC.CHANNELS_19)
    modern = list(modern_channel_names())
    raw_dup = modern + ["F7"]                                 # F7 twice in the RAW HEADER
    tsv_dup = clean + ["F7"]                                  # F7 twice in channels.tsv
    assert CA.adjudicate_channel_source(clean, modern)["verdict"] == "PASS"          # both clean
    r = CA.adjudicate_channel_source(clean, raw_dup)
    assert r["verdict"] == "FAIL" and "F7" in str(r["raw_duplicates"])               # raw header dup → FAIL
    r = CA.adjudicate_channel_source(tsv_dup, modern)
    assert r["verdict"] == "WARN_TSV_DUPLICATE" and r["raw_duplicates"] == []         # tsv dup, raw clean → WARN
    assert CA.adjudicate_channel_source(clean, None)["verdict"] == "PASS"             # no raw header, tsv clean → PASS
    assert CA.adjudicate_channel_source(tsv_dup, None)["verdict"] == "FAIL"           # no raw header, tsv dup → FAIL (cannot adjudicate)
    ok("adjudication: raw-header dup → FAIL; tsv-only dup (raw clean) → WARN; both clean → PASS; no raw + tsv dup → FAIL")


def main():
    print("ACAR v5 Stage-1B11 guard: duplicate F7 raw-header adjudication")
    test_adjudication_matrix()
    print("ALL V5 STAGE1B-DUP-F7-ADJUDICATION GUARDS PASS")


if __name__ == "__main__":
    main()
