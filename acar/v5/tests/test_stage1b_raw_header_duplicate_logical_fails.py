"""Guard (Stage-1B11): a duplicate logical channel in the RAW HEADER is a signal-level ambiguity → FAIL (adjudication AND the DSP
reader). No 'keep-first' / no silent de-dup. Synthetic FakeRaw (dup fails before any interpolation)."""
from __future__ import annotations
from acar.v5.substrate import channel_aliases as CA
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.tests._util import expect_raises, ok, make_fake_raw, modern_channel_names


def test_raw_header_duplicate_fails():
    raw_dup = modern_channel_names() + ["F7"]                 # duplicate logical F7 in the raw header
    r = CA.adjudicate_channel_source(list(PC.CHANNELS_19), raw_dup)
    assert r["verdict"] == "FAIL" and "F7" in str(r["raw_duplicates"])
    # the DSP reader consumes raw header names → a duplicate logical channel fails closed (never de-duped)
    expect_raises(RMR.RealMneReaderError,
                  lambda: RMR.raw_to_windows(make_fake_raw(modern_channel_names() + ["F7"]), "SCZ", "ds004367", "sub-S03"))
    ok("raw-header duplicate logical F7 → adjudication FAIL AND RealMneReaderError (no keep-first / no silent de-dup)")


def main():
    print("ACAR v5 Stage-1B11 guard: raw header duplicate logical fails")
    test_raw_header_duplicate_fails()
    print("ALL V5 STAGE1B-RAW-HEADER-DUP-FAIL GUARDS PASS")


if __name__ == "__main__":
    main()
