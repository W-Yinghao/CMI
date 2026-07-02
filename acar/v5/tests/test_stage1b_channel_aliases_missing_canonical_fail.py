"""Guard (Stage-1B10): if any canonical channel is missing after aliasing (and has no alias source), it FAILS closed. Synthetic
only."""
from __future__ import annotations
from acar.v5.substrate import channel_aliases as CA
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.tests._util import expect_raises, ok, make_fake_raw, modern_channel_names


def test_missing_canonical_fails():
    ch = [c for c in modern_channel_names() if c != "Cz"]     # Cz has no alias → missing canonical
    expect_raises(CA.ChannelAliasError, lambda: CA.resolve_canonical_sources(ch))
    expect_raises(RMR.RealMneReaderError, lambda: RMR.raw_to_windows(make_fake_raw(ch + ["EXG1"], 1024), "PD", "ds002778", "sub-hc1"))
    ok("a recording missing a canonical channel (Cz, no alias) → ChannelAliasError / RealMneReaderError (fail-closed)")


def main():
    print("ACAR v5 Stage-1B10 guard: channel aliases missing canonical fail")
    test_missing_canonical_fails()
    print("ALL V5 STAGE1B-CHANNEL-ALIASES-MISSING GUARDS PASS")


if __name__ == "__main__":
    main()
