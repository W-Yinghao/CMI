"""Guard (Stage-1B10): if two raw channels map to the SAME logical (canonical) channel — e.g. both T3 (canonical) and T7 (alias→T3)
present — it FAILS closed. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import channel_aliases as CA
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.tests._util import expect_raises, ok, make_fake_raw


def test_duplicate_logical_channel_fails():
    ch = list(PC.CHANNELS_19) + ["T7"]                        # T3 canonical AND T7 (→T3) → duplicate logical T3
    expect_raises(CA.ChannelAliasError, lambda: CA.resolve_canonical_sources(ch))
    expect_raises(RMR.RealMneReaderError, lambda: RMR.raw_to_windows(make_fake_raw(ch, 1024), "PD", "ds002778", "sub-hc1"))
    ch2 = list(PC.CHANNELS_19) + ["Cz"]                       # Cz twice → duplicate logical Cz
    expect_raises(CA.ChannelAliasError, lambda: CA.resolve_canonical_sources(ch2))
    ok("two raw channels mapping to the same logical channel (T3+T7, or Cz twice) → ChannelAliasError / RealMneReaderError")


def main():
    print("ACAR v5 Stage-1B10 guard: channel aliases duplicate logical fail")
    test_duplicate_logical_channel_fails()
    print("ALL V5 STAGE1B-CHANNEL-ALIASES-DUP GUARDS PASS")


if __name__ == "__main__":
    main()
