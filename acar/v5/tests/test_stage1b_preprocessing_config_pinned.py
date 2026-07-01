"""Guard (Stage-1B5): the DSP preprocessing config is PINNED in code + deterministically hashed. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.tests._util import ok


def test_pinned_values():
    c = PC.PREPROCESSING_CONFIG
    assert len(PC.CHANNELS_19) == 19 and c["channels"] == list(PC.CHANNELS_19) and c["n_channels"] == 19
    assert c["resample_hz"] == 128 and c["bandpass_hz"] == [0.5, 45.0]
    assert c["window_sec"] == 4.0 and c["window_samples"] == 512
    assert c["bad_channel_policy"] == "fail_closed" and c["trial_normalization"] == "per_trial_zscore"
    ok("preprocessing config pinned: 19ch / 128Hz / 0.5-45Hz / 4s(512) / fail-closed bad channels")


def test_hash_deterministic_64hex():
    h1, h2 = PC.config_sha256(), PC.config_sha256()
    assert h1 == h2 and len(h1) == 64 and all(ch in "0123456789abcdef" for ch in h1)
    ok(f"preprocessing_config_sha256 deterministic 64-hex ({h1[:12]}…)")


def main():
    print("ACAR v5 Stage-1B5 guard: preprocessing config pinned")
    test_pinned_values()
    test_hash_deterministic_64hex()
    print("ALL V5 STAGE1B-PREPROCESSING-CONFIG GUARDS PASS")


if __name__ == "__main__":
    main()
