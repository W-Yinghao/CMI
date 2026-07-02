"""Guard (Stage-1B11): a recording missing MORE canonical channels than the cohort whitelist allows FAILS closed; and the pinned
max_interpolated equals the largest whitelist (so the cap is consistent). Synthetic FakeRaw. No real DEV read."""
from __future__ import annotations
from acar.v5.substrate import montage_completion as MC
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.tests._util import expect_raises, ok, make_fake_raw, modern_channel_names


def test_too_many_missing_fails():
    # ds004000 whitelist = {F3,F4,P3,P4}; also drop P7 (→ T5 missing) → missing set exceeds the whitelist
    present = [c for c in modern_channel_names() if c not in {"F3", "F4", "P3", "P4", "P7"}]
    expect_raises(MC.MontageCompletionError, lambda: MC.complete_missing_channels(make_fake_raw(present), "SCZ", "ds004000", mne=None))
    ok("missing canonical set beyond the cohort whitelist → MontageCompletionError")


def test_max_interpolated_is_consistent_with_whitelist():
    cfg = PC.PREPROCESSING_CONFIG
    largest = max(len(v) for v in cfg["allowed_missing_by_cohort"].values())
    assert cfg["max_interpolated_canonical_channels_per_recording"] == largest == 4
    ok("max_interpolated_canonical_channels_per_recording == largest cohort whitelist (4) — cap is consistent")


def main():
    print("ACAR v5 Stage-1B11 guard: montage completion too many missing fails")
    test_too_many_missing_fails()
    test_max_interpolated_is_consistent_with_whitelist()
    print("ALL V5 STAGE1B-MONTAGE-COMPLETION-TOOMANY GUARDS PASS")


if __name__ == "__main__":
    main()
