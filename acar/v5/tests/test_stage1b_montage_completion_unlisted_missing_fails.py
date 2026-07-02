"""Guard (Stage-1B11): a missing canonical channel NOT in the cohort whitelist FAILS closed (no interpolation of arbitrary missing
channels). Synthetic FakeRaw (fails before any mne interpolation). No real DEV read."""
from __future__ import annotations
from acar.v5.substrate import montage_completion as MC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.tests._util import expect_raises, ok, make_fake_raw, modern_channel_names


def test_unlisted_missing_fails():
    # ds002778 has NO whitelist → any missing canonical fails
    raw = make_fake_raw([c for c in modern_channel_names() if c != "Cz"])   # Cz missing, ds002778 not whitelisted
    expect_raises(MC.MontageCompletionError, lambda: MC.complete_missing_channels(raw, "PD", "ds002778", mne=None))
    expect_raises(RMR.RealMneReaderError,
                  lambda: RMR.raw_to_windows(make_fake_raw([c for c in modern_channel_names() if c != "Cz"]), "PD", "ds002778", "s"))
    # ds004584 whitelist is ONLY {Pz}; a missing channel outside it (Cz) fails even though the cohort allows Pz
    raw2 = make_fake_raw([c for c in modern_channel_names() if c != "Cz"])
    expect_raises(MC.MontageCompletionError, lambda: MC.complete_missing_channels(raw2, "PD", "ds004584", mne=None))
    ok("a missing canonical channel not in the cohort whitelist → MontageCompletionError / RealMneReaderError")


def main():
    print("ACAR v5 Stage-1B11 guard: montage completion unlisted missing fails")
    test_unlisted_missing_fails()
    print("ALL V5 STAGE1B-MONTAGE-COMPLETION-UNLISTED GUARDS PASS")


if __name__ == "__main__":
    main()
