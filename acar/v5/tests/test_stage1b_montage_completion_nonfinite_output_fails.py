"""Guard (Stage-1B11): if interpolation would produce non-finite data (e.g. a NaN donor), montage completion FAILS closed. Synthetic
mne RawArray with a NaN donor channel. No real DEV read."""
from __future__ import annotations
from acar.v5.substrate import montage_completion as MC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.tests._util import expect_raises, ok, make_mne_raw, modern_channel_names


def test_nonfinite_interpolation_fails():
    present = [c for c in modern_channel_names() if c != "Pz"] + ["AF3", "FC1"]
    raw = make_mne_raw(present, 2048, 256.0, nan_channels=["AF3"])          # a donor is NaN → interp non-finite / mne error
    expect_raises(MC.MontageCompletionError, lambda: MC.complete_missing_channels(raw, "PD", "ds004584", mne=None))
    raw2 = make_mne_raw(present, 2048, 256.0, nan_channels=["AF3"])
    expect_raises(RMR.RealMneReaderError, lambda: RMR.raw_to_windows(raw2, "PD", "ds004584", "sub-001"))
    ok("interpolation producing non-finite data (NaN donor) → MontageCompletionError / RealMneReaderError (fail-closed)")


def main():
    print("ACAR v5 Stage-1B11 guard: montage completion non-finite output fails")
    test_nonfinite_interpolation_fails()
    print("ALL V5 STAGE1B-MONTAGE-COMPLETION-NONFINITE GUARDS PASS")


if __name__ == "__main__":
    main()
