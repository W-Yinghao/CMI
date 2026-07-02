"""Guard (Stage-1B11): ds004000 (missing F3/F4/P3/P4) — the montage-completion layer interpolates ONLY those four canonical channels
(== the per-cohort whitelist == max). Synthetic mne RawArray. No real DEV read."""
from __future__ import annotations
import numpy as np
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.substrate import subject_windows as SW
from acar.v5.tests._util import ok, make_mne_raw, modern_channel_names


def test_ds004000_four_interpolated():
    missing = {"F3", "F4", "P3", "P4"}
    present = [c for c in modern_channel_names() if c not in missing] + ["AF3", "FC1", "CP1", "CP2"]
    sw = RMR.raw_to_windows(make_mne_raw(present, 2048, 256.0), "SCZ", "ds004000", "sub-000")
    assert SW.validate_subject_windows(sw) and sw.channels == PC.CHANNELS_19
    assert sw.montage_completion["interpolated"] == sorted(missing) and sw.montage_completion["n_interpolated"] == 4
    assert bool(np.isfinite(sw.windows).all())
    ok("ds004000: exactly F3/F4/P3/P4 interpolated → canonical 19-channel SubjectWindows (4 == whitelist == max)")


def main():
    print("ACAR v5 Stage-1B11 guard: montage completion ds004000 four only")
    test_ds004000_four_interpolated()
    print("ALL V5 STAGE1B-MONTAGE-COMPLETION-DS004000 GUARDS PASS")


if __name__ == "__main__":
    main()
