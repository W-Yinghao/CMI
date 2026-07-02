"""Guard (Stage-1B11): ds004584 (missing Pz) — the reviewed montage-completion layer interpolates ONLY Pz and yields a canonical
19-channel SubjectWindows. Synthetic mne RawArray (real mne numerics, synthetic signal). No real DEV read."""
from __future__ import annotations
import numpy as np
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.substrate import subject_windows as SW
from acar.v5.tests._util import ok, make_mne_raw, modern_channel_names


def test_ds004584_pz_interpolated():
    present = [c for c in modern_channel_names() if c != "Pz"] + ["AF3", "FC1"]   # modern names, Pz absent, + donors
    sw = RMR.raw_to_windows(make_mne_raw(present, 2048, 256.0), "PD", "ds004584", "sub-001")
    assert SW.validate_subject_windows(sw) and sw.channels == PC.CHANNELS_19 and sw.n_channels == 19
    assert sw.montage_completion["interpolated"] == ["Pz"] and sw.montage_completion["n_interpolated"] == 1
    assert sw.montage_completion["donor_count"] >= PC.PREPROCESSING_CONFIG["min_donor_channels"]
    assert bool(np.isfinite(sw.windows).all())
    ok("ds004584: Pz interpolated (only) → canonical 19-channel SubjectWindows, finite, donors>=min")


def main():
    print("ACAR v5 Stage-1B11 guard: montage completion ds004584 Pz only")
    test_ds004584_pz_interpolated()
    print("ALL V5 STAGE1B-MONTAGE-COMPLETION-DS004584 GUARDS PASS")


if __name__ == "__main__":
    main()
