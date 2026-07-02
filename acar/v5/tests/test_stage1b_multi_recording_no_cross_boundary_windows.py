"""Guard (Stage-1B7): with multiple recordings, each recording is windowed INDEPENDENTLY (windows never span two recordings). The
per-recording window count is the sum of floor(len_i/512); concatenating raws THEN windowing (which would create a cross-boundary
window) is NOT done. Synthetic FakeRaw / fake-mne only."""
from __future__ import annotations
import os
import tempfile
import numpy as np
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.substrate import subject_windows as SW
from acar.v5.tests._util import ok


class FakeRaw:
    def __init__(self, n_times):
        self._ch = list(PC.CHANNELS_19)[::-1] + ["EXTRA"]     # permuted + extra
        data = np.zeros((len(self._ch), n_times), dtype=np.float64)
        for i in range(len(self._ch)):
            data[i, :] = np.linspace(-1.0, 1.0, n_times) * (i + 1) + i
        self._data = data

    @property
    def ch_names(self):
        return list(self._ch)

    def pick(self, names):
        idx = [self._ch.index(n) for n in names]
        self._data = self._data[idx, :]
        self._ch = list(names)
        return self

    def set_eeg_reference(self, ref, projection=False):
        return self

    def filter(self, l_freq, h_freq):
        return self

    def resample(self, sfreq):
        return self                                           # fake: does not change sample count

    def get_data(self, units=None):
        return self._data


class _FakeMne:
    """Maps each recording file path to its own FakeRaw so multi-recording concatenation is exercised."""

    def __init__(self, by_path):
        self._by_path = by_path
        outer = self

        class _IO:
            def read_raw_edf(self, path, preload=True, verbose="ERROR"):
                return outer._by_path[os.path.basename(path)]
        self.io = _IO()


def test_windows_are_per_recording_not_cross_boundary():
    # two recordings of 768 samples each: per-recording → floor(768/512)=1 window each = 2 windows total.
    # concatenate-then-window would give floor(1536/512)=3 windows (one spanning the A/B boundary) — must NOT happen.
    with tempfile.TemporaryDirectory() as sub:
        eeg = os.path.join(sub, "eeg")
        os.makedirs(eeg)
        for name in ("sub-001_run-1_eeg.edf", "sub-001_run-2_eeg.edf"):
            open(os.path.join(eeg, name), "w").close()
        mne = _FakeMne({"sub-001_run-1_eeg.edf": FakeRaw(768), "sub-001_run-2_eeg.edf": FakeRaw(768)})
        sw = RMR.preprocess_subject("PD", "ds002778", "sub-001", sub, mne=mne)
        assert SW.validate_subject_windows(sw)
        assert sw.n_windows == 2, f"per-recording windowing must give 2 windows, got {sw.n_windows} (3 ⇒ cross-boundary!)"
        assert sw.windows.shape == (2, 19, 512)
    ok("two 768-sample recordings → 2 windows (per-recording), NOT 3 (no window spans the recording boundary)")


def test_single_recording_baseline():
    with tempfile.TemporaryDirectory() as sub:
        eeg = os.path.join(sub, "eeg")
        os.makedirs(eeg)
        open(os.path.join(eeg, "sub-001_eeg.edf"), "w").close()
        mne = _FakeMne({"sub-001_eeg.edf": FakeRaw(1024)})    # 1024//512 = 2 windows
        sw = RMR.preprocess_subject("PD", "ds002778", "sub-001", sub, mne=mne)
        assert sw.n_windows == 2 and sw.channels == PC.CHANNELS_19
    ok("single 1024-sample recording → 2 windows (canonical channels), baseline sanity")


def main():
    print("ACAR v5 Stage-1B7 guard: multi-recording no cross-boundary windows")
    test_windows_are_per_recording_not_cross_boundary()
    test_single_recording_baseline()
    print("ALL V5 STAGE1B-MULTI-RECORDING-BOUNDARY GUARDS PASS")


if __name__ == "__main__":
    main()
