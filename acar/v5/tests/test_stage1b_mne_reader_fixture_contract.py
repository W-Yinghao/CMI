"""Guard (Stage-1B6): the real mne DSP seam applies the PINNED pipeline deterministically and fail-closed, driven by a FAKE
mne-Raw / fake-mne adapter (no real mne, no real DEV). It selects+reorders the 19 channels to canonical order, average-references,
bandpasses 0.5-45, resamples 128, windows 4s/512 non-overlap, per-trial z-scores, and returns a validated SubjectWindows with NO
label. Synthetic only."""
from __future__ import annotations
import os
import tempfile
import numpy as np
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.substrate import subject_windows as SW
from acar.v5.tests._util import expect_raises, ok


class FakeRaw:
    def __init__(self, ch_names, data, sfreq=256):
        self._ch = list(ch_names)
        self._data = np.asarray(data, dtype=np.float64)
        self._sfreq = sfreq
        self.calls = []

    @property
    def ch_names(self):
        return list(self._ch)

    def pick(self, names):
        idx = [self._ch.index(n) for n in names]              # KeyError-free: caller guarantees presence (core checks first)
        self._data = self._data[idx, :]
        self._ch = list(names)
        self.calls.append(("pick", tuple(names)))
        return self

    def set_eeg_reference(self, ref, projection=False):
        self.calls.append(("ref", ref, projection))
        return self

    def filter(self, l_freq, h_freq):
        self.calls.append(("filter", l_freq, h_freq))
        return self

    def resample(self, sfreq):
        self._sfreq = sfreq
        self.calls.append(("resample", sfreq))
        return self

    def get_data(self, units=None):
        self.calls.append(("get_data", units))
        return self._data


def _permuted_raw(n_times=1024, drop=None):
    """A FakeRaw whose channels are the 19 montage channels in a PERMUTED order (+ an extra non-montage channel), with varying
    per-channel data. If `drop` is given, that montage channel is omitted (to exercise the missing-channel path)."""
    chans = list(PC.CHANNELS_19)
    perm = chans[::-1] + ["EXTRA"]                            # reversed order + an extra channel
    if drop:
        perm = [c for c in perm if c != drop]
    data = np.zeros((len(perm), n_times), dtype=np.float64)
    for i, _c in enumerate(perm):
        data[i, :] = np.linspace(-1.0, 1.0, n_times) * (i + 1) + i   # varying (nonzero std) per channel
    return FakeRaw(perm, data)


def test_raw_to_windows_valid_and_canonical():
    raw = _permuted_raw(n_times=1024)                         # 1024 // 512 = 2 windows
    sw = RMR.raw_to_windows(raw, "PD", "ds002778", "sub-001")
    assert isinstance(sw, SW.SubjectWindows) and SW.validate_subject_windows(sw)
    assert sw.channels == PC.CHANNELS_19 and sw.n_windows == 2 and sw.n_channels == 19 and sw.n_samples == 512
    assert sw.windows.shape == (2, 19, 512) and sw.windows.dtype.kind == "f"
    assert not SW.has_label_field(sw)
    kinds = [c[0] for c in raw.calls]
    assert kinds.index("pick") < kinds.index("ref") < kinds.index("filter") < kinds.index("resample") < kinds.index("get_data")
    assert ("ref", "average", False) in raw.calls and ("filter", 0.5, 45.0) in raw.calls and ("resample", 128) in raw.calls
    ok("raw_to_windows: pinned DSP order, canonical 19ch, 4s/512 windows, per-trial z-score → validated label-free SubjectWindows")


def test_per_trial_zscore_applied():
    raw = _permuted_raw(n_times=1024)
    sw = RMR.raw_to_windows(raw, "PD", "ds002778", "sub-001")
    m = sw.windows.mean(axis=2)                               # each (window,channel) roughly zero-mean after z-score
    assert float(np.max(np.abs(m))) < 1e-4, float(np.max(np.abs(m)))
    ok("per-trial z-score makes each window/channel ~zero-mean (finite, standardized)")


def test_missing_channel_fails_closed():
    expect_raises(RMR.RealMneReaderError, lambda: RMR.raw_to_windows(_permuted_raw(drop="Cz"), "PD", "ds002778", "sub-001"))
    ok("a recording missing a montage channel (Cz) → RealMneReaderError (fail-closed)")


def test_recording_too_short_fails_closed():
    expect_raises(RMR.RealMneReaderError, lambda: RMR.raw_to_windows(_permuted_raw(n_times=300), "PD", "ds002778", "sub-001"))
    ok("a recording shorter than one 512-sample window → RealMneReaderError")


def test_preprocess_subject_via_fake_mne():
    class _IO:
        def __init__(self, raw):
            self._raw = raw

        def read_raw_edf(self, path, preload=True, verbose="ERROR"):
            return self._raw

    class _Mne:
        def __init__(self, raw):
            self.io = _IO(raw)

        def concatenate_raws(self, raws):
            return raws[0]

    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "sub-001_task-rest_eeg.edf"), "w").close()   # discoverable recording
        sw = RMR.preprocess_subject("PD", "ds002778", "sub-001", d, mne=_Mne(_permuted_raw(1024)))
        assert SW.validate_subject_windows(sw) and sw.subject_key == "PD/ds002778/sub-001"
    ok("preprocess_subject discovers the recording + drives the fake mne adapter → validated SubjectWindows (no real mne)")


def main():
    print("ACAR v5 Stage-1B6 guard: mne reader fixture contract")
    test_raw_to_windows_valid_and_canonical()
    test_per_trial_zscore_applied()
    test_missing_channel_fails_closed()
    test_recording_too_short_fails_closed()
    test_preprocess_subject_via_fake_mne()
    print("ALL V5 STAGE1B-MNE-READER-FIXTURE GUARDS PASS")


if __name__ == "__main__":
    main()
