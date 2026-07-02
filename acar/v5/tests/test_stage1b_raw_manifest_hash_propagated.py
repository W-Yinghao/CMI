"""Guard (Stage-1B8): the raw-recording manifest hash is propagated into SubjectWindows.provenance, cryptographically tying the
Stage-1B signal payload to the exact raw files consumed. Synthetic fake-mne + temp files only."""
from __future__ import annotations
import os
import tempfile
import numpy as np
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.substrate import raw_recording_manifest as RM
from acar.v5.substrate import subject_windows as SW
from acar.v5.tests._util import ok


class FakeRaw:
    def __init__(self, n_times=1024):
        self._ch = list(PC.CHANNELS_19)[::-1] + ["EXTRA"]
        self._data = np.zeros((len(self._ch), n_times), dtype=np.float64)
        for i in range(len(self._ch)):
            self._data[i, :] = np.linspace(-1, 1, n_times) * (i + 1) + i

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

    def resample(self, s):
        return self

    def get_data(self, units=None):
        return self._data


class _Mne:
    def __init__(self, raw):
        outer = self
        self._raw = raw

        class _IO:
            def read_raw_edf(self, path, preload=True, verbose="ERROR"):
                return outer._raw
        self.io = _IO()


def test_provenance_carries_manifest_sha():
    with tempfile.TemporaryDirectory() as sub:
        eeg = os.path.join(sub, "eeg")
        os.makedirs(eeg)
        open(os.path.join(eeg, "sub-001_task-rest_eeg.edf"), "wb").write(b"EDFDATA")
        sw = RMR.preprocess_subject("PD", "ds002778", "sub-001", sub, mne=_Mne(FakeRaw()))
        assert SW.validate_subject_windows(sw)
        man = RM.build_manifest(sub)
        assert man["manifest_sha256"] in sw.provenance and "raw_manifest_sha256=" in sw.provenance
    ok("SubjectWindows.provenance carries raw_manifest_sha256 == build_manifest(subject_dir).manifest_sha256 (raw files tied)")


def main():
    print("ACAR v5 Stage-1B8 guard: raw manifest hash propagated")
    test_provenance_carries_manifest_sha()
    print("ALL V5 STAGE1B-RAW-MANIFEST-HASH-PROPAGATED GUARDS PASS")


if __name__ == "__main__":
    main()
