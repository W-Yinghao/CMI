"""Guard (Stage-1B8): the pinned channel policy in preprocessing_config matches the real_mne_reader behavior — all 19 canonical
channels required (missing → fail), duplicates → fail, extras dropped, output in canonical order (permuted input → canonical
output). Synthetic FakeRaw only (no real mne)."""
from __future__ import annotations
import numpy as np
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.substrate import subject_windows as SW
from acar.v5.tests._util import expect_raises, ok


class FakeRaw:
    def __init__(self, ch_names, n_times=1024):
        self._ch = list(ch_names)
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


def test_config_declares_channel_policy():
    c = PC.PREPROCESSING_CONFIG
    assert c["required_channels"] == "all_19_canonical_present"
    assert c["extra_channel_policy"] == "drop_non_canonical_after_required_19_present"
    assert c["duplicate_channel_policy"] == "fail_closed"
    assert c["channel_output_order"] == "canonical_pinned"
    ok("preprocessing_config declares required-19 / drop-extras / duplicate-fail / canonical-order policy")


def test_extras_dropped_and_output_canonical():
    raw = FakeRaw(list(PC.CHANNELS_19)[::-1] + ["EXTRA1", "EXTRA2"])   # permuted + 2 extras
    sw = RMR.raw_to_windows(raw, "PD", "ds002778", "sub-001")
    assert SW.validate_subject_windows(sw) and sw.channels == PC.CHANNELS_19 and sw.n_channels == 19
    ok("extra non-canonical channels dropped; permuted input → canonical 19-channel output (matches config)")


def test_missing_and_duplicate_channels_fail():
    missing = FakeRaw([c for c in PC.CHANNELS_19 if c != "Cz"])        # missing a canonical channel
    expect_raises(RMR.RealMneReaderError, lambda: RMR.raw_to_windows(missing, "PD", "ds002778", "sub-001"))
    dup = FakeRaw(list(PC.CHANNELS_19) + ["Cz"])                       # duplicate canonical channel
    expect_raises(RMR.RealMneReaderError, lambda: RMR.raw_to_windows(dup, "PD", "ds002778", "sub-001"))
    ok("missing canonical channel → fail; duplicate canonical channel → fail (matches config fail-closed policy)")


def main():
    print("ACAR v5 Stage-1B8 guard: channel policy config matches reader")
    test_config_declares_channel_policy()
    test_extras_dropped_and_output_canonical()
    test_missing_and_duplicate_channels_fail()
    print("ALL V5 STAGE1B-CHANNEL-POLICY GUARDS PASS")


if __name__ == "__main__":
    main()
