"""Guard (Stage-1B6): SubjectWindows now carries the ACTUAL window array; the validator fail-closes on wrong shape / non-float /
NaN / Inf / missing payload, and still rejects the metadata violations. Synthetic numpy only (imported here, not at module load)."""
from __future__ import annotations
import numpy as np
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import subject_windows as SW
from acar.v5.tests._util import expect_raises, ok


_DEFAULT = object()


def _sw(windows=_DEFAULT, **over):
    if windows is _DEFAULT:
        windows = np.zeros((10, 19, 512), dtype=np.float32)
    base = dict(subject_key="PD/ds002778/sub-001", disease="PD", cohort="ds002778", raw_subject_id="sub-001",
                n_windows=10, n_channels=19, n_samples=512, sfreq=128, channels=PC.CHANNELS_19,
                preprocessing_config_sha256=PC.config_sha256(), windows=windows, provenance="synthetic")
    base.update(over)
    return SW.SubjectWindows(**base)


def test_valid_payload_passes():
    assert SW.validate_subject_windows(_sw())
    ok("a SubjectWindows carrying a (n_windows,19,512) finite float32 array validates")


def test_missing_payload_rejected():
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(windows=None)))
    ok("windows=None → rejected (payload is required)")


def test_wrong_shape_rejected():
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(windows=np.zeros((10, 18, 512), np.float32))))
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(windows=np.zeros((9, 19, 512), np.float32))))
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(windows=np.zeros((10, 19, 500), np.float32))))
    ok("windows whose shape != (n_windows,19,512) → rejected")


def test_non_float_and_non_finite_rejected():
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(windows=np.zeros((10, 19, 512), np.int32))))
    nan = np.zeros((10, 19, 512), np.float32); nan[0, 0, 0] = np.nan
    inf = np.zeros((10, 19, 512), np.float32); inf[1, 2, 3] = np.inf
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(windows=nan)))
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(windows=inf)))
    ok("integer dtype / NaN / Inf window payload → rejected")


def test_metadata_violation_still_rejected_with_valid_payload():
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(channels=PC.CHANNELS_19[::-1])))
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(preprocessing_config_sha256="0" * 64)))
    ok("metadata violations (reordered channels / wrong config hash) still rejected even with a valid array")


def main():
    print("ACAR v5 Stage-1B6 guard: subject windows actual payload")
    test_valid_payload_passes()
    test_missing_payload_rejected()
    test_wrong_shape_rejected()
    test_non_float_and_non_finite_rejected()
    test_metadata_violation_still_rejected_with_valid_payload()
    print("ALL V5 STAGE1B-SUBJECT-WINDOWS-PAYLOAD GUARDS PASS")


if __name__ == "__main__":
    main()
