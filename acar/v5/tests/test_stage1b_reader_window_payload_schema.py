"""Guard (Stage-1B5): the reader window payload (SubjectWindows) is schema-validated against the pinned preprocessing config, and
carries NO label field. Synthetic only."""
from __future__ import annotations
import dataclasses
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import subject_windows as SW
from acar.v5.tests._util import expect_raises, ok


def _sw(**over):
    base = dict(subject_key="PD/ds002778/sub-001", disease="PD", cohort="ds002778", raw_subject_id="sub-001",
                n_windows=10, n_channels=19, n_samples=512, sfreq=128, channels=PC.CHANNELS_19,
                preprocessing_config_sha256=PC.config_sha256(), provenance="synthetic")
    base.update(over)
    return SW.SubjectWindows(**base)


def test_valid_payload():
    assert SW.validate_subject_windows(_sw())
    ok("a conforming SubjectWindows validates (19ch order, 128Hz, 512 samples, pinned config hash)")


def test_bad_channels_sfreq_samples_rejected():
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(channels=PC.CHANNELS_19[::-1])))
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(n_channels=18)))
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(sfreq=256)))
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(n_samples=500)))
    ok("reordered channels / wrong n_channels / wrong sfreq / wrong window length → rejected")


def test_namespaced_raw_and_bad_hash_rejected():
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(raw_subject_id="ds002778/sub-001",
                                                                                  subject_key="PD/ds002778/ds002778/sub-001")))
    expect_raises(SW.SubjectWindowsError, lambda: SW.validate_subject_windows(_sw(preprocessing_config_sha256="0" * 64)))
    ok("namespaced raw id / wrong preprocessing_config_sha256 → rejected")


def test_no_label_field_on_payload():
    fields = {f.name for f in dataclasses.fields(SW.SubjectWindows)}
    assert not (fields & {"label", "y", "diagnosis", "target", "case_control"}), fields
    assert not SW.has_label_field(_sw())
    ok("SubjectWindows carries NO label field (labels are a separate FIT-only read)")


def main():
    print("ACAR v5 Stage-1B5 guard: reader window payload schema")
    test_valid_payload()
    test_bad_channels_sfreq_samples_rejected()
    test_namespaced_raw_and_bad_hash_rejected()
    test_no_label_field_on_payload()
    print("ALL V5 STAGE1B-READER-PAYLOAD-SCHEMA GUARDS PASS")


if __name__ == "__main__":
    main()
