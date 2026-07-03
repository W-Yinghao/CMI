"""Guard (Stage-1B14): the feature-dump schema is bumped to V5 — the header adds a per-recording channel-name-repair SUBTYPE map
(pure_eeg_ordinal vs type_prefixed_ordinal + the ordinal prefixes), label-free. A V4-shaped dump (missing the V5 field) is rejected;
a label-like key anywhere in the subtype map is rejected."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import feature_dump_schema as FS
from acar.v5.substrate import stage1b_feature_dump_writer as FDW
from acar.v5.tests._util import ok, expect_raises


def _write(path, **over):
    kw = dict(ref="SCZ/fold0/seed1", disease="SCZ", fold=0, seed=1, preprocessing_config_sha256="0" * 64,
              training_config_sha256="0" * 64, encoder_checkpoint_file_sha256="0" * 64, source_state_file_sha256="0" * 64,
              records=[("SCZ/ds003947/sub-2235A", "train", 0, [1.0, 2.0, 3.0])])
    kw.update(over)
    return FDW.write_feature_dump(path, **kw)


def test_v5_records_subtype_map():
    p = os.path.join(tempfile.mkdtemp(), "feat.npz")
    summ = _write(p, channel_name_repair_subtype_by_recording={
        "SCZ/ds003947/sub-2235A::sub-2235A_task-rest_eeg.vhdr": {
            "subtype": "type_prefixed_ordinal", "ordinal_prefixes": ["EEG"] * 61 + ["EOG"] + ["EEG"] * 2}})
    assert str(FS.SCHEMA_VERSION) == "ACAR_V5_STAGE1B_FEAT_DUMP_V5"
    entry = summ["channel_name_repair_subtype_by_recording"]["SCZ/ds003947/sub-2235A::sub-2235A_task-rest_eeg.vhdr"]
    assert entry["subtype"] == "type_prefixed_ordinal"
    ok("V5 dump round-trips the per-recording channel-name-repair subtype + ordinal-prefixes map")


def test_default_dump_is_valid_v5():
    p = os.path.join(tempfile.mkdtemp(), "feat.npz")
    summ = _write(p)
    assert summ["channel_name_repair_subtype_by_recording"] == {}
    ok("a no-rename dump defaults to an empty subtype map and is valid V5")


def test_label_like_key_in_subtype_map_rejected():
    p = os.path.join(tempfile.mkdtemp(), "feat.npz")
    expect_raises(FS.FeatureDumpSchemaError,
                  lambda: _write(p, channel_name_repair_subtype_by_recording={"x::r.vhdr": {"subtype": "pure_eeg_ordinal",
                                                                                            "y_te": 1}}),
                  "a label-like key nested in the subtype map must be rejected")
    ok("a label-like field nested inside channel_name_repair_subtype_by_recording is rejected (label firewall)")


def test_v4_shaped_dump_missing_v5_field_rejected():
    import numpy as np
    v4 = {
        "schema_version": np.asarray(FS.SCHEMA_VERSION), "ref": np.asarray("SCZ/fold0/seed1"), "disease": np.asarray("SCZ"),
        "fold": np.asarray(0), "seed": np.asarray(1),
        "preprocessing_config_sha256": np.asarray("0" * 64), "training_config_sha256": np.asarray("0" * 64),
        "encoder_checkpoint_file_sha256": np.asarray("0" * 64), "source_state_file_sha256": np.asarray("0" * 64),
        "channel_alias_policy_sha256": np.asarray("0" * 64), "montage_completion_policy_sha256": np.asarray("0" * 64),
        "montage_completion_by_subject": np.asarray("{}"),
        "brainvision_read_repair_policy_sha256": np.asarray("0" * 64), "raw_header_repair_manifest_sha256": np.asarray("0" * 64),
        "brainvision_read_repair_by_recording": np.asarray("{}"),
        "channel_name_repair_policy_sha256": np.asarray("0" * 64), "channel_name_repair_by_recording": np.asarray("{}"),
        "subject_key": np.asarray(["SCZ/ds/x"]), "split_role": np.asarray(["train"]),
        "window_id": np.asarray([0], dtype=np.int64), "embedding": np.asarray([[1.0, 2.0]], dtype=np.float32),
    }
    expect_raises(FS.FeatureDumpSchemaError, lambda: FS.validate_loaded(v4),
                  "a V4-shaped dump lacking the V5 field must be rejected")
    ok("a V4-shaped dump (no subtype map) is rejected by the V5 validator (missing required field)")


def main():
    print("ACAR v5 Stage-1B14 guard: feature-dump V5 records channel-name-repair subtype")
    test_v5_records_subtype_map()
    test_default_dump_is_valid_v5()
    test_label_like_key_in_subtype_map_rejected()
    test_v4_shaped_dump_missing_v5_field_rejected()
    print("ALL V5 STAGE1B14-FEAT-DUMP-V5 GUARDS PASS")


if __name__ == "__main__":
    main()
