"""Guard (Stage-1B13): the feature-dump schema is bumped to V4 — the header carries the channels.tsv channel-NAME repair policy hash
and a per-recording rename map (label-free), alongside the V3 read-repair + V2 completion maps. A V3-shaped dump (missing the V4
fields) is rejected; a label-like key anywhere in the rename map is rejected."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import feature_dump_schema as FS
from acar.v5.substrate import stage1b_feature_dump_writer as FDW
from acar.v5.tests._util import ok, expect_raises


def _write(path, **over):
    kw = dict(ref="SCZ/fold0/seed1", disease="SCZ", fold=0, seed=1, preprocessing_config_sha256="0" * 64,
              training_config_sha256="0" * 64, encoder_checkpoint_file_sha256="0" * 64, source_state_file_sha256="0" * 64,
              records=[("SCZ/ds003944/sub-1448", "train", 0, [1.0, 2.0, 3.0])])
    kw.update(over)
    return FDW.write_feature_dump(path, **kw)


def test_v4_records_channel_name_repair_map():
    p = os.path.join(tempfile.mkdtemp(), "feat.npz")
    summ = _write(p, channel_name_repair_by_recording={
        "SCZ/ds003944/sub-1448::sub-1448_task-Rest_eeg.vhdr": {
            "channel_name_source": "channels.tsv", "channels_tsv_sha256": "a" * 64,
            "channel_name_mapping_sha256": "b" * 64, "original_header_channel_names_sha256": "c" * 64,
            "repaired_header_channel_names_sha256": "d" * 64}})
    assert str(FS.SCHEMA_VERSION) == "ACAR_V5_STAGE1B_FEAT_DUMP_V5"   # V4 fields persist under V5 (superset)
    assert len(summ["channel_name_repair_policy_sha256"]) == 64
    assert "SCZ/ds003944/sub-1448::sub-1448_task-Rest_eeg.vhdr" in summ["channel_name_repair_by_recording"]
    ok("V4 dump round-trips the channel-name repair policy hash + per-recording rename map")


def test_default_no_rename_dump_is_valid_v4():
    p = os.path.join(tempfile.mkdtemp(), "feat.npz")
    summ = _write(p)                                           # no rename args → empty map default
    assert summ["channel_name_repair_by_recording"] == {} and len(summ["channel_name_repair_policy_sha256"]) == 64
    ok("a no-rename dump defaults to an empty rename map and is valid V4")


def test_label_like_key_in_rename_map_rejected():
    p = os.path.join(tempfile.mkdtemp(), "feat.npz")
    expect_raises(FS.FeatureDumpSchemaError,
                  lambda: _write(p, channel_name_repair_by_recording={"x::r.vhdr": {"channel_name_source": "channels.tsv",
                                                                                    "diagnosis": 1}}),
                  "a label-like key nested in the rename map must be rejected")
    ok("a label-like field nested inside channel_name_repair_by_recording is rejected (label firewall)")


def test_v3_shaped_dump_missing_v4_fields_rejected():
    import numpy as np
    v3 = {
        "schema_version": np.asarray(FS.SCHEMA_VERSION), "ref": np.asarray("SCZ/fold0/seed1"), "disease": np.asarray("SCZ"),
        "fold": np.asarray(0), "seed": np.asarray(1),
        "preprocessing_config_sha256": np.asarray("0" * 64), "training_config_sha256": np.asarray("0" * 64),
        "encoder_checkpoint_file_sha256": np.asarray("0" * 64), "source_state_file_sha256": np.asarray("0" * 64),
        "channel_alias_policy_sha256": np.asarray("0" * 64), "montage_completion_policy_sha256": np.asarray("0" * 64),
        "montage_completion_by_subject": np.asarray("{}"),
        "brainvision_read_repair_policy_sha256": np.asarray("0" * 64), "raw_header_repair_manifest_sha256": np.asarray("0" * 64),
        "brainvision_read_repair_by_recording": np.asarray("{}"),
        "subject_key": np.asarray(["SCZ/ds/x"]), "split_role": np.asarray(["train"]),
        "window_id": np.asarray([0], dtype=np.int64), "embedding": np.asarray([[1.0, 2.0]], dtype=np.float32),
    }
    expect_raises(FS.FeatureDumpSchemaError, lambda: FS.validate_loaded(v3),
                  "a V3-shaped dump lacking the V4 fields must be rejected")
    ok("a V3-shaped dump (no channel-name-repair fields) is rejected by the V4 validator (missing required field)")


def main():
    print("ACAR v5 Stage-1B13 guard: feature-dump V4 records channel-name repair")
    test_v4_records_channel_name_repair_map()
    test_default_no_rename_dump_is_valid_v4()
    test_label_like_key_in_rename_map_rejected()
    test_v3_shaped_dump_missing_v4_fields_rejected()
    print("ALL V5 STAGE1B13-FEAT-DUMP-V4 GUARDS PASS")


if __name__ == "__main__":
    main()
