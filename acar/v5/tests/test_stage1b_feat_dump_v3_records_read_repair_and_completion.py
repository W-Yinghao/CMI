"""Guard (Stage-1B12): the feature-dump schema is bumped to V3 — the header carries the BrainVision read-repair policy hash, the
raw_header_repair_manifest hash, and a per-recording read-repair map (label-free), alongside the V2 montage-completion map. A V2-shaped
dump (missing the V3 fields) is rejected; a label-like key anywhere in the repair map is rejected."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import feature_dump_schema as FS
from acar.v5.substrate import stage1b_feature_dump_writer as FDW
from acar.v5.tests._util import ok, expect_raises


def _write(path, **over):
    kw = dict(ref="SCZ/fold0/seed1", disease="SCZ", fold=0, seed=1, preprocessing_config_sha256="0" * 64,
              training_config_sha256="0" * 64, encoder_checkpoint_file_sha256="0" * 64, source_state_file_sha256="0" * 64,
              records=[("SCZ/ds004367/sub-S01", "train", 0, [1.0, 2.0, 3.0]),
                       ("SCZ/ds004000/sub-042", "eval", 0, [4.0, 5.0, 6.0])])
    kw.update(over)
    return FDW.write_feature_dump(path, **kw)


def test_v3_records_repair_and_completion_maps():
    p = os.path.join(tempfile.mkdtemp(), "feat.npz")
    summ = _write(
        p,
        montage_completion_by_subject={"SCZ/ds004000/sub-042": {"interpolated": ["F3", "F4", "P3", "P4"],
                                                                "n_interpolated": 4, "donor_count": 35}},
        brainvision_read_repair_by_recording={"SCZ/ds004000/sub-042::sub-042_task-proposer_run-1_eeg.vhdr":
                                              {"repair_mode": "broken_internal_pointer_rewrite",
                                               "original_header_sha256": "a" * 64, "repaired_header_sha256": "b" * 64}},
        raw_header_repair_manifest_sha256="c" * 64)
    assert len(summ["brainvision_read_repair_policy_sha256"]) == 64
    assert summ["raw_header_repair_manifest_sha256"] == "c" * 64
    assert "SCZ/ds004000/sub-042::sub-042_task-proposer_run-1_eeg.vhdr" in summ["brainvision_read_repair_by_recording"]
    assert summ["montage_completion_by_subject"]["SCZ/ds004000/sub-042"]["n_interpolated"] == 4
    ok("V3 dump round-trips the read-repair policy hash, manifest hash, per-recording repair map, and completion map")


def test_default_no_repair_dump_is_valid_v3():
    p = os.path.join(tempfile.mkdtemp(), "feat.npz")
    summ = _write(p)                                                      # no repair args → defaults (empty map + sentinel hash)
    assert summ["brainvision_read_repair_by_recording"] == {} and len(summ["raw_header_repair_manifest_sha256"]) == 64
    ok("a no-repair dump defaults to an empty repair map + the sentinel manifest hash and is valid V3")


def test_label_like_key_in_repair_map_rejected():
    p = os.path.join(tempfile.mkdtemp(), "feat.npz")
    expect_raises(FS.FeatureDumpSchemaError,
                  lambda: _write(p, brainvision_read_repair_by_recording={"x::r.vhdr": {"repair_mode": "x", "label": 1}}),
                  "a label-like key nested in the repair map must be rejected")
    ok("a label-like field nested inside brainvision_read_repair_by_recording is rejected (label firewall)")


def test_v2_shaped_dump_missing_v3_fields_rejected():
    import numpy as np
    # a V2-style mapping: every V2 header field + record arrays, but NONE of the three V3 fields → must fail "missing required field"
    v2 = {
        "schema_version": np.asarray(FS.SCHEMA_VERSION), "ref": np.asarray("SCZ/fold0/seed1"), "disease": np.asarray("SCZ"),
        "fold": np.asarray(0), "seed": np.asarray(1),
        "preprocessing_config_sha256": np.asarray("0" * 64), "training_config_sha256": np.asarray("0" * 64),
        "encoder_checkpoint_file_sha256": np.asarray("0" * 64), "source_state_file_sha256": np.asarray("0" * 64),
        "channel_alias_policy_sha256": np.asarray("0" * 64), "montage_completion_policy_sha256": np.asarray("0" * 64),
        "montage_completion_by_subject": np.asarray("{}"),
        "subject_key": np.asarray(["SCZ/ds/x"]), "split_role": np.asarray(["train"]),
        "window_id": np.asarray([0], dtype=np.int64), "embedding": np.asarray([[1.0, 2.0]], dtype=np.float32),
    }
    expect_raises(FS.FeatureDumpSchemaError, lambda: FS.validate_loaded(v2),
                  "a V2-shaped dump lacking the V3 fields must be rejected")
    ok("a V2-shaped dump (no read-repair fields) is rejected by the V3 validator (missing required field)")


def main():
    print("ACAR v5 Stage-1B12 guard: feature-dump V3 records read-repair + completion")
    test_v3_records_repair_and_completion_maps()
    test_default_no_repair_dump_is_valid_v3()
    test_label_like_key_in_repair_map_rejected()
    test_v2_shaped_dump_missing_v3_fields_rejected()
    print("ALL V5 STAGE1B12-FEAT-DUMP-V3 GUARDS PASS")


if __name__ == "__main__":
    main()
