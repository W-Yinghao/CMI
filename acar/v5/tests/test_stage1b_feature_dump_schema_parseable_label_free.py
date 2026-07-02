"""Guard (Stage-1B7): the feature dump is a PINNED, parseable, LABEL-FREE schema (not an opaque blob). write→parse round-trips; a
label-like field, an empty dump, a non-finite embedding, an unknown split role, or a wrong schema version all fail closed.
Synthetic temp files + numpy only."""
from __future__ import annotations
import os
import tempfile
import numpy as np
from acar.v5.substrate import feature_dump_schema as FS
from acar.v5.substrate import stage1b_feature_dump_writer as FDW
from acar.v5.tests._util import expect_raises, ok

H = {"preprocessing_config_sha256": "a" * 64, "training_config_sha256": "b" * 64,
     "encoder_checkpoint_file_sha256": "c" * 64, "source_state_file_sha256": "d" * 64}
# V2 header extras for manual validate_loaded payloads (the writer defaults these; manual payloads must include them)
HV2 = {"channel_alias_policy_sha256": "e" * 64, "montage_completion_policy_sha256": "f" * 64,
       "montage_completion_by_subject": "{}"}


def _records():
    return [("PD/ds002778/sub-001", "train", 0, [0.1, 0.2, 0.3]),
            ("PD/ds002778/sub-002", "cal", 0, [0.4, 0.5, 0.6]),
            ("PD/ds002778/sub-003", "eval", 1, [0.7, 0.8, 0.9])]


def test_write_parse_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "feat_dump.npz")
        summ = FDW.write_feature_dump(p, ref="PD/fold0/seed20260711", disease="PD", fold=0, seed=20260711,
                                      records=_records(), **H)
        assert summ["n_records"] == 3 and summ["embedding_dim"] == 3 and summ["ref"] == "PD/fold0/seed20260711"
        assert set(summ["split_roles_present"]) == {"train", "cal", "eval"}
        again = FDW.parse_feature_dump(p)                     # re-parse from disk
        assert again["n_records"] == 3 and again["fold"] == 0 and again["seed"] == 20260711
    ok("write_feature_dump → parse_feature_dump round-trips (records, dim, provenance, split roles)")


def test_forbidden_label_field_rejected():
    payload = {k: np.asarray(v) for k, v in {"schema_version": FS.SCHEMA_VERSION, "ref": "PD/fold0/seed20260711",
               "disease": "PD", "fold": 0, "seed": 20260711, **H, **HV2,
               "subject_key": np.asarray(["PD/ds002778/sub-1"]), "split_role": np.asarray(["train"]),
               "window_id": np.asarray([0]), "embedding": np.zeros((1, 3), np.float32),
               "label": np.asarray([1])}.items()}
    expect_raises(FS.FeatureDumpSchemaError, lambda: FS.validate_loaded(payload))
    ok("a label-like field in the dump → FeatureDumpSchemaError")


def test_empty_and_nonfinite_and_bad_role_rejected():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "f.npz")
        expect_raises(FDW.FeatureDumpWriteError, lambda: FDW.write_feature_dump(p, ref="PD/fold0/seed20260711", disease="PD",
                                                                                fold=0, seed=20260711, records=[], **H))
        expect_raises(FDW.FeatureDumpWriteError,
                      lambda: FDW.write_feature_dump(p, ref="PD/fold0/seed20260711", disease="PD", fold=0, seed=20260711,
                                                     records=[("PD/ds002778/sub-1", "bogus", 0, [0.1])], **H))
        expect_raises((FDW.FeatureDumpWriteError, FS.FeatureDumpSchemaError),
                      lambda: FDW.write_feature_dump(p, ref="PD/fold0/seed20260711", disease="PD", fold=0, seed=20260711,
                                                     records=[("PD/ds002778/sub-1", "train", 0, [float("nan")])], **H))
    ok("empty dump / unknown split_role / non-finite embedding → rejected")


def test_wrong_schema_version_rejected():
    payload = {"schema_version": np.asarray("WRONG_V0"), "ref": np.asarray("PD/fold0/seed20260711"),
               "disease": np.asarray("PD"), "fold": np.asarray(0), "seed": np.asarray(20260711),
               "subject_key": np.asarray(["PD/ds002778/sub-1"]), "split_role": np.asarray(["train"]),
               "window_id": np.asarray([0]), "embedding": np.zeros((1, 3), np.float32),
               **{k: np.asarray(v) for k, v in {**H, **HV2}.items()}}
    expect_raises(FS.FeatureDumpSchemaError, lambda: FS.validate_loaded(payload))
    ok("a wrong schema_version → FeatureDumpSchemaError")


def main():
    print("ACAR v5 Stage-1B7 guard: feature dump schema parseable label-free")
    test_write_parse_roundtrip()
    test_forbidden_label_field_rejected()
    test_empty_and_nonfinite_and_bad_role_rejected()
    test_wrong_schema_version_rejected()
    print("ALL V5 STAGE1B-FEATURE-DUMP-SCHEMA GUARDS PASS")


if __name__ == "__main__":
    main()
