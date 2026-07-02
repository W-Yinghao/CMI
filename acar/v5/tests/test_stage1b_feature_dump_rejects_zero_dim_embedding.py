"""Guard (Stage-1B9): the feature-dump SCHEMA validator itself (dumper-agnostic barrier) rejects a zero-dim embedding — not just the
dumper. Synthetic numpy + temp files only."""
from __future__ import annotations
import os
import tempfile
import numpy as np
from acar.v5.substrate import feature_dump_schema as FS
from acar.v5.substrate import stage1b_feature_dump_writer as FDW
from acar.v5.tests._util import expect_raises, ok

H = dict(preprocessing_config_sha256="a" * 64, training_config_sha256="b" * 64,
         encoder_checkpoint_file_sha256="c" * 64, source_state_file_sha256="d" * 64,
         channel_alias_policy_sha256="e" * 64, montage_completion_policy_sha256="f" * 64)


def test_validate_loaded_rejects_zero_dim():
    payload = {"schema_version": np.asarray(FS.SCHEMA_VERSION), "ref": np.asarray("PD/fold0/seed20260711"),
               "disease": np.asarray("PD"), "fold": np.asarray(0), "seed": np.asarray(20260711),
               "subject_key": np.asarray(["PD/ds/sub-1"]), "split_role": np.asarray(["train"]),
               "window_id": np.asarray([0]), "embedding": np.zeros((1, 0), np.float32),
               "montage_completion_by_subject": np.asarray("{}"),
               **{k: np.asarray(v) for k, v in H.items()}}
    expect_raises(FS.FeatureDumpSchemaError, lambda: FS.validate_loaded(payload))
    ok("validate_loaded rejects a (n,0) zero-dim embedding at the parser/barrier level")


def test_writer_rejects_zero_dim():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "f.npz")
        expect_raises(FS.FeatureDumpSchemaError,
                      lambda: FDW.write_feature_dump(p, ref="PD/fold0/seed20260711", disease="PD", fold=0, seed=20260711,
                                                     records=[("PD/ds/sub-1", "train", 0, [])], **H))
    ok("write_feature_dump refuses to emit a zero-dim embedding dump")


def main():
    print("ACAR v5 Stage-1B9 guard: feature dump rejects zero-dim embedding")
    test_validate_loaded_rejects_zero_dim()
    test_writer_rejects_zero_dim()
    print("ALL V5 STAGE1B-FEATURE-DUMP-ZERO-DIM GUARDS PASS")


if __name__ == "__main__":
    main()
