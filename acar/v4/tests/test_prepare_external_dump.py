"""Guards for acar/v4/prepare_external_dump.py (frozen external-input prep layer). SYNTHETIC FIXTURES ONLY; NO download,
NO real signal, NO encoder run. Proves the pure selectors/parsers/validators/hashers and that prepare_dump is FAIL-CLOSED
on the missing frozen encoder (FrozenEncoderMissingError; never retrains). Run: python -m acar.v4.tests.test_prepare_external_dump
"""
import numpy as np

from acar.v4 import prepare_external_dump as P


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:                       # noqa
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no exception raised")


def test_resting_run_selector():
    runs = [{"task": "rest", "run": 1}, {"task": "walking", "run": 1}, {"task": "restWalk", "run": 1}]
    assert P.resting_run_selector(runs, "ds007526") == [{"task": "rest", "run": 1}]   # walking + restWalk(walk) excluded
    assert P.resting_run_selector([{"task": "eyesClosed_rest"}, {"task": "oddball"}],
                                  "zenodo14808296") == [{"task": "eyesClosed_rest"}]
    _expect(ValueError, lambda: P.resting_run_selector([{"task": "walking"}], "ds007526"))   # no resting → fail-closed
    _expect(ValueError, lambda: P.resting_run_selector([{"task": "rest"}], "unknown_site"))
    # EXACT-token match: 'arrest' / 'prestimulus' are NOT resting (no substring false-positive)
    _expect(ValueError, lambda: P.resting_run_selector([{"task": "arrest"}, {"task": "prestimulus"}], "ds007526"))


def test_parse_diagnosis_map():
    rows = [{"participant_id": "sub-001", "group": "HC"}, {"participant_id": "sub-002", "group": "PD"}]
    assert P.parse_diagnosis_map(rows, "ds007526") == {"sub-001": 0, "sub-002": 1}
    _expect(ValueError, lambda: P.parse_diagnosis_map(rows + [{"participant_id": "sub-003", "group": "??"}], "ds007526"))
    _expect(ValueError, lambda: P.parse_diagnosis_map([{"participant_id": "x", "group": "PD"}], "ds007526"))  # 1 class
    _expect(ValueError, lambda: P.parse_diagnosis_map([{"participant_id": "x", "group": "PD"},
                                                       {"participant_id": "x", "group": "HC"}], "ds007526"))  # dup id
    _expect(ValueError, lambda: P.parse_diagnosis_map([{"participant_id": "x"}], "ds007526"))                 # no field


def test_validate_channels_fs():
    assert P.validate_channels_fs(64, 1000, "zenodo14808296")["n_channels"] == 64
    _expect(ValueError, lambda: P.validate_channels_fs(32, 1000, "zenodo14808296"))     # wrong channels
    _expect(ValueError, lambda: P.validate_channels_fs(64, 500, "zenodo14808296"))      # wrong Fs
    assert P.validate_channels_fs(65, 250, "ds007526")["fs"] == 250                     # ds007526 expected None → records


def test_validate_dump_schema():
    n, d = 5, 16
    ok = {"z_te": np.zeros((n, d), float), "subject_id_te": np.array(["s0", "s1", "s2", "s3", "s4"]),
          "recording_id_te": np.array(["s0", "s1", "s2", "s3", "s4"]), "window_index_te": np.arange(n),
          "y_te": np.array([0, 1, 0, 1, 0]), "feat_hash_te": "a" * 64}
    assert P.validate_dump_schema(ok) == n
    _expect(ValueError, lambda: P.validate_dump_schema({**ok, "y_te": np.array([0, 1, 2, 0, 1])}))    # bad label
    _expect(ValueError, lambda: P.validate_dump_schema({k: v for k, v in ok.items() if k != "z_te"}))  # missing
    _expect(ValueError, lambda: P.validate_dump_schema({**ok, "window_index_te": np.arange(n - 1)}))  # length
    _expect(ValueError, lambda: P.validate_dump_schema({**ok, "z_ev": np.zeros((n, d))}))             # forbidden extra
    _expect(ValueError, lambda: P.validate_dump_schema({**ok, "z_te": np.full((n, d), np.nan)}))      # non-finite z
    _expect(ValueError, lambda: P.validate_dump_schema({**ok, "subject_id_te": np.array(["", "s1", "s2", "s3", "s4"])}))
    dup = {**ok, "subject_id_te": np.array(["s"] * n), "recording_id_te": np.array(["r"] * n),
           "window_index_te": np.zeros(n, int)}
    _expect(ValueError, lambda: P.validate_dump_schema(dup))                                          # duplicate rows
    _expect(ValueError, lambda: P.validate_dump_schema(ok, embedding_dim=8))                          # d mismatch
    _expect(ValueError, lambda: P.validate_dump_schema({**ok, "feat_hash_te": "abc"}))               # bad feat_hash


def test_provenance_hashers():
    assert P.subject_list_sha256(["a", "b"]) == P.subject_list_sha256(["b", "a"])      # permutation-independent
    assert P.subject_list_sha256(["a", "b"]) != P.subject_list_sha256(["a", "c"])
    assert P.diagnosis_mapping_sha256({"a": 1, "b": 0}) == P.diagnosis_mapping_sha256({"b": 0, "a": 1})
    assert P.diagnosis_mapping_sha256({"a": 1}) != P.diagnosis_mapping_sha256({"a": 0})
    p1 = P.raw_pipeline_sha256({"fs": 250, "bandpass": [1, 40]})
    assert p1 == P.raw_pipeline_sha256({"bandpass": [1, 40], "fs": 250}) != P.raw_pipeline_sha256({"fs": 500})
    assert len(p1) == 64
    assert P.resting_selection_sha256([{"task": "rest"}]) == P.resting_selection_sha256([{"task": "rest"}])


def test_prepare_dump_fail_closed_and_encoder_artifact():
    # missing/incomplete encoder artifact → FrozenEncoderMissingError BEFORE any heavy import or raw read
    _expect(P.FrozenEncoderMissingError, lambda: P.prepare_dump("ds007526", "/raw", "/out.npz",
                                                                encoder_artifact={}, raw_pipeline_config=P.FROZEN_PIPELINE))
    _expect(P.FrozenEncoderMissingError, lambda: P.require_encoder_artifact({}))
    full = {f: "x" for f in P.ENCODER_ARTIFACT_FIELDS}; full["embedding_dim"] = 16
    _expect(P.FrozenEncoderMissingError, lambda: P.require_encoder_artifact(full))   # complete but paths not on disk
    bad_dim = dict(full); bad_dim["embedding_dim"] = 8
    _expect(ValueError, lambda: P.require_encoder_artifact(bad_dim))                 # wrong embedding_dim
    # pipeline config must equal the frozen DEV pipeline
    assert P.validate_pipeline_config(P.FROZEN_PIPELINE)
    _expect(ValueError, lambda: P.validate_pipeline_config({**P.FROZEN_PIPELINE, "resample_fs": 250}))
    # NO implicit retrain path: prepare_dump never reaches a training call without a verified encoder (fail-closed above)


def main():
    print("ACAR v4 prepare_external_dump guards (synthetic fixtures only):")
    for t in (test_resting_run_selector, test_parse_diagnosis_map, test_validate_channels_fs, test_validate_dump_schema,
              test_provenance_hashers, test_prepare_dump_fail_closed_and_encoder_artifact):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 PREPARE-EXTERNAL-DUMP GUARDS PASS")


if __name__ == "__main__":
    main()
