"""Guards for acar/v4/prepare_external_dump.py (frozen external-input prep layer). SYNTHETIC FIXTURES ONLY; NO download,
NO real signal, NO encoder run. Proves the pure selectors/parsers/validators/hashers and that the real prepare_dump is a
gated stub. Run: python -m acar.v4.tests.test_prepare_external_dump
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
    kept = P.resting_run_selector(runs, "ds007526")
    assert kept == [{"task": "rest", "run": 1}]                       # walking + restWalk excluded
    z = P.resting_run_selector([{"task": "eyesClosed_rest"}, {"task": "oddball"}], "zenodo14808296")
    assert z == [{"task": "eyesClosed_rest"}]
    _expect(ValueError, lambda: P.resting_run_selector([{"task": "walking"}], "ds007526"))   # no resting → fail-closed
    _expect(ValueError, lambda: P.resting_run_selector([{"task": "rest"}], "unknown_site"))


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
    n, d = 5, 4
    ok = {"z_te": np.zeros((n, d), float), "subject_id_te": np.array(["s"] * n),
          "recording_id_te": np.array(["r"] * n), "window_index_te": np.arange(n),
          "y_te": np.array([0, 1, 0, 1, 0]), "feat_hash_te": "abc"}
    assert P.validate_dump_schema(ok) == n
    _expect(ValueError, lambda: P.validate_dump_schema({**ok, "y_te": np.array([0, 1, 2, 0, 1])}))   # bad label
    bad = dict(ok); del bad["z_te"]
    _expect(ValueError, lambda: P.validate_dump_schema(bad))                                          # missing
    _expect(ValueError, lambda: P.validate_dump_schema({**ok, "window_index_te": np.arange(n - 1)}))  # length


def test_provenance_hashers():
    assert P.subject_list_sha256(["a", "b"]) == P.subject_list_sha256(["b", "a"])      # permutation-independent
    assert P.subject_list_sha256(["a", "b"]) != P.subject_list_sha256(["a", "c"])
    assert P.diagnosis_mapping_sha256({"a": 1, "b": 0}) == P.diagnosis_mapping_sha256({"b": 0, "a": 1})
    assert P.diagnosis_mapping_sha256({"a": 1}) != P.diagnosis_mapping_sha256({"a": 0})
    p1 = P.raw_pipeline_sha256({"fs": 250, "bandpass": [1, 40]})
    assert p1 == P.raw_pipeline_sha256({"bandpass": [1, 40], "fs": 250}) != P.raw_pipeline_sha256({"fs": 500})
    assert len(p1) == 64
    assert P.resting_selection_sha256([{"task": "rest"}]) == P.resting_selection_sha256([{"task": "rest"}])


def test_prepare_dump_is_gated_stub():
    _expect(NotImplementedError, lambda: P.prepare_dump("ds007526", "/raw", "/out",
                                                        frozen_pipeline_params={}, frozen_encoder_ref="x"))


def main():
    print("ACAR v4 prepare_external_dump guards (synthetic fixtures only):")
    for t in (test_resting_run_selector, test_parse_diagnosis_map, test_validate_channels_fs, test_validate_dump_schema,
              test_provenance_hashers, test_prepare_dump_is_gated_stub):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 PREPARE-EXTERNAL-DUMP GUARDS PASS")


if __name__ == "__main__":
    main()
