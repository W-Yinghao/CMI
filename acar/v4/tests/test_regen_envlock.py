"""Guards for acar/v4/regen_envlock.py — the regen runtime env-lock schema/validator (NO torch capture). Run:
python -m acar.v4.tests.test_regen_envlock
"""
from acar.v4 import regen_envlock as EL
from acar.v4 import regen_substrate as R


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:                          # noqa
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no exception raised")


def _captured(**over):
    lk = EL.schema_only_template(protocol_commit="a" * 40, pipeline_config_sha256=R.canonical_pipeline_config_sha256())
    lk.update(status="CAPTURED_AND_VERIFIED", python_version="3.13.14", torch_version="2.6.0+cu124",
              torchvision_version="0.21.0+cu124", torchaudio_version="2.6.0+cu124", moabb_version="1.5.0",
              mne_version="1.12.1", skorch_version="1.4.0", braindecode_version="1.5.2", numpy_version="2.4.4",
              scipy_version="1.18.0", sklearn_version="1.9.0", device_kind="cpu", device_name="cpu")
    lk.update(over)
    return lk


def test_schema_only_template_valid():
    t = EL.schema_only_template(protocol_commit="a" * 40, pipeline_config_sha256="b" * 64)
    assert EL.validate_regen_env_lock(t) is t and t["status"] == "SCHEMA_ONLY_NOT_CAPTURED"
    assert set(t) == set(EL.expected_regen_env_fields())            # exact field set


def test_captured_valid_and_hash_canonical():
    lk = _captured()
    assert EL.validate_regen_env_lock(lk) is lk
    h = EL.hash_regen_env_lock(lk)
    assert len(h) == 64 and EL.hash_regen_env_lock(dict(reversed(list(lk.items())))) == h   # key-order independent


def test_missing_and_extra_fields_fail():
    lk = _captured(); del lk["torch_version"]
    _expect(ValueError, lambda: EL.validate_regen_env_lock(lk))                              # missing
    lk2 = _captured(); lk2["bonus"] = 1
    _expect(ValueError, lambda: EL.validate_regen_env_lock(lk2))                             # unknown extra


def test_status_and_device_and_seed_strict():
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(status="WHATEVER")))
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(device_kind="tpu")))
    for s in (1, True, "0", 0.0):
        _expect(ValueError, lambda s=s: EL.validate_regen_env_lock(_captured(seed=s)))       # strict int 0
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(torch_deterministic_algorithms=False)))
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(omp_num_threads=0)))    # positive int


def test_captured_must_have_real_versions():
    # a CAPTURED lock cannot leave version/device fields empty (skeleton can't impersonate a captured runtime)
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(torch_version="")))
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(device_name="")))
    # CAPTURED cuda must fill cuda/cudnn/driver
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(device_kind="cuda", cuda_version="")))
    ok_cuda = _captured(device_kind="cuda", cuda_version="12.4", cudnn_version="9.1", driver_version="550.x",
                        device_name="A100")
    assert EL.validate_regen_env_lock(ok_cuda) is ok_cuda


def test_bad_hashes_fail():
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(pipeline_config_sha256="short")))
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(protocol_commit="x" * 39)))


def test_capture_failed_and_note():
    lk = EL.schema_only_template(protocol_commit="a" * 40, pipeline_config_sha256="b" * 64)
    lk["status"] = "CAPTURE_FAILED"; lk["capture_note"] = "braindecode EEGNetv4 import FAILED: ..."
    assert EL.validate_regen_env_lock(lk) is lk             # CAPTURE_FAILED valid with empty versions + a note
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(capture_note=123)))   # note must be str


def test_captured_requires_threads_pinned_to_one():
    # the lock must capture the SAME single-thread deterministic runtime training uses (interop=20 was the 6d0277b defect)
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(torch_interop_threads=20)))
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(torch_intraop_threads=2)))
    _expect(ValueError, lambda: EL.validate_regen_env_lock(_captured(omp_num_threads=4)))


def test_captured_requires_import_critical_versions():
    # torchvision/torchaudio/moabb are import-critical (the env failure was a torchaudio/torch + braindecode/moabb mismatch)
    for f in ("torchvision_version", "torchaudio_version", "moabb_version"):
        _expect(ValueError, lambda f=f: EL.validate_regen_env_lock(_captured(**{f: ""})))
    # they are in _REQUIRED, hence in the canonical hash
    assert all(f in EL.expected_regen_env_fields() for f in ("torchvision_version", "torchaudio_version", "moabb_version"))


def main():
    print("ACAR v4 regen_envlock guards (schema/validator; NO torch capture):")
    for t in (test_schema_only_template_valid, test_captured_valid_and_hash_canonical, test_missing_and_extra_fields_fail,
              test_status_and_device_and_seed_strict, test_captured_must_have_real_versions, test_bad_hashes_fail,
              test_capture_failed_and_note, test_captured_requires_threads_pinned_to_one,
              test_captured_requires_import_critical_versions):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 REGEN-ENVLOCK GUARDS PASS")


if __name__ == "__main__":
    main()
