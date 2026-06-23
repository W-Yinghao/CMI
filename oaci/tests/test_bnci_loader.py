"""B1a: strict MI preprocessing block, exact MotorImagery mapping, offline-root env handling, and the
missing-data-cannot-download guard. CPU only -- the real six-subject load is exercised by the dedicated
BNCI preflight job (B1b), not by normal CI.

Standalone (``python -m oaci.tests.test_bnci_loader``) and pytest-compatible.
"""
from __future__ import annotations

import os
import tempfile

import yaml

import oaci.protocol
from oaci.data.eeg.bnci import LOADER_CODE_VERSION, load_moabb_confirmatory, motor_imagery_kwargs
from oaci.data.eeg.offline import moabb_offline_root
from oaci.data.eeg.registry import OfflineDownloadError
from oaci.protocol.manifest_v2 import MIPreprocessingBlock, load_v2

_SMOKE = os.path.join(os.path.dirname(oaci.protocol.__file__), "smoke_v1.yaml")
_CH = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
       "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]


def _pp():
    return load_v2(_SMOKE).enabled_datasets()["BNCI2014_001"].preprocessing


def _load_tampered(mutate):
    with open(_SMOKE) as f:
        raw = yaml.safe_load(f)
    mutate(raw)
    p = os.path.join(tempfile.mkdtemp(), "m.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(raw, f)
    return load_v2(p)


# ============================ strict MI block ============================
def test_mi_preprocessing_nested_block_is_strict():
    assert isinstance(_pp(), MIPreprocessingBlock) and _pp().code_version == LOADER_CODE_VERSION


def test_mi_preprocessing_unknown_key_fails():
    try:
        _load_tampered(lambda r: r["datasets"]["BNCI2014_001"]["preprocessing"].__setitem__("fmim", 4.0))
    except ValueError:
        pass
    else:
        raise AssertionError("a misspelled MI preprocessing key must be rejected")


def test_smoke_preprocessing_maps_exactly_to_motor_imagery():
    kw = motor_imagery_kwargs(_pp(), ["left_hand", "right_hand", "feet", "tongue"], _CH)
    assert kw == {"n_classes": 4, "events": ["left_hand", "right_hand", "feet", "tongue"],
                  "fmin": 4.0, "fmax": 38.0, "tmin": 0.5, "tmax": 3.5, "baseline": None,
                  "channels": _CH, "resample": 128.0}


def test_epoch_length_is_exact_not_tolerant():
    pp = _pp()
    exp = int(round((pp.epoch_tmax - pp.epoch_tmin) * pp.resample_sfreq)) + 1   # MNE includes both ends
    assert exp == 385
    try:
        _load_tampered(lambda r: r["datasets"]["BNCI2014_001"].__setitem__("expected_n_times", 384)).validate_complete()
    except ValueError:
        pass
    else:
        raise AssertionError("an off-by-one expected_n_times must be rejected (no tolerance)")


def test_mi_preprocessing_rejects_bad_ranges():
    for bad in (lambda r: r["datasets"]["BNCI2014_001"]["preprocessing"].__setitem__("fmax", 70.0),  # >= sfreq/2
                lambda r: r["datasets"]["BNCI2014_001"]["preprocessing"].__setitem__("channel_interpolation", True),
                lambda r: r["datasets"]["BNCI2014_001"]["preprocessing"].__setitem__("epoch_tmax", 0.5)):
        try:
            _load_tampered(bad).validate_complete()
        except ValueError:
            pass
        else:
            raise AssertionError("an out-of-range MI preprocessing value must be rejected")


# ============================ offline root env handling ============================
def test_moabb_offline_root_sets_and_restores_env():
    before = {k: os.environ.get(k) for k in ("MNE_DATA", "MNE_DATASETS_BNCI_PATH", "MOABB_OFFLINE")}
    root = tempfile.mkdtemp()
    with moabb_offline_root(root):
        assert os.environ["MNE_DATASETS_BNCI_PATH"] == root and os.environ["MOABB_OFFLINE"] == "1"
    assert {k: os.environ.get(k) for k in before} == before        # exact restore


def test_offline_root_does_not_leak_warning_filters():
    import warnings
    before = list(warnings.filters)
    with moabb_offline_root(tempfile.mkdtemp()):
        warnings.filterwarnings("ignore", message="anything")
    assert list(warnings.filters) == before                        # global filters restored


# ============================ missing-data guard ============================
def test_missing_datalake_cannot_download():
    # a non-existent datalake -> data_path fails inside the network guard -> OfflineDownloadError, no fetch
    try:
        load_moabb_confirmatory("BNCI2014_001", [1], _pp(), frozen_class_names=["left_hand", "right_hand", "feet", "tongue"],
                                frozen_channels=_CH, expected_sfreq=128.0, expected_n_times=385,
                                datalake_root=os.path.join(tempfile.mkdtemp(), "nonexistent"))
    except OfflineDownloadError:
        pass
    else:
        raise AssertionError("a missing datalake must raise OfflineDownloadError, never download")


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.data.eeg.bnci  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} bnci-loader tests")


if __name__ == "__main__":
    _run_all()
