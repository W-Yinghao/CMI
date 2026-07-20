"""B1 real-data blockers (CPU, no MOABB download, no real data needed):
strict nested deleted-cell block, disjoint+complete subject roles, the hard offline network guard,
full-SHA-256 real-data hashes, and the ShallowConvNet geometry validator.

Standalone (``python -m oaci.tests.test_real_data_blockers``) and pytest-compatible.
"""
from __future__ import annotations

import os
import socket
import tempfile
import urllib.request

import numpy as np
import yaml

import oaci.protocol
from oaci.data.eeg.offline import forbid_network, new_network_counter
from oaci.data.eeg.registry import OfflineDownloadError
from oaci.models.shallow import validate_shallow_geometry
from oaci.protocol.manifest_v2 import DeletedCellBlock, load_v2

_SMOKE = os.path.join(os.path.dirname(oaci.protocol.__file__), "smoke_v1.yaml")


def _load_tampered(mutate):
    with open(_SMOKE) as f:
        raw = yaml.safe_load(f)
    mutate(raw)
    p = os.path.join(tempfile.mkdtemp(), "m.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(raw, f)
    return load_v2(p)


# ============================ blocker 1: strict nested blocks + subject roles ============================
def test_smoke_nested_preprocessing_and_deleted_cell_are_strict():
    m = load_v2(_SMOKE)
    assert isinstance(m.smoke.deleted_cell_level1, DeletedCellBlock)
    assert m.smoke.deleted_cell_level1.domain_id == "BNCI2014_001|subject-004"
    try:
        _load_tampered(lambda r: r["smoke"].__setitem__("deleted_cell_level1", {"domain_subject": 4, "class": "feet"}))
    except ValueError:
        pass
    else:
        raise AssertionError("a misspelled deleted_cell key must be rejected by the strict parser")


def test_smoke_subject_roles_are_disjoint_and_complete():
    load_v2(_SMOKE).validate_complete()                       # the canonical roles validate
    for bad in (lambda r: r["smoke"].__setitem__("source_train_subjects", [1, 5, 6]),   # overlaps target
                lambda r: r["smoke"].__setitem__("source_train_subjects", [4, 5])):       # union != subjects
        try:
            _load_tampered(bad).validate_complete()
        except ValueError:
            pass
        else:
            raise AssertionError("inconsistent smoke subject roles must be rejected")


# ============================ blocker 2: hard offline guard ============================
def test_missing_raw_data_cannot_trigger_network():
    cnt = new_network_counter()
    with forbid_network(cnt):
        for attempt in (lambda: urllib.request.urlopen("http://127.0.0.1:9/x"),
                        lambda: socket.create_connection(("127.0.0.1", 9)),
                        lambda: socket.socket().connect(("127.0.0.1", 9))):
            try:
                attempt()
            except OfflineDownloadError:
                pass
            else:
                raise AssertionError("a network attempt must raise OfflineDownloadError offline")
    assert cnt.attempts == 3


def test_offline_guard_restores_network_entry_points():
    before = socket.create_connection
    with forbid_network():
        assert socket.create_connection is not before
    assert socket.create_connection is before                 # restored on exit


# ============================ blocker 3: full SHA-256 ============================
def test_real_hashes_are_full_sha256():
    from oaci.data.eeg.audit import canonical_hash, split_manifest_hash, tensor_hash
    from oaci.data.eeg.cache import cache_key
    from oaci.data.eeg.preprocess import PreprocessSpec
    from oaci.data.eeg.schema import tensor_content_hash
    X = np.zeros((2, 3, 4), dtype=np.float32)
    assert len(tensor_content_hash(X)) == 64 and len(tensor_hash(X)) == 64
    assert len(PreprocessSpec().hash()) == 64 and len(canonical_hash({"a": 1})) == 64
    assert len(cache_key("fp", ["c0"], {"k": 1}, "v1")) == 64


# ============================ blocker 5: shallow geometry ============================
def test_shallow_geometry_accepts_real_bnci_shape_and_reports_dims():
    class BB:
        temporal_filters, temporal_kernel_samples = 40, 25
        pool_kernel_samples, pool_stride_samples = 75, 15
        dropout, safe_log_eps = 0.5, 1e-6
    g = validate_shallow_geometry(22, 385, BB)
    assert g["post_temporal_times"] == 361 and g["pooled_times"] == 20 and g["feat_dim"] == 800


def test_shallow_geometry_rejects_degenerate_inputs():
    class BB:
        temporal_filters, temporal_kernel_samples = 40, 25
        pool_kernel_samples, pool_stride_samples = 75, 15
        dropout, safe_log_eps = 0.5, 1e-6
    for ic, it, attr, val in ((22, 20, None, None),               # temporal kernel > in_times
                              (22, 385, "pool_kernel_samples", 400),   # pool kernel > post-temporal
                              (22, 385, "dropout", 1.0),               # dropout out of range
                              (22, 385, "safe_log_eps", 0.0)):         # non-positive eps
        b = BB()
        if attr:
            setattr(b, attr, val)
        try:
            validate_shallow_geometry(ic, it, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"degenerate geometry ({ic},{it},{attr}={val}) must be rejected")


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.data.eeg.offline  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} real-data-blocker tests")


if __name__ == "__main__":
    _run_all()
