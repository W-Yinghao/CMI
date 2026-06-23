"""B1b CPU unit tests: the BNCI adapter's identity/structure on a SYNTHETIC bundle (no real load), plus
the three real-data residuals. The exact 144 / 3456 contract is verified by the dedicated preflight job.

Standalone (``python -m oaci.tests.test_bnci_runner_adapter``) and pytest-compatible.
"""
from __future__ import annotations

import os

import numpy as np

import oaci.protocol
from oaci.data.eeg.bnci import MOABBLoadEvidence, MOABBLoadResult, RawHeaderRecord, _evidence_hash
from oaci.data.eeg.preprocess import PreprocessSpec, apply_normalization
from oaci.data.eeg.schema import EEGBundle
from oaci.protocol.manifest_v2 import load_v2, manifest_payload_hash
from oaci.runner.bnci_data import build_bnci_fold_from_bundle, target_seen_by_fit

_SMOKE = os.path.join(os.path.dirname(oaci.protocol.__file__), "smoke_v1.yaml")
_CLASSES = ["left_hand", "right_hand", "feet", "tongue"]
_CH = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
       "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]


def _synthetic_load_result():
    rows = []
    for subj in range(1, 7):
        dom = f"BNCI2014_001|subject-{subj:03d}"
        for rec in range(3):
            g = f"{dom}|session-0train|run-{rec}"
            k = 0
            for c in range(4):
                for _ in range(4):                            # 4 trials per (recording, class)
                    rows.append((f"{g}|trial-{k:03d}", dom, g, c)); k += 1
    rng = np.random.default_rng(0)
    sid = [r[0] for r in rows]
    X = rng.standard_normal((len(rows), 22, 8)).astype(np.float32)
    b = EEGBundle(
        X=X, y=np.array([r[3] for r in rows]), sample_id=np.array(sid, dtype=object), dataset_id="BNCI2014_001",
        site_id=np.array(["BNCI2014_001"] * len(rows), dtype=object), subject_id=np.array([r[1] for r in rows], dtype=object),
        session_id=np.array(["0train"] * len(rows), dtype=object), run_id=np.array(["0"] * len(rows), dtype=object),
        recording_id=np.array([r[2] for r in rows], dtype=object), trial_id=np.array(sid, dtype=object),
        support_unit_id=np.array(sid, dtype=object), eval_unit_id=np.array(sid, dtype=object), sfreq=128.0,
        ch_names=list(_CH), class_names=list(_CLASSES), preprocess_hash="resolved" + "0" * 56,
        raw_data_fingerprint="fp" + "0" * 62).validate()
    ev = MOABBLoadEvidence(
        dataset_id="BNCI2014_001", subjects=(1, 2, 3, 4, 5, 6), raw_logical_paths=("BNCI2014_001/a",),
        raw_file_count=1, raw_data_fingerprint=b.raw_data_fingerprint, raw_data_fingerprint_after=b.raw_data_fingerprint,
        header_records=(RawHeaderRecord("subject-001", "0train", "0", 250.0, tuple(_CH), 100, "h0"),),
        header_record_count=1, common_eeg_channels=tuple(_CH), actual_sfreq=128.0, actual_n_times=8,
        actual_shape=tuple(X.shape), output_dtype="float32", class_names=tuple(_CLASSES),
        class_count_table=(("left_hand", 72),), recording_count_table=(("BNCI2014_001|subject-001", 3),),
        library_versions=(("moabb", "1.2.0"),), declared_preprocess_hash="d" * 64, resolved_preprocess_hash=b.preprocess_hash,
        network_attempt_count=0, excluded_recordings=(), evidence_hash="")
    ev = MOABBLoadEvidence(**{**ev.__dict__, "evidence_hash": _evidence_hash(ev, b.tensor_content_hash)})
    return MOABBLoadResult(bundle=b, evidence=ev)


_C = {}


def _fold():
    if "f" not in _C:
        m = load_v2(_SMOKE); m.validate_complete()
        _C["f"] = build_bnci_fold_from_bundle(m, _synthetic_load_result())
    return _C["f"]


# ============================ residuals ============================
def test_manifest_normalization_eps_is_actually_used():
    X = np.ones((2, 3, 4), dtype=np.float32) + np.array([0, 0, 0, 1e-3])  # tiny per-row std
    a = apply_normalization(X, None, PreprocessSpec(normalization="zscore_sample", normalization_eps=1e-8))
    b = apply_normalization(X, None, PreprocessSpec(normalization="zscore_sample", normalization_eps=1.0))
    assert not np.allclose(a, b)                                # the epsilon actually changes the transform


def test_load_evidence_hash_binds_headers_counts_versions_and_network_state():
    ev = _fold().load_result.evidence
    base = _evidence_hash(ev, "T")
    assert len(base) == 64
    import dataclasses
    for over in (dict(network_attempt_count=5), dict(header_record_count=99),
                 dict(actual_shape=(1, 1, 1)), dict(library_versions=(("moabb", "0.0"),))):
        assert _evidence_hash(dataclasses.replace(ev, **over), "T") != base


def test_bnci_adapter_never_uses_temporary_domain_or_group_codes():
    fd = _fold().fold_data
    assert all(str(d).startswith("BNCI2014_001|subject-") for d in fd.domain_id)
    assert all("|run-" in str(g) for g in fd.group_id)         # recording strings, not 0/1/2 codes


# ============================ structure / identity ============================
def test_bnci_trial_mass_is_one():
    fd = _fold().fold_data
    assert float(np.asarray(fd.sample_mass).min()) == 1.0 and float(np.asarray(fd.sample_mass).max()) == 1.0


def test_bnci_group_spans_classes_not_subjects_or_roles():
    fd = _fold().fold_data
    by_g = {}
    for i in range(len(fd.sample_id)):
        by_g.setdefault(fd.group_id[i], set()).add((fd.domain_id[i], int(fd.y[i])))
    assert any(len({y for _, y in v}) == 4 for v in by_g.values())     # a recording has all 4 classes
    for v in by_g.values():
        assert len({d for d, _ in v}) == 1                            # one subject per recording


def test_bnci_split_hash_is_model_seed_independent():
    m = load_v2(_SMOKE); m.validate_complete()
    a = build_bnci_fold_from_bundle(m, _synthetic_load_result())
    b = build_bnci_fold_from_bundle(m, _synthetic_load_result())
    assert a.split_manifest_hash == b.split_manifest_hash and "model_seed" not in a.split_manifest_hash


def test_bnci_fold_scope_is_model_seed_independent():
    a, b = _fold(), build_bnci_fold_from_bundle(load_v2(_SMOKE), _synthetic_load_result())
    assert a.fold_scope.fold_scope_hash == b.fold_scope.fold_scope_hash


def test_bnci_model_spec_uses_actual_22_by_385_shape():
    ms = _fold().model_spec
    assert ms.factory_id == "shallow_convnet" and tuple(ms.input_shape) == (22, 385) and ms.n_classes == 4


def test_bnci_shallow_geometry_is_exact():
    assert _fold().shallow_geometry == {"post_temporal_times": 361, "pooled_times": 20, "feat_dim": 800}


def test_bnci_target_never_enters_fit_ids():
    fd = _fold().fold_data
    assert not fd.preprocess_fit_ids and target_seen_by_fit(fd) is False


def test_bnci_manifest_payload_roundtrips_to_manifest_hash():
    f = _fold()
    assert manifest_payload_hash(f.manifest_payload) == f.manifest_hash


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.runner.bnci_data  # noqa: F401
    import oaci.runner.bnci_preflight  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} bnci-runner-adapter tests")


if __name__ == "__main__":
    _run_all()
