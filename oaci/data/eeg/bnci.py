"""Offline-first BNCI/MOABB Motor-Imagery loader (B1a).

Strict preflight order: resolve raw paths offline -> fingerprint -> scan EVERY recording header ->
validate channel/sfreq structure -> build MotorImagery from the manifest -> epoch -> validate the exact
output -> sample-wise normalize -> stable IDs + canonical sort -> recompute the raw fingerprint
(unchanged) -> resolved preprocessing hash -> EEGBundle. Every MOABB call runs inside
``moabb_offline_root`` + ``forbid_network`` so a missing file cannot download. The bundle's
``preprocess_hash`` is the RESOLVED identity (raw fingerprint + actual shape + library versions), not
the declared spec hash.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
from dataclasses import dataclass

import numpy as np

from .offline import forbid_network, moabb_offline_root, new_network_counter
from .preprocess import PreprocessSpec, apply_normalization
from .registry import OfflineDownloadError, get_entry
from .schema import EEGBundle, tensor_content_hash

LOADER_CODE_VERSION = "oaci-bnci-loader-v1"


@dataclass(frozen=True)
class RawHeaderRecord:
    subject_id: str
    session_id: str
    run_id: str
    sfreq: float
    eeg_ch_names: tuple
    n_raw_samples: int
    header_hash: str


@dataclass(frozen=True)
class MOABBLoadEvidence:
    dataset_id: str
    subjects: tuple
    raw_logical_paths: tuple
    raw_file_count: int
    raw_data_fingerprint: str
    raw_data_fingerprint_after: str
    header_records: tuple
    header_record_count: int
    common_eeg_channels: tuple
    actual_sfreq: float
    actual_n_times: int
    actual_shape: tuple
    output_dtype: str
    class_names: tuple
    class_count_table: tuple
    recording_count_table: tuple
    library_versions: tuple
    declared_preprocess_hash: str
    resolved_preprocess_hash: str
    network_attempt_count: int
    excluded_recordings: tuple
    evidence_hash: str


@dataclass(frozen=True)
class MOABBLoadResult:
    bundle: EEGBundle
    evidence: MOABBLoadEvidence


def _sha(s) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()


def _library_versions() -> tuple:
    import moabb
    import mne
    import scipy
    return (("moabb", moabb.__version__), ("mne", mne.__version__), ("numpy", np.__version__),
            ("scipy", scipy.__version__), ("loader_code_version", LOADER_CODE_VERSION))


def _file_content_sha(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _resolve_paths(ds, dataset_id, subjects, datalake_root) -> tuple:
    """data_path() per subject (no download / no path update), each file checked under the datalake."""
    root = os.path.realpath(str(datalake_root))
    paths, logical = [], []
    for s in subjects:
        try:
            pp = ds.data_path(int(s), path=str(datalake_root), force_update=False, update_path=False)
        except Exception as e:
            raise OfflineDownloadError(f"data_path failed for {dataset_id} subject {s}: {e}")
        for p in (pp if isinstance(pp, (list, tuple)) else [pp]):
            rp = os.path.realpath(str(p))
            if not os.path.exists(rp) or not os.path.isfile(rp):
                raise OfflineDownloadError(f"raw file missing for subject {s}: {p}")
            if os.path.commonpath([rp, root]) != root:
                raise ValueError(f"raw path escapes the datalake root: {rp}")
            paths.append(rp)
            logical.append(f"{dataset_id}/{os.path.relpath(rp, root)}")
    order = sorted(range(len(logical)), key=lambda i: logical[i])      # canonical by logical path
    return tuple(paths[i] for i in order), tuple(logical[i] for i in order)


def _fingerprint(paths, logical) -> str:
    items = [f"{k}|{os.path.getsize(p)}|{_file_content_sha(p)}" for k, p in sorted(zip(logical, paths))]
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def _iter_recordings(raw_tree):
    """MOABB get_data() returns {subject: {session: {run: Raw}}}; yield (subj, sess, run, Raw)."""
    for subj, sessions in raw_tree.items():
        for sess, runs in sessions.items():
            for run, raw in runs.items():
                yield str(subj), str(sess), str(run), raw


def _scan_headers(ds, subjects, frozen_channels) -> tuple:
    records = []
    for s in subjects:
        tree = ds.get_data(subjects=[int(s)])
        for subj, sess, run, raw in _iter_recordings(tree):
            eeg = list(raw.copy().pick("eeg").ch_names)
            if list(eeg) != list(frozen_channels):
                raise ValueError(f"subject {subj} session {sess} run {run}: EEG order {eeg} != frozen {list(frozen_channels)}")
            if len(set(eeg)) != len(eeg) or any(not c for c in eeg):
                raise ValueError("channel names must be unique and non-empty")
            sf = float(raw.info["sfreq"])
            hh = _sha(f"{subj}|{sess}|{run}|{sf}|{raw.n_times}|{'|'.join(eeg)}")
            records.append(RawHeaderRecord(subject_id=f"subject-{int(subj):03d}", session_id=str(sess),
                                           run_id=str(run), sfreq=sf, eeg_ch_names=tuple(eeg),
                                           n_raw_samples=int(raw.n_times), header_hash=hh))
    records.sort(key=lambda r: (r.subject_id, r.session_id, r.run_id))
    return tuple(records)


def _stable_ids(dataset_id, subj, sess, run):
    """Canonical stable ids; trial ordinal resets within a recording, in event (row) order."""
    sample_id = np.empty(len(subj), dtype=object)
    recording = np.empty(len(subj), dtype=object)
    domain = np.empty(len(subj), dtype=object)
    counters = {}
    for i in range(len(subj)):
        d = f"{dataset_id}|subject-{int(subj[i]):03d}"
        rec = f"{d}|session-{sess[i]}|run-{run[i]}"
        k = counters.get(rec, 0); counters[rec] = k + 1
        domain[i] = d; recording[i] = rec
        sample_id[i] = f"{rec}|trial-{k:03d}"
    return sample_id, recording, domain


def motor_imagery_kwargs(preprocessing, frozen_class_names, frozen_channels) -> dict:
    """The exact MotorImagery kwargs the manifest maps to (testable without constructing the paradigm)."""
    return {"n_classes": len(frozen_class_names), "events": list(frozen_class_names),
            "fmin": float(preprocessing.fmin), "fmax": float(preprocessing.fmax),
            "tmin": float(preprocessing.epoch_tmin), "tmax": float(preprocessing.epoch_tmax),
            "baseline": preprocessing.baseline, "channels": list(frozen_channels),
            "resample": float(preprocessing.resample_sfreq)}


def load_moabb_confirmatory(dataset_id, subjects, preprocessing, *, frozen_class_names, frozen_channels,
                            expected_sfreq, expected_n_times, datalake_root) -> MOABBLoadResult:
    entry = get_entry(dataset_id)
    if entry.loader != "moabb":
        raise ValueError(f"{dataset_id} is not a MOABB dataset")
    subjects = tuple(sorted(int(s) for s in subjects))            # canonical
    frozen_channels = tuple(str(c) for c in frozen_channels)
    counter = new_network_counter()

    with moabb_offline_root(datalake_root), forbid_network(counter):
        import moabb.datasets as mds
        from moabb.paradigms import MotorImagery
        ds = getattr(mds, entry.moabb_id)()

        paths, logical = _resolve_paths(ds, dataset_id, subjects, datalake_root)
        fp_before = _fingerprint(paths, logical)
        headers = _scan_headers(ds, subjects, frozen_channels)

        paradigm = MotorImagery(**motor_imagery_kwargs(preprocessing, frozen_class_names, frozen_channels))
        epochs, labels, meta = paradigm.get_data(dataset=ds, subjects=list(subjects), return_epochs=True)
        X = np.asarray(epochs.get_data(), dtype=np.float32)
        actual_sfreq = float(epochs.info["sfreq"])
        actual_ch = tuple(epochs.ch_names)
        n_times = int(X.shape[2])
        paths_after, logical_after = _resolve_paths(ds, dataset_id, subjects, datalake_root)
        fp_after = _fingerprint(paths_after, logical_after)

    if list(actual_ch) != list(frozen_channels):
        raise ValueError(f"epoch channel order {actual_ch} != frozen {frozen_channels}")
    if actual_sfreq != float(expected_sfreq) or n_times != int(expected_n_times):
        raise ValueError(f"actual (sfreq={actual_sfreq}, n_times={n_times}) != expected "
                         f"({expected_sfreq}, {expected_n_times})")
    if fp_after != fp_before:
        raise RuntimeError("raw fingerprint changed during load (loader mutated raw files)")

    subj = meta["subject"].astype(str).to_numpy()
    sess = meta["session"].astype(str).to_numpy()
    run = meta["run"].astype(str).to_numpy()
    y_str = np.asarray(labels, dtype=object)
    cls_index = {c: i for i, c in enumerate(frozen_class_names)}
    bad = sorted(set(y_str.tolist()) - set(cls_index))
    if bad:
        raise ValueError(f"loaded labels {bad} not in frozen class map {list(frozen_class_names)}")
    y = np.array([cls_index[c] for c in y_str], dtype=int)

    if preprocessing.normalization == "zscore_sample":            # apply with the MANIFEST epsilon
        X = apply_normalization(X, None, PreprocessSpec(normalization="zscore_sample",
                                                        normalization_eps=float(preprocessing.normalization_eps)))
    if not np.isfinite(X).all():
        raise ValueError("non-finite values after normalization")
    Xd = X.astype(np.float64)                                     # float32 mean/std are noisy
    m, sd = Xd.mean(axis=2), Xd.std(axis=2)                       # per (trial, channel)
    # std deviates from 1 by the fixed epsilon in apply_normalization (EEG is volts-scale); tolerate it
    if np.abs(m).max() > 1e-2 or np.abs(sd[sd > 1e-3] - 1.0).max() > 5e-2:
        raise ValueError("zscore_sample not correctly applied (per-trial/channel mean~0, std~1)")

    sample_id, recording, domain = _stable_ids(dataset_id, subj, sess, run)
    order = np.argsort(sample_id.astype(str))                     # canonical row order by sample_id
    X, y, sample_id, recording, domain = X[order], y[order], sample_id[order], recording[order], domain[order]
    subj_o, sess_o, run_o = subj[order], sess[order], run[order]

    full_tensor = tensor_content_hash(X)
    lib = _library_versions()
    declared = preprocessing_declared_hash(preprocessing)
    resolved = _resolved_hash(dataset_id, preprocessing, fp_before, frozen_class_names, actual_ch,
                              actual_sfreq, n_times, str(X.dtype), lib)

    bundle = EEGBundle(
        X=X, y=y, sample_id=sample_id, dataset_id=dataset_id,
        site_id=np.array([dataset_id] * len(y), dtype=object), subject_id=domain, session_id=sess_o,
        run_id=run_o, recording_id=recording, trial_id=sample_id, support_unit_id=sample_id,
        eval_unit_id=sample_id, sfreq=actual_sfreq, ch_names=list(actual_ch),
        class_names=list(frozen_class_names), preprocess_hash=resolved, raw_data_fingerprint=fp_before,
    ).validate()

    cct = tuple((c, int((y == cls_index[c]).sum())) for c in frozen_class_names)
    rct = tuple((str(d), len({r for r in recording.tolist() if r.startswith(d + "|")}))
                for d in sorted(set(domain.tolist())))
    ev = MOABBLoadEvidence(
        dataset_id=dataset_id, subjects=subjects, raw_logical_paths=tuple(logical), raw_file_count=len(logical),
        raw_data_fingerprint=fp_before, raw_data_fingerprint_after=fp_after, header_records=headers,
        header_record_count=len(headers), common_eeg_channels=tuple(actual_ch), actual_sfreq=actual_sfreq,
        actual_n_times=n_times, actual_shape=tuple(int(x) for x in X.shape), output_dtype=str(X.dtype),
        class_names=tuple(frozen_class_names), class_count_table=cct, recording_count_table=rct,
        library_versions=lib, declared_preprocess_hash=declared, resolved_preprocess_hash=resolved,
        network_attempt_count=counter.attempts, excluded_recordings=(), evidence_hash="")
    return MOABBLoadResult(bundle=bundle, evidence=dataclasses.replace(ev, evidence_hash=_evidence_hash(ev, full_tensor)))


def _evidence_hash(ev: MOABBLoadEvidence, full_tensor_hash) -> str:
    from ..eeg.audit import canonical_hash
    return canonical_hash({
        "dataset_id": ev.dataset_id, "subjects": list(ev.subjects), "raw_logical_paths": list(ev.raw_logical_paths),
        "raw_file_count": ev.raw_file_count, "raw_data_fingerprint": ev.raw_data_fingerprint,
        "raw_data_fingerprint_after": ev.raw_data_fingerprint_after,
        "header_hashes": [r.header_hash for r in ev.header_records], "header_record_count": ev.header_record_count,
        "common_eeg_channels": list(ev.common_eeg_channels), "actual_sfreq": ev.actual_sfreq,
        "actual_n_times": ev.actual_n_times, "actual_shape": list(ev.actual_shape), "output_dtype": ev.output_dtype,
        "class_count_table": [list(x) for x in ev.class_count_table],
        "recording_count_table": [list(x) for x in ev.recording_count_table],
        "library_versions": [list(x) for x in ev.library_versions],
        "declared_preprocess_hash": ev.declared_preprocess_hash, "resolved_preprocess_hash": ev.resolved_preprocess_hash,
        "network_attempt_count": ev.network_attempt_count, "excluded_recordings": list(ev.excluded_recordings),
        "full_tensor_hash": full_tensor_hash})


def preprocessing_declared_hash(pp) -> str:
    from ..eeg.audit import canonical_hash
    return canonical_hash({"kind": pp.kind, "fmin": pp.fmin, "fmax": pp.fmax,
                           "resample_sfreq": pp.resample_sfreq, "epoch_tmin": pp.epoch_tmin,
                           "epoch_tmax": pp.epoch_tmax, "baseline": pp.baseline,
                           "normalization": pp.normalization, "normalization_eps": pp.normalization_eps,
                           "channel_interpolation": pp.channel_interpolation, "code_version": pp.code_version})


def _resolved_hash(dataset_id, pp, raw_fp, class_names, channels, sfreq, n_times, dtype, lib) -> str:
    payload = {
        "dataset_id": dataset_id, "preprocessing": preprocessing_declared_hash(pp),
        "motor_imagery_kwargs": {"events": list(class_names), "fmin": pp.fmin, "fmax": pp.fmax,
                                 "tmin": pp.epoch_tmin, "tmax": pp.epoch_tmax, "baseline": pp.baseline,
                                 "channels": list(channels), "resample": pp.resample_sfreq},
        "raw_data_fingerprint": raw_fp, "class_order": list(class_names), "channel_order": list(channels),
        "actual_sfreq": float(sfreq), "actual_n_times": int(n_times), "output_dtype": dtype,
        "normalization": pp.normalization, "normalization_eps": pp.normalization_eps,
        "library_versions": [list(x) for x in lib]}
    from ..eeg.audit import canonical_hash
    return canonical_hash(payload)
