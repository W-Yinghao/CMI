"""Offline MOABB loader (BNCI / Cho / Lee MI), with confirmatory-grade provenance.

Fixes vs the first draft:
* ``trial_id`` = ``dataset|subject|session|run|within-recording ordinal`` (stable across which
  subjects you load and in what order — NOT a global enumerate of the returned array);
* class map = the FROZEN full order from the registry (not the sort of the loaded subset);
* ``raw_data_fingerprint`` = SHA-256 of the actual raw FILE listing (path|size), separate from
  ``preprocess_hash``;
* channel header that is unreadable or inconsistent with X => confirmatory FAIL (no ``ch0`` fallback);
* a declared ``zscore_sample`` normalization is ACTUALLY applied.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from .preprocess import PreprocessSpec, apply_normalization
from .registry import OfflineDownloadError, get_entry, set_offline_env
from .schema import EEGBundle


# ---- pure, unit-testable helpers (no MOABB needed) -------------------------------------
def trial_ids(dataset_id, subj, sess, run) -> np.ndarray:
    """``dataset|s{subject}|{session}|{run}|t{within-recording ordinal}`` — ordinals reset per
    recording, so a subject's IDs do not depend on the loaded subset / order."""
    subj, sess, run = map(lambda a: np.asarray(a, dtype=object), (subj, sess, run))
    rec = np.array([f"{dataset_id}|s{a}|{b}|{c}" for a, b, c in zip(subj, sess, run)], dtype=object)
    seen: dict = {}
    out = []
    for r in rec:
        k = seen.get(r, 0)
        out.append(f"{r}|t{k}")
        seen[r] = k + 1
    return rec, np.array(out, dtype=object)


def map_classes(y_str, frozen_classes) -> np.ndarray:
    """Map string labels to indices via the FROZEN registry class order. Unknown label -> error
    (never silently renumber from the loaded subset)."""
    frozen = list(frozen_classes)
    if not frozen:
        raise ValueError("registry class map is empty")
    idx = {c: i for i, c in enumerate(frozen)}
    bad = sorted(set(str(v) for v in y_str) - set(frozen))
    if bad:
        raise ValueError(f"loaded labels {bad} not in frozen registry class map {frozen}")
    return np.array([idx[str(v)] for v in y_str], dtype=int)


def resolve_channels(raw_ch_names, n_channels: int, confirmatory: bool = True) -> list:
    """Channel names must be readable and consistent with X. In confirmatory mode a mismatch is a
    hard FAIL (no generic ``ch0..`` fallback)."""
    if raw_ch_names is None or len(raw_ch_names) == 0:
        if confirmatory:
            raise ValueError("channel header unreadable; refusing a generic channel fallback in confirmatory mode")
        return [f"ch{i}" for i in range(n_channels)]
    if len(raw_ch_names) != n_channels:
        if confirmatory:
            raise ValueError(f"channel header ({len(raw_ch_names)}) inconsistent with X ({n_channels} channels)")
        return (list(raw_ch_names) + [f"ch{i}" for i in range(n_channels)])[:n_channels]
    return list(raw_ch_names)


def _file_content_sha(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def raw_file_fingerprint(paths, confirmatory: bool = True) -> str:
    """SHA-256 of a canonical manifest of ``basename|size|content_sha256`` over the actual raw
    files — hashes the raw BYTES (independent of the datalake mount root and of the preprocessing
    spec). In confirmatory mode an empty path list or an unreadable file is a hard FAIL (no silent
    swallow)."""
    paths = [str(p) for p in paths]
    if confirmatory and not paths:
        raise ValueError("no raw files to fingerprint; refusing an empty confirmatory fingerprint")
    items = []
    for p in sorted(paths):
        try:
            items.append(f"{os.path.basename(p)}|{os.path.getsize(p)}|{_file_content_sha(p)}")
        except OSError as e:
            if confirmatory:
                raise ValueError(f"raw file unreadable in confirmatory mode: {p} ({e})")
            items.append(f"{os.path.basename(p)}|?|?")
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def _data_paths(ds, subjects) -> list:
    paths = []
    for s in subjects:
        try:
            pp = ds.data_path(int(s))
            paths.extend(pp if isinstance(pp, (list, tuple)) else [pp])
        except Exception:
            pass
    return paths


# ---- the loader ------------------------------------------------------------------------
def load_moabb(dataset_id: str, subjects, spec: PreprocessSpec | None = None,
               confirmatory: bool = True) -> EEGBundle:
    spec = spec or PreprocessSpec()
    entry = get_entry(dataset_id)
    if entry.loader != "moabb":
        raise ValueError(f"{dataset_id} is not a MOABB dataset")
    if not entry.classes:
        raise ValueError(f"registry has no frozen class map for {dataset_id}")
    set_offline_env()
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import moabb.datasets as mds
        from moabb.paradigms import MotorImagery
    except Exception as e:  # pragma: no cover
        raise OfflineDownloadError(f"moabb unavailable: {e}")

    ds = getattr(mds, entry.moabb_id)()
    paradigm = MotorImagery(fmin=spec.l_freq, fmax=spec.h_freq, resample=spec.resample_sfreq)
    try:
        X, y, meta = paradigm.get_data(dataset=ds, subjects=list(subjects))
    except Exception as e:
        raise OfflineDownloadError(f"offline load failed for {dataset_id} subjects={subjects}: {e}")

    X = np.asarray(X, dtype=np.float32)
    subj = meta["subject"].astype(str).to_numpy()
    sess = meta["session"].astype(str).to_numpy() if "session" in meta else np.array(["0"] * len(y))
    run = meta["run"].astype(str).to_numpy() if "run" in meta else np.array(["0"] * len(y))
    rec, trial = trial_ids(dataset_id, subj, sess, run)
    y_idx = map_classes(np.asarray(y, dtype=object), entry.classes)   # frozen registry order
    site = np.array([dataset_id] * len(y), dtype=object)

    try:
        raws = ds.get_data(subjects=[int(list(subjects)[0])])
        raw_ch = [c for c in _first_raw(raws).copy().pick("eeg").ch_names]
    except Exception:
        raw_ch = None
    ch_names = resolve_channels(raw_ch, X.shape[1], confirmatory=confirmatory)

    if spec.normalization == "zscore_sample":            # actually APPLY the declared transform
        X = apply_normalization(X, None, spec)

    fp = raw_file_fingerprint(_data_paths(ds, subjects), confirmatory=confirmatory)
    return EEGBundle(
        X=X, y=y_idx, sample_id=trial, dataset_id=dataset_id, site_id=site, subject_id=subj,
        session_id=sess, run_id=run, recording_id=rec, trial_id=trial, support_unit_id=trial,
        eval_unit_id=trial, sfreq=float(spec.resample_sfreq), ch_names=list(ch_names),
        class_names=list(entry.classes), preprocess_hash=spec.hash(), raw_data_fingerprint=fp,
    ).validate()


def _first_raw(obj):
    cur = obj
    while isinstance(cur, dict):
        cur = next(iter(cur.values()))
    return cur
