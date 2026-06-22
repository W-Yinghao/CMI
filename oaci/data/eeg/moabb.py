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


def paradigm_kwargs(spec, classes) -> dict:
    """MotorImagery kwargs from the FROZEN spec — frequency band, resample, full class map
    (events + n_classes), trial epoch window (tmin/tmax) and the frozen channel order."""
    kw = dict(fmin=float(spec.l_freq), fmax=float(spec.h_freq), resample=float(spec.resample_sfreq),
              n_classes=len(classes), events=list(classes))
    if spec.epoch_tmin is not None:
        kw["tmin"] = float(spec.epoch_tmin)
    if spec.epoch_tmax is not None:
        kw["tmax"] = float(spec.epoch_tmax)
    if spec.channels:
        kw["channels"] = list(spec.channels)
    return kw


def validate_epoch_n_times(n_times: int, sfreq: float, tmin: float, tmax: float, tol: int = 1) -> int:
    """Verify the produced epoch length matches the frozen window (±tol samples for MOABB's
    endpoint convention)."""
    expected = int(round((tmax - tmin) * sfreq))
    if abs(int(n_times) - expected) > tol:
        raise ValueError(f"epoch n_times {n_times} != expected {expected} (±{tol}) for "
                         f"[{tmin},{tmax}]s @ {sfreq}Hz")
    return expected


def validate_channel_order(ch_names, frozen, confirmatory: bool = True) -> bool:
    if list(ch_names) != list(frozen):
        if confirmatory:
            raise ValueError(f"channel order {list(ch_names)} != frozen {list(frozen)}")
        return False
    return True


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


def raw_file_fingerprint(paths, logical_paths=None, confirmatory: bool = True) -> str:
    """SHA-256 of a canonical manifest of ``logical_path|size|content_sha256`` over the actual raw
    files. ``logical_paths`` are caller-provided DATASET-RELATIVE keys (e.g. ``sub-01/eeg.fif``):
    stable across the datalake mount root, and they keep two files that share a basename DISTINCT.
    With no logical paths the basename is used (back-compat). In confirmatory mode an empty list or
    an unreadable file is a hard FAIL."""
    paths = [str(p) for p in paths]
    if confirmatory and not paths:
        raise ValueError("no raw files to fingerprint; refusing an empty confirmatory fingerprint")
    keys = ([str(k) for k in logical_paths] if logical_paths is not None
            else [os.path.basename(p) for p in paths])
    if len(keys) != len(paths):
        raise ValueError("logical_paths length must match paths")
    items = []
    for key, p in sorted(zip(keys, paths)):
        try:
            items.append(f"{key}|{os.path.getsize(p)}|{_file_content_sha(p)}")
        except OSError as e:
            if confirmatory:
                raise ValueError(f"raw file unreadable in confirmatory mode: {p} ({e})")
            items.append(f"{key}|?|?")
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
    paradigm = MotorImagery(**paradigm_kwargs(spec, entry.classes))   # tmin/tmax/events/channels
    try:
        X, y, meta = paradigm.get_data(dataset=ds, subjects=list(subjects))
    except Exception as e:
        raise OfflineDownloadError(f"offline load failed for {dataset_id} subjects={subjects}: {e}")
    if spec.epoch_tmin is not None and spec.epoch_tmax is not None:
        validate_epoch_n_times(np.asarray(X).shape[2], spec.resample_sfreq, spec.epoch_tmin, spec.epoch_tmax)

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
    if spec.channels:                                    # exact frozen channel order required
        validate_channel_order(ch_names, spec.channels, confirmatory=confirmatory)

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
