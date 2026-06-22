"""Offline MOABB loader (BNCI / Cho / Lee MI). Reads the cached datalake; never downloads.

Each motor-imagery TRIAL is the sample = support unit = eval unit; the recording (subject-session-
run) is the clustered-inference group. Stable string IDs throughout.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .preprocess import PreprocessSpec
from .registry import OfflineDownloadError, get_entry, set_offline_env
from .schema import EEGBundle


def load_moabb(dataset_id: str, subjects, spec: PreprocessSpec | None = None) -> EEGBundle:
    spec = spec or PreprocessSpec()
    entry = get_entry(dataset_id)
    if entry.loader != "moabb":
        raise ValueError(f"{dataset_id} is not a MOABB dataset")
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

    X = np.asarray(X, dtype=np.float32)                       # [N, C, T]
    y_str = np.asarray(y, dtype=object)
    subj = meta["subject"].astype(str).to_numpy()
    sess = meta["session"].astype(str).to_numpy() if "session" in meta else np.array(["0"] * len(y))
    run = meta["run"].astype(str).to_numpy() if "run" in meta else np.array(["0"] * len(y))

    classes = sorted(set(y_str.tolist()))
    y_idx = np.array([classes.index(v) for v in y_str], dtype=int)
    rec = np.array([f"{dataset_id}|s{a}|{b}|{c}" for a, b, c in zip(subj, sess, run)], dtype=object)
    trial = np.array([f"{r}|t{i}" for i, r in enumerate(rec)], dtype=object)
    site = np.array([dataset_id] * len(y), dtype=object)      # one site for within-dataset MI

    # ch_names from one cached raw (offline); fall back to generic if unavailable.
    try:
        raws = ds.get_data(subjects=[int(list(subjects)[0])])
        raw = _first_raw(raws)
        ch_names = [c for c in raw.copy().pick("eeg").ch_names][: X.shape[1]]
    except Exception:
        ch_names = [f"ch{i}" for i in range(X.shape[1])]
    if len(ch_names) != X.shape[1]:
        ch_names = (ch_names + [f"ch{i}" for i in range(X.shape[1])])[: X.shape[1]]

    fp = hashlib.sha256(f"{entry.moabb_id}|{sorted(map(int, subjects))}|{spec.hash()}".encode()).hexdigest()[:16]
    return EEGBundle(
        X=X, y=y_idx, sample_id=trial, dataset_id=dataset_id, site_id=site, subject_id=subj,
        session_id=sess, run_id=run, recording_id=rec, trial_id=trial, support_unit_id=trial,
        eval_unit_id=trial, sfreq=float(spec.resample_sfreq), ch_names=list(ch_names),
        class_names=classes, preprocess_hash=spec.hash(), raw_data_fingerprint=fp,
    ).validate()


def _first_raw(obj):
    """Descend MOABB's subject->session->run nested dict to the first Raw."""
    cur = obj
    while isinstance(cur, dict):
        cur = next(iter(cur.values()))
    return cur
