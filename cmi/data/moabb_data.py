"""Offline MOABB motor-imagery loading + leave-one-subject-out / cross-session splits.

Importing `cmi.paths` first points MNE/MOABB at the read-only datalake cache so
`get_data` runs with no network. Labels are returned as int64; metadata keeps the
MOABB columns (subject, session, run) used to define the domain D.
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import LabelEncoder

from cmi.paths import configure_offline_moabb
configure_offline_moabb()
import mne  # noqa: E402
mne.set_log_level("ERROR")

# name -> default (binary?, band)  — band per common practice for each task
DATASET_DEFAULTS = {
    "BNCI2014_001": dict(binary=False, fmin=4, fmax=38),   # 2a, 4-class
    "BNCI2014_004": dict(binary=True,  fmin=8, fmax=30),   # 2b, L/R
    # BNCI2015_001 is 2-class MI but right_hand vs FEET (not left/right hand), so LeftRightImagery is
    # invalid for it; use the MotorImagery paradigm restricted to its two classes.
    "BNCI2015_001": dict(binary=True,  fmin=8, fmax=30, paradigm="motor",
                         events=["right_hand", "feet"], n_classes=2),
    "Lee2019_MI":   dict(binary=True,  fmin=8, fmax=30),   # OpenBMI L/R
    "Cho2017":      dict(binary=True,  fmin=8, fmax=30),
    "Schirrmeister2017": dict(binary=False, fmin=4, fmax=38),  # HGD, 4-class
    "Stieger2021":  dict(binary=False, fmin=4, fmax=38,  # longitudinal 4-class MI lockbox (MotorImagery, all classes)
                         channels=['AF3', 'AF4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP7', 'TP8']),  # 60 common channels (uniform across all 62 subjects)
    "Shin2017A":    dict(binary=True,  fmin=8, fmax=30),  # 3-session L/R MI lockbox (needs accept=True)
}


def construct_dataset(name):
    """Build a MOABB dataset by name, handling license-gated datasets (accept=True) uniformly."""
    import moabb.datasets as D
    cls = getattr(D, name)
    try:
        return cls(accept=True)          # license-gated (e.g. Shin2017A/B)
    except TypeError:
        return cls()                     # no accept parameter


import hashlib, os
import pandas as pd
_CACHE_DIR = os.environ.get("CMI_EPOCH_CACHE", "/home/infres/yinwang/cmi_epoch_cache")


def _cache_key(name, tmin, tmax, resample, normalize, channels):
    raw = f"{name}|{tmin}|{tmax}|{resample}|{normalize}|{channels}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load(name, subjects=None, tmin=0.5, tmax=3.5, resample=128,
         binary=None, fmin=None, fmax=None, normalize="trial_zscore", cache=True):
    """Return (X[float32, n_trials,n_ch,n_times], y[int64], meta[DataFrame], classes). Epoching is fold- and seed-
    INDEPENDENT, so the ALL-subjects load is disk-cached (CMI_EPOCH_CACHE) and re-used across LOSO folds/seeds — the
    expensive MOABB epoching runs ONCE per (dataset,tmin,tmax,resample,normalize,channels), not once per fold."""
    import moabb.datasets as D
    from moabb.paradigms import MotorImagery, LeftRightImagery

    d = DATASET_DEFAULTS.get(name, dict(binary=False, fmin=4, fmax=38))
    binary = d["binary"] if binary is None else binary
    fmin = d["fmin"] if fmin is None else fmin
    fmax = d["fmax"] if fmax is None else fmax
    channels = d.get("channels")                      # optional fixed channel set (variable-channel datasets)

    # disk cache: full-dataset epoched X/y/meta (only when loading ALL subjects, which is the LOSO case)
    ckey = _cache_key(name, tmin, tmax, resample, normalize, channels)
    cpath = os.path.join(_CACHE_DIR, f"{name}_{ckey}.npz")
    if cache and subjects is None and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        meta = pd.DataFrame(dict(subject=z["m_subject"], session=z["m_session"], run=z["m_run"]))
        return z["X"], z["y"], meta, list(z["classes"])

    ds = construct_dataset(name)
    if subjects is None:
        subjects = ds.subject_list
    # Paradigm selection. Default reproduces the original behaviour EXACTLY: binary -> LeftRightImagery,
    # else MotorImagery (all classes). A dataset may set paradigm="motor" (+ optional events/n_classes)
    # to force MotorImagery even when binary — required for BNCI2015_001 (right_hand vs feet), which
    # LeftRightImagery rejects. Only datasets that explicitly opt in are affected.
    paradigm_kind = d.get("paradigm", "left_right" if binary else "motor")
    base_kw = dict(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=resample)
    if channels:
        base_kw["channels"] = list(channels)         # fixed common channel set (variable-channel datasets)
    if paradigm_kind == "left_right":
        para = LeftRightImagery(**base_kw)
    else:
        mkw = dict(base_kw)
        if d.get("events"):
            mkw["events"] = list(d["events"])
        if d.get("n_classes"):
            mkw["n_classes"] = int(d["n_classes"])
        para = MotorImagery(**mkw)
    X, y, meta = para.get_data(ds, subjects=subjects)
    X = X.astype("float32")
    if normalize == "trial_zscore":          # per-trial, per-channel z-score over time (no leakage)
        m = X.mean(axis=2, keepdims=True)
        s = X.std(axis=2, keepdims=True) + 1e-7
        X = (X - m) / s
    le = LabelEncoder()
    y = le.fit_transform(y).astype("int64")
    meta = meta.reset_index(drop=True); classes = list(le.classes_)
    if cache and subjects is None:           # bank the full-dataset epoching for reuse across folds/seeds
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = cpath + f".tmp{os.getpid()}"
        np.savez_compressed(tmp, X=X, y=y, classes=np.array(classes),
                            m_subject=meta["subject"].to_numpy(), m_session=meta["session"].astype(str).to_numpy(),
                            m_run=meta["run"].astype(str).to_numpy() if "run" in meta else np.array([""] * len(y)))
        os.replace(tmp, cpath)               # atomic
    return X, y, meta, classes


def paradigm_info(name):
    """Which MOABB paradigm / event restriction the loader uses for `name` (metadata-only; no load)."""
    d = DATASET_DEFAULTS.get(name, dict(binary=False))
    binary = d.get("binary", False)
    kind = d.get("paradigm", "left_right" if binary else "motor")
    return dict(moabb_paradigm=("LeftRightImagery" if kind == "left_right" else "MotorImagery"),
                events=(list(d["events"]) if d.get("events") else None),
                n_classes_hint=d.get("n_classes"))


def domain_labels(meta, mode="subject"):
    """Integer domain id per trial. mode: 'subject' (LOSO) | 'subject_session' (cross-session)."""
    if mode == "subject":
        keys = meta["subject"].astype(str)
    elif mode == "subject_session":
        keys = meta["subject"].astype(str) + "|" + meta["session"].astype(str)
    else:
        raise ValueError(mode)
    uniq = {k: i for i, k in enumerate(sorted(keys.unique()))}
    return keys.map(uniq).to_numpy(), uniq


def loso_splits(meta):
    """Yield (target_subject, train_mask, test_mask) leaving one subject out."""
    subs = sorted(meta["subject"].unique())
    for tgt in subs:
        test = (meta["subject"] == tgt).to_numpy()
        yield tgt, ~test, test


def leave_one_session_splits(meta):
    """Yield (target_session, train_mask, test_mask) leaving one session out (cross-session DG)."""
    sess = sorted(meta["session"].unique())
    for tgt in sess:
        test = (meta["session"] == tgt).to_numpy()
        yield tgt, ~test, test
