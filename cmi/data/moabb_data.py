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
}


def load(name, subjects=None, tmin=0.5, tmax=3.5, resample=128,
         binary=None, fmin=None, fmax=None, normalize="trial_zscore"):
    """Return (X[float32, n_trials,n_ch,n_times], y[int64], meta[DataFrame], classes)."""
    import moabb.datasets as D
    from moabb.paradigms import MotorImagery, LeftRightImagery

    d = DATASET_DEFAULTS.get(name, dict(binary=False, fmin=4, fmax=38))
    binary = d["binary"] if binary is None else binary
    fmin = d["fmin"] if fmin is None else fmin
    fmax = d["fmax"] if fmax is None else fmax

    ds = getattr(D, name)()
    if subjects is None:
        subjects = ds.subject_list
    # Paradigm selection. Default reproduces the original behaviour EXACTLY: binary -> LeftRightImagery,
    # else MotorImagery (all classes). A dataset may set paradigm="motor" (+ optional events/n_classes)
    # to force MotorImagery even when binary — required for BNCI2015_001 (right_hand vs feet), which
    # LeftRightImagery rejects. Only datasets that explicitly opt in are affected.
    paradigm_kind = d.get("paradigm", "left_right" if binary else "motor")
    if paradigm_kind == "left_right":
        para = LeftRightImagery(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=resample)
    else:
        mkw = dict(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=resample)
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
    return X, y, meta.reset_index(drop=True), list(le.classes_)


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
