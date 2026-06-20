"""Disease-detection (SCPS) EEG loader — ADFTD (Alzheimer/FTD/Control).

SCPS = Single-Class-Per-Subject: each subject has ONE fixed label, so subject identity
is (near-)perfectly correlated with the label. This is the regime where conditional
I(Z;D|Y) vs marginal I(Z;D) matters most: removing all subject info ≈ removing the label,
which marginal alignment / DANN cannot avoid but our label-conditional criterion can.

ADFTD (OpenNeuro ds004504): 88 subjects (A=36 Alzheimer, F=23 FTD, C=29 Control), 19ch
@500Hz, ~10 min eyes-closed resting. Long recordings cut into fixed windows (<=max_per_subject).
Domain D = subject (LOSO is the only valid SCPS protocol).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

ADFTD_DIR = Path("/projects/EEG-foundation-model/datalake/raw/ADFTD/ds004504-1.0.8")
MUMTAZ_DIR = Path("/projects/EEG-foundation-model/datalake/raw/mumtaz/edf")
GROUP3 = {"A": 0, "C": 1, "F": 2}          # 3-class
GROUP_BIN = {"A": 1, "C": 0}               # Alzheimer vs Control (drop FTD)


def _windows(x, win, max_n):
    T = x.shape[1]; n = T // win
    if n == 0:
        return []
    starts = np.arange(n) * win
    if max_n and n > max_n:
        starts = starts[np.linspace(0, n - 1, max_n).round().astype(int)]
    return [x[:, s:s + win] for s in starts]


MUMTAZ_CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
             "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]   # canonical 19 (10-20)


def _load_mumtaz(subjects, win_sec, max_per_subject, resample, fmin, fmax, normalize):
    """mumtaz MDD vs Healthy, eyes-closed resting (.edf). SCPS, 2-class. Restrict to a fixed
    19-channel 10-20 montage (names like 'EEG Fp1-LE') so all subjects align."""
    import mne
    mne.set_log_level("ERROR")
    win = int(win_sec * resample)
    Xs, ys, subs, seen = [], [], [], set()
    for f in sorted(MUMTAZ_DIR.glob("*EC.edf")):
        m = re.match(r"^(?:\d+_)?(H|MDD) S(\d+) EC", f.name)
        if not m:
            continue
        grp, snum = m.group(1), int(m.group(2))
        sid = (0 if grp == "H" else 1000) + snum          # unique int id across groups
        if sid in seen or (subjects and sid not in subjects):
            continue
        raw = mne.io.read_raw_edf(str(f), preload=True)
        avail = {c.replace("EEG ", "").replace("-LE", "").strip().upper(): c for c in raw.ch_names}
        sel = [avail[ch.upper()] for ch in MUMTAZ_CH if ch.upper() in avail]
        if len(sel) != len(MUMTAZ_CH):                    # skip files missing canonical channels
            continue
        seen.add(sid)
        raw.pick(sel)                                     # reorders to canonical order
        raw.filter(fmin, fmax, verbose="ERROR").resample(resample, verbose="ERROR")
        data = raw.get_data().astype("float32")
        for w in _windows(data, win, max_per_subject):
            Xs.append(w); ys.append(0 if grp == "H" else 1); subs.append(sid)
    X = np.stack(Xs).astype("float32")
    if normalize == "trial_zscore":
        mu = X.mean(axis=2, keepdims=True); sd = X.std(axis=2, keepdims=True) + 1e-7
        X = (X - mu) / sd
    meta = pd.DataFrame({"subject": subs, "session": 1, "run": 0})
    return X, np.array(ys, dtype="int64"), meta, ["Healthy", "MDD"]


TUAB_PROC = Path("/projects/EEG-foundation-model/datalake/processed/5e77943a/TUAB")
TUAB_CH = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
           "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]        # 19-ch 10-20


def _load_tuab(subjects, win_sec, max_per_subject, resample, fmin, fmax, normalize, n_per_class=40):
    """TUH EEG Abnormal (TUAB), from the lab-PREPROCESSED datalake (processed/5e77943a/TUAB), NOT raw
    .edf: per-recording [T,30] npy @200Hz, already band-passed 0.3-75 + 60Hz notch. Normal(0) vs
    Abnormal(1), SCPS, 2-class. Subject-level label (drop the few mixed-label subjects); one recording
    per subject; subsample n_per_class subjects/class for tractable LOSO; crop 60-240s then resample."""
    import ast
    from scipy.signal import resample_poly
    meta = pd.read_parquet(TUAB_PROC / "metadata.parquet")
    ev = pd.read_parquet(TUAB_PROC / "events.parquet").set_index("recording_id")["event_code"]
    meta["label"] = meta["recording_id"].map(ev)
    sl = meta.groupby("subject")["label"].nunique()
    good = set(sl[sl == 1].index)                          # subject == one label (SCPS)
    win = int(win_sec * resample)
    Xs, ys, subs, seen = [], [], [], set()
    picked = {0: 0, 1: 0}
    for _, row in meta.iterrows():
        subj, lab = int(row["subject"]), int(row["label"])
        if subj not in good or subj in seen or (subjects and subj not in subjects) or picked[lab] >= n_per_class:
            continue
        ch = ast.literal_eval(row["channels"]) if isinstance(row["channels"], str) else list(row["channels"])
        avail = {c.replace("EEG ", "").replace("-REF", "").strip().upper(): i for i, c in enumerate(ch)}
        sel = [avail[c] for c in TUAB_CH if c in avail]
        if len(sel) != len(TUAB_CH):
            continue
        data = np.load(TUAB_PROC / row["filepath"]).T.astype("float32")[sel]   # [19, T] @200Hz
        data = data[:, 12000:48000]                                           # 60-240s @200Hz
        if data.shape[1] < 2000:
            continue
        if resample != 200:
            data = resample_poly(data, resample, 200, axis=1).astype("float32")
        ws = _windows(data, win, max_per_subject)
        if not ws:
            continue
        seen.add(subj); picked[lab] += 1
        for w in ws:
            Xs.append(w); ys.append(lab); subs.append(subj)
    X = np.stack(Xs).astype("float32")
    if normalize == "trial_zscore":
        mu = X.mean(axis=2, keepdims=True); sd = X.std(axis=2, keepdims=True) + 1e-7
        X = (X - mu) / sd
    meta_df = pd.DataFrame({"subject": subs, "session": 1, "run": 0})
    return X, np.array(ys, dtype="int64"), meta_df, ["Normal", "Abnormal"]


def load(name="ADFTD", subjects=None, win_sec=4.0, max_per_subject=60, resample=128,
         binary=False, fmin=0.5, fmax=45.0, normalize="trial_zscore", **_):
    if name == "MUMTAZ":
        return _load_mumtaz(subjects, win_sec, max_per_subject, resample, fmin, fmax, normalize)
    if name == "TUAB":
        return _load_tuab(subjects, win_sec, max_per_subject, resample, fmin, fmax, normalize)
    import mne
    mne.set_log_level("ERROR")
    part = pd.read_csv(ADFTD_DIR / "participants.tsv", sep="\t")
    gmap = GROUP_BIN if binary else GROUP3
    win = int(win_sec * resample)
    Xs, ys, subs = [], [], []
    for _, row in part.iterrows():
        sid = row["participant_id"]                    # e.g. sub-001
        grp = row["Group"]
        if grp not in gmap:
            continue
        snum = int(sid.split("-")[1])
        if subjects and snum not in subjects:
            continue
        f = ADFTD_DIR / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set"
        if not f.exists():
            continue
        raw = mne.io.read_raw_eeglab(str(f), preload=True)
        raw.filter(fmin, fmax, verbose="ERROR").resample(resample, verbose="ERROR")
        data = raw.get_data().astype("float32")        # [19, T]
        for w in _windows(data, win, max_per_subject):
            Xs.append(w); ys.append(int(gmap[grp])); subs.append(snum)
    X = np.stack(Xs).astype("float32")
    if normalize == "trial_zscore":
        m = X.mean(axis=2, keepdims=True); s = X.std(axis=2, keepdims=True) + 1e-7
        X = (X - m) / s
    meta = pd.DataFrame({"subject": subs, "session": 1, "run": 0})
    classes = ["Control", "Alzheimer"] if binary else ["Alzheimer", "Control", "FTD"]
    return X, np.array(ys, dtype="int64"), meta, classes
