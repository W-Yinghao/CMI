"""Generic loader for the lab-PREPROCESSED datalake (processed/5e77943a/<name>): per-recording [T,C]
npy @200 Hz (already band-passed + notched) + metadata.parquet (subject, channels, filepath) +
events.parquet (onset_sample, event_code). EPOCHED datasets (MI: Stieger2021 51-subj, PhysionetMI
109-subj) -> one window per event at its onset. Used for SCALE credibility (more subjects than 2a/2b).
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROC = Path("/projects/EEG-foundation-model/datalake/processed/5e77943a")


def load(name, subjects=None, win_sec=4.0, max_per_subject=80, resample=128, max_subjects=50,
         normalize="trial_zscore", **_):
    from scipy.signal import resample_poly
    root = PROC / name
    infos = json.load(open(root / "infos.json"))
    sfreq = int(infos["sfreq"])
    eid = infos.get("event_id", {})                    # name -> code
    codes = sorted(set(eid.values())) if eid else None
    code2lab = {c: i for i, c in enumerate(codes)} if codes else None
    classes = [k for k, _ in sorted(eid.items(), key=lambda kv: kv[1])] if eid else None
    meta = pd.read_parquet(root / "metadata.parquet")
    ev = pd.read_parquet(root / "events.parquet")
    win200 = int(win_sec * sfreq)

    subj_all = sorted(meta["subject"].unique())
    if subjects:
        subj_all = [s for s in subj_all if s in subjects]
    keep = set(subj_all[:max_subjects])
    evg = {rid: g for rid, g in ev.groupby("recording_id")}

    Xs, ys, subs, per = [], [], [], {}
    for _, row in meta.iterrows():
        subj = int(row["subject"])
        if subj not in keep or row["recording_id"] not in evg:
            continue
        data = np.load(root / row["filepath"]).T.astype("float32")   # [C, T]
        T = data.shape[1]
        for _, e in evg[row["recording_id"]].iterrows():
            if per.get(subj, 0) >= max_per_subject:
                break
            on = int(e["onset_sample"])
            if on + win200 > T:
                continue
            w = data[:, on:on + win200]
            if resample != sfreq:
                w = resample_poly(w, resample, sfreq, axis=1).astype("float32")
            lab = code2lab[int(e["event_code"])] if code2lab else int(e["event_code"])
            Xs.append(w); ys.append(lab); subs.append(subj); per[subj] = per.get(subj, 0) + 1
    X = np.stack(Xs).astype("float32")
    if normalize == "trial_zscore":
        mu = X.mean(axis=2, keepdims=True); sd = X.std(axis=2, keepdims=True) + 1e-7
        X = (X - mu) / sd
    meta_df = pd.DataFrame({"subject": subs, "session": 1, "run": 0})
    return X, np.array(ys, dtype="int64"), meta_df, classes
