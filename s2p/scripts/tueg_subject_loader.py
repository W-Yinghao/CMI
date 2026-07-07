#!/usr/bin/env python
"""S2P — TUEG subject-subset loader (see S2P_02/03). Selects a SUBJECT subset from the processed TUEG corpus
(4704743c), restricts to 19-common 10-20 covered recordings (canonical S2P-v1 corpus), enforces a per-subject
HOURS BUDGET (fixed: H0/N; growing: cap), splits subjects DISJOINTLY into pretrain-train vs pretrain-val, and
yields 30 s windows (C=19, S=30 patches, 200/patch) with subject ids. Deterministic (seeded). No target labels.
Validates go/no-go gates G1-G4 (sample-by-subject, hours-budget, subject-disjoint train/val)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

TUEG = "/projects/EEG-foundation-model/datalake/processed/4704743c/TUEG"
SFREQ, PATCH, N_PATCH = 200, 200, 30            # 30 s window = 30 x 1 s patches
COMMON19 = ["EEG FP1-LE","EEG FP2-LE","EEG F3-LE","EEG F4-LE","EEG C3-LE","EEG C4-LE","EEG P3-LE","EEG P4-LE",
            "EEG O1-LE","EEG O2-LE","EEG F7-LE","EEG F8-LE","EEG T3-LE","EEG T4-LE","EEG T5-LE","EEG T6-LE",
            "EEG FZ-LE","EEG CZ-LE","EEG PZ-LE"]
_META = None


def _meta():
    global _META
    if _META is None:
        m = pd.read_parquet(f"{TUEG}/metadata.parquet")
        m["hours"] = m["n_timepoints"] / SFREQ / 3600.0
        m["cov"] = m["channels"].map(lambda s: set(COMMON19) <= set(json.loads(s)))
        _META = m[m["cov"]].reset_index(drop=True)   # 19-common covered recordings only
    return _META


def build_subset(n_subjects, hours_budget, condition, seed, val_frac=0.15):
    """returns dict(train=[recording rows], val=[...], subjects_train, subjects_val, manifest). Hours-budget enforced
    by capping per-subject recordings; subjects split disjointly into train/val."""
    m = _meta()
    subj_hours = m.groupby("subject")["hours"].sum()
    cap = (hours_budget / n_subjects) if condition == "fixed_hours" else float(hours_budget)  # growing: hours_budget=per-subject cap
    eligible = subj_hours[subj_hours >= cap].index.to_numpy()
    if len(eligible) < n_subjects:
        raise RuntimeError(f"infeasible: {len(eligible)} subjects have >= {cap:.3f}h, need {n_subjects}")
    rng = np.random.default_rng(10000 + seed)
    subjects = np.sort(rng.choice(eligible, n_subjects, replace=False))
    rng.shuffle(subjects)
    nval = max(1, int(round(n_subjects * val_frac)))
    sub_val, sub_train = sorted(subjects[:nval].tolist()), sorted(subjects[nval:].tolist())

    cap_windows = max(1, int(round(cap * 3600 / (N_PATCH * PATCH / SFREQ))))   # WINDOW-level budget (fixes fixed-hours bug)

    def pick(subs):
        rows = []
        for s in subs:
            r = m[m["subject"] == s].sort_values("recording_id"); acc_w = 0
            for _, row in r.iterrows():
                if acc_w >= cap_windows:
                    break
                avail_w = int(row["n_timepoints"]) // (N_PATCH * PATCH)         # non-overlapping 30 s windows
                take_w = min(avail_w, cap_windows - acc_w)
                if take_w <= 0:
                    continue
                rows.append(dict(subject=int(s), recording_id=int(row["recording_id"]), filepath=row["filepath"],
                                 channels=row["channels"], take_windows=int(take_w),
                                 hours=round(take_w * (N_PATCH * PATCH / SFREQ) / 3600, 4)))
                acc_w += take_w
        return rows
    tr, va = pick(sub_train), pick(sub_val)
    man = dict(n_subjects=n_subjects, hours_budget=hours_budget, condition=condition, seed=seed, per_subject_cap_h=round(cap, 4),
               n_subjects_train=len(sub_train), n_subjects_val=len(sub_val),
               n_recordings_train=len(tr), n_recordings_val=len(va),
               train_hours=round(sum(r["hours"] for r in tr), 2), val_hours=round(sum(r["hours"] for r in va), 2),
               subjects_disjoint=bool(set(sub_train).isdisjoint(sub_val)))
    return dict(train=tr, val=va, subjects_train=sub_train, subjects_val=sub_val, manifest=man)


def windows_for(rows, max_windows_per_rec=None):
    """yield (X (n_win,19,30,200) float32, subj (n_win,)) — 19-common channels in canonical order, 30 s windows."""
    for r in rows:
        chn = json.loads(r["channels"]); idx = [chn.index(c) for c in COMMON19]
        a = np.load(f"{TUEG}/{r['filepath']}", mmap_mode="r")          # (T, 33) T_C
        T = a.shape[0]; wlen = N_PATCH * PATCH                          # 6000
        nwin = T // wlen
        if r.get("take_windows"):                                      # window-level budget (fixed-hours enforcement)
            nwin = min(nwin, int(r["take_windows"]))
        if max_windows_per_rec:
            nwin = min(nwin, max_windows_per_rec)
        if nwin == 0:
            continue
        x = np.asarray(a[:nwin * wlen, idx], dtype=np.float32)          # (nwin*6000, 19)
        x = x.reshape(nwin, N_PATCH, PATCH, 19).transpose(0, 3, 1, 2)   # (nwin,19,30,200)
        # per-window per-channel z-score
        x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
        yield x.astype(np.float32), np.full(nwin, r["subject"], dtype=int)
