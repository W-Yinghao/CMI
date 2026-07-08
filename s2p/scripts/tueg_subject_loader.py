#!/usr/bin/env python
"""S2P — TUEG subject-subset loader (see S2P_02/03/06 v3). Selects SUBJECT subsets from the processed TUEG corpus
(4704743c), restricted to the 19-common 10-20 covered recordings (canonical S2P-v1 corpus), for the P1
MATCHED-EXPOSURE SUBJECT-SCALING pilot. Yields 30 s windows (C=19, S=30 patches, 200/patch) with subject ids.
Deterministic (seeded). No target labels.

P1 design (S2P_06 v3, PM 2026-07-08): three within-pool matched-exposure NESTED pairs. For a fixed per-subject
exposure e (hours/subject) and a common eligibility pool (subjects with >= e usable data), draw a large-N subset and
a nested small-N subset (small ⊂ large) so each pair asks: *same per-subject depth, more subjects added*. A FIXED
per-contrast pretrain-val subject pool (disjoint from every training subject, invariant to the subset seed) enables
comparable best-val-loss checkpoint selection WITHIN a pair. Eligibility + budget are enforced at WINDOW granularity
(MJ-8: floored available windows, not summed hours); each selected subject contributes EXACTLY cap_windows windows
(subject-contribution Gini ~0 by construction). Checkpoint selection by pretrain-val loss only; target labels never
touch subset/checkpoint/PCA/head/rank/probe.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

TUEG = "/projects/EEG-foundation-model/datalake/processed/4704743c/TUEG"
SFREQ, PATCH, N_PATCH = 200, 200, 30            # 30 s window = 30 x 1 s patches
WLEN = N_PATCH * PATCH                          # 6000 samples / 30 s window
WIN_H = WLEN / SFREQ / 3600.0                   # hours per window
COMMON19 = ["EEG FP1-LE","EEG FP2-LE","EEG F3-LE","EEG F4-LE","EEG C3-LE","EEG C4-LE","EEG P3-LE","EEG P4-LE",
            "EEG O1-LE","EEG O2-LE","EEG F7-LE","EEG F8-LE","EEG T3-LE","EEG T4-LE","EEG T5-LE","EEG T6-LE",
            "EEG FZ-LE","EEG CZ-LE","EEG PZ-LE"]
_META = None
_SUBJWIN = None


def _meta():
    global _META
    if _META is None:
        m = pd.read_parquet(f"{TUEG}/metadata.parquet")
        m["cov"] = m["channels"].map(lambda s: set(COMMON19) <= set(json.loads(s)))
        m = m[m["cov"]].reset_index(drop=True)          # 19-common covered recordings only
        m["avail_w"] = (m["n_timepoints"] // WLEN).astype(int)   # FLOORED 30 s windows per recording (MJ-8)
        _META = m
    return _META


def _subject_windows():
    """per-subject total FLOORED available windows across 19-common recordings (MJ-8 eligibility basis)."""
    global _SUBJWIN
    if _SUBJWIN is None:
        _SUBJWIN = _meta().groupby("subject")["avail_w"].sum()
    return _SUBJWIN


def cap_windows_for(exposure_h):
    """per-subject window budget for a given per-subject exposure (hours)."""
    return max(1, int(round(exposure_h * 3600.0 / (WLEN / SFREQ))))


def eligible_pool(exposure_h):
    """subjects whose FLOORED available windows >= the per-subject window cap for `exposure_h` (MJ-8).
    A subject in this pool can supply the full exact cap, so per-subject contribution is uniform."""
    capw = cap_windows_for(exposure_h)
    sw = _subject_windows()
    return np.sort(sw[sw >= capw].index.to_numpy())


def _pick(subs, cap_windows):
    """recording rows for `subs`, truncating each subject to EXACTLY cap_windows (in recording_id order)."""
    m = _meta(); rows = []
    for s in subs:
        r = m[m["subject"] == s].sort_values("recording_id"); acc = 0
        for _, row in r.iterrows():
            if acc >= cap_windows:
                break
            take = min(int(row["avail_w"]), cap_windows - acc)
            if take <= 0:
                continue
            rows.append(dict(subject=int(s), recording_id=int(row["recording_id"]), filepath=row["filepath"],
                             channels=row["channels"], take_windows=int(take),
                             hours=round(take * WIN_H, 6)))
            acc += take
    return rows


def build_matched_exposure_pair(exposure_h, n_low, n_high, subset_seed, n_val=64):
    """Build a within-pool NESTED matched-exposure pair (BL-5/BL-6 fix).

    exposure_h : per-subject exposure (hours), shared by BOTH cells of the pair.
    n_low,n_high : nested subject counts (n_low <= n_high); the n_low training subjects are a SUBSET of n_high.
    subset_seed : controls the subject draw ONLY (init/training seed is separate, applied by the trainer; MJ-5).
    n_val : size of the FIXED per-contrast pretrain-val subject pool (invariant to subset_seed; disjoint from train).

    returns dict(exposure_h, cap_windows, pool_size, high/low/val recording-rows + subject lists, manifest).
    Each subject (train and val) contributes EXACTLY cap_windows windows.
    """
    assert n_low <= n_high
    capw = cap_windows_for(exposure_h)
    pool = eligible_pool(exposure_h)
    econtrast = int(round(exposure_h * 1e6))                    # stable per-exposure integer key

    # FIXED per-contrast pretrain-val subjects (invariant to subset_seed) -> comparable checkpoint selection in-pair
    rng_val = np.random.default_rng(700000 + econtrast)
    if len(pool) < n_val + n_high:
        raise RuntimeError(f"infeasible pair e={exposure_h:.4f}h: pool {len(pool)} < n_val {n_val} + n_high {n_high}")
    val_subj = np.sort(rng_val.choice(pool, n_val, replace=False))
    remaining = np.setdiff1d(pool, val_subj, assume_unique=False)   # train pool disjoint from val

    # nested train draw: pick n_high, then n_low is a PREFIX (subset) of the n_high draw
    rng_sub = np.random.default_rng(900000 + econtrast * 17 + subset_seed)
    high = rng_sub.choice(remaining, n_high, replace=False)
    rng_sub.shuffle(high)
    sub_high = sorted(high.tolist())
    sub_low = sorted(high[:n_low].tolist())                        # low ⊂ high (nested)

    rows_high, rows_low, rows_val = _pick(sub_high, capw), _pick(sub_low, capw), _pick(val_subj.tolist(), capw)

    def _hours(rows): return round(sum(r["hours"] for r in rows), 4)
    def _wsub(rows):
        d = {}
        for r in rows:
            d[r["subject"]] = d.get(r["subject"], 0) + r["take_windows"]
        return d
    wl, wh, wv = _wsub(rows_low), _wsub(rows_high), _wsub(rows_val)
    man = dict(
        exposure_h=round(exposure_h, 6), cap_windows=capw, pool_size=int(len(pool)),
        n_low=n_low, n_high=n_high, n_val=n_val, subset_seed=subset_seed,
        low_H0_h=round(n_low * exposure_h, 3), high_H0_h=round(n_high * exposure_h, 3),
        low_total_hours=_hours(rows_low), high_total_hours=_hours(rows_high), val_total_hours=_hours(rows_val),
        nested_low_subset_of_high=bool(set(sub_low) <= set(sub_high)),
        train_val_disjoint=bool(set(sub_high).isdisjoint(val_subj.tolist())),
        low_win_min=min(wl.values()), low_win_max=max(wl.values()),
        high_win_min=min(wh.values()), high_win_max=max(wh.values()),
        val_win_min=min(wv.values()), val_win_max=max(wv.values()))
    return dict(exposure_h=exposure_h, cap_windows=capw, pool_size=int(len(pool)),
                subjects_high=sub_high, subjects_low=sub_low, subjects_val=val_subj.tolist(),
                train_high=rows_high, train_low=rows_low, val=rows_val, manifest=man)


def build_subset(n_subjects, hours_budget, condition, seed, val_frac=0.15):
    """Legacy single-cell builder (9B-0 smoke). Eligibility on FLOORED windows (MJ-8). Kept for the smoke path;
    P1 uses build_matched_exposure_pair. Splits subjects disjointly into pretrain-train vs pretrain-val."""
    cap = (hours_budget / n_subjects) if condition == "fixed_hours" else float(hours_budget)
    capw = cap_windows_for(cap)
    pool = eligible_pool(cap)
    if len(pool) < n_subjects:
        raise RuntimeError(f"infeasible: {len(pool)} subjects have >= {capw} windows ({cap:.3f}h), need {n_subjects}")
    rng = np.random.default_rng(10000 + seed)
    subjects = rng.choice(pool, n_subjects, replace=False); rng.shuffle(subjects)
    nval = max(1, int(round(n_subjects * val_frac)))
    sub_val, sub_train = sorted(subjects[:nval].tolist()), sorted(subjects[nval:].tolist())
    tr, va = _pick(sub_train, capw), _pick(sub_val, capw)
    man = dict(n_subjects=n_subjects, hours_budget=hours_budget, condition=condition, seed=seed,
               per_subject_cap_h=round(cap, 4), cap_windows=capw,
               n_subjects_train=len(sub_train), n_subjects_val=len(sub_val),
               n_recordings_train=len(tr), n_recordings_val=len(va),
               train_hours=round(sum(r["hours"] for r in tr), 2), val_hours=round(sum(r["hours"] for r in va), 2),
               subjects_disjoint=bool(set(sub_train).isdisjoint(sub_val)))
    return dict(train=tr, val=va, subjects_train=sub_train, subjects_val=sub_val, manifest=man)


def windows_for(rows, max_windows_per_rec=None):
    """yield (X (n_win,19,30,200) float32, subj (n_win,)) — 19-common channels in canonical order, 30 s windows."""
    m = _meta()
    for r in rows:
        chn = json.loads(r["channels"]); idx = [chn.index(c) for c in COMMON19]
        a = np.load(f"{TUEG}/{r['filepath']}", mmap_mode="r")          # (T, 33) T_C
        T = a.shape[0]; nwin = T // WLEN
        if r.get("take_windows"):                                      # window-level budget (exact per-subject cap)
            nwin = min(nwin, int(r["take_windows"]))
        if max_windows_per_rec:
            nwin = min(nwin, max_windows_per_rec)
        if nwin == 0:
            continue
        x = np.asarray(a[:nwin * WLEN, idx], dtype=np.float32)         # (nwin*6000, 19)
        x = x.reshape(nwin, N_PATCH, PATCH, 19).transpose(0, 3, 1, 2)  # (nwin,19,30,200)
        x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)   # per-window per-channel z-score
        yield x.astype(np.float32), np.full(nwin, r["subject"], dtype=int)
