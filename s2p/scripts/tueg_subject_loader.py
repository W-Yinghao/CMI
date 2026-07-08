#!/usr/bin/env python
"""S2P — TUEG subject-subset loader (see S2P_02/03/06 v3). Selects SUBJECT subsets from the processed TUEG corpus
(4704743c), restricted to the 19-common 10-20 covered recordings (canonical S2P-v1 corpus), for the P1
MATCHED-EXPOSURE SUBJECT-SCALING pilot. Yields 30 s windows (C=19, S=30 patches, 200/patch) with subject ids.
Deterministic (seeded). No target labels.

P1 design (S2P_06 v4, PM 2026-07-08): FIXED-BUDGET SUBJECT-vs-DEPTH FRONTIER. The v3 matched-exposure design was
killed by the identifiability triangle T = N·e (BL-9: at fixed per-subject exposure, doubling N doubles TOTAL data,
so subject count is collinear with total-data). "Pure subject diversity" is unidentifiable at pretraining scale. So
P1 fixes the TOTAL budget T (=200 h) and varies N ∈ {128,256,512,1024,2048}; per-subject exposure e = T/N compensates.
The estimand is the ALLOCATION effect — *given a fixed data budget, spend it on more subjects (shallower) or fewer
(deeper)?* — NOT a diversity effect. Total windows are held EXACTLY constant (24000 = 200 h) via a remainder
distribution (each subject gets base or base+1 windows, max−min ≤ 1). A FIXED GLOBAL pretrain-val subject pool
(seed-independent, disjoint from EVERY frontier training draw) enables comparable best-val-loss checkpoint selection
across all N. Eligibility + budget at WINDOW granularity (MJ-8: floored available windows). Checkpoint selection by
pretrain-val loss only; target labels never touch subset/checkpoint/PCA/head/rank/probe.
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


def _pick(subs, cap):
    """recording rows for `subs`, truncating each subject to EXACTLY its budget (in recording_id order).
    `cap` is either an int (same budget for all subjects) or a dict {subject: budget}."""
    m = _meta(); rows = []
    for s in subs:
        capw = cap[s] if isinstance(cap, dict) else cap
        r = m[m["subject"] == s].sort_values("recording_id"); acc = 0
        for _, row in r.iterrows():
            if acc >= capw:
                break
            take = min(int(row["avail_w"]), capw - acc)
            if take <= 0:
                continue
            rows.append(dict(subject=int(s), recording_id=int(row["recording_id"]), filepath=row["filepath"],
                             channels=row["channels"], take_windows=int(take),
                             hours=round(take * WIN_H, 6)))
            acc += take
    return rows


FRONTIER_T_H = 200.0                       # fixed total pretraining budget (hours)
FRONTIER_N = [128, 256, 512, 1024, 2048]   # subject-count axis
_FRONT_VAL = None


def _deepest_need_windows(total_hours=FRONTIER_T_H, n_grid=FRONTIER_N):
    """window budget per subject at the DEEPEST (smallest-N) frontier endpoint = the eligibility floor a val subject
    must stay BELOW to be guaranteed absent from every training pool. Derived (not hardcoded) so a T/grid change can
    never silently split val from its intended shallowness (MN-12)."""
    Nmin = min(n_grid); WT = int(round(total_hours * 120))
    return WT // Nmin + (1 if WT % Nmin else 0)


def _frontier_val(n_val=128, val_cap_windows=24, deep_need_windows=None):
    """FIXED GLOBAL pretrain-val subjects (seed-independent): eligible at val_cap_windows but SHALLOWER than the
    deepest training endpoint (windows < deep_need_windows) so they can NEVER be drawn into any frontier training
    cell (incl. the smallest N). Each val subject contributes exactly val_cap_windows windows -> val identical
    across all N. (Disjointness is ALSO enforced by setdiff in build_frontier_cell; this keeps val genuinely shallow.)"""
    global _FRONT_VAL
    if deep_need_windows is None:
        deep_need_windows = _deepest_need_windows()
    if _FRONT_VAL is None:
        sw = _subject_windows()
        cand = np.sort(sw[(sw >= val_cap_windows) & (sw < deep_need_windows)].index.to_numpy())
        if len(cand) < n_val:
            raise RuntimeError(f"infeasible val: {len(cand)} shallow-eligible subjects < n_val {n_val}")
        rng = np.random.default_rng(500000)          # fixed, seed- AND N-independent
        _FRONT_VAL = (np.sort(rng.choice(cand, n_val, replace=False)), int(val_cap_windows))
    return _FRONT_VAL


def build_frontier_cell(n_subjects, subset_seed, total_hours=FRONTIER_T_H, n_val=128):
    """Build one fixed-budget frontier cell (BL-9 fix): fixed TOTAL budget, variable N.

    n_subjects  : subjects to cover; per-subject exposure e = total_hours/n_subjects (window-quantized).
    subset_seed : controls the subject draw ONLY (trainer init seed is separate; MJ-5).
    Total windows are held EXACTLY at WT = round(total_hours*120) via a remainder distribution (each subject gets
    `base` or `base+1` windows, max−min = 1). Training subjects are disjoint from the FIXED GLOBAL val pool.
    returns dict(n_subjects, exposure_h, WT, pool_size, subjects/rows for train+val, manifest).
    """
    WT = int(round(total_hours * 120))
    base, rem = WT // n_subjects, WT - (WT // n_subjects) * n_subjects
    need_w = base + (1 if rem > 0 else 0)
    val_subj, val_capw = _frontier_val(n_val=n_val)
    sw = _subject_windows()
    pool = np.setdiff1d(np.sort(sw[sw >= need_w].index.to_numpy()), val_subj)   # disjoint from global val
    if len(pool) < n_subjects:
        raise RuntimeError(f"infeasible N={n_subjects}: pool {len(pool)} (>= {need_w}w, minus val) < {n_subjects}")
    rng = np.random.default_rng(400000 + subset_seed * 13 + n_subjects)
    subs = np.sort(rng.choice(pool, n_subjects, replace=False)); rng.shuffle(subs)
    # exact-budget allocation: first `rem` subjects (by sorted id, deterministic) get base+1, the rest base
    subs_sorted = sorted(subs.tolist())
    alloc = {s: (base + 1 if i < rem else base) for i, s in enumerate(subs_sorted)}
    rows_tr = _pick(subs_sorted, alloc)
    rows_val = _pick(val_subj.tolist(), val_capw)

    def _wsub(rows):
        d = {}
        for r in rows:
            d[r["subject"]] = d.get(r["subject"], 0) + r["take_windows"]
        return d
    wt, wv = _wsub(rows_tr), _wsub(rows_val)
    tot_w = sum(wt.values())
    man = dict(
        n_subjects=n_subjects, total_hours=total_hours, exposure_h=round(total_hours / n_subjects, 6),
        WT=WT, base_windows=base, plus1_subjects=rem, subset_seed=subset_seed, pool_size=int(len(pool)),
        train_total_windows=int(tot_w), train_total_hours=round(tot_w * WIN_H, 4),
        pct_off_budget=round(100 * (tot_w - WT) / WT, 5),
        train_win_min=min(wt.values()), train_win_max=max(wt.values()), train_win_maxmin=max(wt.values()) - min(wt.values()),
        n_val=n_val, val_cap_windows=val_capw, val_total_hours=round(sum(wv.values()) * WIN_H, 4),
        train_val_disjoint=bool(set(wt).isdisjoint(set(wv))))
    return dict(n_subjects=n_subjects, exposure_h=total_hours / n_subjects, WT=WT, pool_size=int(len(pool)),
                subjects_train=subs_sorted, subjects_val=val_subj.tolist(),
                train=rows_tr, val=rows_val, manifest=man)


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
            want = int(r["take_windows"])
            if nwin < want:                                            # MJ-13: on-disk array shorter than metadata
                raise RuntimeError(f"load-time budget shortfall: rec {r['recording_id']} has {nwin} windows on disk "
                                   f"< take_windows {want} (metadata n_timepoints mismatch) — exact-budget contract broken")
            nwin = want
        if max_windows_per_rec:
            nwin = min(nwin, max_windows_per_rec)
        if nwin == 0:
            continue
        x = np.asarray(a[:nwin * WLEN, idx], dtype=np.float32)         # (nwin*6000, 19)
        x = x.reshape(nwin, N_PATCH, PATCH, 19).transpose(0, 3, 1, 2)  # (nwin,19,30,200)
        x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)   # per-PATCH (200-sample) per-channel z-score
        yield x.astype(np.float32), np.full(nwin, r["subject"], dtype=int)
