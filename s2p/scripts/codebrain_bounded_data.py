#!/usr/bin/env python
"""Frozen data contract for the bounded CodeBrain Stage-2 experiment.

The contract samples unique 30 s windows from the processed TUEG 19-common
corpus.  Budgets are nested prefixes of one fixed window permutation; the
same subsets are used for both model-initialization seeds.  No downstream
labels are read here.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import tueg_subject_loader as L


BUDGETS_H = (200, 1000, 2000)
SUBSET_SEED = 941731
VAL_SUBJECTS = 128
VAL_WINDOWS_PER_SUBJECT = 24
WINDOWS_PER_HOUR = 120


def _sha_int64(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype="<i8")
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    cols = ["recording_id", "subject", "session", "run", "filepath", "channels", "avail_w"]
    h = hashlib.sha256()
    for row in frame[cols].itertuples(index=False, name=None):
        h.update(json.dumps(row, separators=(",", ":"), ensure_ascii=True).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def corpus_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return positive-window full/train/validation recording frames."""
    full = L._meta().copy()
    full = full[full["avail_w"] > 0].sort_values("recording_id").reset_index(drop=True)
    val_subjects, val_cap = L._budget_floor_val(
        n_val=VAL_SUBJECTS, val_cap_windows=VAL_WINDOWS_PER_SUBJECT
    )
    if int(val_cap) != VAL_WINDOWS_PER_SUBJECT:
        raise RuntimeError(f"unexpected val cap {val_cap}")
    val_set = set(map(int, val_subjects))
    train = full[~full["subject"].isin(val_set)].reset_index(drop=True)
    val = full[full["subject"].isin(val_set)].reset_index(drop=True)
    if set(train["subject"]).intersection(set(val["subject"])):
        raise RuntimeError("pretrain train/val subject overlap")
    return full, train, val


def train_offsets(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    counts = train["avail_w"].to_numpy(dtype=np.int64)
    ends = np.cumsum(counts, dtype=np.int64)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
    return starts, ends


def train_window_permutation(train: pd.DataFrame) -> np.ndarray:
    total = int(train["avail_w"].sum())
    return np.random.default_rng(SUBSET_SEED).permutation(total).astype(np.int64, copy=False)


def resolve_train_window_ids(train: pd.DataFrame, window_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(window_ids, dtype=np.int64)
    starts, ends = train_offsets(train)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= int(ends[-1])):
        raise IndexError("global train window id outside corpus")
    rows = np.searchsorted(ends, ids, side="right")
    local = ids - starts[rows]
    return rows.astype(np.int64), local.astype(np.int64)


def fixed_val_window_pairs(val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    row_positions, local_positions = [], []
    for subject, frame in val.groupby("subject", sort=True):
        remaining = VAL_WINDOWS_PER_SUBJECT
        for row_pos, row in frame.sort_values("recording_id").iterrows():
            take = min(int(row["avail_w"]), remaining)
            row_positions.extend([int(row_pos)] * take)
            local_positions.extend(range(take))
            remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError(f"validation subject {subject} is short by {remaining} windows")
    return np.asarray(row_positions, dtype=np.int64), np.asarray(local_positions, dtype=np.int64)


def gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if not arr.size or np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    return float(2 * np.sum(np.arange(1, n + 1) * arr) / (n * arr.sum()) - (n + 1) / n)


def budget_summaries() -> tuple[list[dict], dict]:
    full, train, val = corpus_frames()
    permutation = train_window_permutation(train)
    total_train = len(permutation)
    rows = []
    previous = np.asarray([], dtype=np.int64)
    for budget_h in BUDGETS_H:
        target = int(budget_h * WINDOWS_PER_HOUR)
        if target > total_train:
            raise RuntimeError(f"H{budget_h} needs {target} windows but only {total_train} are available")
        selected = permutation[:target]
        rec_rows, _ = resolve_train_window_ids(train, selected)
        rec_counts = np.bincount(rec_rows, minlength=len(train))
        subject_counts = pd.Series(rec_counts, index=train["subject"]).groupby(level=0).sum()
        subject_counts = subject_counts[subject_counts > 0].sort_index()
        selected_record_counts = rec_counts[rec_counts > 0]
        nested = bool(np.isin(previous, selected, assume_unique=True).all()) if previous.size else True
        rows.append({
            "budget_h": int(budget_h),
            "target_windows": target,
            "exact_window_budget_feasible": True,
            "no_window_reuse": bool(np.unique(selected).size == selected.size),
            "nested_with_previous_budget": nested,
            "subset_seed": SUBSET_SEED,
            "subset_shared_across_init_seeds": True,
            "selected_window_order_sha256": _sha_int64(selected),
            "selected_window_set_sha256": _sha_int64(np.sort(selected)),
            "n_subjects": int(len(subject_counts)),
            "n_recordings": int(len(selected_record_counts)),
            "subject_windows_min": int(subject_counts.min()),
            "subject_windows_median": float(subject_counts.median()),
            "subject_windows_p90": float(subject_counts.quantile(0.9)),
            "subject_windows_max": int(subject_counts.max()),
            "subject_contribution_gini": round(gini(subject_counts.to_numpy()), 8),
            "mean_exposure_per_selected_subject_h": float(target / WINDOWS_PER_HOUR / len(subject_counts)),
            "optimizer_epochs_planned": 10,
            "optimizer_updates_depend_on_budget": True,
        })
        previous = selected
    val_rows, val_local = fixed_val_window_pairs(val)
    authority = {
        "corpus": "TUEG_processed_4704743c_19common",
        "channel_order": L.COMMON19,
        "channel_order_sha256": hashlib.sha256(json.dumps(L.COMMON19).encode()).hexdigest(),
        "window_shape": [19, 30, 200],
        "window_seconds": 30,
        "normalization_contract": "source_volts_times_1e6_then_native_divide_100",
        "full_positive_window_recordings": int(len(full)),
        "full_positive_window_subjects": int(full["subject"].nunique()),
        "full_positive_windows": int(full["avail_w"].sum()),
        "train_recordings_after_val_exclusion": int(len(train)),
        "train_subjects_after_val_exclusion": int(train["subject"].nunique()),
        "train_windows_after_val_exclusion": int(train["avail_w"].sum()),
        "train_hours_after_val_exclusion": float(train["avail_w"].sum() / WINDOWS_PER_HOUR),
        "val_subjects": int(val["subject"].nunique()),
        "val_windows": int(len(val_rows)),
        "val_hours": float(len(val_rows) / WINDOWS_PER_HOUR),
        "train_val_subject_disjoint": True,
        "full_metadata_sha256": _frame_sha256(full),
        "train_metadata_sha256": _frame_sha256(train),
        "val_window_pairs_sha256": hashlib.sha256(
            np.stack((val_rows, val_local), axis=1).astype("<i8").tobytes()
        ).hexdigest(),
        "selection_algorithm": "PCG64 permutation of all unique train windows; nested budget prefixes",
        "subset_seed": SUBSET_SEED,
    }
    return rows, authority


def diagnostic_samples(n_per_stratum: int = 256, gate_seed: int = 941799) -> list[dict]:
    """Sample disjoint train budget strata plus fixed validation windows."""
    _, train, val = corpus_frames()
    permutation = train_window_permutation(train)
    boundaries = [("H200", 0, 200 * 120),
                  ("H1000_increment", 200 * 120, 1000 * 120),
                  ("H2000_increment", 1000 * 120, 2000 * 120)]
    out = []
    rng = np.random.default_rng(gate_seed)
    for name, lo, hi in boundaries:
        positions = rng.choice(np.arange(lo, hi, dtype=np.int64), n_per_stratum, replace=False)
        ids = permutation[positions]
        rec_rows, local = resolve_train_window_ids(train, ids)
        for rr, ll, gid in zip(rec_rows, local, ids):
            out.append({"stratum": name, "split": "train", "frame_row": int(rr),
                        "local_window": int(ll), "global_window": int(gid)})
    val_rows, val_local = fixed_val_window_pairs(val)
    pick = rng.choice(len(val_rows), n_per_stratum, replace=False)
    for p in pick:
        out.append({"stratum": "pretrain_val", "split": "val", "frame_row": int(val_rows[p]),
                    "local_window": int(val_local[p]), "global_window": -1})
    return out


class CodeBrainWindowDataset(Dataset):
    """Read selected windows as microvolt tensors; native trainer applies /100."""

    def __init__(self, samples: list[dict], cache_size: int = 16):
        _, self.train, self.val = corpus_frames()
        self.samples = samples
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _array(self, path: str) -> np.ndarray:
        if path in self._cache:
            arr = self._cache.pop(path)
            self._cache[path] = arr
            return arr
        arr = np.load(path, mmap_mode="r")
        self._cache[path] = arr
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arr

    def __getitem__(self, index: int):
        sample = self.samples[index]
        frame = self.train if sample["split"] == "train" else self.val
        row = frame.iloc[int(sample["frame_row"])]
        channels = json.loads(row["channels"])
        channel_idx = [channels.index(name) for name in L.COMMON19]
        arr = self._array(str(Path(L.TUEG) / row["filepath"]))
        lo = int(sample["local_window"]) * L.WLEN
        x = np.asarray(arr[lo:lo + L.WLEN, channel_idx], dtype=np.float32)
        if x.shape != (L.WLEN, 19):
            raise RuntimeError(f"short or malformed window: {x.shape}, recording={row['recording_id']}")
        x = x.reshape(L.N_PATCH, L.PATCH, 19).transpose(2, 0, 1) * 1e6
        if not np.isfinite(x).all():
            raise RuntimeError(f"non-finite EEG window in recording={row['recording_id']}")
        return torch.from_numpy(x.copy()), sample["stratum"]
