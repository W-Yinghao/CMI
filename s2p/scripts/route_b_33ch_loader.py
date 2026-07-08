"""Route B 33-channel fixed-group-mix TUEG loader.

No labels. Exact 30 s window budgets are allocated by channel_order_hash x
reference_scheme group according to the 9D-1 contract.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TUEG = "/projects/EEG-foundation-model/datalake/processed/4704743c/TUEG"
WLEN = 6000
PATCH = 200
N_PATCH = 30
WIN_H = 30.0 / 3600.0
VAL_CAP_W = 24
N_VAL = 128


def _ch_hash(channels):
    return hashlib.sha256(json.dumps(channels, separators=(",", ":")).encode()).hexdigest()[:16]


def _reference_scheme(channels):
    eeg = [c for c in channels if c.startswith("EEG ")]
    if eeg and all(c.endswith("-LE") for c in eeg):
        return "LE"
    if eeg and all(c.endswith("-REF") for c in eeg):
        return "REF"
    return "mixed_or_nonstandard"


def _meta():
    m = pd.read_parquet(f"{TUEG}/metadata.parquet")
    m = m[m["n_channels"] == 33].copy()
    m["avail_w"] = (m["n_timepoints"] // WLEN).astype(int)
    m = m[m["avail_w"] > 0].copy()
    m["channels_list"] = m["channels"].map(json.loads)
    m["channel_order_hash"] = m["channels_list"].map(_ch_hash)
    m["reference_scheme"] = m["channels_list"].map(_reference_scheme)
    m["group_id"] = m["channel_order_hash"] + "_" + m["reference_scheme"]
    return m.reset_index(drop=True)


def fixed_val_subjects():
    sw = _meta().groupby("subject")["avail_w"].sum().sort_index()
    cand = np.sort(sw[sw >= VAL_CAP_W].index.to_numpy())
    if len(cand) < N_VAL:
        raise RuntimeError(f"Route B val infeasible: {len(cand)} subjects >= {VAL_CAP_W}w < {N_VAL}")
    rng = np.random.default_rng(920000)
    return np.sort(rng.choice(cand, N_VAL, replace=False)).astype(int)


def _contract_group_mix(contract_dir, budget_h):
    mix = pd.read_csv(Path(contract_dir) / "route_b_group_mix_by_budget.csv")
    rows = mix[mix["budget_h"].astype(float) == float(budget_h)]
    if rows.empty:
        raise RuntimeError(f"no group mix rows for budget_h={budget_h}")
    return {r["group_id"]: int(r["take_windows"]) for _, r in rows.iterrows()}


def _group_orders(contract_dir):
    spec = json.loads((Path(contract_dir) / "route_b_canonical_channel_order.json").read_text())
    return {g["group_id"]: list(g["channels"]) for g in spec["groups"]}


def _take_from_frame(frame, target_w, rng):
    idx = np.arange(len(frame))
    rng.shuffle(idx)
    rows, remaining = [], int(target_w)
    for j in idx:
        if remaining <= 0:
            break
        r = frame.iloc[int(j)]
        take = min(int(r["avail_w"]), remaining)
        if take <= 0:
            continue
        rows.append({
            "subject": int(r["subject"]),
            "recording_id": int(r["recording_id"]),
            "filepath": r["filepath"],
            "channels": r["channels"],
            "group_id": r["group_id"],
            "take_windows": int(take),
            "hours": round(take * WIN_H, 6),
        })
        remaining -= take
    if remaining != 0:
        raise RuntimeError(f"group allocation shortfall: remaining={remaining}")
    return rows


def build_route_b_cell(budget_h, subset_seed, contract_dir="results/s2p_route_b_33ch_contract"):
    m = _meta()
    val = set(map(int, fixed_val_subjects()))
    train = m[~m["subject"].isin(val)].copy()
    group_mix = _contract_group_mix(contract_dir, budget_h)
    rows_train = []
    for gid, target_w in sorted(group_mix.items()):
        frame = train[train["group_id"] == gid].reset_index(drop=True)
        seed = 970000 + int(float(budget_h) * 10) + int(subset_seed) * 101 + int(hashlib.sha256(gid.encode()).hexdigest()[:8], 16)
        rows_train.extend(_take_from_frame(frame, target_w, np.random.default_rng(seed)))

    rows_val = []
    for s in sorted(val):
        frame = m[m["subject"] == s].sort_values(["recording_id"]).reset_index(drop=True)
        remaining = VAL_CAP_W
        for _, r in frame.iterrows():
            if remaining <= 0:
                break
            take = min(int(r["avail_w"]), remaining)
            if take <= 0:
                continue
            rows_val.append({
                "subject": int(r["subject"]),
                "recording_id": int(r["recording_id"]),
                "filepath": r["filepath"],
                "channels": r["channels"],
                "group_id": r["group_id"],
                "take_windows": int(take),
                "hours": round(take * WIN_H, 6),
            })
            remaining -= take
        if remaining != 0:
            raise RuntimeError(f"val subject {s} shortfall: remaining={remaining}")

    wt = int(round(float(budget_h) * 120))
    train_w = int(sum(r["take_windows"] for r in rows_train))
    val_w = int(sum(r["take_windows"] for r in rows_val))
    train_subjects = sorted({r["subject"] for r in rows_train})
    val_subjects = sorted({r["subject"] for r in rows_val})
    manifest = {
        "route": "B_33ch_cbramod_only",
        "budget_h": float(budget_h),
        "subset_seed": int(subset_seed),
        "WT": wt,
        "train_total_windows": train_w,
        "train_total_hours": round(train_w * WIN_H, 6),
        "val_total_windows": val_w,
        "val_total_hours": round(val_w * WIN_H, 6),
        "n_train_subjects": len(train_subjects),
        "n_val_subjects": len(val_subjects),
        "train_val_disjoint": bool(set(train_subjects).isdisjoint(set(val_subjects))),
        "group_mix_windows": group_mix,
        "selected_subjects_sha": hashlib.sha256(json.dumps(train_subjects).encode()).hexdigest()[:16],
    }
    if train_w != wt:
        raise RuntimeError(f"train window budget mismatch: {train_w} != {wt}")
    if not manifest["train_val_disjoint"]:
        raise RuntimeError("train/val subjects overlap")
    return {"train": rows_train, "val": rows_val, "manifest": manifest}


def windows_for(rows, contract_dir="results/s2p_route_b_33ch_contract"):
    orders = _group_orders(contract_dir)
    for r in rows:
        chn = json.loads(r["channels"])
        target = orders[r["group_id"]]
        idx = [chn.index(c) for c in target]
        arr = np.load(f"{TUEG}/{r['filepath']}", mmap_mode="r")
        nwin = int(r["take_windows"])
        if arr.shape[0] // WLEN < nwin:
            raise RuntimeError(f"recording {r['recording_id']} shorter than take_windows={nwin}")
        x = np.asarray(arr[: nwin * WLEN, idx], dtype=np.float32)
        x = x.reshape(nwin, N_PATCH, PATCH, 33).transpose(0, 3, 1, 2)
        x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
        yield x.astype(np.float32), np.full(nwin, int(r["subject"]), dtype=int)
