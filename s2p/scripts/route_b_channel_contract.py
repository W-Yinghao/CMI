#!/usr/bin/env python
"""S2P 9D-1 Route B channel/reference sampling contract.

CPU metadata-only for the TUEG side. It pins the fixed 33-channel group orders,
builds a fixed group-mixture exact-window sampling contract for B1 budgets, and
combines it with the downstream sanity result if present.

It does not train and does not submit SLURM jobs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TUEG = "/projects/EEG-foundation-model/datalake/processed/4704743c/TUEG"
WLEN = 6000
WIN_H = 30.0 / 3600.0
BUDGETS = [200, 500, 1000, 2000]
SEEDS = [0, 1]
N_VAL = 128
VAL_CAP_W = 24


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def ch_hash(channels):
    return hashlib.sha256(json.dumps(channels, separators=(",", ":")).encode()).hexdigest()[:16]


def reference_scheme(channels):
    eeg = [c for c in channels if c.startswith("EEG ")]
    if eeg and all(c.endswith("-LE") for c in eeg):
        return "LE"
    if eeg and all(c.endswith("-REF") for c in eeg):
        return "REF"
    return "mixed_or_nonstandard"


def fixed_val_subjects(subject_windows):
    cand = np.sort(subject_windows[subject_windows >= VAL_CAP_W].index.to_numpy())
    if len(cand) < N_VAL:
        raise RuntimeError(f"Route B val infeasible: {len(cand)} subjects >= {VAL_CAP_W}w < {N_VAL}")
    rng = np.random.default_rng(920000)
    return np.sort(rng.choice(cand, N_VAL, replace=False))


def allocate_by_largest_remainder(total_windows, group_caps):
    total_cap = int(sum(group_caps.values()))
    if total_windows > total_cap:
        raise RuntimeError(f"budget {total_windows}w exceeds available {total_cap}w")
    quotas = {g: total_windows * cap / total_cap for g, cap in group_caps.items()}
    alloc = {g: int(np.floor(q)) for g, q in quotas.items()}
    rem = int(total_windows - sum(alloc.values()))
    order = sorted(group_caps, key=lambda g: (quotas[g] - alloc[g], group_caps[g], g), reverse=True)
    for g in order[:rem]:
        alloc[g] += 1
    for g, cap in group_caps.items():
        if alloc[g] > cap:
            raise RuntimeError(f"group {g} allocation {alloc[g]} exceeds cap {cap}")
    return alloc, quotas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/s2p_route_b_33ch_contract")
    ap.add_argument("--downstream-sanity", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(f"{TUEG}/metadata.parquet")
    meta = meta[meta["n_channels"] == 33].copy()
    meta["avail_w"] = (meta["n_timepoints"] // WLEN).astype(int)
    meta = meta[meta["avail_w"] > 0].copy()
    meta["channels_list"] = meta["channels"].map(json.loads)
    meta["channel_order_hash"] = meta["channels_list"].map(ch_hash)
    meta["reference_scheme"] = meta["channels_list"].map(reference_scheme)
    meta["group_id"] = meta["channel_order_hash"] + "_" + meta["reference_scheme"]

    subject_windows = meta.groupby("subject")["avail_w"].sum().sort_index()
    val_subjects = fixed_val_subjects(subject_windows)
    meta_train = meta[~meta["subject"].isin(set(map(int, val_subjects)))].copy()

    canonical = {
        "route": "B_33ch_cbramod_only",
        "corpus": "TUEG_processed_exact_33ch",
        "contract_scope": "group_specific_pinned_33_channel_orders",
        "single_global_33_name_order_feasible": False,
        "single_global_33_name_order_reason": (
            "The exact-33ch subset contains six distinct channel-name sets/orders across LE and REF recordings; "
            "a single global 33-name order would require dropping/imputing channels, which is not authorized."
        ),
        "canonicalization_rule": (
            "Within each channel_order_hash x reference_scheme group, reorder records to that group's pinned 33-channel "
            "name order. Preserve the fixed group mixture across budgets."
        ),
        "n_groups": int(meta["group_id"].nunique()),
        "budgets_h": BUDGETS,
        "seeds": SEEDS,
        "target_labels_used_for_selection": False,
    }

    group_rows = []
    group_caps = {}
    group_subjects = {}
    for gid, frame in meta_train.groupby("group_id"):
        channels = json.loads(frame["channels"].iloc[0])
        cap = int(frame["avail_w"].sum())
        group_caps[gid] = cap
        group_subjects[gid] = int(frame["subject"].nunique())
        group_rows.append({
            "group_id": gid,
            "channel_order_hash": frame["channel_order_hash"].iloc[0],
            "reference_scheme": frame["reference_scheme"].iloc[0],
            "n_channels": 33,
            "n_recordings_train_after_val": int(len(frame)),
            "n_subjects_train_after_val": int(frame["subject"].nunique()),
            "available_train_windows": cap,
            "available_train_hours": round(cap * WIN_H, 6),
            "proportion_by_windows": cap / int(meta_train["avail_w"].sum()),
            "channels_json": json.dumps(channels, separators=(",", ":")),
        })
    group_rows = sorted(group_rows, key=lambda r: r["group_id"])
    write_csv(out / "route_b_channel_group_manifest.csv", group_rows)

    ref_rows = []
    for ref, frame in meta_train.groupby("reference_scheme"):
        cap = int(frame["avail_w"].sum())
        ref_rows.append({
            "reference_scheme": ref,
            "n_groups": int(frame["group_id"].nunique()),
            "n_recordings_train_after_val": int(len(frame)),
            "n_subjects_train_after_val": int(frame["subject"].nunique()),
            "available_train_windows": cap,
            "available_train_hours": round(cap * WIN_H, 6),
            "proportion_by_windows": cap / int(meta_train["avail_w"].sum()),
        })
    write_csv(out / "route_b_reference_scheme_manifest.csv", ref_rows)

    canonical["groups"] = [
        {
            "group_id": r["group_id"],
            "channel_order_hash": r["channel_order_hash"],
            "reference_scheme": r["reference_scheme"],
            "channels": json.loads(r["channels_json"]),
        }
        for r in group_rows
    ]
    canonical["sha256_16"] = hashlib.sha256(json.dumps(canonical["groups"], sort_keys=True).encode()).hexdigest()[:16]
    dump_json(out / "route_b_canonical_channel_order.json", canonical)

    mix_rows = []
    check_rows = []
    fixed_mix_ok = True
    for h in BUDGETS:
        wt = int(round(h * 120))
        alloc, quotas = allocate_by_largest_remainder(wt, group_caps)
        for gid in sorted(group_caps):
            target_prop = group_caps[gid] / int(meta_train["avail_w"].sum())
            realized_prop = alloc[gid] / wt
            mix_rows.append({
                "budget_h": h,
                "group_id": gid,
                "target_windows": wt,
                "take_windows": alloc[gid],
                "available_train_windows": group_caps[gid],
                "target_proportion_by_windows": target_prop,
                "realized_proportion_by_windows": realized_prop,
                "absolute_proportion_error": abs(realized_prop - target_prop),
                "quota_windows": quotas[gid],
                "reference_scheme": gid.split("_")[-1],
            })
        max_abs_err = max(abs((alloc[g] / wt) - (group_caps[g] / int(meta_train["avail_w"].sum()))) for g in group_caps)
        ok = bool(sum(alloc.values()) == wt and all(alloc[g] <= group_caps[g] for g in group_caps) and max_abs_err <= (1.0 / wt))
        fixed_mix_ok = fixed_mix_ok and ok
        check_rows.append({
            "budget_h": h,
            "target_windows": wt,
            "allocated_windows": int(sum(alloc.values())),
            "exact_window_budget_feasible": bool(sum(alloc.values()) == wt),
            "all_group_allocations_within_capacity": bool(all(alloc[g] <= group_caps[g] for g in group_caps)),
            "max_abs_group_proportion_error": max_abs_err,
            "fixed_group_mix_feasible": ok,
        })
    write_csv(out / "route_b_group_mix_by_budget.csv", mix_rows)
    write_csv(out / "route_b_sampling_contract_check.csv", check_rows)
    task_rows = [
        {"task_id": i, "cell": f"H{h}_s{s}", "budget_h": h, "seed": s}
        for i, (h, s) in enumerate((h, s) for h in BUDGETS for s in SEEDS)
    ]
    write_csv(out / "route_b_b1_training_tasks.csv", task_rows)

    single_group_feasible = False
    largest = max(group_rows, key=lambda r: r["available_train_windows"])
    if largest["available_train_windows"] >= 2000 * 120:
        # Keep the same n_val=128 rule for the fallback. The largest group must
        # independently support both val and train to qualify.
        single_group_feasible = largest["n_subjects_train_after_val"] >= (N_VAL + 1)

    sanity_path = Path(args.downstream_sanity) if args.downstream_sanity else out / "route_b_downstream_sanity.csv"
    downstream_primary = "none"
    released_pass = None
    if sanity_path.exists():
        sanity = pd.read_csv(sanity_path)
        native = sanity[(sanity["channel_mode"] == "native32") & (sanity["sanity_pass"] == True)]
        common = sanity[(sanity["channel_mode"] == "19common") & (sanity["sanity_pass"] == True)]
        if len(native):
            downstream_primary = "SHU_MI_native32"
            released_pass = True
        elif len(common):
            downstream_primary = "SHU_MI_19common"
            released_pass = True
        else:
            released_pass = False

    selected = "fixed_group_mix" if fixed_mix_ok else ("single_group" if single_group_feasible else "none")
    launch = bool(
        selected != "none"
        and downstream_primary != "none"
        and released_pass is True
        and all(r["fixed_group_mix_feasible"] for r in check_rows)
    )
    go = {
        "route": "B_33ch_cbramod_only",
        "canonical_channel_order_pinned": True,
        "canonical_channel_order_scope": "group_specific_pinned_orders",
        "single_global_33_name_order_feasible": False,
        "reference_scheme_groups_pinned": True,
        "fixed_group_mix_feasible": bool(fixed_mix_ok),
        "largest_single_group_feasible_ge2000h": bool(single_group_feasible),
        "selected_sampling_contract": selected,
        "budgets_h": BUDGETS,
        "seeds": SEEDS,
        "h4000_included": False,
        "downstream_primary": downstream_primary,
        "released_checkpoint_downstream_sanity_pass": released_pass,
        "target_labels_used_for_selection": False,
        "launch_route_b_b1": launch,
        "training_launched": False,
        "notes": (
            "Route B uses fixed group-mixture sampling across channel_order_hash x reference_scheme groups. "
            "Training remains a separate action and is not launched by this script."
        ),
    }
    dump_json(out / "route_b_training_go_nogo.json", go)
    print(json.dumps(go, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
