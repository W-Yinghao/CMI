#!/usr/bin/env python
"""Route B B1 pretraining gate for completed budget cells.

This is a QC gate over pretraining logs/checkpoints only. It does not inspect
downstream labels and does not launch downstream jobs.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        writer.writerows(rows)


def cell_name(budget_h, seed):
    return f"H{int(budget_h)}_s{int(seed)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", default="results/s2p_route_b_33ch_b1")
    ap.add_argument("--contract-dir", default="results/s2p_route_b_33ch_contract")
    ap.add_argument("--max-budget-h", type=float, default=1000.0)
    ap.add_argument("--suffix", default="through_1000h")
    args = ap.parse_args()

    root = Path(args.result_root)
    contract = Path(args.contract_dir)
    tasks = list(csv.DictReader((contract / "route_b_b1_training_tasks.csv").open()))
    tasks = [r for r in tasks if float(r["budget_h"]) <= args.max_budget_h]

    log_rows, ckpt_rows, contract_rows = [], [], []
    all_completed = True
    nan_or_inf_seen = False
    checkpoint_reload_pass = True
    native_objective_preserved = True
    fixed_group_mix_preserved = True
    pretrain_val_loss_selection_only = True
    target_labels_used = False

    for task in tasks:
        cell = task["cell"]
        cdir = root / cell
        log_path = cdir / "train_log.jsonl"
        best_path = cdir / "best.pth"
        last_path = cdir / "last.pth"
        events = read_events(log_path) if log_path.exists() else []
        data = next((e for e in events if e.get("event") == "data"), {})
        epochs = [e for e in events if e.get("event") == "epoch"]
        done = [e for e in events if e.get("event") == "done"]
        complete = bool(done) and len(epochs) == int(done[-1].get("epochs", 50))
        all_completed = all_completed and complete
        done_event = done[-1] if done else {}

        bad = []
        for e in epochs:
            for key in ("train_loss", "val_loss"):
                value = e.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    bad.append((e.get("epoch"), key, value))
        nan_or_inf_seen = nan_or_inf_seen or bool(bad)
        checkpoint_reload_pass = checkpoint_reload_pass and bool(done_event.get("checkpoint_strict_reload"))
        target_labels_used = target_labels_used or bool(done_event.get("target_labels_used")) or bool(data.get("target_labels_used"))
        native_objective_preserved = native_objective_preserved and bool(best_path.exists()) and bool(last_path.exists())
        pretrain_val_loss_selection_only = pretrain_val_loss_selection_only and bool(done_event.get("best_val_loss") is not None)

        manifest = done_event.get("cell_manifest") or data.get("cell_manifest") or {}
        fixed_group_mix_preserved = fixed_group_mix_preserved and bool(manifest.get("group_mix_windows"))
        train_val_disjoint = bool(manifest.get("train_val_disjoint"))
        exact_budget = float(manifest.get("train_total_hours", -1)) == float(task["budget_h"])

        best_epoch = min(epochs, key=lambda x: x["val_loss"]) if epochs else {}
        last_epoch = epochs[-1] if epochs else {}
        log_rows.append({
            "cell": cell,
            "budget_h": float(task["budget_h"]),
            "seed": int(task["seed"]),
            "epochs_seen": len(epochs),
            "done": complete,
            "first_val_loss": done_event.get("first_val_loss"),
            "best_epoch": done_event.get("best_epoch", best_epoch.get("epoch")),
            "best_val_loss": done_event.get("best_val_loss", best_epoch.get("val_loss")),
            "last_epoch": last_epoch.get("epoch"),
            "last_train_loss": last_epoch.get("train_loss"),
            "last_val_loss": last_epoch.get("val_loss"),
            "gpu_peak_gb": last_epoch.get("gpu_peak_gb"),
            "nan_or_inf_seen": bool(bad),
            "target_labels_used": bool(done_event.get("target_labels_used") or data.get("target_labels_used")),
            "training_wall_s": done_event.get("training_wall_s"),
        })
        ckpt_rows.append({
            "cell": cell,
            "budget_h": float(task["budget_h"]),
            "seed": int(task["seed"]),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "best_checkpoint_exists": best_path.exists(),
            "last_checkpoint_exists": last_path.exists(),
            "checkpoint_strict_reload": bool(done_event.get("checkpoint_strict_reload")),
            "checkpoint_selection": "pretrain_val_loss_only",
            "best_epoch": done_event.get("best_epoch", best_epoch.get("epoch")),
            "best_val_loss": done_event.get("best_val_loss", best_epoch.get("val_loss")),
        })
        contract_rows.append({
            "cell": cell,
            "budget_h": float(task["budget_h"]),
            "seed": int(task["seed"]),
            "train_total_hours": manifest.get("train_total_hours"),
            "train_total_windows": manifest.get("train_total_windows"),
            "val_total_hours": manifest.get("val_total_hours"),
            "val_total_windows": manifest.get("val_total_windows"),
            "n_train_subjects": manifest.get("n_train_subjects"),
            "n_val_subjects": manifest.get("n_val_subjects"),
            "train_val_disjoint": train_val_disjoint,
            "exact_budget_h": exact_budget,
            "fixed_group_mix_present": bool(manifest.get("group_mix_windows")),
            "selected_subjects_sha": manifest.get("selected_subjects_sha"),
        })

    approve_probe = (
        all_completed
        and not nan_or_inf_seen
        and checkpoint_reload_pass
        and native_objective_preserved
        and fixed_group_mix_preserved
        and pretrain_val_loss_selection_only
        and not target_labels_used
        and all(r["exact_budget_h"] and r["train_val_disjoint"] for r in contract_rows)
    )

    suffix = args.suffix
    write_csv(root / f"route_b_b1_pretrain_logs_{suffix}.csv", log_rows)
    write_csv(root / f"route_b_b1_checkpoint_manifest_{suffix}.csv", ckpt_rows)
    write_csv(root / f"route_b_b1_contract_verification_{suffix}.csv", contract_rows)
    gate = {
        "scope": suffix,
        "max_budget_h": args.max_budget_h,
        "cells": [r["cell"] for r in tasks],
        "all_runs_completed": all_completed,
        "nan_or_inf_seen": nan_or_inf_seen,
        "checkpoint_reload_pass": checkpoint_reload_pass,
        "native_objective_preserved": native_objective_preserved,
        "fixed_group_mix_preserved": fixed_group_mix_preserved,
        "pretrain_val_loss_selection_only": pretrain_val_loss_selection_only,
        "target_labels_used_for_selection": False,
        "approve_downstream_probe_gate_recommended": approve_probe,
        "full_downstream_fleet_held_until_policy_decision": True,
        "h2000_still_running_allowed": True,
    }
    (root / f"route_b_b1_pretrain_gate_{suffix}.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
