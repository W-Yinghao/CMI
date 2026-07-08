#!/usr/bin/env python
"""S2P 9C v2 high-budget floor calibration feasibility.

CPU-only, SLURM-launched. It resolves the high-coverage allocation rule at
window granularity and writes the PM-requested go/no-go artifacts. It does not
train, inspect SHU-MI labels, or touch target labels.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

S2P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S2P / "s2p" / "scripts"))
import tueg_subject_loader as L


CANDIDATE_BUDGETS = [200, 500, 1000, 2000, 4000]
HIGH_SCAN = [4000, 3500, 3000, 2500, 2000]
SEEDS = [0, 1]
BASE_200H_WALL_H = 4.5
WALLTIME_LIMIT_H = 96.0
WALLTIME_SAFETY = 1.15


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def try_plan(hours, min_exposure_h, n_val, val_cap_windows):
    try:
        p = L.high_coverage_plan(hours, min_exposure_h=min_exposure_h, n_val=n_val,
                                 val_cap_windows=val_cap_windows)
        m = dict(p["manifest"])
        m["feasible_exact_window"] = True
        m["reason"] = "ok"
        return m, p
    except Exception as e:
        return {
            "total_hours": hours,
            "feasible_exact_window": False,
            "reason": str(e),
        }, None


def compute_row(manifest):
    h = float(manifest["total_hours"])
    est = BASE_200H_WALL_H * (h / 200.0) * WALLTIME_SAFETY
    return {
        "budget_h": h,
        "train_windows": int(round(h * 120)),
        "epochs": 50,
        "batch_size": 64,
        "loader_mode": "streaming",
        "estimated_wall_h": round(est, 2),
        "planned_partition": "A40",
        "planned_time_h": WALLTIME_LIMIT_H,
        "compute_budget_acceptable": bool(est <= WALLTIME_LIMIT_H),
        "estimate_basis": f"{BASE_200H_WALL_H}h_per_200h_x{WALLTIME_SAFETY}_safety",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/s2p_budget_floor_calibration_v2")
    ap.add_argument("--min-exposure-hours", type=float, default=0.25)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--val-cap-windows", type=int, default=24)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    grid_rows, plans = [], {}
    for h in sorted(set(CANDIDATE_BUDGETS + HIGH_SCAN)):
        m, p = try_plan(h, args.min_exposure_hours, args.n_val, args.val_cap_windows)
        c = compute_row(m) if m.get("feasible_exact_window") else {
            "budget_h": h,
            "train_windows": int(round(h * 120)),
            "epochs": 50,
            "batch_size": 64,
            "loader_mode": "streaming",
            "estimated_wall_h": None,
            "planned_partition": "A40",
            "planned_time_h": WALLTIME_LIMIT_H,
            "compute_budget_acceptable": False,
            "estimate_basis": "not_estimated_exact_window_failed",
        }
        row = dict(m)
        row.update(c)
        row["candidate_budget"] = h in CANDIDATE_BUDGETS
        row["h_high_scan"] = h in HIGH_SCAN
        row["overall_feasible"] = bool(row.get("feasible_exact_window") and row.get("compute_budget_acceptable"))
        grid_rows.append(row)
        if p is not None:
            plans[h] = p

    h4000_row = next(r for r in grid_rows if int(r["budget_h"]) == 4000)
    h4000_feasible = bool(h4000_row["overall_feasible"])
    h_high = None
    for h in HIGH_SCAN:
        row = next(r for r in grid_rows if int(r["budget_h"]) == h)
        if row["overall_feasible"]:
            h_high = h
            break

    stop_reasons = []
    if h_high is None or h_high < 2000:
        stop_reasons.append("stop_rule_1_no_h_high_ge_2000_feasible")

    training_budgets = []
    if h_high is not None:
        for h in [500, 1000, 2000, h_high]:
            if h not in training_budgets:
                training_budgets.append(h)

    for h in training_budgets:
        row = next((r for r in grid_rows if int(r["budget_h"]) == h), None)
        if row is None or not row["feasible_exact_window"]:
            stop_reasons.append(f"stop_rule_2_exact_window_budget_fails_H{h}")
        if row is None or not row["compute_budget_acceptable"]:
            stop_reasons.append(f"stop_rule_4_compute_budget_exceeds_limits_H{h}")

    subject_rows, window_rows, task_rows = [], [], []
    exposure_rows, val_rows = [], []
    n_by_budget, exposure_by_budget = {}, {}
    if not stop_reasons:
        first_val_done = False
        for h in training_budgets:
            plan = plans[h]
            pm = plan["manifest"]
            n_by_budget[str(h)] = int(pm["n_subjects"])
            exposure_by_budget[str(h)] = float(pm["exposure_h"])
            exposure_rows.append({
                "budget_h": h,
                "n_subjects": pm["n_subjects"],
                "exposure_h": pm["exposure_h"],
                "base_windows": pm["base_windows"],
                "plus1_subjects": pm["plus1_subjects"],
                "need_w": pm["need_w"],
                "pool_size_after_val_exclusion": pm["pool_size_after_val_exclusion"],
            })
            if not first_val_done:
                for s in plan["subjects_val"]:
                    val_rows.append({
                        "val_subject": int(s),
                        "take_windows": int(plan["val_cap_windows"]),
                        "hours": round(plan["val_cap_windows"] * L.WIN_H, 6),
                    })
                first_val_done = True
            for seed in SEEDS:
                cell = L.build_high_coverage_cell(h, seed, min_exposure_h=args.min_exposure_hours,
                                                  n_val=args.n_val, val_cap_windows=args.val_cap_windows,
                                                  expected_n_subjects=pm["n_subjects"])
                cm = cell["manifest"]
                tag = f"H{h}_s{seed}"
                task_rows.append({
                    "cell": tag,
                    "budget_h": h,
                    "seed": seed,
                    "n_subjects": cm["n_subjects"],
                    "exposure_h": cm["exposure_h"],
                })
                window_rows.append({
                    "cell": tag,
                    "budget_h": h,
                    "seed": seed,
                    "WT": cm["WT"],
                    "train_total_windows": cm["train_total_windows"],
                    "pct_off_budget": cm["pct_off_budget"],
                    "train_win_min": cm["train_win_min"],
                    "train_win_max": cm["train_win_max"],
                    "train_win_maxmin": cm["train_win_maxmin"],
                    "train_val_disjoint": cm["train_val_disjoint"],
                    "selected_subjects_sha": cm["selected_subjects_sha"],
                })
                for r in cell["train"]:
                    subject_rows.append({
                        "cell": tag,
                        "budget_h": h,
                        "seed": seed,
                        "subject": r["subject"],
                        "recording_id": r["recording_id"],
                        "take_windows": r["take_windows"],
                        "hours": r["hours"],
                    })

    write_csv(out / "budget_grid_feasibility.csv", grid_rows)
    write_csv(out / "high_coverage_subject_plan.csv", subject_rows,
              fieldnames=["cell", "budget_h", "seed", "subject", "recording_id", "take_windows", "hours"])
    write_csv(out / "budget_exposure_table.csv", exposure_rows,
              fieldnames=["budget_h", "n_subjects", "exposure_h", "base_windows", "plus1_subjects", "need_w",
                          "pool_size_after_val_exclusion"])
    write_csv(out / "pretrain_val_pool_plan.csv", val_rows,
              fieldnames=["val_subject", "take_windows", "hours"])
    write_csv(out / "window_budget_check.csv", window_rows,
              fieldnames=["cell", "budget_h", "seed", "WT", "train_total_windows", "pct_off_budget",
                          "train_win_min", "train_win_max", "train_win_maxmin", "train_val_disjoint",
                          "selected_subjects_sha"])
    write_csv(out / "compute_budget_estimate.csv", [r for r in grid_rows if r.get("candidate_budget") or r.get("h_high_scan")])
    write_csv(out / "budget_training_tasks.csv", task_rows, fieldnames=["cell", "budget_h", "seed", "n_subjects", "exposure_h"])

    hmax_decision = {
        "h4000_exact_window_feasible": bool(h4000_row.get("feasible_exact_window")),
        "h4000_compute_budget_acceptable": bool(h4000_row.get("compute_budget_acceptable")),
        "h4000_feasible_19common": h4000_feasible,
        "h_high_selected_h": h_high,
        "h_high_selection_rule": "4000_if_feasible_else_largest_19common_endpoint_rounded_down",
        "scan_h": HIGH_SCAN,
        "notes": "Feasible means exact 19-common window budget, subject-disjoint pretrain-val, and planned A40 walltime estimate <= 96h.",
    }
    dump_json(out / "hmax_19common_decision.json", hmax_decision)

    exact_ok = bool(not stop_reasons and all(r["train_total_windows"] == r["WT"] and r["train_win_maxmin"] <= 1
                                             and str(r["train_val_disjoint"]).lower() == "true"
                                             for r in window_rows))
    compute_ok = bool(not stop_reasons and all(r["compute_budget_acceptable"] for r in grid_rows
                                               if int(r["budget_h"]) in training_budgets))
    subject_val_ok = bool(not stop_reasons and all(str(r["train_val_disjoint"]).lower() == "true" for r in window_rows))
    go = bool(not stop_reasons and exact_ok and compute_ok and subject_val_ok)

    go_nogo = {
        "phase": "9C_v2_high_budget_floor_calibration",
        "primary_model": "CBraMod",
        "primary_corpus": "TUEG_19_common",
        "candidate_budgets_h": CANDIDATE_BUDGETS,
        "reuse_200h_baseline": True,
        "min_exposure_per_subject_h": args.min_exposure_hours,
        "h4000_feasible_19common": h4000_feasible,
        "h_high_selected_h": h_high,
        "h_high_selection_rule": "4000_if_feasible_else_largest_19common_endpoint_rounded_down",
        "training_budgets_h": training_budgets if go else None,
        "n_subjects_by_budget": n_by_budget if go else None,
        "exposure_by_budget": exposure_by_budget if go else None,
        "subject_disjoint_pretrain_val_feasible": subject_val_ok,
        "exact_window_budget_feasible": exact_ok,
        "compute_budget_acceptable": compute_ok,
        "target_labels_used": False,
        "auto_launch_training_if_pass": True,
        "GO": go,
        "stop_reasons": stop_reasons,
    }
    dump_json(out / "budget_floor_v2_go_nogo.json", go_nogo)

    report = [
        "# S2P 9C v2 feasibility report",
        "",
        f"GO: {go}",
        f"H_high selected: {h_high}",
        f"Training budgets: {training_budgets if go else None}",
        f"Stop reasons: {stop_reasons if stop_reasons else 'none'}",
        "",
        "No target labels were read or used. This job only inspected TUEG 19-common metadata/window counts.",
    ]
    (out / "slurm_feasibility_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(go_nogo, indent=2))
    raise SystemExit(0 if go else 2)


if __name__ == "__main__":
    main()
