#!/usr/bin/env python
"""Aggregate S2P 9C v2 high-budget calibration outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha16(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def parse_logs(out, tasks):
    rows = []
    for _, t in tasks.iterrows():
        log = out / t.cell / "train_log.jsonl"
        if not log.exists():
            rows.append({"cell": t.cell, "budget_h": t.budget_h, "seed": t.seed, "event": "missing_log"})
            continue
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append({"cell": t.cell, "budget_h": t.budget_h, "seed": t.seed, **rec})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/s2p_budget_floor_calibration_v2")
    args = ap.parse_args()
    out = Path(args.out_dir)
    patch = out / "downstream_patch"
    window = out / "downstream_window_ref"
    tasks = pd.read_csv(out / "budget_training_tasks.csv")
    raw = pd.read_csv(patch / "p1_task_and_frontier_raw.csv")
    raw["tag"] = raw["tag"].replace({"random": "random_init"})
    if "budget_h" not in raw:
        raw["budget_h"] = raw["tag"].str.extract(r"H([0-9.]+)_").astype(float)

    # Pretraining manifests.
    run_rows, ckpt_rows = [], []
    for _, t in tasks.iterrows():
        cell_dir = out / t.cell
        rs = read_json(cell_dir / "run_summary.json", {})
        ck = cell_dir / "best.pth"
        run_rows.append({
            "cell": t.cell,
            "budget_h": t.budget_h,
            "seed": t.seed,
            "n_subjects": t.n_subjects,
            "exposure_h": t.exposure_h,
            "best_epoch": rs.get("best_epoch"),
            "first_val_loss": rs.get("first_val_loss"),
            "best_val_loss": rs.get("best_val_loss"),
            "val_rel_decrease": rs.get("val_rel_decrease"),
            "loader_mode": rs.get("loader_mode"),
            "target_labels_used": rs.get("target_labels_used"),
            "git": rs.get("git"),
        })
        ckpt_rows.append({
            "cell": t.cell,
            "budget_h": t.budget_h,
            "seed": t.seed,
            "checkpoint": str(ck),
            "checkpoint_exists": ck.exists(),
            "checkpoint_sha16": sha16(ck),
            "best_val_loss": rs.get("best_val_loss"),
            "checkpoint_selection": "pretrain_val_loss_only",
        })
    pd.DataFrame(run_rows).to_csv(out / "budget_pretrain_run_manifest.csv", index=False)
    pd.DataFrame(ckpt_rows).to_csv(out / "budget_checkpoint_manifest.csv", index=False)
    parse_logs(out, tasks).to_csv(out / "budget_pretrain_logs.csv", index=False)

    budget = raw[raw.tag.str.startswith("H")].copy()
    refs = raw[raw.tag.isin(["random_init", "released"])].copy()
    budget.to_csv(out / "budget_downstream_task_performance.csv", index=False)
    budget[[c for c in budget.columns if c.startswith("l1_") or c in ("tag", "budget_h", "seed")]].to_csv(
        out / "budget_pairwise_subject_separability.csv", index=False)
    budget[[c for c in budget.columns if c in ("tag", "budget_h", "seed", "l4_alignment", "subject_subspace_var_frac")]].to_csv(
        out / "budget_l4_task_alignment.csv", index=False)
    budget[[c for c in budget.columns if c.startswith("l5_") or c in ("tag", "budget_h", "seed")]].to_csv(
        out / "budget_l5_replay.csv", index=False)
    budget[[c for c in budget.columns if c in ("tag", "budget_h", "seed", "target_bacc", "target_macro_f1", "target_nll")]].to_csv(
        out / "budget_l6_target_consequence.csv", index=False)

    ref_rows = []
    for norm, frame in [("patch", refs), ("window", pd.read_csv(window / "p1_task_and_frontier_raw.csv"))]:
        frame["tag"] = frame["tag"].replace({"random": "random_init"})
        for _, r in frame[frame.tag.isin(["random_init", "released"])].iterrows():
            ref_rows.append({
                "tag": r.tag,
                "norm": norm,
                "target_bacc": r.target_bacc,
                "source_val_bacc": r.source_val_bacc,
                "l1_pairwise_bacc_mean": r.get("l1_l1_pairwise_bacc_mean"),
                "l4_alignment": r.get("l4_alignment"),
                "l5_reliance_z": r.get("l5_l5_reliance_z"),
            })
    ref_df = pd.DataFrame(ref_rows)
    ref_df.to_csv(out / "budget_random_released_references.csv", index=False)

    fw = read_json(patch / "p1_target_label_firewall.json", {})
    fw.update({
        "phase": "9C_v2_high_budget_floor_calibration",
        "target_labels_used_for_selection": False,
        "target_labels_used": False,
        "target_labels_final_scoring_only": True,
    })
    dump_json(out / "budget_target_label_firewall.json", fw)

    rand_patch = float(ref_df[(ref_df.tag == "random_init") & (ref_df.norm == "patch")].target_bacc.iloc[0])
    rel_patch = float(ref_df[(ref_df.tag == "released") & (ref_df.norm == "patch")].target_bacc.iloc[0])
    rel_window = float(ref_df[(ref_df.tag == "released") & (ref_df.norm == "window")].target_bacc.iloc[0])
    by_budget = {}
    l1_by_budget = {}
    for h, g in budget.groupby("budget_h"):
        hkey = str(int(h) if float(h).is_integer() else h)
        target_mean = float(g.target_bacc.mean())
        src_gate = bool((g.source_val_bacc >= 0.58).all())
        by_budget[hkey] = {
            "mean": target_mean,
            "sd": float(g.target_bacc.std(ddof=0)),
            "n": int(len(g)),
            "source_val_gate_all_seeds": src_gate,
            "criterion_A_random_plus_0p02": bool(target_mean >= rand_patch + 0.02),
            "criterion_B_above_0p55": bool(target_mean >= 0.55),
            "criterion_C_source_val_gate": src_gate,
        }
        l1_by_budget[hkey] = float(g.l1_l1_pairwise_bacc_mean.mean())
    crossed = [float(h) for h, r in by_budget.items() if r["criterion_A_random_plus_0p02"] and r["criterion_C_source_val_gate"]]
    usable = [float(h) for h, r in by_budget.items() if r["criterion_A_random_plus_0p02"] and r["criterion_B_above_0p55"] and r["criterion_C_source_val_gate"]]
    summary = {
        "primary_question": "when_does_from_scratch_cbramod_exit_frozen_transfer_floor",
        "budget_grid_h": sorted(float(h) for h in budget.budget_h.dropna().unique()),
        "target_bacc_by_budget": by_budget,
        "random_floor_bacc": rand_patch,
        "released_reference_bacc_patch": rel_patch,
        "released_reference_bacc_window": rel_window,
        "budget_floor_crossed": bool(crossed),
        "first_budget_above_random_plus_0p02": min(crossed) if crossed else None,
        "first_budget_above_0p55": min(usable) if usable else None,
        "l1_subject_separability_by_budget": l1_by_budget,
        "p2_allocation_study_recommended": bool(crossed),
        "target_labels_used": False,
    }
    dump_json(out / "budget_floor_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
