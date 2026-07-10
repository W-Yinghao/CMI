#!/usr/bin/env python
"""D2-2 FACED H2000 downstream update.

This downstream-only updater evaluates H2000_s0/H2000_s1 and merges those
rows into the already committed FACED through-1000h audit tables. It reuses the
existing random/released references and does not launch pretraining or
fine-tuning.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import route_b_faced_downstream_audit as faced


DEFAULT_CELLS = ["H2000_s0", "H2000_s1"]
TASK_FIELDS = {
    "tag", "checkpoint", "budget_h", "seed", "source_val_kappa", "source_val_weighted_f1",
    "source_val_macro_f1", "source_val_bacc", "source_val_nll", "target_kappa",
    "target_weighted_f1", "target_macro_f1", "target_bacc", "target_nll",
    "selected_pca_dim", "selected_C", "selection_metric", "task_gate_pass",
    "l4_l5_l6_status", "target_labels_used_for_selection", "fine_tuning_used",
    "codebrain_used", "h2000_included",
}
L1_FIELDS = {
    "tag", "budget_h", "seed", "split", "l1_classifier", "l1_pairwise_subject_bacc_mean",
    "l1_pairwise_subject_bacc_sd", "l1_n_pairs", "target_split",
    "target_l1_classifier", "target_l1_pairwise_subject_bacc_mean",
    "target_l1_pairwise_subject_bacc_sd", "target_l1_n_pairs",
}
L4_FIELDS = {
    "tag", "budget_h", "seed", "task_gate_pass", "l4_l5_l6_status",
    "subject_subspace_rank", "subject_subspace_var_frac",
    "l4_task_head_subject_subspace_energy",
}
L6_FIELDS = {
    "tag", "budget_h", "seed", "task_gate_pass", "l4_l5_l6_status",
    "l6_target_kappa_base", "l6_target_kappa_subject_removed",
    "l6_target_bacc_base", "l6_target_bacc_subject_removed",
    "l6_target_weighted_f1_base", "l6_target_weighted_f1_subject_removed",
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fobj:
        while True:
            chunk = fobj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_immutable_contract(ckpt_root, manifest_path, go_nogo_path):
    go_nogo = json.loads(Path(go_nogo_path).read_text())
    if go_nogo.get("status") != "PASS" or go_nogo.get("faced_reaudit_unlocked") is not True:
        raise RuntimeError(f"H2000 immutable closure is not PASS: {go_nogo}")
    manifest_obj = json.loads(Path(manifest_path).read_text())
    rows = {row["tag"]: row for row in manifest_obj.get("checkpoints", [])}
    if set(rows) != set(DEFAULT_CELLS):
        raise RuntimeError(f"immutable manifest cells mismatch: {sorted(rows)}")
    verified = {}
    for tag in DEFAULT_CELLS:
        row = rows[tag]
        link = Path(ckpt_root) / tag / "best.pth"
        payload = Path(row["immutable_checkpoint"])
        if not link.is_symlink():
            raise RuntimeError(f"immutable checkpoint link missing for {tag}: {link}")
        if link.resolve() != payload.resolve():
            raise RuntimeError(f"immutable checkpoint path mismatch for {tag}")
        if payload.stat().st_mode & 0o222:
            raise RuntimeError(f"immutable checkpoint is writable for {tag}: {payload}")
        digest = sha256_file(payload)
        if digest != row["sha256"]:
            raise RuntimeError(f"immutable checkpoint SHA mismatch for {tag}")
        verified[tag] = {**row, "audit_path": str(link), "sha256_verified": digest}
    return verified


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    faced.write_csv(Path(path), rows, fieldnames=fieldnames)


def row_subset(row, fields):
    return {k: v for k, v in row.items() if k in fields}


def merge_by_tag(existing, new_rows):
    new_tags = {str(r["tag"]) for r in new_rows}
    rows = [r for r in existing if str(r.get("tag")) not in new_tags]
    rows.extend(new_rows)
    return rows


def f(row, key):
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    return float(value)


def reference_row(rows, tag):
    found = [r for r in rows if r.get("tag") == tag]
    if not found:
        raise RuntimeError(f"missing required reference row: {tag}")
    return found[0]


def budget_groups(task_rows):
    out = {}
    for r in task_rows:
        if not r.get("tag", "").startswith("H"):
            continue
        h = int(float(r["budget_h"]))
        out.setdefault(h, []).append(r)
    return out


def metric_mean(rows, key):
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def metric_sd(rows, key):
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if v is not None]
    return float(np.std(vals)) if vals else None


def budget_mean_table(task_rows):
    groups = budget_groups(task_rows)
    table = {}
    for h, rows in sorted(groups.items()):
        table[h] = {
            "n": len(rows),
            "target_kappa_mean": metric_mean(rows, "target_kappa"),
            "target_kappa_sd": metric_sd(rows, "target_kappa"),
            "target_bacc_mean": metric_mean(rows, "target_bacc"),
            "target_bacc_sd": metric_sd(rows, "target_bacc"),
            "target_weighted_f1_mean": metric_mean(rows, "target_weighted_f1"),
            "source_val_kappa_mean": metric_mean(rows, "source_val_kappa"),
        }
    return table


def build_summary(task_rows, l1_rows, elapsed_s):
    random = reference_row(task_rows, "random")
    released = reference_row(task_rows, "released")
    means = budget_mean_table(task_rows)
    random_kappa = f(random, "target_kappa")
    random_bacc = f(random, "target_bacc")
    released_kappa = f(released, "target_kappa")
    released_bacc = f(released, "target_bacc")
    h1000 = means.get(1000)
    h2000 = means.get(2000)
    best_budget = None
    if means:
        best_h, best_vals = max(means.items(), key=lambda kv: kv[1]["target_kappa_mean"])
        best_budget = {
            "budget_h": best_h,
            "mean_target_kappa": best_vals["target_kappa_mean"],
            "mean_target_bacc": best_vals["target_bacc_mean"],
            "descriptive_only_not_optimality_claim": True,
        }
    pretrained_l1 = [
        f(r, "l1_pairwise_subject_bacc_mean")
        for r in l1_rows
        if r.get("tag", "").startswith("H") and f(r, "l1_pairwise_subject_bacc_mean") is not None
    ]
    h2000_seed_rows = sorted(budget_groups(task_rows).get(2000, []), key=lambda r: int(float(r["seed"])))
    h2000_seed_difference = None
    if len(h2000_seed_rows) == 2:
        h2000_seed_difference = {
            "target_kappa_s1_minus_s0": f(h2000_seed_rows[1], "target_kappa") - f(h2000_seed_rows[0], "target_kappa"),
            "target_bacc_s1_minus_s0": f(h2000_seed_rows[1], "target_bacc") - f(h2000_seed_rows[0], "target_bacc"),
        }
    released_gap_h2000 = None
    if h2000:
        released_gap_h2000 = {
            "released_minus_h2000_mean_kappa": released_kappa - h2000["target_kappa_mean"],
            "released_minus_h2000_mean_bacc": released_bacc - h2000["target_bacc_mean"],
        }
    released_band_reached = False
    if best_budget is not None:
        released_band_reached = (
            abs(best_budget["mean_target_kappa"] - released_kappa) <= 0.005
            or abs(best_budget["mean_target_bacc"] - released_bacc) <= 0.005
        )
    return {
        "downstream_dataset": "FACED",
        "protocol": "frozen_encoder_source_only_probe",
        "not_full_finetuning_reproduction": True,
        "phase": "h2000_immutable_reaudit",
        "train_subjects": "1-80",
        "val_subjects": "81-100",
        "test_subjects": "101-123",
        "primary_metric": "cohen_kappa",
        "evaluated_budgets_h": sorted(means),
        "evaluated_seeds": sorted({int(float(r["seed"])) for r in task_rows if r.get("seed") not in ("", None)}),
        "budget_means": {str(k): v for k, v in means.items()},
        "random_reference": {
            "target_kappa": random_kappa,
            "target_bacc": random_bacc,
            "source_val_kappa": f(random, "source_val_kappa"),
            "source_val_bacc": f(random, "source_val_bacc"),
        },
        "released_reference": {
            "target_kappa": released_kappa,
            "target_bacc": released_bacc,
            "source_val_kappa": f(released, "source_val_kappa"),
            "source_val_bacc": f(released, "source_val_bacc"),
        },
        "h1000_mean": None if not h1000 else {
            "target_kappa": h1000["target_kappa_mean"],
            "target_bacc": h1000["target_bacc_mean"],
            "target_weighted_f1": h1000["target_weighted_f1_mean"],
        },
        "h2000_mean": None if not h2000 else {
            "target_kappa": h2000["target_kappa_mean"],
            "target_bacc": h2000["target_bacc_mean"],
            "target_weighted_f1": h2000["target_weighted_f1_mean"],
        },
        "h2000_seed_difference": h2000_seed_difference,
        "floor_crossed_by_1000": None if not h1000 else bool(h1000["target_kappa_mean"] >= random_kappa + 0.02),
        "floor_crossed_by_2000": None if not h2000 else bool(h2000["target_kappa_mean"] >= random_kappa + 0.02),
        "floor_crossed_kappa": bool(any(v["target_kappa_mean"] >= random_kappa + 0.02 for v in means.values())),
        "floor_crossed_bacc": bool(any(v["target_bacc_mean"] >= random_bacc + 0.02 for v in means.values())),
        "monotonic_scaling_established": False,
        "best_budget_descriptive": best_budget,
        "released_band_reached": bool(released_band_reached),
        "released_band_rule": "budget mean within 0.005 of released Kappa or bAcc; descriptive band, not superiority claim",
        "released_gap_h2000_mean": released_gap_h2000,
        "l1_ceiling_saturated": bool(pretrained_l1 and min(pretrained_l1) >= 0.95),
        "full_finetuning_reproduction": False,
        "target_labels_used_for_selection": False,
        "h2000_included": True,
        "h4000_included": False,
        "fine_tuning_used": False,
        "codebrain_used": False,
        "elapsed_s": round(elapsed_s, 1),
    }


def update_notes(path, summary):
    lines = [
        "# FACED Frozen-Probe Verifier Notes",
        "",
        "- Phase: immutable H2000 re-audit.",
        "- Dataset: FACED native32 LMDB, 10s, 200Hz, 9 classes.",
        "- Split: train subjects 1-80, validation subjects 81-100, test subjects 101-123.",
        "- Encoder is frozen; no fine-tuning or pretraining is launched by this script.",
        "- PCA, classifier, subject subspace, and rank are source train/val only.",
        "- FACED test labels are used only for final scoring.",
        "- H2000_s0 and H2000_s1 use SHA-pinned, read-only immutable checkpoints.",
        "- Random and released rows are reused from the D2-1 FACED audit.",
        "- H4000, CodeBrain, fine-tuning, and any extra dataset are excluded.",
        "- H500/H1000/H2000 comparisons are descriptive budget-floor calibration, not a monotonic scaling-law claim.",
        f"- Floor crossed by 1000h: {summary['floor_crossed_by_1000']}.",
        f"- Floor crossed by 2000h: {summary['floor_crossed_by_2000']}.",
        f"- Best descriptive budget: {summary['best_budget_descriptive']}.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default=faced.B1_CKPT_ROOT)
    ap.add_argument("--out-dir", default="results/s2p_route_b_33ch_b1_faced")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--pca-dims", nargs="+", type=int, default=[32, 64, 128])
    ap.add_argument("--cells", nargs="*", default=DEFAULT_CELLS)
    ap.add_argument("--immutable-manifest", required=True)
    ap.add_argument("--closure-go-nogo", required=True)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = args.cells
    if cells != DEFAULT_CELLS:
        raise RuntimeError(f"D2-2 only approves {DEFAULT_CELLS}; got {cells}")
    immutable = load_immutable_contract(
        args.ckpt_root, args.immutable_manifest, args.closure_go_nogo
    )

    task_existing = read_csv(out / "faced_task_performance.csv")
    if not task_existing:
        raise RuntimeError("missing existing through-1000h FACED task table")
    random_ref = reference_row(task_existing, "random")
    random_source_val_kappa = f(random_ref, "source_val_kappa")

    ckpt_rows_existing = read_csv(out / "faced_checkpoint_manifest.csv")
    feature_existing = read_csv(out / "faced_feature_manifest.csv")
    l1_existing = read_csv(out / "faced_pairwise_subject_separability.csv")
    l4_existing = read_csv(out / "faced_l4_task_alignment.csv")
    l5_existing = read_csv(out / "faced_l5_replay.csv")
    l6_existing = read_csv(out / "faced_l6_target_consequence.csv")

    protocol = faced.protocol_json("h2000", cells, args)
    protocol["phase"] = "D2_h2000_update"
    protocol["reuse_existing_references"] = ["random", "released"]
    protocol["forbidden"]["h2000_downstream"] = False
    protocol["forbidden"]["h4000_downstream"] = True
    protocol["immutable_h2000_required"] = True
    protocol["immutable_manifest"] = str(Path(args.immutable_manifest).resolve())
    faced.write_json(out / "faced_downstream_protocol.json", protocol)

    ckpt_rows_new = []
    for tag in cells:
        ckpt = str(Path(args.ckpt_root) / tag / "best.pth")
        budget_h, seed = faced.budget_seed(tag)
        ckpt_rows_new.append({
            "tag": tag,
            "checkpoint": ckpt,
            "budget_h": budget_h,
            "seed": seed,
            "reference": False,
            "checkpoint_exists": Path(ckpt).exists(),
            "checkpoint_selection": "pretrain_val_loss_only",
            "h2000_included": True,
            "immutable_checkpoint": True,
            "checkpoint_sha256": immutable[tag]["sha256_verified"],
            "checkpoint_epoch": immutable[tag]["checkpoint_epoch"],
            "pretrain_best_val_loss": immutable[tag]["best_val_loss"],
        })
    missing = [r for r in ckpt_rows_new if not r["checkpoint_exists"]]
    if missing:
        raise FileNotFoundError(f"missing H2000 checkpoints: {missing}")

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    print(f"loading FACED from {faced.FACED_LMDB}", flush=True)
    X, y, subj, split, segment_id, item_index, dataset_rows = faced.load_faced_lmdb(faced.FACED_LMDB)
    faced.write_csv(out / "faced_dataset_manifest.csv", dataset_rows)
    faced.write_csv(out / "faced_split_manifest.csv", faced.split_manifest_rows(y, subj, split, item_index))
    print(f"FACED loaded shape={X.shape} labels={sorted(np.unique(y).tolist())}", flush=True)

    records = []
    feature_rows_new = []
    for tag in cells:
        ckpt = str(Path(args.ckpt_root) / tag / "best.pth")
        sha_before = sha256_file(Path(ckpt).resolve())
        if sha_before != immutable[tag]["sha256_verified"]:
            raise RuntimeError(f"{tag} immutable SHA changed before feature extraction")
        print(f"FACED D2-2 audit tag={tag} ckpt={ckpt}", flush=True)
        model, loaded = faced.build_encoder(tag, ckpt, device)
        det = faced.deterministic_batch_check(model, X, device)
        feat = faced.extract_features(model, X, device, args.batch_size)
        sha_after = sha256_file(Path(ckpt).resolve())
        if sha_after != sha_before:
            raise RuntimeError(f"{tag} immutable SHA changed during feature extraction")
        feature_rows_new.append({
            "tag": tag,
            "checkpoint": ckpt,
            "loaded": str(loaded),
            "feature_shape": "x".join(map(str, feat.shape)),
            "feature_dim": int(feat.shape[1]),
            "feature_sha16": hashlib.sha256(feat[: min(len(feat), 128)].astype(np.float32).tobytes()).hexdigest()[:16],
            "determinism_max_abs_diff": det,
            "checkpoint_sha256": sha_after,
            "immutable_checkpoint": True,
        })
        rec = faced.audit_one(tag, ckpt, feat, y, subj, split, item_index, args.pca_dims, random_source_val_kappa)
        rec["h2000_included"] = True
        records.append(rec)
        print(
            f"{tag}: target_kappa={rec['target_kappa']:.4f} target_bacc={rec['target_bacc']:.4f} "
            f"source_val_kappa={rec['source_val_kappa']:.4f} L1={rec['l1_pairwise_subject_bacc_mean']:.4f}",
            flush=True,
        )

    task_rows_new = [row_subset(r, TASK_FIELDS) for r in records]
    l1_rows_new = [row_subset(r, L1_FIELDS) for r in records]
    l4_rows_new = [row_subset(r, L4_FIELDS) for r in records]
    l5_rows_new = [
        {k: v for k, v in r.items() if k.startswith("l5_") or k in {"tag", "budget_h", "seed", "task_gate_pass", "l4_l5_l6_status"}}
        for r in records
    ]
    l6_rows_new = [row_subset(r, L6_FIELDS) for r in records]

    task_rows = merge_by_tag(task_existing, task_rows_new)
    l1_rows = merge_by_tag(l1_existing, l1_rows_new)
    write_csv(out / "faced_task_performance.csv", task_rows)
    write_csv(out / "faced_pairwise_subject_separability.csv", l1_rows)
    write_csv(out / "faced_l4_task_alignment.csv", merge_by_tag(l4_existing, l4_rows_new))
    write_csv(out / "faced_l5_replay.csv", merge_by_tag(l5_existing, l5_rows_new))
    write_csv(out / "faced_l6_target_consequence.csv", merge_by_tag(l6_existing, l6_rows_new))
    write_csv(out / "faced_feature_manifest.csv", merge_by_tag(feature_existing, feature_rows_new))
    write_csv(out / "faced_checkpoint_manifest.csv", merge_by_tag(ckpt_rows_existing, ckpt_rows_new))

    summary = build_summary(task_rows, l1_rows, time.time() - t0)
    summary["h2000_checkpoint_provenance"] = "immutable_sha256_pinned"
    summary["h2000_checkpoint_sha256_by_tag"] = {
        tag: immutable[tag]["sha256_verified"] for tag in cells
    }
    summary["historical_mutable_d2_2_superseded"] = True
    faced.write_json(out / "faced_budget_summary.json", summary)
    faced.write_json(out / "faced_target_label_firewall.json", {
        "target_labels_in_pca_fit": False,
        "target_labels_in_head_fit": False,
        "target_labels_in_checkpoint_selection": False,
        "target_labels_in_subject_subspace": False,
        "target_labels_in_l5_rank_or_null_selection": False,
        "target_labels_in_normalization_selection": False,
        "target_labels_final_scoring_only": True,
        "source_train_subjects": "1-80",
        "source_val_subjects": "81-100",
        "target_test_subjects": "101-123",
        "checkpoint_selection": "pretrain_val_loss_only",
        "normalization": "per_channel_per_1s_patch_zscore",
        "d2_2_reused_reference_rows": ["random", "released"],
        "h2000_immutable_checkpoints": True,
        "h2000_checkpoint_sha256_by_tag": {
            tag: immutable[tag]["sha256_verified"] for tag in cells
        },
    })
    update_notes(out / "faced_verifier_notes.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
