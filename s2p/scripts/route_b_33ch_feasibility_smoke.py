#!/usr/bin/env python
"""S2P 9D Route B: 33-channel CBraMod-only feasibility + tiny smoke.

This is not training. It performs metadata/window-count feasibility for the
fixed n_channels==33 processed TUEG subset, then runs a single synthetic
B x 33 x 30 x 200 native-CBraMod forward/loss/checkpoint/feature smoke.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

S2P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S2P / "s2p" / "scripts"))
CBRAMOD = Path(os.path.expanduser("~/eeg2025/CBraMod"))
sys.path.insert(0, str(CBRAMOD))

import tueg_subject_loader as L
from models.cbramod import CBraMod
from utils.util import generate_mask


TUEG = "/projects/EEG-foundation-model/datalake/processed/4704743c/TUEG"
ROUTE_B_BUDGETS = [500, 1000, 2000, 4000]
MIN_W = L.cap_windows_for(0.25)
N_VAL = 128
VAL_CAP_W = 24
BASE_200H_WALL_H = 4.5
WALLTIME_LIMIT_H = 96.0
WALLTIME_SAFETY = 1.15


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


def gini(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    return float((2.0 * np.sum(np.arange(1, n + 1) * arr) / (n * np.sum(arr))) - ((n + 1.0) / n))


def channel_hash(channels):
    return hashlib.sha256(json.dumps(channels, separators=(",", ":")).encode()).hexdigest()[:16]


def classify_reference(channels):
    eeg = [c for c in channels if c.startswith("EEG ")]
    if eeg and all(c.endswith("-LE") for c in eeg):
        return "LE"
    if eeg and all(c.endswith("-REF") for c in eeg):
        return "REF"
    return "mixed_or_nonstandard"


def fixed_val(sw):
    cand = np.sort(sw[sw >= VAL_CAP_W].index.to_numpy())
    if len(cand) < N_VAL:
        raise RuntimeError(f"not enough 33ch val subjects: {len(cand)} < {N_VAL}")
    rng = np.random.default_rng(920000)
    return np.sort(rng.choice(cand, N_VAL, replace=False))


def exact_highcoverage(sw_train, hours):
    wt = int(round(hours * 120))
    eligible = sw_train[sw_train >= MIN_W]
    upper = min(int(len(eligible)), int(wt // MIN_W))
    for n_subjects in range(upper, 0, -1):
        base = wt // n_subjects
        rem = wt - base * n_subjects
        need = base + (1 if rem else 0)
        pool_size = int((sw_train >= need).sum())
        if pool_size >= n_subjects:
            return True, int(n_subjects), int(need), pool_size
    return False, None, None, None


def compute_estimate(hours):
    est = BASE_200H_WALL_H * (hours / 200.0) * WALLTIME_SAFETY * (33.0 / 19.0)
    return round(est, 2), bool(est <= WALLTIME_LIMIT_H)


def metadata_audit(out):
    meta = pd.read_parquet(f"{TUEG}/metadata.parquet")
    meta = meta[meta["n_channels"] == 33].copy()
    meta["avail_w"] = (meta["n_timepoints"] // L.WLEN).astype(int)
    meta = meta[meta["avail_w"] > 0].copy()
    meta["hours"] = meta["avail_w"] * L.WIN_H
    meta["channels_list"] = meta["channels"].map(json.loads)
    meta["channel_order_hash"] = meta["channels_list"].map(channel_hash)
    meta["reference_scheme"] = meta["channels_list"].map(classify_reference)

    sw = meta.groupby("subject")["avail_w"].sum().sort_index()
    val = fixed_val(sw)
    sw_train = sw.drop(labels=[s for s in val if s in sw.index], errors="ignore")
    total_train_w = int(sw_train.sum())

    order_rows = []
    for chstr, count in meta["channels"].value_counts().items():
        channels = json.loads(chstr)
        order_rows.append({
            "channel_order_hash": channel_hash(channels),
            "n_recordings": int(count),
            "n_channels": len(channels),
            "reference_scheme": classify_reference(channels),
            "first_channels": "|".join(channels[:5]),
            "last_channels": "|".join(channels[-5:]),
            "channels_json": json.dumps(channels, separators=(",", ":")),
        })
    write_csv(out / "route_b_33ch_channel_order_diagnostics.csv", order_rows)

    values = sw_train.to_numpy(dtype=float)
    sorted_desc = np.sort(values)[::-1]
    top1 = max(1, int(np.ceil(0.01 * len(sorted_desc))))
    top5 = max(1, int(np.ceil(0.05 * len(sorted_desc))))
    pop = {
        "corpus": "TUEG_processed_exact_33ch",
        "n_recordings": int(len(meta)),
        "n_subjects": int(len(sw)),
        "n_train_subjects_after_val": int(len(sw_train)),
        "n_val_subjects": int(len(val)),
        "total_usable_windows": int(sw.sum()),
        "total_usable_hours": round(float(sw.sum() * L.WIN_H), 3),
        "train_windows_after_val": total_train_w,
        "train_hours_after_val": round(float(total_train_w * L.WIN_H), 3),
        "eligible_subjects_min_exposure_after_val": int((sw_train >= MIN_W).sum()),
        "subject_window_median": float(np.median(values)),
        "subject_window_p90": float(np.percentile(values, 90)),
        "subject_window_p99": float(np.percentile(values, 99)),
        "subject_window_max": int(values.max()),
        "subject_window_gini": round(float(gini(values)), 6),
        "top_1pct_subject_window_share": round(float(sorted_desc[:top1].sum() / sorted_desc.sum()), 6),
        "top_5pct_subject_window_share": round(float(sorted_desc[:top5].sum() / sorted_desc.sum()), 6),
        "unique_channel_orders": int(meta["channel_order_hash"].nunique()),
        "reference_scheme_counts": meta["reference_scheme"].value_counts().to_dict(),
        "channel_order_consistent": bool(meta["channel_order_hash"].nunique() == 1),
        "requires_channel_order_grouping_or_canonicalization": bool(meta["channel_order_hash"].nunique() > 1),
    }
    write_csv(out / "route_b_33ch_population_diagnostics.csv", [pop])
    dump_json(out / "route_b_33ch_population_diagnostics.json", pop)

    rows = []
    for h in ROUTE_B_BUDGETS:
        wt = int(round(h * 120))
        data_ok = bool(total_train_w >= wt)
        exact_hc_ok, n_hc, need_hc, pool_hc = exact_highcoverage(sw_train, h)
        est, compute_ok = compute_estimate(h)
        rows.append({
            "budget_h": h,
            "target_windows": wt,
            "target_hours": h,
            "data_volume_exact_window_feasible": data_ok,
            "exact_highcoverage_feasible": exact_hc_ok,
            "exact_highcoverage_n_subjects": n_hc,
            "exact_highcoverage_need_windows": need_hc,
            "exact_highcoverage_pool_size": pool_hc,
            "subject_disjoint_pretrain_val_feasible": True,
            "train_windows_after_val": total_train_w,
            "eligible_subjects_min_exposure_after_val": int((sw_train >= MIN_W).sum()),
            "estimated_wall_h_33ch": est,
            "planned_time_h": WALLTIME_LIMIT_H,
            "compute_budget_acceptable": compute_ok,
            "overall_route_b_metadata_feasible": bool(data_ok and compute_ok),
        })
    write_csv(out / "route_b_33ch_feasibility.csv", rows)
    return pop, rows


def smoke(out, batch_size):
    torch.manual_seed(940033)
    np.random.seed(940033)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8).to(device)
    model.train()
    x = torch.randn(batch_size, 33, 30, 200, device=device)
    mask = generate_mask(batch_size, 33, 30, mask_ratio=0.5, device=device)
    y = model(x, mask=mask)
    loss = torch.nn.functional.mse_loss(y[mask == 1], x[mask == 1])
    loss.backward()
    grad_finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())

    with tempfile.NamedTemporaryFile(suffix=".pth", dir=str(out), delete=False) as tmp:
        ckpt_path = Path(tmp.name)
    torch.save({"model_state": model.state_dict(), "input_shape": list(x.shape)}, ckpt_path)
    reloaded = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8).to(device)
    reloaded.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    ckpt_path.unlink(missing_ok=True)

    reloaded.eval()
    with torch.no_grad():
        patch_emb = reloaded.patch_embedding(x, mask=mask)
        feats = reloaded.encoder(patch_emb)
        feature_probe = feats.mean(dim=(1, 2)).detach().cpu().numpy().astype(np.float32)
    feature_sha = hashlib.sha256(feature_probe.tobytes()).hexdigest()[:16]

    trainer_path = CBRAMOD / "pretrain_trainer.py"
    trainer_text = trainer_path.read_text()
    hardcode_detected = "input_size=(1, 19, 30, 200)" in trainer_text and "(19, 30, 200)" in trainer_text

    result = {
        "smoke_passed": bool(
            list(x.shape) == [batch_size, 33, 30, 200]
            and list(mask.shape) == [batch_size, 33, 30]
            and list(y.shape) == [batch_size, 33, 30, 200]
            and torch.isfinite(y).all().item()
            and torch.isfinite(loss).item()
            and grad_finite
        ),
        "device": str(device),
        "input_shape": list(x.shape),
        "mask_shape": list(mask.shape),
        "output_shape": list(y.shape),
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "output_finite": bool(torch.isfinite(y).all().item()),
        "gradients_finite": bool(grad_finite),
        "checkpoint_save_reload": True,
        "checkpoint_retained": False,
        "feature_dump_shape": list(feature_probe.shape),
        "feature_dump_sha16": feature_sha,
        "native_generate_mask_used": True,
        "native_cbramod_used": True,
        "summary_ptflops_19ch_reporting_hardcode_detected": bool(hardcode_detected),
        "summary_ptflops_bypassed_for_smoke": True,
        "target_labels_used": False,
        "training_launched": False,
    }
    dump_json(out / "route_b_33ch_smoke.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/s2p_budget_floor_calibration_v2/relaxation_audit")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pop, feas = metadata_audit(out)
    smk = smoke(out, args.batch_size)
    summary = {
        "route": "B",
        "phase": "9D_route_B_33ch_feasibility_smoke",
        "corpus": "TUEG_processed_exact_33ch",
        "metadata_feasible_budgets_h": [int(r["budget_h"]) for r in feas if r["overall_route_b_metadata_feasible"]],
        "data_volume_feasible_budgets_h": [int(r["budget_h"]) for r in feas if r["data_volume_exact_window_feasible"]],
        "exact_highcoverage_feasible_budgets_h": [int(r["budget_h"]) for r in feas if r["exact_highcoverage_feasible"]],
        "smoke_passed": bool(smk["smoke_passed"]),
        "training_requires_pm_approval": True,
        "training_launched": False,
        "target_labels_used": False,
        "claim_allowed": "CBraMod 33-channel full-corpus high-budget calibration if PM approves training.",
        "claim_forbidden": "direct CodeBrain 19-common comparison or channel-invariance claim.",
        "population_summary": pop,
    }
    dump_json(out / "route_b_33ch_feasibility_smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
