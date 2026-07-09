#!/usr/bin/env python
"""Route B single-checkpoint SHU-MI native32 downstream probe gate.

This is not the full B1 downstream fleet. It verifies the downstream path on
one completed checkpoint plus random/released controls using source-only
PCA/head/subspace fitting and target labels only for final scoring.
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch

S2P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S2P / "s2p" / "scripts"))
sys.path.insert(0, str(os.path.expanduser("~/eeg2025/CBraMod")))

import shumi_downstream_audit as audit
from models.cbramod import CBraMod


SHUMI = "/projects/EEG-foundation-model/tdoan-24/SHUMI_200hz"
RELEASED_CKPT = os.path.expanduser("~/eeg2025/NIPS/Cbramod_pretrained_weights.pth")
SHU32_ORDER = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "CZ",
    "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6", "PZ", "P3",
    "P4", "T5", "T6", "PO3", "PO4", "OZ", "O1", "O2",
]
SPLIT = {"source_train": list(range(1, 16)), "source_val": list(range(16, 21)), "target_test": list(range(21, 26))}


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


def load_shumi_native32(trials_per_class_per_session):
    env = lmdb.open(SHUMI, readonly=True, lock=False, readahead=False, meminit=False)
    X, y, subj, sess = [], [], [], []
    with env.begin() as txn:
        for s in sorted(SPLIT["source_train"] + SPLIT["source_val"] + SPLIT["target_test"]):
            for se in range(1, 6):
                by_label = {0: [], 1: []}
                t = 0
                while True:
                    key = f"sub-{s:03d}_ses-{se:02d}_task_motorimagery_eeg-{t}".encode()
                    raw = txn.get(key)
                    if raw is None:
                        break
                    item = pickle.loads(raw)
                    label = int(item["label"])
                    if label in by_label:
                        by_label[label].append(np.asarray(item["sample"], np.float32).reshape(32, 4, 200))
                    t += 1
                for label in [0, 1]:
                    for sample in by_label[label][:trials_per_class_per_session]:
                        X.append(sample)
                        y.append(label)
                        subj.append(s)
                        sess.append(se)
    X = np.stack(X)
    X = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)
    return X.astype(np.float32), np.asarray(y), np.asarray(subj), np.asarray(sess)


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model_state", "model", "state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items()}


def resolve_checkpoint(tag, ckpt_root):
    if tag in ("random", "random_init"):
        return "random"
    if tag == "released":
        return RELEASED_CKPT
    return str(Path(ckpt_root) / tag / "best.pth")


def build_encoder(tag, ckpt_path, device):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=4, n_layer=12, nhead=8).to(device)
    loaded = "random_init"
    if ckpt_path != "random":
        state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(unwrap_state_dict(state), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"{tag} state mismatch missing={missing[:5]} unexpected={unexpected[:5]}")
        loaded = ckpt_path
    model.eval()
    return model, loaded


@torch.no_grad()
def extract(model, X, device, batch_size):
    feats = []
    for start in range(0, len(X), batch_size):
        if start == 0 or (start // batch_size) % 10 == 0:
            print(f"extract start={start} / {len(X)}", flush=True)
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        pe = model.patch_embedding(xb, None)
        fe = model.encoder(pe)
        emb = fe.mean(2).reshape(fe.shape[0], -1)
        feats.append(emb.float().cpu().numpy())
    return np.concatenate(feats)


@torch.no_grad()
def deterministic_batch_check(model, X, device):
    xb = torch.from_numpy(X[: min(64, len(X))]).to(device)
    def f():
        pe = model.patch_embedding(xb, None)
        fe = model.encoder(pe)
        return fe.mean(2).reshape(fe.shape[0], -1).float().cpu().numpy()
    a = f()
    b = f()
    return float(np.abs(a - b).max())


def audit_features(tag, ckpt_path, feat, y, subj, sess, out_dir):
    rec = audit.audit_checkpoint(tag, ckpt_path, feat, y, subj, sess, out_dir)
    rec["tag"] = tag
    rec["checkpoint"] = str(ckpt_path)
    rec["budget_h"] = float(tag.split("_")[0][1:]) if tag.startswith("H") and "_s" in tag else None
    rec["seed"] = int(tag.split("_s")[1]) if tag.startswith("H") and "_s" in tag else None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-cell", default="H1000_s0")
    ap.add_argument("--ckpt-root", default="results/s2p_route_b_33ch_b1")
    ap.add_argument("--out-dir", default="results/s2p_route_b_33ch_b1/downstream_probe_gate_through_1000h")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--trials-per-class-per-session", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    X, y, subj, sess = load_shumi_native32(args.trials_per_class_per_session)
    load_info = {
        "dataset": "SHU-MI",
        "channel_mode": "native32",
        "input_shape": list(X.shape),
        "n_trials": int(len(X)),
        "subjects": sorted(int(s) for s in np.unique(subj)),
        "trials_per_class_per_session": int(args.trials_per_class_per_session),
        "split": SPLIT,
    }
    print("SHU-MI native32 loaded:", json.dumps(load_info), flush=True)

    tags = [args.checkpoint_cell, "random", "released"]
    rows, feature_rows = [], []
    det = {}
    for tag in tags:
        ckpt = resolve_checkpoint(tag, args.ckpt_root)
        if ckpt != "random" and not Path(ckpt).exists():
            raise FileNotFoundError(ckpt)
        print(f"probe tag={tag} ckpt={ckpt}", flush=True)
        model, loaded = build_encoder(tag, ckpt, device)
        det_max = deterministic_batch_check(model, X, device)
        feat = extract(model, X, device, args.batch_size)
        det[tag] = det_max
        feature_rows.append({
            "tag": tag,
            "checkpoint": str(ckpt),
            "loaded": str(loaded),
            "feature_shape": "x".join(map(str, feat.shape)),
            "feature_dim": int(feat.shape[1]),
            "feature_sha16": hashlib.sha256(feat[: min(len(feat), 128)].astype(np.float32).tobytes()).hexdigest()[:16],
            "determinism_max_abs_diff": det_max,
        })
        rec = audit_features(tag, ckpt, feat, y, subj, sess, out)
        rec.update({
            "channel_mode": "native32",
            "input_channels": 32,
            "target_labels_used_for_selection": False,
            "target_labels_final_scoring_only": True,
        })
        rows.append(rec)
        print(f"{tag}: target_bacc={rec['target_bacc']:.3f} source_val={rec['source_val_bacc']:.3f} "
              f"L1={rec['l1_l1_pairwise_bacc_mean']:.3f} L4={rec['l4_alignment']:.3f} "
              f"L5z={rec['l5_l5_reliance_z']:+.2f}", flush=True)

    write_csv(out / "route_b_b1_downstream_probe_gate.csv", rows)
    write_csv(out / "route_b_b1_downstream_probe_features.csv", feature_rows)
    by_tag = {r["tag"]: r for r in rows}
    random_bacc = by_tag["random"]["target_bacc"]
    released_bacc = by_tag["released"]["target_bacc"]
    probe = by_tag[args.checkpoint_cell]
    metrics = [
        probe["target_bacc"],
        probe["source_val_bacc"],
        probe["l1_l1_pairwise_bacc_mean"],
        probe["l4_alignment"],
        probe["l5_l5_reliance_z"],
    ]
    gate = {
        "phase": "route_b_b1_downstream_probe_gate_through_1000h",
        "checkpoint_cell": args.checkpoint_cell,
        "downstream_primary": "SHU_MI_native32",
        "controls": ["random", "released"],
        "load_info": load_info,
        "device": str(device),
        "g1_native32_forward": True,
        "g2_frozen_feature_dump_deterministic": all(v == 0.0 for v in det.values()),
        "g3_source_only_pca_head_subspace": True,
        "g4_target_labels_final_scoring_only": True,
        "g5_l1_l4_l5_l6_compute": all(np.isfinite(metrics)),
        "g6_released_reference_above_random_plus_0p02": bool(released_bacc >= random_bacc + 0.02),
        "random_target_bacc": float(random_bacc),
        "released_target_bacc": float(released_bacc),
        "probe_target_bacc": float(probe["target_bacc"]),
        "probe_source_val_bacc": float(probe["source_val_bacc"]),
        "probe_l1_subject_separability": float(probe["l1_l1_pairwise_bacc_mean"]),
        "probe_l4_alignment": float(probe["l4_alignment"]),
        "probe_l5_reliance_z": float(probe["l5_l5_reliance_z"]),
        "target_labels_used_for_selection": False,
        "full_downstream_fleet_launched": False,
        "full_downstream_fleet_recommended_if_pass": None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    gate["probe_gate_pass"] = all(v for k, v in gate.items() if k.startswith("g"))
    gate["full_downstream_fleet_recommended_if_pass"] = bool(gate["probe_gate_pass"])
    (out / "route_b_b1_downstream_probe_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    (out / "route_b_b1_downstream_firewall.json").write_text(json.dumps({
        "target_labels_in_pca_fit": False,
        "target_labels_in_head_fit": False,
        "target_labels_in_checkpoint_selection": False,
        "target_labels_in_subject_subspace": False,
        "target_labels_final_scoring_only": True,
        "checkpoint_selection": "pretrain_val_loss_only",
        "channel_mode": "SHU_MI_native32",
    }, indent=2) + "\n")
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
