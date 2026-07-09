#!/usr/bin/env python
"""Route B FACED frozen-probe downstream audit.

This is downstream-only. It evaluates completed Route B CBraMod checkpoints on
FACED native32 with source-only PCA/head/subspace fitting. FACED test labels are
used only for final scoring.
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
)

S2P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(os.path.expanduser("~/eeg2025/CBraMod")))
from models.cbramod import CBraMod


FACED_LMDB = "/projects/EEG-foundation-model/FACED_data/processed"
RELEASED_CKPT = os.path.expanduser("~/eeg2025/NIPS/Cbramod_pretrained_weights.pth")
B1_CKPT_ROOT = "/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1"
N_CLASSES = 9
LABELS = list(range(N_CLASSES))
SPLIT = {
    "source_train": list(range(1, 81)),
    "source_val": list(range(81, 101)),
    "target_test": list(range(101, 124)),
}
PHASE_CELLS = {
    "probe": ["random", "released", "H1000_s0", "H1000_s1"],
    "fleet": [
        "random",
        "released",
        "H200_s0",
        "H200_s1",
        "H500_s0",
        "H500_s1",
        "H1000_s0",
        "H1000_s1",
    ],
}
CHANNEL_TOKENS = [f"FACED_NATIVE32_ARRAY_INDEX_{i:02d}" for i in range(32)]
CHANNEL_HASH = hashlib.sha256(json.dumps(CHANNEL_TOKENS, separators=(",", ":")).encode()).hexdigest()[:16]


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


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def parse_faced_key(key):
    if isinstance(key, bytes):
        key = key.decode()
    m = re.match(r"sub(?P<sub>\d+)\.pkl-(?P<label>\d+)-(?P<seg>\d+)$", key)
    if not m:
        raise ValueError(f"unexpected FACED key format: {key}")
    return {
        "key": key,
        "subject": int(m.group("sub")) + 1,
        "condition_id": int(m.group("label")),
        "segment_id": int(m.group("seg")),
    }


def expected_split_name(subject):
    if subject in SPLIT["source_train"]:
        return "source_train"
    if subject in SPLIT["source_val"]:
        return "source_val"
    if subject in SPLIT["target_test"]:
        return "target_test"
    raise ValueError(f"subject outside fixed FACED split: {subject}")


def load_faced_lmdb(path):
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    X, y, subj, split, segment_id, item_index = [], [], [], [], [], []
    dataset_rows = []
    with env.begin() as txn:
        keys_by_split = pickle.loads(txn.get(b"__keys__"))
        if set(keys_by_split) != {"train", "val", "test"}:
            raise RuntimeError(f"unexpected FACED split keys: {sorted(keys_by_split)}")
        lmdb_to_protocol = {"train": "source_train", "val": "source_val", "test": "target_test"}
        for lmdb_split in ["train", "val", "test"]:
            protocol_split = lmdb_to_protocol[lmdb_split]
            keys = keys_by_split[lmdb_split]
            per_subject_ordinal = {}
            label_counts = {str(k): 0 for k in LABELS}
            condition_counts = {}
            subjects_seen = []
            shapes_seen = set()
            dtypes_seen = set()
            for key in keys:
                parsed = parse_faced_key(key)
                if expected_split_name(parsed["subject"]) != protocol_split:
                    raise RuntimeError(f"FACED key {parsed['key']} violates fixed subject split")
                raw = txn.get(key if isinstance(key, bytes) else key.encode())
                if raw is None:
                    raise RuntimeError(f"missing FACED sample: {parsed['key']}")
                obj = pickle.loads(raw)
                sample = np.asarray(obj["sample"], dtype=np.float32)
                label = int(obj["label"])
                if sample.shape != (32, 10, 200):
                    raise RuntimeError(f"FACED sample {parsed['key']} shape {sample.shape} != (32,10,200)")
                if label not in LABELS:
                    raise RuntimeError(f"FACED label outside 0..8: {label}")
                ordinal = per_subject_ordinal.get(parsed["subject"], 0)
                per_subject_ordinal[parsed["subject"]] = ordinal + 1
                X.append(sample)
                y.append(label)
                subj.append(parsed["subject"])
                split.append(protocol_split)
                segment_id.append(parsed["segment_id"])
                item_index.append(ordinal)
                label_counts[str(label)] += 1
                condition_counts[str(parsed["condition_id"])] = condition_counts.get(str(parsed["condition_id"]), 0) + 1
                subjects_seen.append(parsed["subject"])
                shapes_seen.add("x".join(map(str, sample.shape)))
                dtypes_seen.add(str(sample.dtype))
            subjects_seen = sorted(set(subjects_seen))
            dataset_rows.append({
                "dataset": "FACED",
                "lmdb_path": path,
                "lmdb_split": lmdb_split,
                "protocol_split": protocol_split,
                "n_segments": len(keys),
                "n_subjects": len(subjects_seen),
                "subject_min": min(subjects_seen),
                "subject_max": max(subjects_seen),
                "expected_subjects": f"{SPLIT[protocol_split][0]}-{SPLIT[protocol_split][-1]}",
                "sample_shape": ";".join(sorted(shapes_seen)),
                "dtype_after_load": ";".join(sorted(dtypes_seen)),
                "sfreq_hz": 200,
                "window_s": 10,
                "n_classes": N_CLASSES,
                "label_counts_json": json.dumps(label_counts, sort_keys=True),
                "condition_counts_json": json.dumps(condition_counts, sort_keys=True),
            })
    X = np.stack(X).astype(np.float32)
    X = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)
    return (
        X,
        np.asarray(y, dtype=int),
        np.asarray(subj, dtype=int),
        np.asarray(split),
        np.asarray(segment_id, dtype=int),
        np.asarray(item_index, dtype=int),
        dataset_rows,
    )


def split_manifest_rows(y, subj, split, item_index):
    rows = []
    for split_name in ["source_train", "source_val", "target_test"]:
        for s in SPLIT[split_name]:
            m = (split == split_name) & (subj == s)
            label_counts = {str(k): int(((y == k) & m).sum()) for k in LABELS}
            rows.append({
                "dataset": "FACED",
                "split": split_name,
                "subject": int(s),
                "n_segments": int(m.sum()),
                "min_item_index": int(item_index[m].min()) if m.any() else None,
                "max_item_index": int(item_index[m].max()) if m.any() else None,
                "label_counts_json": json.dumps(label_counts, sort_keys=True),
            })
    return rows


def channel_manifest():
    return {
        "dataset": "FACED",
        "channel_mode": "native32",
        "channel_count": 32,
        "channel_names_available_in_local_lmdb": False,
        "channel_order_tokens_are_array_indices": True,
        "channel_order_tokens": CHANNEL_TOKENS,
        "channel_order_sha256_16": CHANNEL_HASH,
        "multiple_channel_orders_observed": False,
        "evidence": [
            "All selected FACED LMDB samples have shape (32,10,200).",
            "Local FACED_data tree exposes processed LMDB and per-subject pkl files but no channel-name metadata.",
            "No interpolation, zero-padding, or unrecorded channel reorder is applied.",
        ],
        "metadata_search_paths": [
            "/projects/EEG-foundation-model/FACED_data",
            "/projects/EEG-foundation-model/FACED_data/processed",
            "/projects/EEG-foundation-model/FACED_data/Processed_data",
        ],
    }


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
    if tag.startswith("H2000"):
        raise RuntimeError("H2000 FACED downstream is explicitly held and must not be evaluated here")
    return str(Path(ckpt_root) / tag / "best.pth")


def budget_seed(tag):
    if tag.startswith("H") and "_s" in tag:
        left, seed = tag.split("_s")
        return float(left[1:]), int(seed)
    return None, None


def checkpoint_manifest(tags, ckpt_root):
    rows = []
    for tag in tags:
        ckpt = resolve_checkpoint(tag, ckpt_root)
        budget_h, seed = budget_seed(tag)
        rows.append({
            "tag": tag,
            "checkpoint": ckpt,
            "budget_h": budget_h,
            "seed": seed,
            "reference": tag in ("random", "released"),
            "checkpoint_exists": ckpt == "random" or Path(ckpt).exists(),
            "checkpoint_selection": "pretrain_val_loss_only" if tag.startswith("H") else "reference",
            "h2000_included": False,
        })
    return rows


def build_encoder(tag, ckpt_path, device):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=10,
        n_layer=12,
        nhead=8,
    ).to(device)
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
def extract_features(model, X, device, batch_size):
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
    xb = torch.from_numpy(X[: min(32, len(X))]).to(device)

    def run_once():
        pe = model.patch_embedding(xb, None)
        fe = model.encoder(pe)
        return fe.mean(2).reshape(fe.shape[0], -1).float().cpu().numpy()

    a = run_once()
    b = run_once()
    return float(np.abs(a - b).max())


def masks(split):
    return {
        "source_train": split == "source_train",
        "source_val": split == "source_val",
        "target_test": split == "target_test",
    }


def metric_block(y_true, pred, prob, prefix):
    return {
        f"{prefix}_kappa": float(cohen_kappa_score(y_true, pred)),
        f"{prefix}_weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        f"{prefix}_macro_f1": float(f1_score(y_true, pred, average="macro")),
        f"{prefix}_bacc": float(balanced_accuracy_score(y_true, pred)),
        f"{prefix}_nll": float(log_loss(y_true, prob, labels=LABELS)),
    }


def task_probe(feat, y, split, pca_dims):
    m = masks(split)
    best = None
    for n_pca in pca_dims:
        n = min(int(n_pca), feat[m["source_train"]].shape[0] - 1, feat.shape[1])
        pca = PCA(n_components=n, svd_solver="randomized", random_state=0)
        Ztr = pca.fit_transform(feat[m["source_train"]])
        Zva = pca.transform(feat[m["source_val"]])
        for c in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=c, max_iter=2000, n_jobs=1).fit(Ztr, y[m["source_train"]])
            pred_va = clf.predict(Zva)
            kappa = cohen_kappa_score(y[m["source_val"]], pred_va)
            bacc = balanced_accuracy_score(y[m["source_val"]], pred_va)
            score = (kappa, bacc)
            if best is None or score > best["score"]:
                best = {"score": score, "n_pca": n, "C": c, "pca": pca, "clf": clf}
    pca = best["pca"]
    clf = best["clf"]
    Ztr = pca.transform(feat[m["source_train"]])
    Zva = pca.transform(feat[m["source_val"]])
    Zte = pca.transform(feat[m["target_test"]])
    pred_va = clf.predict(Zva)
    prob_va = clf.predict_proba(Zva)
    pred_te = clf.predict(Zte)
    prob_te = clf.predict_proba(Zte)
    rec = {
        "selected_pca_dim": int(best["n_pca"]),
        "selected_C": float(best["C"]),
        "selection_metric": "source_val_cohen_kappa_then_bacc",
        "source_train_n": int(m["source_train"].sum()),
        "source_val_n": int(m["source_val"].sum()),
        "target_test_n": int(m["target_test"].sum()),
    }
    rec.update(metric_block(y[m["source_val"]], pred_va, prob_va, "source_val"))
    rec.update(metric_block(y[m["target_test"]], pred_te, prob_te, "target"))
    pack = {
        "pca": pca,
        "clf": clf,
        "Z": {"source_train": Ztr, "source_val": Zva, "target_test": Zte},
    }
    return rec, pack


def l1_pairwise_nearest_centroid(Z, subj, split, item_index, split_name):
    m = split == split_name
    subjects = sorted(int(s) for s in np.unique(subj[m]))
    accs = []
    for i, a in enumerate(subjects):
        for b in subjects[i + 1:]:
            pair = m & np.isin(subj, [a, b])
            holdout = (item_index % 5) == 0
            train = pair & ~holdout
            test = pair & holdout
            if test.sum() < 4 or len(np.unique(subj[train])) < 2:
                continue
            za = Z[train & (subj == a)].mean(0)
            zb = Z[train & (subj == b)].mean(0)
            zt = Z[test]
            pred_a = np.sum((zt - za) ** 2, axis=1) <= np.sum((zt - zb) ** 2, axis=1)
            pred = np.where(pred_a, a, b)
            accs.append(balanced_accuracy_score(subj[test], pred))
    return {
        "split": split_name,
        "l1_classifier": "nearest_centroid_source_pca_space_item_index_mod5_holdout",
        "l1_pairwise_subject_bacc_mean": float(np.mean(accs)) if accs else np.nan,
        "l1_pairwise_subject_bacc_sd": float(np.std(accs)) if accs else np.nan,
        "l1_n_pairs": int(len(accs)),
    }


def subject_subspace_source_pca(Z_source_train, subj_train, k=5):
    subjects = sorted(int(s) for s in np.unique(subj_train))
    means = np.stack([Z_source_train[subj_train == s].mean(0) for s in subjects])
    centered = means - means.mean(0)
    _, svals, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(k, vt.shape[0])
    denom = float((svals ** 2).sum()) + 1e-12
    return vt[:k], float(((svals[:k] ** 2).sum()) / denom)


def projection_energy(W, basis):
    denom = float((W ** 2).sum()) + 1e-12
    return float(((W @ basis.T) ** 2).sum() / denom)


def remove_subspace(Z, basis):
    return Z - (Z @ basis.T) @ basis


def score_z(clf, Z, y_true):
    pred = clf.predict(Z)
    prob = clf.predict_proba(Z)
    out = metric_block(y_true, pred, prob, "target")
    return out


def l4_l5_l6(pack, y, subj, split, n_null=50):
    m = masks(split)
    ztr = pack["Z"]["source_train"]
    zte = pack["Z"]["target_test"]
    yte = y[m["target_test"]]
    basis, ss_var = subject_subspace_source_pca(ztr, subj[m["source_train"]], k=5)
    clf = pack["clf"]
    l4 = {
        "subject_subspace_rank": int(basis.shape[0]),
        "subject_subspace_var_frac": float(ss_var),
        "l4_task_head_subject_subspace_energy": projection_energy(clf.coef_, basis),
    }
    base = score_z(clf, zte, yte)
    removed = score_z(clf, remove_subspace(zte, basis), yte)
    rng = np.random.default_rng(92014)
    null_delta_kappa = []
    null_delta_bacc = []
    dim, rank = basis.shape[1], basis.shape[0]
    for _ in range(n_null):
        q, _ = np.linalg.qr(rng.standard_normal((dim, rank)))
        null = score_z(clf, remove_subspace(zte, q.T), yte)
        null_delta_kappa.append(base["target_kappa"] - null["target_kappa"])
        null_delta_bacc.append(base["target_bacc"] - null["target_bacc"])
    null_delta_kappa = np.asarray(null_delta_kappa)
    null_delta_bacc = np.asarray(null_delta_bacc)
    subj_delta_kappa = base["target_kappa"] - removed["target_kappa"]
    subj_delta_bacc = base["target_bacc"] - removed["target_bacc"]
    l5 = {
        "l5_subject_removal_delta_kappa": float(subj_delta_kappa),
        "l5_subject_removal_delta_bacc": float(subj_delta_bacc),
        "l5_null_delta_kappa_mean": float(null_delta_kappa.mean()),
        "l5_null_delta_kappa_sd": float(null_delta_kappa.std()),
        "l5_null_delta_bacc_mean": float(null_delta_bacc.mean()),
        "l5_null_delta_bacc_sd": float(null_delta_bacc.std()),
        "l5_subject_removal_kappa_z": float((subj_delta_kappa - null_delta_kappa.mean()) / (null_delta_kappa.std() + 1e-9)),
        "l5_subject_removal_bacc_z": float((subj_delta_bacc - null_delta_bacc.mean()) / (null_delta_bacc.std() + 1e-9)),
        "l5_removed_variance_frac": float(((zte @ basis.T) ** 2).sum() / ((zte ** 2).sum() + 1e-12)),
    }
    l6 = {
        "l6_target_kappa_base": float(base["target_kappa"]),
        "l6_target_kappa_subject_removed": float(removed["target_kappa"]),
        "l6_target_bacc_base": float(base["target_bacc"]),
        "l6_target_bacc_subject_removed": float(removed["target_bacc"]),
        "l6_target_weighted_f1_base": float(base["target_weighted_f1"]),
        "l6_target_weighted_f1_subject_removed": float(removed["target_weighted_f1"]),
    }
    return l4, l5, l6


def audit_one(tag, ckpt_path, feat, y, subj, split, item_index, pca_dims, random_source_val_kappa=None):
    task, pack = task_probe(feat, y, split, pca_dims)
    pca_all = np.zeros((len(feat), int(task["selected_pca_dim"])), dtype=np.float32)
    pca_all[split == "source_train"] = pack["Z"]["source_train"]
    pca_all[split == "source_val"] = pack["pca"].transform(feat[split == "source_val"])
    pca_all[split == "target_test"] = pack["Z"]["target_test"]
    l1_source = l1_pairwise_nearest_centroid(pca_all, subj, split, item_index, "source_train")
    l1_target = l1_pairwise_nearest_centroid(pca_all, subj, split, item_index, "target_test")
    l4, l5, l6 = l4_l5_l6(pack, y, subj, split)
    budget_h, seed = budget_seed(tag)
    task_gate = None
    if random_source_val_kappa is not None and tag != "random":
        task_gate = bool(task["source_val_kappa"] >= random_source_val_kappa + 0.02 and task["source_val_kappa"] >= 0.05)
    status = "PASS" if task_gate else "WEAK_TASK_DIAGNOSTIC_ONLY"
    if tag in ("random", "released"):
        status = "REFERENCE"
    base = {
        "tag": tag,
        "checkpoint": str(ckpt_path),
        "budget_h": budget_h,
        "seed": seed,
        "downstream_dataset": "FACED",
        "channel_mode": "native32",
        "input_channels": 32,
        "input_shape": "32x10x200",
        "frozen_encoder": True,
        "fine_tuning_used": False,
        "codebrain_used": False,
        "h2000_included": False,
        "target_labels_used_for_selection": False,
        "task_gate_pass": task_gate,
        "l4_l5_l6_status": status,
    }
    return {**base, **task, **l1_source, **{f"target_{k}": v for k, v in l1_target.items()}, **l4, **l5, **l6}


def protocol_json(phase, cells, args):
    return {
        "phase": f"D2_{phase}",
        "downstream_dataset": "FACED",
        "protocol": "frozen_encoder_source_only_probe",
        "not_full_finetuning_reproduction": True,
        "model": "CBraMod",
        "route": "B_33ch_cbramod_only",
        "cells": cells,
        "train_subjects": "1-80",
        "val_subjects": "81-100",
        "test_subjects": "101-123",
        "input": {"channels": 32, "window_s": 10, "sfreq_hz": 200, "patch_size": 200, "patches": 10},
        "normalization": "per_channel_per_1s_patch_zscore",
        "primary_metric": "cohen_kappa",
        "secondary_metrics": ["weighted_f1", "balanced_accuracy", "macro_f1"],
        "pca_head_selection": "source_train_fit_source_val_select",
        "pca_dims": args.pca_dims,
        "target_labels_final_scoring_only": True,
        "forbidden": {
            "pretraining": False,
            "fine_tuning": False,
            "codebrain": False,
            "h2000_downstream": False,
            "target_labels_for_selection": False,
        },
    }


def verifier_notes(path, phase, probe_gate, random_ref, released_ref):
    lines = [
        "# FACED Frozen-Probe Verifier Notes",
        "",
        f"- Phase: D2-{phase}",
        "- Dataset: FACED native32 LMDB, 10s, 200Hz, 9 classes.",
        "- Split: train subjects 1-80, validation subjects 81-100, test subjects 101-123.",
        "- Encoder is frozen; no fine-tuning or pretraining is launched by this script.",
        "- PCA, classifier, subject subspace, and rank are source train/val only.",
        "- FACED test labels are used only for final scoring.",
        "- H2000, CodeBrain, and any extra dataset are excluded.",
        "- Channel-name metadata was not present in the selected local LMDB tree; the native array order is pinned by hash.",
    ]
    if random_ref and released_ref:
        lines.append(f"- Random target kappa: {random_ref.get('target_kappa'):.6f}; target bAcc: {random_ref.get('target_bacc'):.6f}.")
        lines.append(f"- Released target kappa: {released_ref.get('target_kappa'):.6f}; target bAcc: {released_ref.get('target_bacc'):.6f}.")
    if probe_gate is not None:
        lines.append(f"- D2-0 probe gate pass: {probe_gate}.")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["probe", "fleet"], default="probe")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--faced-lmdb", default=FACED_LMDB)
    ap.add_argument("--ckpt-root", default=B1_CKPT_ROOT)
    ap.add_argument("--out-dir", default="results/s2p_route_b_33ch_b1_faced")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--pca-dims", nargs="+", type=int, default=[32, 64, 128])
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = args.cells if args.cells else PHASE_CELLS[args.phase]
    if any(c.startswith("H2000") for c in cells):
        raise RuntimeError("H2000 FACED downstream is not approved for this run")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    write_json(out / "faced_downstream_protocol.json", protocol_json(args.phase, cells, args))
    write_json(out / "faced_channel_order_manifest.json", channel_manifest())
    ckpt_rows = checkpoint_manifest(cells, args.ckpt_root)
    write_csv(out / "faced_checkpoint_manifest.csv", ckpt_rows)
    missing = [r for r in ckpt_rows if not r["checkpoint_exists"]]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")

    print(f"loading FACED from {args.faced_lmdb}", flush=True)
    X, y, subj, split, segment_id, item_index, dataset_rows = load_faced_lmdb(args.faced_lmdb)
    write_csv(out / "faced_dataset_manifest.csv", dataset_rows)
    write_csv(out / "faced_split_manifest.csv", split_manifest_rows(y, subj, split, item_index))
    print(f"FACED loaded shape={X.shape} labels={sorted(np.unique(y).tolist())}", flush=True)

    records = []
    feature_rows = []
    random_source_val_kappa = None
    random_ref = None
    released_ref = None
    for tag in cells:
        ckpt = resolve_checkpoint(tag, args.ckpt_root)
        print(f"FACED audit tag={tag} ckpt={ckpt}", flush=True)
        model, loaded = build_encoder(tag, ckpt, device)
        det = deterministic_batch_check(model, X, device)
        feat = extract_features(model, X, device, args.batch_size)
        feature_rows.append({
            "tag": tag,
            "checkpoint": str(ckpt),
            "loaded": str(loaded),
            "feature_shape": "x".join(map(str, feat.shape)),
            "feature_dim": int(feat.shape[1]),
            "feature_sha16": hashlib.sha256(feat[: min(len(feat), 128)].astype(np.float32).tobytes()).hexdigest()[:16],
            "determinism_max_abs_diff": det,
        })
        rec = audit_one(tag, ckpt, feat, y, subj, split, item_index, args.pca_dims, random_source_val_kappa)
        records.append(rec)
        if tag == "random":
            random_source_val_kappa = float(rec["source_val_kappa"])
            random_ref = rec
        if tag == "released":
            released_ref = rec
        print(
            f"{tag}: target_kappa={rec['target_kappa']:.4f} target_bacc={rec['target_bacc']:.4f} "
            f"source_val_kappa={rec['source_val_kappa']:.4f} L1={rec['l1_pairwise_subject_bacc_mean']:.4f}",
            flush=True,
        )

    # Recompute task gates after random is known, preserving reference rows.
    if random_ref is not None:
        for rec in records:
            if rec["tag"] not in ("random", "released"):
                gate = bool(rec["source_val_kappa"] >= random_ref["source_val_kappa"] + 0.02 and rec["source_val_kappa"] >= 0.05)
                rec["task_gate_pass"] = gate
                rec["l4_l5_l6_status"] = "PASS" if gate else "WEAK_TASK_DIAGNOSTIC_ONLY"

    write_csv(out / "faced_feature_manifest.csv", feature_rows)
    write_csv(out / "faced_task_performance.csv", [
        {k: v for k, v in r.items() if k in {
            "tag", "checkpoint", "budget_h", "seed", "source_val_kappa", "source_val_weighted_f1",
            "source_val_macro_f1", "source_val_bacc", "source_val_nll", "target_kappa",
            "target_weighted_f1", "target_macro_f1", "target_bacc", "target_nll",
            "selected_pca_dim", "selected_C", "selection_metric", "task_gate_pass",
            "l4_l5_l6_status", "target_labels_used_for_selection", "fine_tuning_used",
            "codebrain_used", "h2000_included",
        }}
        for r in records
    ])
    write_csv(out / "faced_pairwise_subject_separability.csv", [
        {k: v for k, v in r.items() if k in {
            "tag", "budget_h", "seed", "split", "l1_classifier", "l1_pairwise_subject_bacc_mean",
            "l1_pairwise_subject_bacc_sd", "l1_n_pairs", "target_split",
            "target_l1_classifier", "target_l1_pairwise_subject_bacc_mean",
            "target_l1_pairwise_subject_bacc_sd", "target_l1_n_pairs",
        }}
        for r in records
    ])
    write_csv(out / "faced_l4_task_alignment.csv", [
        {k: v for k, v in r.items() if k in {
            "tag", "budget_h", "seed", "task_gate_pass", "l4_l5_l6_status",
            "subject_subspace_rank", "subject_subspace_var_frac",
            "l4_task_head_subject_subspace_energy",
        }}
        for r in records
    ])
    write_csv(out / "faced_l5_replay.csv", [
        {k: v for k, v in r.items() if k.startswith("l5_") or k in {
            "tag", "budget_h", "seed", "task_gate_pass", "l4_l5_l6_status",
        }}
        for r in records
    ])
    write_csv(out / "faced_l6_target_consequence.csv", [
        {k: v for k, v in r.items() if k.startswith("l6_") or k in {
            "tag", "budget_h", "seed", "task_gate_pass", "l4_l5_l6_status",
        }}
        for r in records
    ])
    write_csv(out / "faced_random_released_references.csv", [r for r in records if r["tag"] in ("random", "released")])

    by_tag = {r["tag"]: r for r in records}
    probe_gate = None
    if args.phase == "probe":
        if random_ref is None or released_ref is None:
            raise RuntimeError("probe phase requires random and released references")
        probe_gate = bool(
            released_ref["target_kappa"] >= random_ref["target_kappa"] + 0.02
            or released_ref["target_bacc"] >= random_ref["target_bacc"] + 0.02
        )
        probe_rows = []
        for r in records:
            row = dict(r)
            row["released_minus_random_kappa"] = released_ref["target_kappa"] - random_ref["target_kappa"]
            row["released_minus_random_bacc"] = released_ref["target_bacc"] - random_ref["target_bacc"]
            row["probe_gate_pass"] = probe_gate
            row["probe_gate_rule"] = "released >= random + 0.02 on target Cohen kappa OR target balanced accuracy"
            probe_rows.append(row)
        write_csv(out / "faced_probe_gate_results.csv", probe_rows)
    elif (out / "faced_probe_gate_results.csv").exists():
        # Preserve the committed D2-0 gate file across D2-1.
        pass
    else:
        write_csv(out / "faced_probe_gate_results.csv", [])

    budget_records = [r for r in records if isinstance(r.get("budget_h"), float)]
    h1000 = [r for r in budget_records if r["budget_h"] == 1000.0]
    best = max(budget_records, key=lambda r: r["target_kappa"]) if budget_records else None
    random_out = None if random_ref is None else {
        "target_kappa": random_ref["target_kappa"],
        "target_bacc": random_ref["target_bacc"],
        "source_val_kappa": random_ref["source_val_kappa"],
        "source_val_bacc": random_ref["source_val_bacc"],
    }
    released_out = None if released_ref is None else {
        "target_kappa": released_ref["target_kappa"],
        "target_bacc": released_ref["target_bacc"],
        "source_val_kappa": released_ref["source_val_kappa"],
        "source_val_bacc": released_ref["source_val_bacc"],
    }
    summary = {
        "downstream_dataset": "FACED",
        "protocol": "frozen_encoder_source_only_probe",
        "not_full_finetuning_reproduction": True,
        "phase": args.phase,
        "train_subjects": "1-80",
        "val_subjects": "81-100",
        "test_subjects": "101-123",
        "primary_metric": "cohen_kappa",
        "evaluated_budgets_h": sorted({int(r["budget_h"]) for r in budget_records}),
        "evaluated_seeds": sorted({int(r["seed"]) for r in budget_records}),
        "random_reference": random_out,
        "released_reference": released_out,
        "h1000_mean": None if not h1000 else {
            "target_kappa": float(np.mean([r["target_kappa"] for r in h1000])),
            "target_bacc": float(np.mean([r["target_bacc"] for r in h1000])),
            "target_weighted_f1": float(np.mean([r["target_weighted_f1"] for r in h1000])),
        },
        "floor_crossed_kappa": None if random_ref is None or not budget_records else bool(max(r["target_kappa"] for r in budget_records) >= random_ref["target_kappa"] + 0.02),
        "floor_crossed_bacc": None if random_ref is None or not budget_records else bool(max(r["target_bacc"] for r in budget_records) >= random_ref["target_bacc"] + 0.02),
        "released_gap": None if released_ref is None or best is None else {
            "best_tag": best["tag"],
            "released_minus_best_target_kappa": float(released_ref["target_kappa"] - best["target_kappa"]),
            "released_minus_best_target_bacc": float(released_ref["target_bacc"] - best["target_bacc"]),
        },
        "probe_gate_pass": probe_gate,
        "target_labels_used_for_selection": False,
        "h2000_included": False,
        "fine_tuning_used": False,
        "codebrain_used": False,
        "elapsed_s": round(time.time() - t0, 1),
    }
    write_json(out / "faced_budget_summary.json", summary)
    write_json(out / "faced_target_label_firewall.json", {
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
    })
    verifier_notes(out / "faced_verifier_notes.md", args.phase, probe_gate, random_ref, released_ref)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
