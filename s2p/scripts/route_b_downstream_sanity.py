#!/usr/bin/env python
"""Route B tiny downstream sanity on SHU-MI native32 and 19-common.

Frozen random-init and released CBraMod are evaluated with source-only
PCA/head selection and target labels only for final scoring. This is a launch
sanity, not a full downstream audit.
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

S2P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(os.path.expanduser("~/eeg2025/CBraMod")))
from models.cbramod import CBraMod


SHUMI = "/projects/EEG-foundation-model/tdoan-24/SHUMI_200hz"
RELEASED_CKPT = os.path.expanduser("~/eeg2025/NIPS/Cbramod_pretrained_weights.pth")
SHU32_ORDER = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "CZ",
    "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6", "PZ", "P3",
    "P4", "T5", "T6", "PO3", "PO4", "OZ", "O1", "O2",
]
COMMON19 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
CH_IDX_19 = [SHU32_ORDER.index(c) for c in COMMON19]
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
        for row in rows:
            writer.writerow(row)


def load_shumi_all(trials_per_class_per_session):
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
                    lab = int(item["label"])
                    if lab in by_label:
                        by_label[lab].append(np.asarray(item["sample"], np.float32).reshape(32, 4, 200))
                    t += 1
                for lab in [0, 1]:
                    for sample in by_label[lab][:trials_per_class_per_session]:
                        X.append(sample)
                        y.append(lab)
                        subj.append(s)
                        sess.append(se)
    X = np.stack(X)
    X = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)
    return X.astype(np.float32), np.asarray(y), np.asarray(subj), np.asarray(sess)


def build_encoder(tag, seq_len, device):
    torch.manual_seed(0)
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=seq_len, n_layer=12, nhead=8).to(device)
    if tag == "released":
        sd = torch.load(RELEASED_CKPT, map_location=device)
        if isinstance(sd, dict) and "model_state" in sd:
            sd = sd["model_state"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"released state mismatch missing={missing[:3]} unexpected={unexpected[:3]}")
    elif tag != "random":
        raise ValueError(tag)
    model.eval()
    return model


@torch.no_grad()
def extract(model, X, device, batch_size):
    feats = []
    for start in range(0, len(X), batch_size):
        if start == 0 or (start // batch_size) % 10 == 0:
            print(f"extract {len(feats)} start={start} / {len(X)}", flush=True)
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        pe = model.patch_embedding(xb, None)
        fe = model.encoder(pe)
        emb = fe.mean(2).reshape(fe.shape[0], -1)
        feats.append(emb.float().cpu().numpy())
    return np.concatenate(feats)


def task_probe(feat, y, subj):
    tr = np.isin(subj, SPLIT["source_train"])
    va = np.isin(subj, SPLIT["source_val"])
    te = np.isin(subj, SPLIT["target_test"])
    pca = PCA(n_components=64, svd_solver="randomized", random_state=0)
    Ztr = pca.fit_transform(feat[tr])
    Zva = pca.transform(feat[va])
    Zte = pca.transform(feat[te])
    best = None
    for c in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=c, max_iter=2000).fit(Ztr, y[tr])
        bacc = balanced_accuracy_score(y[va], clf.predict(Zva))
        if best is None or bacc > best[0]:
            best = (bacc, c, clf)
    source_val_bacc, c, clf = best
    pred = clf.predict(Zte)
    prob = clf.predict_proba(Zte)
    return {
        "source_val_bacc": float(source_val_bacc),
        "selected_C": float(c),
        "target_bacc": float(balanced_accuracy_score(y[te], pred)),
        "target_macro_f1": float(f1_score(y[te], pred, average="macro")),
        "target_nll": float(log_loss(y[te], prob, labels=[0, 1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/s2p_route_b_33ch_contract")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--trials-per-class-per-session", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    X32, y, subj, sess = load_shumi_all(args.trials_per_class_per_session)
    mode_data = {
        "native32": X32,
        "19common": X32[:, CH_IDX_19],
    }
    rows = []
    for mode, X in mode_data.items():
        for tag in ["random", "released"]:
            print(f"downstream sanity: mode={mode} checkpoint={tag} n={len(X)}", flush=True)
            model = build_encoder(tag, seq_len=4, device=device)
            feat = extract(model, X, device, args.batch_size)
            rec = task_probe(feat, y, subj)
            rec.update({
                "channel_mode": mode,
                "checkpoint": tag,
                "input_channels": int(X.shape[1]),
                "input_shape": "x".join(map(str, X.shape)),
                "feature_dim": int(feat.shape[1]),
                "feature_sha16": hashlib.sha256(feat[: min(len(feat), 128)].astype(np.float32).tobytes()).hexdigest()[:16],
                "device": str(device),
                "trials_per_class_per_session": int(args.trials_per_class_per_session),
                "n_trials": int(len(X)),
                "target_labels_used_for_selection": False,
            })
            rows.append(rec)

    by_mode = {m: {r["checkpoint"]: r for r in rows if r["channel_mode"] == m} for m in mode_data}
    for mode, pair in by_mode.items():
        random_bacc = pair["random"]["target_bacc"]
        released_bacc = pair["released"]["target_bacc"]
        sanity_pass = released_bacc >= random_bacc + 0.02
        for tag in ["random", "released"]:
            pair[tag]["random_floor_bacc"] = random_bacc
            pair[tag]["released_minus_random_bacc"] = released_bacc - random_bacc
            pair[tag]["sanity_pass"] = bool(sanity_pass)
            pair[tag]["sanity_rule"] = "released_target_bacc >= random_target_bacc + 0.02"

    write_csv(out / "route_b_downstream_sanity.csv", rows)
    manifest = {
        "dataset": "SHU-MI",
        "split": SPLIT,
        "channel_modes": {
            "native32": SHU32_ORDER,
            "19common": COMMON19,
        },
        "target_labels_used_for_selection": False,
        "target_labels_final_scoring_only": True,
        "primary_selection_rule": "native32 if released sanity passes, else 19common if released sanity passes, else none",
    }
    (out / "route_b_downstream_sanity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
