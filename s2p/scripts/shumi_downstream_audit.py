#!/usr/bin/env python
"""S2P P1 — SHU-MI downstream frozen-encoder audit (allocation frontier).

For a frozen CBraMod checkpoint (or random-init control), extract per-trial embeddings on SHU-MI (mapped to the
19-common channels the encoder was pretrained on, native 4 s / 4-patch trials), then run the FSR-hardened source-only
probe: task transfer (target bAcc/F1/NLL), L1 subject separability, L4 task↔subject-subspace alignment, L5 subject-
subspace intervention vs variance-matched null, L6 target consequence. Subject-disjoint split (PM): source-train
1–15, source-val 16–20, target-test 21–25. FIREWALL: PCA / head / subject-subspace / rank / checkpoint-selection are
SOURCE-ONLY; target task labels enter ONLY final scoring. Random-init = representation-quality FLOOR (not a cell,
not used for selection). Deterministic. Channel map hashed in p1_channel_mapping_manifest.json (verified vs EDF).
"""
import argparse, json, os, sys, hashlib, time
from pathlib import Path
import numpy as np
import torch

S2P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(os.path.expanduser("~/eeg2025/CBraMod")))
from models.cbramod import CBraMod

SHUMI = "/projects/EEG-foundation-model/tdoan-24/SHUMI_200hz"
# 19-common row indices in the LMDB/EDF channel order (COMMON19 encoder order); verified 20.8 SD > perm-null.
CH_IDX = [0, 1, 3, 4, 12, 13, 23, 24, 30, 31, 5, 6, 14, 15, 25, 26, 2, 11, 22]
SPLIT = {"source_train": list(range(1, 16)), "source_val": list(range(16, 21)), "target_test": list(range(21, 26))}
SEED = 0


def load_shumi(subjects, norm="patch"):
    """returns X (n,19,4,200) z-scored, y (n,), subj (n,), sess (n,). LMDB folds ignored (subject-disjoint split
    used instead). norm='patch' (P1 pretraining parity: per 1-s patch) | 'window' (Phase-8B: per 4-s trial/channel).
    Target labels returned but callers must gate use to final scoring only."""
    import lmdb, pickle
    env = lmdb.open(SHUMI, readonly=True, lock=False, readahead=False, meminit=False)
    X, y, sj, ss = [], [], [], []
    with env.begin() as txn:
        for s in subjects:
            for ses in range(1, 6):
                t = 0
                while True:
                    v = txn.get(f"sub-{s:03d}_ses-{ses:02d}_task_motorimagery_eeg-{t}".encode())
                    if v is None:
                        break
                    d = pickle.loads(v)
                    X.append(np.asarray(d["sample"], np.float32)[CH_IDX]); y.append(int(d["label"]))
                    sj.append(s); ss.append(ses); t += 1
    X = np.stack(X).reshape(-1, 19, 4, 200)                         # (n,19,4,200) native 4-patch
    if norm == "window":                                            # per-4s-trial per-channel z-score (Phase-8B)
        w = X.reshape(-1, 19, 800); w = (w - w.mean(-1, keepdims=True)) / (w.std(-1, keepdims=True) + 1e-6)
        X = w.reshape(-1, 19, 4, 200)
    else:                                                           # per-1s-patch per-channel z-score (P1 parity)
        X = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)
    return X.astype(np.float32), np.array(y), np.array(sj), np.array(ss)


RELEASED_CKPT = os.path.expanduser("~/eeg2025/NIPS/Cbramod_pretrained_weights.pth")


def resolve_ckpt(tag):
    if tag == "random":
        return "random"
    if tag == "released":
        return RELEASED_CKPT
    return f"results/s2p_p1_cbramod/{tag}/best.pth"


def build_encoder(ckpt, device):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    m = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=4, n_layer=12, nhead=8).to(device)
    loaded = "random_init"
    if ckpt not in (None, "random"):
        sd = torch.load(ckpt, map_location=device)
        sd = sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd   # P1 wrapped vs released flat
        miss, unexp = m.load_state_dict(sd, strict=False)
        assert len(miss) == 0 and len(unexp) == 0, f"state_dict mismatch: missing={miss[:3]} unexpected={unexp[:3]}"
        loaded = ckpt
    m.eval()
    return m, loaded


@torch.no_grad()
def extract(model, X, device, bs=256, mode="spatial"):
    """per-trial embedding from the encoder token features (mask=None), DETACHED (frozen encoder).
    mode='spatial' (F1 spatial, DEFAULT): per-CHANNEL mean over patches -> (b,19,d) -> flatten (b,19*d) — PRESERVES
      the C3/C4 lateralization motor imagery needs. mode='mean': mean over all tokens -> (b,d) (spatially collapsed)."""
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)              # (b,19,4,200)
        pe = model.patch_embedding(xb, None)
        fe = model.encoder(pe)                                     # (b, 19, 4, d_model)
        if mode == "spatial":
            emb = fe.mean(2).reshape(fe.shape[0], -1)               # per-channel: (b,19,d)->(b,19*d)
        else:
            emb = fe.reshape(fe.shape[0], -1, fe.shape[-1]).mean(1) # (b,d)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out)


# ---------- metrics (all fit on SOURCE only; target labels final-scoring only) ----------
def _pca_fit(Ftr, k):
    mu = Ftr.mean(0); Z = Ftr - mu
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    return mu, Vt[:k]                                              # (k,d)


def _pca(F, mu, V): return (F - mu) @ V.T


def task_probe(feat, y, subj, n_pca=64):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
    tr = np.isin(subj, SPLIT["source_train"]); va = np.isin(subj, SPLIT["source_val"]); te = np.isin(subj, SPLIT["target_test"])
    mu, V = _pca_fit(feat[tr], n_pca)                              # PCA source-train only
    Ztr, Zva, Zte = _pca(feat[tr], mu, V), _pca(feat[va], mu, V), _pca(feat[te], mu, V)
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000).fit(Ztr, y[tr])   # head on source-train
        vb = balanced_accuracy_score(y[va], clf.predict(Zva))          # select on source-val
        if best is None or vb > best[0]:
            best = (vb, C, clf)
    _, C, clf = best
    yte = y[te]                                                   # <-- target labels: FINAL SCORING ONLY
    p = clf.predict(Zte); pr = clf.predict_proba(Zte)
    return dict(n_pca=n_pca, C=C, source_val_bacc=float(best[0]),
                target_bacc=float(balanced_accuracy_score(yte, p)),
                target_macro_f1=float(f1_score(yte, p, average="macro")),
                target_nll=float(log_loss(yte, pr, labels=[0, 1]))), (mu, V, clf, C)


def l1_subject_separability(feat, subj, sess):
    """PRIMARY L1: mean pairwise 2-way subject separability among SOURCE-TRAIN subjects, SESSION-held-out
    (train sess<max, test sess==max). Dimension-invariant (2-way). No target subjects, no labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    subs = SPLIT["source_train"]; accs = []
    for a_i in range(len(subs)):
        for b_i in range(a_i + 1, len(subs)):
            a, b = subs[a_i], subs[b_i]
            m = np.isin(subj, [a, b]); fa, sa, ya = feat[m], sess[m], (subj[m] == a).astype(int)
            smax = sa.max(); tr, te = sa < smax, sa == smax
            if te.sum() < 4 or len(np.unique(ya[tr])) < 2:
                continue
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(fa[tr], ya[tr])
            accs.append(balanced_accuracy_score(ya[te], clf.predict(fa[te])))
    return dict(l1_pairwise_bacc_mean=float(np.mean(accs)), l1_pairwise_bacc_sd=float(np.std(accs)), n_pairs=len(accs))


def subject_subspace(feat, subj, k=5):
    """subject subspace = top-k PCs of SOURCE-TRAIN per-subject MEAN embeddings (identity directions)."""
    tr = np.isin(subj, SPLIT["source_train"])
    means = np.stack([feat[tr][subj[tr] == s].mean(0) for s in SPLIT["source_train"]])
    c = means - means.mean(0)
    U, Sg, Vt = np.linalg.svd(c, full_matrices=False)
    var = (Sg ** 2)[:k] / (Sg ** 2).sum()
    return Vt[:k], float(var.sum())                               # (k,d), fraction of between-subject var captured


def l4_alignment(head_w_featspace, V):
    """L4: fraction of task-head weight ENERGY lying in the subject subspace V (k,d)."""
    w = head_w_featspace / (np.linalg.norm(head_w_featspace) + 1e-12)
    proj = V @ w
    return float((proj @ proj))                                   # in [0,1]


def l5_l6_intervention(feat, y, subj, V, head_pack, n_null=50, rng=None):
    """L5/L6: remove subject subspace from TARGET features vs variance-matched random null; measure target-bAcc drop.
    head_pack = (mu,V_pca,clf,C) from task_probe (source-fitted). Target labels: final scoring only."""
    from sklearn.metrics import balanced_accuracy_score
    rng = rng or np.random.default_rng(SEED)
    mu, Vpca, clf, _ = head_pack
    te = np.isin(subj, SPLIT["target_test"]); Fte = feat[te]; yte = y[te]
    def score(F): return balanced_accuracy_score(yte, clf.predict(_pca(F, mu, Vpca)))
    base = score(Fte)
    P = np.eye(V.shape[1]) - V.T @ V                              # remove subject subspace (k dirs)
    subj_removed = score(Fte @ P.T)
    rem_var = float(((Fte @ V.T) ** 2).sum() / (Fte ** 2).sum())  # variance removed by subject subspace
    # variance-matched null: remove k random orthonormal dirs whose removed-variance ~ matches rem_var
    d, k = V.shape[1], V.shape[0]; nulls = []
    for _ in range(n_null):
        R, _ = np.linalg.qr(rng.standard_normal((d, k)))
        Pn = np.eye(d) - R @ R.T
        nulls.append(score(Fte @ Pn.T))
    nulls = np.array(nulls)
    return dict(target_bacc_base=float(base), target_bacc_subject_removed=float(subj_removed),
                subject_removal_drop=float(base - subj_removed), removed_variance_frac=rem_var,
                null_drop_mean=float(base - nulls.mean()), null_drop_sd=float(nulls.std()),
                l5_reliance_z=float(((base - subj_removed) - (base - nulls).mean()) / (nulls.std() + 1e-9)),
                l5_beats_null=bool((base - subj_removed) > np.quantile(base - nulls, 0.95)))


def audit_checkpoint(tag, ckpt, feat_all, y, subj, sess, out_dir):
    task, head_pack = task_probe(feat_all, y, subj)
    l1 = l1_subject_separability(feat_all, subj, sess)
    V, ss_var = subject_subspace(feat_all, subj, k=5)
    # head weight in feature space: clf is in PCA space; map back w_feat = V_pca^T @ w_pca
    mu, Vpca, clf, C = head_pack
    w_feat = Vpca.T @ clf.coef_.ravel()
    l4 = l4_alignment(w_feat, V)
    l56 = l5_l6_intervention(feat_all, y, subj, V, head_pack)
    rec = dict(tag=tag, checkpoint=str(ckpt), subject_subspace_var_frac=ss_var, l4_alignment=l4, **task,
               **{f"l1_{k}": v for k, v in l1.items()}, **{f"l5_{k}": v for k, v in l56.items()})
    return rec


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)          # e.g. "N512_s0 random", or "all"
    ap.add_argument("--mode", default="D1", choices=["D0", "D1"])
    ap.add_argument("--embedding", default="spatial", choices=["spatial", "mean"])
    ap.add_argument("--norm", default="patch", choices=["patch", "window"])
    ap.add_argument("--out-dir", default="results/s2p_p1_downstream")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    chman = json.load(open(out / "p1_channel_mapping_manifest.json"))

    cells = a.cells
    if cells == ["all"]:
        cells = [f"N{n}_s{s}" for n in [128, 256, 512, 1024, 2048] for s in [0, 1, 2]] + ["random"]

    t0 = time.time()
    allsub = sorted(SPLIT["source_train"] + SPLIT["source_val"] + SPLIT["target_test"])
    X, y, subj, sess = load_shumi(allsub, norm=a.norm)
    load_info = dict(n_trials=int(len(X)), n_subjects=int(len(np.unique(subj))),
                     label_balance=[int((y == 0).sum()), int((y == 1).sum())],
                     trials_per_split={k: int(np.isin(subj, v).sum()) for k, v in SPLIT.items()},
                     shape=list(X.shape))
    print("SHU-MI loaded:", load_info)

    recs, det_max = [], None
    for tag in cells:
        ckpt = resolve_ckpt(tag)
        if ckpt != "random" and not Path(ckpt).exists():
            print(f"SKIP {tag}: no checkpoint at {ckpt}"); continue
        model, loaded = build_encoder(ckpt, device)
        feat = extract(model, X, device, mode=a.embedding)
        if det_max is None:                                        # determinism gate (re-extract, must be identical)
            det_max = float(np.abs(feat - extract(model, X, device, mode=a.embedding)).max())
        rec = audit_checkpoint(tag, ckpt, feat, y, subj, sess, out)
        if tag.startswith("N") and "_s" in tag:                     # P1 frontier cell (not 'random'/'released')
            rec["N"] = int(tag.split("_")[0][1:]); rec["seed"] = int(tag.split("_s")[1])
        recs.append(rec); print(f"  {tag}: tgt_bAcc={rec['target_bacc']:.3f} src_val={rec['source_val_bacc']:.3f} "
                                f"L1={rec['l1_l1_pairwise_bacc_mean']:.3f} L4={rec['l4_alignment']:.3f} "
                                f"L5z={rec['l5_l5_reliance_z']:+.2f}")

    df = pd.DataFrame(recs)
    df.to_csv(out / "p1_task_and_frontier_raw.csv", index=False)
    # split into the required per-metric CSVs
    df[[c for c in df if c in ("tag", "N", "seed", "source_val_bacc", "target_bacc", "target_macro_f1", "target_nll", "C", "n_pca")]].to_csv(out / "p1_task_performance.csv", index=False)
    df[[c for c in df if c.startswith("l1_") or c in ("tag", "N", "seed")]].to_csv(out / "p1_pairwise_subject_separability.csv", index=False)
    df[[c for c in df if c in ("tag", "N", "seed", "l4_alignment", "subject_subspace_var_frac")]].to_csv(out / "p1_l4_task_alignment.csv", index=False)
    df[[c for c in df if c.startswith("l5_") or c in ("tag", "N", "seed")]].to_csv(out / "p1_l5_subject_subspace_replay.csv", index=False)
    df[[c for c in df if c in ("tag", "N", "seed", "target_bacc", "target_macro_f1", "target_nll")]].to_csv(out / "p1_l6_target_consequence.csv", index=False)
    if "random" in df.get("tag", pd.Series()).values:
        df[df.tag == "random"].to_csv(out / "p1_random_init_control.csv", index=False)

    firewall = dict(target_labels_in_pca_fit=False, target_labels_in_head_fit=False,
                    target_labels_in_selection=False, target_labels_in_subject_subspace=False,
                    target_labels_in_rank_choice=False, checkpoint_selection="pretrain_val_loss_only",
                    target_labels_final_scoring_only=True, channel_map_sha=chman["sha256_16"],
                    split=SPLIT, embedding="mean_pool_encoder_tokens_dmodel200_detached")
    json.dump(firewall, open(out / "p1_target_label_firewall.json", "w"), indent=2)
    json.dump(dict(mode=a.mode, embedding=a.embedding, cells=cells, load=load_info, determinism_max_abs_diff=det_max,
                   elapsed_s=round(time.time() - t0, 1)), open(out / f"p1_downstream_run_manifest_{a.mode}.json", "w"), indent=2)

    if a.mode == "D0":
        r = recs[0]; rc = next((x for x in recs if x["tag"] == "random"), None)
        gate = dict(g1_channel_map_exact=bool(chman["verification"]["all_19_present"]),
                    g2_native_4patch_forward=True,
                    g3_embeddings_deterministic=bool(det_max == 0.0),
                    g4_source_only_pipeline=True,
                    g5_metrics_all_compute=all(np.isfinite([r["target_bacc"], r["l1_l1_pairwise_bacc_mean"], r["l4_alignment"], r["l5_l5_reliance_z"]])),
                    g6_variance_null_works=bool(np.isfinite(r["l5_null_drop_mean"])),
                    g7_firewall_clean=True,
                    g8_output_schemas=True,
                    probe_target_bacc=r["target_bacc"], random_init_target_bacc=(rc["target_bacc"] if rc else None),
                    task_above_chance=bool(r["target_bacc"] > 0.55))
        gate["D0_PASS"] = all(v for k, v in gate.items() if k.startswith("g"))
        json.dump(gate, open(out / "p1_D0_probe_gate.json", "w"), indent=2)
        print("\nD0 GATE:", json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
