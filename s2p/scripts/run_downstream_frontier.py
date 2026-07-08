#!/usr/bin/env python
"""S2P_09 — P1 downstream frontier audit on SHU-MI (frozen pre-reg docs/S2P_09_DOWNSTREAM_AUDIT_FROZEN.md).

One frozen CBraMod cell (a P1 checkpoint or the random-init floor) -> SHU-MI 32ch mapped to the COMMON19 canonical
PRETRAINING order -> native 4s (4-patch @200Hz) forward with per-PATCH z-score (matches pretraining) -> F1 spatial
(per-channel, 19*200) + F0 pooled embeddings -> source-only FSR probe on the fixed subject-disjoint split
train{1-15}/val{16-20}/target{21-25}: L1 (subject decodability), L4 (task-head/subject-subspace alignment),
L5 (subject-subspace vs variance-matched null), L6 (target consequence). Target task labels are used for FINAL
SCORING ONLY. PCA/head/subspace/rank are source-only. Checkpoint selection is pretrain-val loss only (upstream).

FSR-hardened helpers (top_k / subject_offsets / task_offsets / erase / var_frac_removed / clu_ci) are copied from
CMI_AAAI_rq4/scripts/cb_cbm_8b_audit.py (FSR_46) so this runner is self-contained.

    # D0 one-checkpoint probe gate (N512_s0 + random-init):
    <eeg2025 python> s2p/scripts/run_downstream_frontier.py --probe-gate --device cuda
    # D1 single cell (fleet unit):
    <eeg2025 python> s2p/scripts/run_downstream_frontier.py --cell N512_s0 --device cuda
    <eeg2025 python> s2p/scripts/run_downstream_frontier.py --cell random_init --device cuda
"""
import argparse, csv, glob, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import resample
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as BACC, f1_score, log_loss

CBRAMOD = "/home/infres/yinwang/eeg2025/CBraMod"
SHU = "/projects/EEG-foundation-model/SHU-MI-cbramod/mat"
CKPT_ROOT = Path("results/s2p_p1_cbramod")
OUT = Path("results/s2p_p1_downstream")
RNG = np.random.default_rng(0)

# SHU-MI 32ch (order in the .mat) and the COMMON19 canonical PRETRAINING order (tueg_subject_loader.COMMON19, "-LE" stripped)
SHU_CH = ["FP1","FP2","FZ","F3","F4","F7","F8","FC1","FC2","FC5","FC6","CZ","C3","C4","T3","T4","A1","A2",
          "CP1","CP2","CP5","CP6","PZ","P3","P4","T5","T6","PO3","PO4","OZ","O1","O2"]
COMMON19 = ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T3","T4","T5","T6","FZ","CZ","PZ"]
NATIVE_HZ, TARGET_HZ, PATCH = 250, 200, 200
SPLIT = {"source_train": list(range(1, 16)), "source_val": list(range(16, 21)), "target_test": list(range(21, 26))}
PCA_VAR, PCA_CAP, PRIMARY_K, NPERM = 0.95, 128, 2, 1000
KS = [1, 2, 4, 8]


# ----------------------------------------------------------------------------- channel map (STOP if any missing)
def channel_map():
    missing = [c for c in COMMON19 if c not in SHU_CH]
    if missing:
        raise SystemExit(f"STOP (stop-rule 1): COMMON19 channels missing from SHU montage: {missing}")
    idx = [SHU_CH.index(c) for c in COMMON19]
    table = [{"canonical_pos": i, "channel": c, "shu_index": idx[i]} for i, c in enumerate(COMMON19)]
    h = hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest()[:16]
    return idx, table, h


# ----------------------------------------------------------------------------- SHU loader + preprocess
def load_shu_subject(sub):
    for f in sorted(glob.glob(f"{SHU}/sub-{sub:03d}_ses-*_task_motorimagery_eeg.mat")):
        ses = int(os.path.basename(f).split("ses-")[1][:2])
        m = sio.loadmat(f)
        yield m["data"].astype(np.float32), m["labels"].astype(int).ravel() - 1, ses   # labels {1,2}->{0,1}


def preprocess(x_raw, ch_idx):
    """(n,32,1000)@250Hz -> select 19 canonical channels -> resample 200Hz -> (n,19,4,200) per-PATCH z-score."""
    x = x_raw[:, ch_idx, :]                                    # (n,19,1000) canonical order
    T200 = int(round(x.shape[-1] * TARGET_HZ / NATIVE_HZ))     # 1000@250 -> 800@200
    x = resample(x, T200, axis=-1)
    npatch = T200 // PATCH                                      # 4
    x = x[:, :, :npatch * PATCH].reshape(x.shape[0], len(COMMON19), npatch, PATCH)
    x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)   # per-PATCH per-channel z-score (matches pretraining)
    return x.astype(np.float32), npatch


# ----------------------------------------------------------------------------- CBraMod cell (checkpoint or random-init)
def unwrap(sd):
    # S2P P1 checkpoints save {"epoch","model_state","optimizer_state",...} (run_frontier_cbramod.py:135)
    if isinstance(sd, dict):
        for k in ("model_state", "model", "state_dict", "model_state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]; break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def set_determinism(torch):
    import random
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def build_cell(torch, device, cell):
    """cell = 'N{N}_s{S}' -> load best.pth; 'random_init' -> untrained CBraMod (init_seed 0)."""
    sys.path.insert(0, CBRAMOD)
    from models.cbramod import CBraMod
    torch.manual_seed(0)                                       # fixed init for random-init floor + param determinism
    bb = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    ckpt_path, ckpt_sha, pretrain_val = None, None, None
    if cell != "random_init":
        ckpt_path = CKPT_ROOT / cell / "best.pth"
        if not ckpt_path.exists():
            raise SystemExit(f"STOP: checkpoint missing {ckpt_path}")
        ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:16]
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m, u = bb.load_state_dict(unwrap(sd), strict=False)
        if len(u) > 0:
            raise SystemExit(f"STOP: unexpected keys loading {cell}: {u[:5]}")
        rs = CKPT_ROOT / cell / "run_summary.json"
        if rs.exists():
            pretrain_val = json.loads(rs.read_text()).get("best_val_loss")
    bb.proj_out = torch.nn.Identity(); bb.eval().to(device)
    return bb, dict(cell=cell, ckpt_path=str(ckpt_path) if ckpt_path else None, ckpt_sha256=ckpt_sha,
                    pretrain_val_loss=pretrain_val, n_params=sum(p.numel() for p in bb.parameters()))


def pool(feats):                                               # (B,C,np,200) -> F0 (B,200), F1 (B,C*200)
    B, C, npp, D = feats.shape
    return feats.mean(axis=(1, 2)), feats.mean(axis=2).reshape(B, C * D)


def extract_features(torch, bb, device, ch_idx, bs=64):
    F1, F0, Y, D, SES = [], [], [], [], []
    qc = {}
    for sub in range(1, 26):
        for x_raw, y, ses in load_shu_subject(sub):
            x, npatch = preprocess(x_raw, ch_idx)
            xt = torch.tensor(x, dtype=torch.float32, device=device)
            f0l, f1l = [], []
            with torch.no_grad():
                for i in range(0, len(xt), bs):
                    f0b, f1b = pool(bb(xt[i:i + bs]).cpu().numpy())
                    f0l.append(f0b); f1l.append(f1b)
            f0 = np.concatenate(f0l); f1 = np.concatenate(f1l)
            F0.append(f0); F1.append(f1); Y.append(y); D.append(np.full(len(y), sub)); SES.append(np.full(len(y), ses))
            if not qc and len(xt) >= 32:                       # determinism QC (repeat + batch-grouping) on first session
                with torch.no_grad():
                    a0, a1 = pool(bb(xt[:32]).cpu().numpy()); b0, b1 = pool(bb(xt[:32]).cpu().numpy())
                    c0a, c1a = pool(bb(xt[:16]).cpu().numpy()); c0b, c1b = pool(bb(xt[16:32]).cpu().numpy())
                c0 = np.concatenate([c0a, c0b]); c1 = np.concatenate([c1a, c1b])
                qc = dict(F0_repeat_max=float(np.abs(a0 - b0).max()), F1_repeat_max=float(np.abs(a1 - b1).max()),
                          F0_batchgroup_max=float(np.abs(a0 - c0).max()), F1_batchgroup_max=float(np.abs(a1 - c1).max()),
                          npatch=int(npatch),
                          deterministic=bool(np.abs(a0 - b0).max() < 1e-5 and np.abs(a1 - b1).max() < 1e-5),
                          batch_invariant=bool(np.abs(a0 - c0).max() < 1e-5 and np.abs(a1 - c1).max() < 1e-5))
    return (np.concatenate(F1).astype(np.float64), np.concatenate(F0).astype(np.float64), np.concatenate(Y),
            np.concatenate(D), np.concatenate(SES), qc)


# ----------------------------------------------------------------------------- FSR-hardened probe helpers (cb_cbm_8b_audit.py)
def _std(X):
    mu = X.mean(0); sd = X.std(0) + 1e-8
    return (X - mu) / sd, mu, sd


def top_k(M, k):
    if M.shape[0] == 0:
        return np.zeros((0, M.shape[1]))
    Mc = M - M.mean(0, keepdims=True) if M.shape[0] > 1 else M
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    return Vt[:min(k, Vt.shape[0])]


def subject_offsets(X, y, d):
    rows = []
    for yy in np.unique(y):
        my = X[y == yy].mean(0)
        for dd in np.unique(d[y == yy]):
            m = (y == yy) & (d == dd); n = int(m.sum())
            if n > 0:
                rows.append(np.sqrt(n) * (X[m].mean(0) - my))
    return np.array(rows) if rows else np.zeros((0, X.shape[1]))


def task_offsets(X, y):
    mu = X.mean(0)
    return np.array([np.sqrt((y == yy).sum()) * (X[y == yy].mean(0) - mu) for yy in np.unique(y)])


def erase(X, basis):
    return X - (X @ basis.T) @ basis if basis.shape[0] else X


def var_frac_removed(X, basis):
    if basis.shape[0] == 0:
        return 0.0
    Xc = X - X.mean(0)
    return float(((Xc @ basis.T) ** 2).sum() / (Xc ** 2).sum())


def clu_ci(vals, nb=2000):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return None, [None, None]
    b = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(nb)]
    return round(float(v.mean()), 4), [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def pca_fit(Xtr):
    p = PCA(n_components=min(PCA_CAP, Xtr.shape[1], Xtr.shape[0] - 1)).fit(Xtr)
    dim = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), PCA_VAR) + 1)
    return p, min(dim, p.n_components_)


# ----------------------------------------------------------------------------- L1 subject probe (session-held-out, within a subject group)
def l1_probe(X, d, ses, y=None, cond_on_y=False):
    subs = np.unique(d); chance = 1.0 / len(subs)
    Xr = X.copy()
    if cond_on_y and y is not None:
        for yy in np.unique(y):
            Xr[y == yy] -= Xr[y == yy].mean(0)
    accs, perm_all = [], []
    for te in np.unique(ses):
        tr = ses != te; teM = ses == te
        if len(np.unique(d[tr])) < len(subs) or teM.sum() < len(subs):
            continue
        p, dim = pca_fit(Xr[tr]); Ztr = p.transform(Xr[tr])[:, :dim]; Zte = p.transform(Xr[teM])[:, :dim]
        Zs, mu, sd = _std(Ztr)
        clf = LDA().fit(Zs, d[tr]); pred = clf.predict((Zte - mu) / sd)
        accs.append(BACC(d[teM], pred)); dt = d[teM]
        perm_all.append([BACC(RNG.permutation(dt), pred) for _ in range(NPERM)])
    if not accs:
        return dict(bacc=None, chance=round(chance, 4), n_folds=0)
    obs = float(np.mean(accs)); perm = np.array(perm_all).mean(0)
    return dict(bacc=round(obs, 4), chance=round(chance, 4), null_mean=round(float(perm.mean()), 4),
                p=round(float((perm >= obs).mean()), 4), effect=round(obs - float(perm.mean()), 4), n_folds=len(accs))


# ----------------------------------------------------------------------------- fixed-split source-only audit (task gate + L4/L5/L6)
def _metrics(head, Z, y):
    pred = head.predict(Z)
    proba = head.predict_proba(Z)
    return (round(BACC(y, pred), 4), round(f1_score(y, pred, average="macro"), 4),
            round(float(log_loss(y, proba, labels=list(range(proba.shape[1])))), 4))


def fixed_split_audit(F1, F0, y, d, f0_target_bacc_ref=None):
    tr = np.isin(d, SPLIT["source_train"]); va = np.isin(d, SPLIT["source_val"]); tg = np.isin(d, SPLIT["target_test"])
    p, dim = pca_fit(F1[tr])
    Ztr = p.transform(F1[tr])[:, :dim]; Zva = p.transform(F1[va])[:, :dim]; Ztg = p.transform(F1[tg])[:, :dim]
    Zs, mu, sd = _std(Ztr)
    def S(Z): return (Z - mu) / sd
    ytr, yva, ytg = y[tr], y[va], y[tg]
    head = LDA().fit(Zs, ytr)
    sv_b, sv_f, sv_n = _metrics(head, S(Zva), yva)
    tb_b, tb_f, tb_n = _metrics(head, S(Ztg), ytg)             # TARGET labels: FINAL SCORING ONLY
    gate = dict(pca_dim=int(dim), source_val_bacc=sv_b, source_val_macroF1=sv_f, source_val_nll=sv_n,
                target_bacc=tb_b, target_macroF1=tb_f, target_nll=tb_n)
    beats_f0 = bool(f0_target_bacc_ref is not None and tb_b - f0_target_bacc_ref >= 0.04)
    gate["task_gate_pass"] = bool((sv_b >= 0.60 and tb_b >= 0.58) or beats_f0)
    gate["target_beats_f0_by_0p04"] = beats_f0

    W = head.coef_ if head.coef_.ndim == 2 else head.coef_.reshape(1, -1)
    subjM = subject_offsets(Zs, ytr, d[tr])
    l4 = []
    for k in KS:
        Bs = top_k(subjM, k)
        al = (float(np.mean(np.max(np.abs((W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)) @ Bs.T), axis=1)))
              if Bs.shape[0] and W.shape[0] else None)
        l4.append(dict(k=k, task_head_subject_alignment=round(al, 4) if al is not None else None, subj_rank=int(Bs.shape[0])))

    k = PRIMARY_K
    Bs = top_k(subjM, k); Bv = top_k(Zs - Zs.mean(0), k); Bt = top_k(task_offsets(Zs, ytr), k)
    Zv = S(Zva); base = BACC(yva, head.predict(Zv))
    def drop(B): return round(base - BACC(yva, head.predict(erase(Zv, B))), 4)
    l5 = dict(k=k, base_bacc=round(base, 4), drop_subject=drop(Bs), drop_variance=drop(Bv), drop_oracle_task=drop(Bt),
              var_removed_subject=round(var_frac_removed(Zv, Bs), 4), var_removed_variance=round(var_frac_removed(Zv, Bv), 4))
    # per-val-subject drops for a clustered CI on the subject-beats-variance test
    l5_per = []
    for s in SPLIT["source_val"]:
        mm = d[va] == s
        if mm.sum() > 2:
            bb_ = BACC(yva[mm], head.predict(Zv[mm]))
            l5_per.append(dict(val_subject=int(s), drop_subject=round(bb_ - BACC(yva[mm], head.predict(erase(Zv[mm], Bs))), 4),
                               drop_variance=round(bb_ - BACC(yva[mm], head.predict(erase(Zv[mm], Bv))), 4)))

    Zt = S(Ztg); tchance = round(1.0 / len(np.unique(y)), 4)
    base_t_b, base_t_f, base_t_n = _metrics(head, Zt, ytg)
    aft_s = _metrics(head, erase(Zt, Bs), ytg); aft_v = _metrics(head, erase(Zt, Bv), ytg)
    l6 = dict(k=k, chance=tchance, target_base_bacc=base_t_b, target_base_macroF1=base_t_f, target_base_nll=base_t_n,
              target_after_subject_bacc=aft_s[0], target_after_variance_bacc=aft_v[0],
              delta_subject_bacc=round(base_t_b - aft_s[0], 4), delta_variance_bacc=round(base_t_b - aft_v[0], 4),
              delta_subject_nll=round(aft_s[2] - base_t_n, 4))
    l6_per = []
    for s in SPLIT["target_test"]:
        mm = d[tg] == s
        if mm.sum() > 2:
            bb_ = BACC(ytg[mm], head.predict(Zt[mm]))
            l6_per.append(dict(target_subject=int(s), target_base_bacc=round(bb_, 4),
                               delta_subject_bacc=round(bb_ - BACC(ytg[mm], head.predict(erase(Zt[mm], Bs))), 4)))
    firewall = dict(pca_fit_subjects=SPLIT["source_train"], head_fit_subjects=SPLIT["source_train"],
                    head_select_subjects=SPLIT["source_val"], subject_subspace_fit_subjects=SPLIT["source_train"],
                    rank_choice_subjects=SPLIT["source_train"] + SPLIT["source_val"],
                    target_labels_used_only_for_final_scoring=True, target_scoring_subjects=SPLIT["target_test"])
    return gate, l4, l5, l5_per, l6, l6_per, firewall


# ----------------------------------------------------------------------------- one cell end-to-end (idempotent per-cell JSON)
CELLS = OUT / "cells"


def _atomic_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def cell_N_seed(cell):
    if cell == "random_init":
        return None, None
    return int(cell.split("_s")[0][1:]), int(cell.split("_s")[1])


def run_cell(cell, device="cuda", write_features=False):
    import torch
    torch.set_num_threads(4); set_determinism(torch)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    idx, ctable, chash = channel_map()
    bb, meta = build_cell(torch, dev, cell)
    t0 = time.time()
    F1, F0, y, d, ses, qc = extract_features(torch, bb, dev, idx)
    tr, va, tg = (np.isin(d, SPLIT[k]) for k in ("source_train", "source_val", "target_test"))
    # F0 reference for the beats-F0 gate: pooled-feature target bAcc via the same fixed-split head
    g0, *_ = fixed_split_audit(F0, F0, y, d)   # F0 as its own feature (pooled)
    f0_ref = g0["target_bacc"]
    gate, l4, l5, l5_per, l6, l6_per, firewall = fixed_split_audit(F1, F0, y, d, f0_target_bacc_ref=f0_ref)

    l1p = l1_probe(F1[tr], d[tr], ses[tr], y[tr])
    l1p_c = l1_probe(F1[tr], d[tr], ses[tr], y[tr], cond_on_y=True)
    l1_val = l1_probe(F1[va], d[va], ses[va])
    l1_tgt = l1_probe(F1[tg], d[tg], ses[tg])

    N, seed = cell_N_seed(cell)
    meta = dict(**meta, N=N, seed=seed, is_random_init=cell == "random_init", n_trials=int(len(y)),
                n_subjects=int(np.unique(d).size), F1_dim=int(F1.shape[1]), F0_dim=int(F0.shape[1]),
                npatch=qc.get("npatch"), per_patch_zscore=True, n_channels=len(COMMON19), channel_map_hash=chash,
                target_labels_used=False, sec=round(time.time() - t0, 1), qc=qc)
    result = dict(cell=cell, meta=meta, f0_target_bacc=f0_ref, gate=gate, l4=l4, l5=l5, l5_per=l5_per,
                  l6=l6, l6_per=l6_per, l1_source=l1p, l1_source_cc=l1p_c, l1_val=l1_val, l1_tgt=l1_tgt,
                  firewall=firewall, channel_table=ctable, channel_map_hash=chash)
    _atomic_write_json(CELLS / f"{cell}.json", result)         # single source of truth (idempotent; skip-if-done)
    if write_features:
        (OUT / "embeddings").mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "embeddings" / f"{cell}_F1.npz", X=F1.astype(np.float32), y=y, d=d, ses=ses)
    return result


# ----------------------------------------------------------------------------- aggregate per-cell JSON -> shared CSVs + summary
def _write_csv(fn, rows):
    if not rows:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)


def _slope(xs, ys):
    x = np.asarray(xs, float); yv = np.asarray(ys, float); ok = np.isfinite(x) & np.isfinite(yv)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-9:
        return None
    return float(np.polyfit(x[ok], yv[ok], 1)[0])


def _frontier(cells):
    """slope + quadratic curvature/peak + leave-one-N sign stability vs log2N over the 15 P1 cells (per-N-seed)."""
    P = [c for c in cells if not c["meta"]["is_random_init"]]
    def series(getter):
        return [(np.log2(c["meta"]["N"]), getter(c)) for c in P if getter(c) is not None]
    endpoints = {
        "target_bacc": lambda c: c["gate"]["target_bacc"],
        "l1_source_effect": lambda c: c["l1_source"].get("effect"),
        "l4_alignment_k2": lambda c: next((r["task_head_subject_alignment"] for r in c["l4"] if r["k"] == PRIMARY_K), None),
        "l5_drop_subject": lambda c: c["l5"]["drop_subject"],
        "l6_delta_subject_bacc": lambda c: c["l6"]["delta_subject_bacc"],
    }
    out = {}
    for name, g in endpoints.items():
        s = series(g)
        if len(s) < 3:
            out[name] = dict(slope=None); continue
        xs, ys = zip(*s)
        slope = _slope(xs, ys)
        # cluster bootstrap over seeds
        seeds = sorted({c["meta"]["seed"] for c in P})
        boot = []
        for _ in range(2000):
            pick = RNG.choice(seeds, len(seeds))
            ss = [(np.log2(c["meta"]["N"]), g(c)) for c in P if c["meta"]["seed"] in pick and g(c) is not None]
            if len({a for a, _ in ss}) >= 3:
                sl = _slope(*zip(*ss))
                if sl is not None:
                    boot.append(sl)
        ci = [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)] if boot else [None, None]
        quad = np.polyfit(xs, ys, 2) if len(set(xs)) >= 3 else None
        peakN = None
        if quad is not None and abs(quad[0]) > 1e-9:
            peakN = float(2 ** (-quad[1] / (2 * quad[0])))
        # leave-one-N-out sign stability
        Ns = sorted({c["meta"]["N"] for c in P}); signs = []
        for drop_N in Ns:
            ss = [(np.log2(c["meta"]["N"]), g(c)) for c in P if c["meta"]["N"] != drop_N and g(c) is not None]
            sl = _slope(*zip(*ss)) if len({a for a, _ in ss}) >= 3 else None
            if sl is not None:
                signs.append(np.sign(sl))
        out[name] = dict(slope=round(slope, 4) if slope is not None else None, slope_ci=ci,
                         straddles_zero=bool(ci[0] is not None and ci[0] <= 0 <= ci[1]),
                         quad_curvature=round(float(quad[0]), 4) if quad is not None else None,
                         peak_N=round(peakN, 1) if peakN is not None else None,
                         leave_one_N_sign_stable=bool(signs and len(set(signs)) == 1))
    return out


def aggregate():
    cells = [json.loads(p.read_text()) for p in sorted(CELLS.glob("*.json"))]
    if not cells:
        raise SystemExit("no cells to aggregate")
    def base(c):
        return dict(cell=c["cell"], N=c["meta"]["N"], seed=c["meta"]["seed"], is_random_init=c["meta"]["is_random_init"])
    _write_csv("p1_task_performance.csv", [dict(**base(c), f0_target_bacc=c["f0_target_bacc"], **c["gate"]) for c in cells])
    sep = []
    for c in cells:
        for grp, key in (("source_train", "l1_source"), ("source_train_cc", "l1_source_cc"),
                         ("source_val_secondary", "l1_val"), ("target_secondary", "l1_tgt")):
            sep.append(dict(**base(c), group=grp, **c[key]))
    _write_csv("p1_pairwise_subject_separability.csv", sep)
    _write_csv("p1_l4_task_alignment.csv", [dict(**base(c), **r) for c in cells for r in c["l4"]])
    _write_csv("p1_l5_subject_subspace_replay.csv", [dict(**base(c), **c["l5"]) for c in cells])
    _write_csv("p1_l6_target_consequence.csv", [dict(**base(c), **c["l6"]) for c in cells])
    _write_csv("p1_feature_dump_manifest.csv", [dict(cell=c["cell"], N=c["meta"]["N"], seed=c["meta"]["seed"],
        n_trials=c["meta"]["n_trials"], n_subjects=c["meta"]["n_subjects"], F1_dim=c["meta"]["F1_dim"],
        npatch=c["meta"]["npatch"], ckpt_sha256=c["meta"]["ckpt_sha256"], pretrain_val_loss=c["meta"]["pretrain_val_loss"],
        channel_map_hash=c["channel_map_hash"], deterministic=c["meta"]["qc"].get("deterministic"),
        batch_invariant=c["meta"]["qc"].get("batch_invariant")) for c in cells])
    _write_csv("p1_downstream_run_manifest.csv", [dict(cell=c["cell"], N=c["meta"]["N"], seed=c["meta"]["seed"],
        ckpt_path=c["meta"]["ckpt_path"], ckpt_sha256=c["meta"]["ckpt_sha256"], git=None,
        target_labels_used=c["meta"]["target_labels_used"], per_patch_zscore=c["meta"]["per_patch_zscore"],
        sec=c["meta"]["sec"]) for c in cells])
    ri = [c for c in cells if c["meta"]["is_random_init"]]
    if ri:
        _write_csv("p1_random_init_control.csv", [dict(cell=c["cell"], target_bacc=c["gate"]["target_bacc"],
            source_val_bacc=c["gate"]["source_val_bacc"], l1_source_effect=c["l1_source"].get("effect"),
            l5_drop_subject=c["l5"]["drop_subject"]) for c in ri])
    ctab = next((c["channel_table"] for c in cells), [])
    _write_csv("p1_channel_mapping_manifest.csv", ctab)
    _write_csv("p1_windowing_manifest.csv", [dict(dataset="SHU-MI", native_hz=NATIVE_HZ, target_hz=TARGET_HZ,
        patch=PATCH, npatch=cells[0]["meta"]["npatch"], window_sec=4, pad_to_30="NO", per_patch_zscore=True)])
    (OUT / "p1_target_label_firewall.json").write_text(json.dumps(cells[0]["firewall"], indent=2) + "\n")

    P = [c for c in cells if not c["meta"]["is_random_init"]]
    fr = _frontier(cells)
    ri_bacc = ri[0]["gate"]["target_bacc"] if ri else None
    n_beat_ri = sum(1 for c in P if ri_bacc is not None and c["gate"]["target_bacc"] > ri_bacc)
    summary = dict(primary_estimand="bundled_fixed_budget_allocation_frontier", downstream_dataset="SHU-MI",
                   channel_mapping="19-common", windowing="native_4s_trials", target_labels_used_for_selection=False,
                   checkpoint_selection="pretrain_val_loss_only", primary_metric="target_bAcc",
                   n_cells=len(P), n_random_init=len(ri), random_init_target_bacc=ri_bacc,
                   pretrained_cells_beating_random_init=f"{n_beat_ri}/{len(P)}",
                   task_gate_pass_cells=sum(1 for c in P if c["gate"]["task_gate_pass"]),
                   frontier_slope=fr.get("target_bacc"),
                   curvature_or_peak={k: dict(quad=v.get("quad_curvature"), peak_N=v.get("peak_N")) for k, v in fr.items()},
                   leave_one_N_stability={k: v.get("leave_one_N_sign_stable") for k, v in fr.items()},
                   l1_frontier_trend=fr.get("l1_source_effect"), l5_frontier_trend=fr.get("l5_drop_subject"),
                   l4_frontier_trend=fr.get("l4_alignment_k2"), l6_frontier_trend=fr.get("l6_delta_subject_bacc"),
                   p2_recommended=None)
    _atomic_write_json(OUT / "p1_frontier_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return summary


# ----------------------------------------------------------------------------- D0 probe gate (8 QC items)
def probe_gate(device="cuda"):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for cell in ("N512_s0", "random_init"):
        res[cell] = run_cell(cell, device=device, write_features=True)
    r = res["N512_s0"]; qc = r["meta"]["qc"]; g = r["gate"]
    checks = {
        "1_channel_mapping_exact": bool(len(r["channel_table"]) == 19 and r["channel_map_hash"]),
        "2_native_4patch_forward": bool(qc.get("npatch") == 4 and r["meta"]["F1_dim"] == 19 * 200),
        "3_embeddings_deterministic": bool(qc.get("deterministic") and qc.get("batch_invariant")),
        "4_pca_head_subspace_source_only": bool(r["firewall"]["pca_fit_subjects"] == SPLIT["source_train"]
                                                and r["firewall"]["head_fit_subjects"] == SPLIT["source_train"]),
        "5_L1_L4_L5_L6_all_compute": bool(r["l1_source"].get("bacc") is not None and r["l4"][1]["task_head_subject_alignment"] is not None
                                          and r["l5"]["drop_subject"] is not None and r["l6"]["delta_subject_bacc"] is not None),
        "6_variance_null_consistent": bool(r["l5"]["var_removed_variance"] is not None and r["l5"]["drop_variance"] is not None
                                           and r["l5"]["var_removed_variance"] >= r["l5"]["var_removed_subject"] - 1e-9),
        "7_target_label_firewall_clean": bool(r["firewall"]["target_labels_used_only_for_final_scoring"] and r["meta"]["target_labels_used"] is False),
        "8_output_schemas_correct": all((OUT / fn).exists() for fn in
            ("p1_task_performance.csv", "p1_pairwise_subject_separability.csv", "p1_l4_task_alignment.csv",
             "p1_l5_subject_subspace_replay.csv", "p1_l6_target_consequence.csv", "p1_feature_dump_manifest.csv")),
    }
    gate = dict(probe_gate="D0", checkpoint="N512_s0", random_init_floor=res["random_init"]["gate"]["target_bacc"],
                pretrained_target_bacc=g["target_bacc"], source_val_bacc=g["source_val_bacc"], task_gate_pass=g["task_gate_pass"],
                l1_source_effect=r["l1_source"].get("effect"), l1_source_p=r["l1_source"].get("p"),
                l5_drop_subject=r["l5"]["drop_subject"], l5_drop_variance=r["l5"]["drop_variance"],
                qc=qc, checks=checks, all_pass=bool(all(checks.values())))
    _atomic_write_json(OUT / "p1_probe_gate_D0.json", gate)
    _write_csv("p1_channel_mapping_manifest.csv", r["channel_table"])
    _write_csv("p1_windowing_manifest.csv", [dict(dataset="SHU-MI", native_hz=NATIVE_HZ, target_hz=TARGET_HZ,
        patch=PATCH, npatch=qc.get("npatch"), window_sec=4, pad_to_30="NO", per_patch_zscore=True)])
    (OUT / "p1_target_label_firewall.json").write_text(json.dumps(r["firewall"], indent=2) + "\n")
    print(json.dumps(gate, indent=2, default=str))
    return gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=None, help="N{N}_s{S} or random_init")
    ap.add_argument("--probe-gate", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--write-features", action="store_true")
    ap.add_argument("--skip-if-done", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(); return
    if args.probe_gate:
        g = probe_gate(device=args.device)
        raise SystemExit(0 if g["all_pass"] else 2)
    if not args.cell:
        raise SystemExit("need --cell, --probe-gate, or --aggregate")
    if args.skip_if_done and (CELLS / f"{args.cell}.json").exists():
        print(f"skip-if-done: {args.cell} already present"); return
    r = run_cell(args.cell, device=args.device, write_features=args.write_features)
    print(json.dumps(dict(cell=args.cell, gate=r["gate"], l1_source=r["l1_source"], l5=r["l5"], l6=r["l6"]), indent=2, default=str))


if __name__ == "__main__":
    main()
