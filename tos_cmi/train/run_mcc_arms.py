"""MCC 3-arm continuation trainer (GPU, env `icml`). Per (dataset, target_subject, seed): train ONE ERM warm-up
(the existing recipe) and cache it, then fork THREE arms that continue 20 epochs from that SAME warm-up:
  A = ERM-continue (task only)   B = MCC-true   C = MCC-shuffle (matched control).
Each arm dumps frozen Z/logits/labels in the EXACT feature_dump blob format so the frozen mechanism oracle and the
geometry/DG aggregator re-audit them. Checkpoint SELECTION = source-only validation. Target X/Y = evaluation only,
NEVER in training. Manuscript FROZEN; only the project owner stops a scientific line.

  python -m tos_cmi.train.run_mcc_arms --dataset BNCI2014_001 --subject 1 --seed 0 --device cuda \
      --out-dir results/cmi_trace_mcc --warmup-epochs 300 --cont-epochs 20
  # smoke: --warmup-epochs 2 --cont-epochs 2 --smoke
"""
from __future__ import annotations
import argparse, copy, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from cmi.paths import configure_offline_moabb
from cmi.data.moabb_data import load, domain_labels
from cmi.models.backbones import build_backbone
from cmi.train.trainer import train_model
from tos_cmi.eeg.feature_dump import _forward_dump, _remap, _git_sha
from tos_cmi.train.mechanism_consistency import mcc_loss, BalancedSubjectClassSampler, contrast_norm, effective_rank

ARMS = {"A_erm_continue": "task_only", "B_mcc_true": "mcc_true", "C_mcc_shuffle": "mcc_shuffle"}
SUBJECTS = {"BNCI2014_001": list(range(1, 10)), "BNCI2015_001": list(range(1, 13))}   # 9 + 12 LOSO subjects
SEEDS = [0, 1, 2]


def enumerate_bundles():
    """The 63 (dataset, subject, seed) bundles = 2 datasets x (9+12) subjects x 3 seeds. Each bundle -> 1 warm-up
    -> 3 arms = 189 continuation cells. Deterministic order for the SLURM array index."""
    return [(ds, s, sd) for ds in ("BNCI2014_001", "BNCI2015_001") for s in SUBJECTS[ds] for sd in SEEDS]


def _state_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def _source_val_split(dtr, ytr, frac=0.2, seed=0):
    """Stratified per (subject,class) holdout so BOTH continuation-train and source-val cover every cell."""
    rng = np.random.default_rng(seed); tr, va = [], []
    for s in np.unique(dtr):
        for c in np.unique(ytr):
            idx = np.where((dtr == s) & (ytr == c))[0]; rng.shuffle(idx)
            k = max(1, int(round(len(idx) * frac)))
            va += idx[:k].tolist(); tr += idx[k:].tolist()
    return np.array(sorted(tr)), np.array(sorted(va))


@torch.no_grad()
def _bacc(bb, X, y, device, bs=256):
    bb.eval(); from sklearn.metrics import balanced_accuracy_score
    pred = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32).to(device)
        pred.append(bb(xb)[0].argmax(1).cpu().numpy())
    return float(balanced_accuracy_score(y, np.concatenate(pred)))


def _warmup(dataset, target_subject, seed, device, warmup_epochs, bs, cache_dir):
    """Train (or load) the ONE ERM warm-up for this cell. Returns (backbone, Xtr, ytr, dtr, n_cls, Xte, yte, meta, warm_hash)."""
    configure_offline_moabb()
    X, y, meta, classes = load(dataset)
    n_cls = len(classes); dom_all, _ = domain_labels(meta, "subject")
    subj = meta["subject"].to_numpy(); sess = meta["session"].astype(str).to_numpy()
    te = subj == target_subject; tr = ~te
    if te.sum() == 0:
        raise ValueError(f"target_subject {target_subject} not in {dataset}")
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    dtr, n_dom = _remap(dom_all[tr])
    cache = None
    if cache_dir:
        cache = Path(cache_dir) / f"warmup_{dataset}_sub{target_subject}_seed{seed}_ep{warmup_epochs}.pt"
    torch.manual_seed(seed); np.random.seed(seed)
    bb = build_backbone("EEGNet", X.shape[1], X.shape[2], n_cls, device=device)
    if cache and cache.exists():
        bb.load_state_dict(torch.load(cache, map_location=device)); warm_hash = _state_hash(bb.state_dict())
    else:
        bb, _post, _diag = train_model(bb, Xtr, ytr, dtr, n_cls, method="erm", lam=0.0,
                                       epochs=warmup_epochs, bs=bs, warmup=40, device=device, seed=seed)
        warm_hash = _state_hash(bb.state_dict())
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True); torch.save(bb.state_dict(), cache)
    meta_arr = dict(subject_source=subj[tr].astype("int64"), subject_target=subj[te].astype("int64"),
                    session_source=sess[tr], session_target=sess[te], classes=np.array(classes),
                    domain_source=dtr, n_dom=n_dom)
    return bb, Xtr, ytr, dtr, n_cls, Xte, yte, dataset, classes, meta_arr, warm_hash, X.shape


def _continue(warm_sd, Xtr, ytr, dtr, n_cls, X_shape, arm, device, cont_epochs, lr, K, lam_mcc, ramp, seed):
    """Fork a fresh backbone from the warm-up state, continue cont_epochs with the arm's loss; source-only ckpt select."""
    bb = build_backbone("EEGNet", X_shape[1], X_shape[2], n_cls, device=device)
    bb.load_state_dict(copy.deepcopy(warm_sd))
    tr_idx, va_idx = _source_val_split(dtr, ytr, seed=seed)
    samp = BalancedSubjectClassSampler(dtr[tr_idx], ytr[tr_idx], K=K,
                                       n_batches=max(1, len(tr_idx) // (len(np.unique(dtr)) * n_cls * K)), seed=seed)
    opt = torch.optim.AdamW(bb.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed + 777)
    # DIAGNOSTIC 1: full-encoder-trainable assertion (unfreeze is a no-op -- the whole backbone is already trainable)
    params = list(bb.parameters()); total_p = int(sum(p.numel() for p in params))
    train_p = int(sum(p.numel() for p in params if p.requires_grad))
    assert train_p == total_p, f"expected FULL encoder trainable, got {train_p}/{total_p} (frozen params present)"
    names_hash = hashlib.sha256("|".join(sorted(n for n, p in bb.named_parameters() if p.requires_grad)).encode()).hexdigest()[:16]
    best = dict(val=-1.0, sd=None, ep=-1); diag = dict(mcc_loss=[], ce_loss=[], grad_ratio=[], grad_cos=[])
    for ep in range(cont_epochs):
        lam_t = 0.0 if arm == "A_erm_continue" else lam_mcc * min(1.0, (ep + 1) / max(1, ramp))
        bb.train(); first = True
        for local in samp:
            gi = tr_idx[local]
            xb = torch.tensor(Xtr[gi], dtype=torch.float32).to(device)
            yb = torch.tensor(ytr[gi], dtype=torch.long).to(device); db = torch.tensor(dtr[gi], dtype=torch.long).to(device)
            logits, z = bb(xb); ce = F.cross_entropy(logits, yb)
            if arm == "A_erm_continue":
                loss = ce; mcc_v = 0.0
            else:
                mcc, _ = mcc_loss(z, yb, db, shuffle_subjects=(arm == "C_mcc_shuffle"), generator=gen)
                loss = ce + lam_t * mcc; mcc_v = float(mcc.detach())
                # DIAGNOSTIC 2 (read-only, first batch/epoch): is lambda*||grad L_MCC|| non-trivial vs ||grad L_task||?
                if first and lam_t > 0:
                    gt = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
                    gm = torch.autograd.grad(mcc, params, retain_graph=True, allow_unused=True)
                    ft = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gt, params)])
                    fm = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gm, params)])
                    nt = float(ft.norm()); nm = float(fm.norm())
                    diag["grad_ratio"].append(lam_t * nm / (nt + 1e-9)); diag["grad_cos"].append(float((ft @ fm) / (nt * nm + 1e-9)))
            opt.zero_grad(); loss.backward(); opt.step()
            diag["mcc_loss"].append(mcc_v); diag["ce_loss"].append(float(ce.detach())); first = False
        val = _bacc(bb, Xtr[va_idx], ytr[va_idx], device)
        if val > best["val"]:
            best = dict(val=val, sd=copy.deepcopy(bb.state_dict()), ep=ep)
    bb.load_state_dict(best["sd"])
    # collapse guards on the SELECTED model (source-train features)
    with torch.no_grad():
        Zc = torch.tensor(_forward_dump(bb, Xtr[tr_idx][:2048], device)[1], dtype=torch.float32)
    diag.update(selected_epoch=best["ep"], source_val_bacc=best["val"],
                feature_norm=float(Zc.norm(dim=1).mean()), effective_rank=effective_rank(Zc),
                contrast_norm=contrast_norm(Zc, ytr[tr_idx][:2048], dtr[tr_idx][:2048]),
                mean_mcc_loss=float(np.mean(diag["mcc_loss"])) if diag["mcc_loss"] else 0.0,
                mean_ce_loss=float(np.mean(diag["ce_loss"])),
                total_parameter_count=total_p, trainable_parameter_count=train_p,
                trainable_parameter_names_hash=names_hash, full_encoder_trainable=bool(train_p == total_p),
                mean_grad_ratio=float(np.mean(diag["grad_ratio"])) if diag["grad_ratio"] else 0.0,
                mean_grad_cos=float(np.mean(diag["grad_cos"])) if diag["grad_cos"] else float("nan"))
    return bb, diag


def _dump_arm(bb, arm, dataset, target_subject, seed, meta_arr, Xtr, ytr, Xte, yte, classes, device,
              warm_hash, lam_mcc, diag, out_dir):
    lg_src, Z_src = _forward_dump(bb, Xtr, device); lg_tgt, Z_tgt = _forward_dump(bb, Xte, device)
    blob = dict(Z_source=Z_src, Z_target=Z_tgt, logits_source=lg_src, logits_target=lg_tgt,
                y_source=ytr.astype("int64"), y_target=yte.astype("int64"), domain_source=meta_arr["domain_source"],
                subject_source=meta_arr["subject_source"], subject_target=meta_arr["subject_target"],
                session_source=meta_arr["session_source"], session_target=meta_arr["session_target"],
                dataset=dataset, backbone="EEGNet", method=f"mcc_{arm}", lam=np.float64(lam_mcc),
                seed=np.int64(seed), target_subject=np.int64(target_subject), n_cls=np.int64(len(classes)),
                n_dom_source=np.int64(meta_arr["n_dom"]), z_dim=np.int64(Z_src.shape[1]), classes=np.array(classes),
                tmin=np.float64(0.5), tmax=np.float64(3.5), resample=np.int64(128), domain_mode="subject",
                git_sha=_git_sha(REPO), tsmnet_sha="", arm=arm, warmup_hash=warm_hash,
                train_diag=json.dumps({k: v for k, v in diag.items() if isinstance(v, (int, float))}))
    outd = Path(out_dir); outd.mkdir(parents=True, exist_ok=True)
    p = outd / f"{dataset}_sub{target_subject}_seed{seed}_{arm}.npz"
    np.savez_compressed(p, **blob)
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset"); ap.add_argument("--subject")
    ap.add_argument("--bundle-index", type=int, default=None); ap.add_argument("--list-bundles", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/cmi_trace_mcc"); ap.add_argument("--cache-dir", default="results/cmi_trace_mcc/warmup_cache")
    ap.add_argument("--warmup-epochs", type=int, default=300); ap.add_argument("--cont-epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lam-mcc", type=float, default=0.25); ap.add_argument("--ramp", type=int, default=5)
    ap.add_argument("--cont-lr", type=float, default=1e-4); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-warmup-from", default=None)   # prior-round dir; assert warm-up hash matches per bundle
    a = ap.parse_args()
    bundles = enumerate_bundles()
    if a.list_bundles:
        for i, (ds, s, sd) in enumerate(bundles):
            print(f"{i}\t{ds}\tsub{s}\tseed{sd}")
        print(f"# {len(bundles)} bundles x 3 arms = {len(bundles)*3} continuation cells"); return
    if a.bundle_index is not None:
        a.dataset, a.subject, a.seed = bundles[a.bundle_index][0], str(bundles[a.bundle_index][1]), bundles[a.bundle_index][2]
    if not a.dataset or a.subject is None:
        ap.error("need --dataset+--subject, or --bundle-index, or --list-bundles")
    subj = int(a.subject) if str(a.subject).isdigit() else a.subject
    print(f"[mcc] {a.dataset} sub{subj} seed{a.seed} device={a.device} warmup_ep={a.warmup_epochs} cont_ep={a.cont_epochs}", flush=True)
    bb, Xtr, ytr, dtr, n_cls, Xte, yte, dataset, classes, meta_arr, warm_hash, X_shape = _warmup(
        a.dataset, subj, a.seed, a.device, a.warmup_epochs, a.bs, a.cache_dir)
    warm_sd = copy.deepcopy(bb.state_dict())
    print(f"[mcc] warm-up done, hash={warm_hash}", flush=True)
    # reuse warm-ups across rounds ONLY with exact hash match (PM requirement): compare to the prior-round manifest.
    if a.verify_warmup_from:
        prev = Path(a.verify_warmup_from) / f"{a.dataset}_sub{subj}_seed{a.seed}.manifest.json"
        if prev.exists():
            pw = json.loads(prev.read_text()).get("warmup_hash")
            assert pw == warm_hash, f"warm-up hash mismatch vs prior round: {warm_hash} != {pw} (stale/retrained warm-up)"
            print(f"[mcc] warm-up hash verified == prior round ({pw})", flush=True)
    manifest = dict(dataset=a.dataset, subject=str(subj), seed=a.seed, warmup_hash=warm_hash, lam_mcc=a.lam_mcc, arms={})
    for arm in ARMS:
        b2, diag = _continue(warm_sd, Xtr, ytr, dtr, n_cls, X_shape, arm, a.device,
                             a.cont_epochs, a.cont_lr, a.K, a.lam_mcc, a.ramp, a.seed)
        p = _dump_arm(b2, arm, a.dataset, subj, a.seed, meta_arr, Xtr, ytr, Xte, yte, classes, a.device,
                      warm_hash, a.lam_mcc, diag, a.out_dir)
        tgt_bacc = _bacc(b2, Xte, yte, a.device)
        manifest["arms"][arm] = dict(dump=Path(p).name, target_bacc=tgt_bacc, selected_epoch=diag["selected_epoch"],
                                     source_val_bacc=diag["source_val_bacc"], eff_rank=diag["effective_rank"],
                                     contrast_norm=diag["contrast_norm"], mean_mcc_loss=diag["mean_mcc_loss"],
                                     total_parameter_count=diag["total_parameter_count"],
                                     trainable_parameter_count=diag["trainable_parameter_count"],
                                     trainable_parameter_names_hash=diag["trainable_parameter_names_hash"],
                                     full_encoder_trainable=diag["full_encoder_trainable"],
                                     mean_grad_ratio=diag["mean_grad_ratio"], mean_grad_cos=diag["mean_grad_cos"])
        print(f"  {arm}: target_bAcc={tgt_bacc:.4f} sel_ep={diag['selected_epoch']} effrank={diag['effective_rank']:.2f} "
              f"cnorm={diag['contrast_norm']:.3f} mcc={diag['mean_mcc_loss']:.4f} trainable={diag['trainable_parameter_count']}/{diag['total_parameter_count']} "
              f"grad_ratio={diag['mean_grad_ratio']:.3f} grad_cos={diag['mean_grad_cos']:+.3f}", flush=True)
    outd = Path(a.out_dir); (outd / f"{a.dataset}_sub{subj}_seed{a.seed}.manifest.json").write_text(json.dumps(manifest, indent=2))
    (outd / f"{a.dataset}_sub{subj}_seed{a.seed}.done").write_text(f"{warm_hash}\t3 arms\n")
    print(f"[mcc] cell done: {a.dataset} sub{subj} seed{a.seed} -> 3 arms + manifest + .done", flush=True)


if __name__ == "__main__":
    main()
