#!/usr/bin/env python
"""CMI-Trace Version 1 Fixed-P Train-Through-Erasure runner (DGCNN, real EEG, GPU).

Per LOSO fold/seed: (1) ERM warm-up; (2) estimate a FIXED source-only projector P_0 per arm; (3) fine-tune the
TOP block + a fresh head THROUGH the erasure (I-P_0) with the lower encoder frozen; (4) score target eval-only.
Arms: full (no erasure control, same fine-tune) / exact_head_null / subject / random (matched rank).

Metrics (pre-frozen success = kept-branch CMI down, source task retained >=-0.02, informed beats random):
target bAcc, source bAcc, kept-branch posterior-KL CMI, kept-branch linear subject residual.

  python scripts/run_tte_v1.py --dataset BNCI2014_001 --folds 0 1 2 --seeds 0 --warmup_epochs 60 --tte_epochs 60 --device cuda
"""
from __future__ import annotations
import argparse, copy, glob, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cmi.data import moabb_data
from cmi.models.graph_task_backbones import build_graph_task_backbone
from cmi.train.trainer import train_model, predict
from cmi.train.train_through_erasure import (train_through_erasure, predict_erased, kept_graph_z,
                                             subject_projector, exact_head_null_projector_gz, random_projector)
from cmi.eval.probe_splits import stratified_trial_split_by_y_d
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

OUT = REPO / "results" / "cmi_trace_tte_v1"
CAND = "dgcnn_forward_graph_adapter"


def _cfg_hash():
    c = REPO / "configs" / "cmi_trace_relaxation_ladder.yaml"
    return hashlib.sha256(c.read_bytes()).hexdigest() if c.exists() else "no_config"


def _remap(d):
    u = {v: i for i, v in enumerate(sorted(np.unique(d)))}
    return np.array([u[v] for v in d], dtype=np.int64), len(u)


@torch.no_grad()
def _graph_z(adapter, X, device, bs=256):
    adapter.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(np.asarray(X[i:i + bs]), dtype=torch.float32).to(device)
        _, gz, _, _ = adapter.forward_graph(xb); out.append(gz.cpu().numpy())
    return np.concatenate(out)


def _linear_subject_residual(Z, y, d, seed=0):
    """label-conditional linear subject decode balanced accuracy (residual leakage witness)."""
    accs = []
    rng = np.random.default_rng(seed)
    for c in np.unique(y):
        m = y == c; zz, dd = Z[m], d[m]
        if len(np.unique(dd)) < 2:
            continue
        idx = rng.permutation(len(zz)); cut = int(0.7 * len(idx)); tr, ev = idx[:cut], idx[cut:]
        if len(np.unique(dd[tr])) < 2 or len(ev) == 0:
            continue
        accs.append(balanced_accuracy_score(dd[ev], LogisticRegression(max_iter=300).fit(zz[tr], dd[tr]).predict(zz[ev])))
    return float(np.mean(accs)) if accs else float("nan")


def _kept_cmi(Zk, y, d, n_cls, n_dom, seed, device, n_perm=20, epochs=60):
    from cmi.eval.conditional_subject_leakage import three_way_support_split, flat_conditional_cmi
    _, pt, pe, _ = three_way_support_split(y.astype(int), d.astype(int), seed=seed)
    r = flat_conditional_cmi(Zk, y.astype(int), d.astype(int), n_cls, n_dom, pt, pe,
                             n_perm=n_perm, seed=seed, epochs=epochs, device=device, with_residual=False)
    return r["posterior_kl_nats"], r["excess_over_null"]


def run(args):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[tte] --device cuda but CUDA unavailable (fail closed)")
    X, y, meta, classes = moabb_data.load(args.dataset, tmin=0.5, tmax=3.5, resample=128)
    dom_all, _ = moabb_data.domain_labels(meta, "subject")
    folds = list(moabb_data.loso_splits(meta))
    sel = args.folds if args.folds is not None else list(range(len(folds)))
    n_cls = len(classes)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / f"{args.dataset}_raw_rows.jsonl"
    cfg_hash = _cfg_hash()
    t0 = time.time()
    with open(raw, "a") as fh:
        for fi in sel:
            tgt, tr_mask, te_mask = folds[fi]
            Xs, ys = X[tr_mask], y[tr_mask]; ds, n_dom = _remap(dom_all[tr_mask])
            Xte, yte = X[te_mask], y[te_mask]
            enc_idx, pool_idx, _ = stratified_trial_split_by_y_d(ys, ds, train_frac=0.7, seed=0, min_per_cell=2)
            for seed in args.seeds:
                # (1) ERM warm-up
                torch.manual_seed(seed); np.random.seed(seed)
                base = build_graph_task_backbone(CAND, X.shape[1], X.shape[2], n_cls).to(device)
                base, _, _ = train_model(base, Xs[enc_idx], ys[enc_idx], ds[enc_idx], n_cls, method="erm",
                                         epochs=args.warmup_epochs, bs=args.bs, warmup=max(1, args.warmup_epochs // 5),
                                         device=device, seed=seed)
                gz_pool = _graph_z(base, Xs[pool_idx], device)
                W = base.net.head.weight.detach().cpu().numpy()
                d = gz_pool.shape[1]
                k = min(n_dom - 1, d)
                # (2) source-only projectors
                P_ehn, rank_ehn = exact_head_null_projector_gz(gz_pool, ys[pool_idx], ds[pool_idx], W, k)
                P_subj = subject_projector(gz_pool, ys[pool_idx], ds[pool_idx], rank_ehn or k)
                P_rand = random_projector(d, rank_ehn or k, seed=1000 + seed)
                arms = {"full": np.zeros((d, d)), "exact_head_null": P_ehn, "subject": P_subj, "random": P_rand}
                for arm, P0 in arms.items():
                    # (3) train-through-erasure (fresh clone of the warm-up)
                    adap = copy.deepcopy(base)
                    adap = train_through_erasure(adap, P0, Xs[enc_idx], ys[enc_idx], freeze_lower=args.freeze_lower,
                                                 reinit_head=True, epochs=args.tte_epochs, bs=args.bs,
                                                 device=device, seed=seed)
                    # (4) eval
                    tgt_b = balanced_accuracy_score(yte, predict_erased(adap, Xte, P0, device).argmax(1))
                    src_b = balanced_accuracy_score(ys[pool_idx], predict_erased(adap, Xs[pool_idx], P0, device).argmax(1))
                    Zk = kept_graph_z(adap, Xs[pool_idx], P0, device)
                    kl, ex = _kept_cmi(Zk, ys[pool_idx], ds[pool_idx], n_cls, n_dom, seed, device,
                                       n_perm=args.n_perm)
                    subj_res = _linear_subject_residual(Zk, ys[pool_idx], ds[pool_idx], seed=seed)
                    row = dict(dataset=args.dataset, backbone=CAND, fold=int(fi), heldout_subject=str(tgt),
                               seed=int(seed), arm=arm, projector_rank=int(rank_ehn if arm != "full" else 0),
                               target_bacc=float(tgt_b), source_bacc=float(src_b),
                               kept_branch_cmi=float(kl), kept_branch_cmi_excess=float(ex),
                               kept_branch_subject_residual=float(subj_res),
                               freeze_lower=bool(args.freeze_lower), warmup_epochs=args.warmup_epochs,
                               tte_epochs=args.tte_epochs, config_hash=cfg_hash)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"  fold{fi} sub{tgt} seed{seed} {arm:16s} tgt={tgt_b:.3f} src={src_b:.3f} "
                          f"keptCMI={kl:.3f} subjRes={subj_res:.3f} rank={row['projector_rank']} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[tte] wrote -> {raw}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--warmup_epochs", type=int, default=60)
    ap.add_argument("--tte_epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n_perm", type=int, default=20)
    ap.add_argument("--freeze_lower", action="store_true", default=True)
    ap.add_argument("--no_freeze_lower", dest="freeze_lower", action="store_false")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    run(ap.parse_args())


if __name__ == "__main__":
    main()
