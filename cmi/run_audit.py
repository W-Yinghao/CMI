"""Run the leakage audit across methods: train on source (a few LOSO folds), extract frozen Z,
and run the probe-ensemble + permutation + grouped audit. Adds random-encoder and raw-covariance
baselines (expected leakage: raw high, random low). CPU/GPU.

  python -m cmi.run_audit --dataset BNCI2014_001 --configs erm:0 lpc_prior:0.3 lpc_uniform:0.5 cdann:1
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch

from cmi.run_loso import load, _remap
from cmi.data.moabb_data import domain_labels, loso_splits
from cmi.models.backbones import build_backbone
from cmi.train.trainer import train_model, embed
from cmi.eval.leakage_audit import audit


def parse_cfg(c):
    p = c.split(":"); m = p[0]; nums = [float(x) for x in p[1:]]
    if m == "lpc_supcon":
        return m, nums[0], nums[1]
    return m, (nums[0] if nums else 0.0), 0.0


def _raw_cov(X):
    """Per-trial covariance upper-triangle — a 'raw' feature that strongly encodes the subject."""
    out = []
    iu = np.triu_indices(X.shape[1])
    for x in X:
        c = np.cov(x)
        out.append(c[iu])
    return np.asarray(out, dtype="float32")


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, meta, classes = load(args.dataset, tmin=args.tmin, tmax=args.tmax, resample=args.resample)
    dom_all, _ = domain_labels(meta, "subject")
    sess = meta["session"].astype(str).to_numpy() if "session" in meta else None
    n_cls = len(classes); n_ch, n_t = X.shape[1], X.shape[2]
    print(f"[{args.dataset}] X={X.shape} classes={classes} sessions={sess is not None}", flush=True)

    agg = {}                                            # cfg -> list of (random-split audit, grouped audit)
    t0 = time.time()
    for tgt, tr, te in list(loso_splits(meta))[:args.n_folds]:
        Xs, ys = X[tr], y[tr]
        ds, n_dom = _remap(dom_all[tr])
        gs = sess[tr] if sess is not None else None
        for cfg in list(args.configs) + ["raw", "random"]:
            if cfg == "raw":
                Z = _raw_cov(Xs)
            elif cfg == "random":
                Z = embed(build_backbone(args.backbone, n_ch, n_t, n_cls, device=device), Xs, device)
            else:
                m, lam, gamma = parse_cfg(cfg)
                bb = build_backbone(args.backbone, n_ch, n_t, n_cls, device=device)
                bb, _, _ = train_model(bb, Xs, ys, ds, n_cls, method=m, lam=lam, gamma=gamma,
                                       epochs=args.epochs, bs=args.bs, warmup=args.warmup,
                                       n_inner=args.n_inner, sampler=args.sampler, device=device, seed=args.seed)
                Z = embed(bb, Xs, device)
            a = audit(Z, ys, ds, n_cls, n_dom, seed=args.seed)
            ag = audit(Z, ys, ds, n_cls, n_dom, seed=args.seed, groups=gs) if gs is not None else {}
            agg.setdefault(cfg, []).append((a, ag))
            print(f"  tgt={tgt} {cfg:14s} mlp_l_adv={a.get('mlp_l_adv',0):+.3f} rf_adv={a.get('rf_adv',0):+.3f} "
                  f"knn={a.get('knn_cmi',0):.2f} perm_null={a.get('perm_null_adv',0):+.3f} ({time.time()-t0:.0f}s)", flush=True)

    keys = ["linear_adv", "mlp_s_adv", "mlp_l_adv", "rf_adv", "hgbm_adv", "hsic", "knn_cmi", "perm_null_adv", "prior_bacc"]
    summary = {}
    print(f"\n=== {args.dataset} leakage audit (avg over {args.n_folds} folds; D=subject; adv = domain bacc - prior) ===")
    print(f"{'method':14s} " + " ".join(f"{k.replace('_adv',''):>9s}" for k in keys) + f" {'grp_mlpl':>9s}")
    for cfg in agg:
        rs = [a for a, _ in agg[cfg]]
        summary[cfg] = {k: float(np.nanmean([r.get(k, np.nan) for r in rs])) for k in keys}
        grp = [g.get("mlp_l_adv", np.nan) for _, g in agg[cfg] if g]
        summary[cfg]["grouped_mlp_l_adv"] = float(np.nanmean(grp)) if grp else float("nan")
        s = summary[cfg]
        print(f"{cfg:14s} " + " ".join(f"{s[k]:9.3f}" for k in keys) + f" {s['grouped_mlp_l_adv']:9.3f}")
    if args.out:
        json.dump(dict(config=vars(args), classes=classes, summary=summary), open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--configs", nargs="+", default=["erm:0", "lpc_prior:0.3", "lpc_uniform:0.5", "cdann:1"])
    ap.add_argument("--n_folds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--n_inner", type=int, default=2)
    ap.add_argument("--sampler", default="classbal")
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
