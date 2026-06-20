"""Protocol C: cross-dataset binary MI (leave-one-dataset-out).

Train on all subjects of K-1 datasets (each source dataset:subject is a domain D), test on
an entirely unseen dataset. Same harness/frameworks/metrics as cmi.run_loso.

  python -m cmi.run_cross_dataset --backbone EEGNet \
      --configs erm:0 lpc_prior:0.3 cdann:1 --epochs 300 --out results/xdata_EEGNet.json
  python -m cmi.run_cross_dataset --backbone EEGNet \
      --configs erm:0 lpc_prior:0.1 dualpc:0.1:0.05 --epochs 4 --max_subj 2 --device cpu
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch

from cmi.data.cross_dataset import load_cross
from cmi.models.backbones import build_backbone
from cmi.train.trainer import train_model, predict, resolve_dec_margin
from cmi.eval.metrics import (classification_metrics, leakage_probe, marginal_leakage_probe,
                              decoder_leakage_probe, label_separability, add_decoder_valid_means)


def parse_cfg(c):
    parts = c.split(":"); method = parts[0]; nums = [float(x) for x in parts[1:]]
    lam_edge = 0.0
    z_margin = 0.0
    dec_scale = 1.0
    if method == "supcon":
        lam, gamma = 0.0, nums[0]
    elif method in ("lpc_supcon", "lpc_simclr", "lpc_byol"):
        lam, gamma = nums[0], nums[1]
    elif method == "graphcmi":
        lam, gamma, lam_edge = nums[0], (nums[1] if len(nums) > 1 else 0.0), (nums[2] if len(nums) > 2 else 0.0)
    elif method in ("dual", "dualc", "dualpc", "dualpc_hinge", "dualpc_marginal"):
        lam, gamma = nums[0], (nums[1] if len(nums) > 1 else nums[0])
        if method == "dualpc_hinge":
            z_margin = nums[2] if len(nums) > 2 else 0.0
            dec_scale = nums[3] if len(nums) > 3 else 1.0
    else:
        lam, gamma = (nums[0] if nums else 0.0), 0.0
    return c, method, lam, gamma, lam_edge, z_margin, dec_scale


def run(args):
    from cmi.train.trainer import ALL_METHODS
    bad = [c for c in args.configs if c.split(":")[0] not in ALL_METHODS]
    if bad:
        raise ValueError(f"unknown method(s) in configs: {bad}; allowed: {sorted(ALL_METHODS)}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    X, y, meta, classes = load_cross(tuple(args.datasets), tmin=args.tmin, tmax=args.tmax,
                                     resample=args.resample, max_subj=args.max_subj or None)
    n_cls = len(classes); n_ch, n_t = X.shape[1], X.shape[2]
    print(f"[xdata] X={X.shape} classes={classes} datasets={args.datasets} device={device} "
          f"sizes={meta.groupby('dataset').size().to_dict()}", flush=True)
    configs = [parse_cfg(c) for c in args.configs]
    results = {lbl: [] for lbl, *_ in configs}
    t0 = time.time()
    for tgt in args.datasets:
        te = (meta["dataset"] == tgt).to_numpy(); tr = ~te
        Xtr, ytr = X[tr], y[tr]
        dom_src = meta["domain"][tr].to_numpy()
        uniq = {v: i for i, v in enumerate(sorted(set(dom_src)))}
        dtr = np.array([uniq[v] for v in dom_src], dtype=np.int64)
        Xte, yte = X[te], y[te]
        rng = np.random.default_rng(args.seed); idx = rng.permutation(len(Xtr)); cut = int(0.7 * len(idx))
        pi, ei = idx[:cut], idx[cut:]
        for lbl, method, lam, gamma, lam_edge, z_margin, dec_scale in configs:
            bb = build_backbone(args.backbone, n_ch, n_t, n_cls, device=device)
            bb, _, diag = train_model(bb, Xtr, ytr, dtr, n_cls, method=method, lam=lam, gamma=gamma,
                                lam_edge=lam_edge,
                                dec_margin=resolve_dec_margin(method, args.dec_margin),
                                z_margin=z_margin, dec_scale=dec_scale,
                                epochs=args.epochs, bs=args.bs, warmup=args.warmup,
                                n_inner=args.n_inner, sampler=args.sampler,
                                weight_decay=args.weight_decay, device=device, seed=args.seed)
            cm = classification_metrics(predict(bb, Xte, device), yte)
            lk = leakage_probe(bb, Xtr[pi], ytr[pi], dtr[pi], Xtr[ei], ytr[ei], dtr[ei], n_cls, device=device)
            lk_rw = leakage_probe(bb, Xtr[pi], ytr[pi], dtr[pi], Xtr[ei], ytr[ei], dtr[ei],
                                  n_cls, device=device, reweight=True)
            mlk = marginal_leakage_probe(bb, Xtr[pi], ytr[pi], dtr[pi], Xtr[ei], ytr[ei], dtr[ei],
                                         n_cls, device=device)
            mlk_rw = marginal_leakage_probe(bb, Xtr[pi], ytr[pi], dtr[pi], Xtr[ei], ytr[ei], dtr[ei],
                                            n_cls, device=device, reweight=True)
            dlk = decoder_leakage_probe(bb, Xtr[pi], ytr[pi], dtr[pi], Xtr[ei], ytr[ei], dtr[ei],
                                        n_cls, device=device)
            dlk_rw = decoder_leakage_probe(bb, Xtr[pi], ytr[pi], dtr[pi], Xtr[ei], ytr[ei], dtr[ei],
                                           n_cls, device=device, reweight=True)
            ls = label_separability(bb, Xtr[pi], ytr[pi], Xtr[ei], ytr[ei], device)
            results[lbl].append(dict(target=tgt, **cm, **lk, label_sep=ls,
                                     leakage_kl_rw=lk_rw["leakage_kl"],
                                     leakage_advantage_rw=lk_rw["leakage_advantage"],
                                     marginal_leakage_kl=mlk["marginal_leakage_kl"],
                                     marginal_leakage_kl_rw=mlk_rw["marginal_leakage_kl"],
                                     marginal_leakage_advantage=mlk["marginal_leakage_advantage"],
                                     marginal_leakage_advantage_rw=mlk_rw["marginal_leakage_advantage"],
                                     decoder_cmi=dlk["decoder_cmi"],
                                     decoder_cmi_rw=dlk_rw["decoder_cmi"],
                                     decoder_cmi_res=dlk["decoder_cmi_res"],
                                     decoder_cmi_res_rw=dlk_rw["decoder_cmi_res"],
                                     decoder_js_res=dlk["decoder_js_res"],
                                     decoder_js_res_rw=dlk_rw["decoder_js_res"],
                                     decoder_valid=bool(dlk["decoder_valid"]),
                                     decoder_n_domains=dlk["decoder_n_domains"],
                                     decoder_min_domain_classes=dlk["decoder_min_domain_classes"],
                                     decoder_mean_domain_classes=dlk["decoder_mean_domain_classes"],
                                     decoder_single_class_frac=dlk["decoder_single_class_frac"],
                                     decoder_domain_class_spans=dlk["decoder_domain_class_spans"],
                                     decoder_domain_counts=dlk["decoder_domain_counts"],
                                     inloop_reg=diag["inloop_reg"],
                                     inloop_dec=diag.get("inloop_dec", 0.0),
                                     inloop_dec_loss=diag.get("inloop_dec_loss", 0.0),
                                     train_dec_margin=diag.get("dec_margin", resolve_dec_margin(method, args.dec_margin)),
                                     train_sampler=diag.get("sampler", args.sampler),
                                     stepA_dom_acc=diag["stepA_dom_acc"]))
            print(f"  target={tgt:14s} {lbl:14s} bAcc={cm['balanced_acc']*100:5.1f} "
                  f"leakKL={lk['leakage_kl']:.3f} labelSep={ls*100:4.1f} ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    print(f"\n=== cross-dataset / {args.backbone} (leave-one-dataset-out) ===")
    print(f"{'config':16s} {'BalAcc':>11s} {'WorstDS':>7s} {'MacroF1':>8s} {'LeakKL':>7s} {'LabelSep':>8s}")
    for m in results:
        ba = np.array([r["balanced_acc"] for r in results[m]])
        summary[m] = dict(balanced_acc_mean=float(ba.mean()), balanced_acc_std=float(ba.std()),
                          worst_dataset=float(ba.min()),
                          macro_f1=float(np.mean([r["macro_f1"] for r in results[m]])),
                          leakage_kl=float(np.mean([r["leakage_kl"] for r in results[m]])),
                          leakage_kl_rw=float(np.mean([r["leakage_kl_rw"] for r in results[m]])),
                          marginal_leakage_kl=float(np.mean([r["marginal_leakage_kl"] for r in results[m]])),
                          marginal_leakage_kl_rw=float(np.mean([r["marginal_leakage_kl_rw"] for r in results[m]])),
                          marginal_leakage_advantage=float(np.mean([r["marginal_leakage_advantage"] for r in results[m]])),
                          marginal_leakage_advantage_rw=float(np.mean([r["marginal_leakage_advantage_rw"] for r in results[m]])),
                          decoder_cmi=float(np.mean([r["decoder_cmi"] for r in results[m]])),
                          decoder_cmi_rw=float(np.mean([r["decoder_cmi_rw"] for r in results[m]])),
                          decoder_cmi_res=float(np.mean([r["decoder_cmi_res"] for r in results[m]])),
                          decoder_cmi_res_rw=float(np.mean([r["decoder_cmi_res_rw"] for r in results[m]])),
                          decoder_js_res=float(np.mean([r["decoder_js_res"] for r in results[m]])),
                          decoder_js_res_rw=float(np.mean([r["decoder_js_res_rw"] for r in results[m]])),
                          decoder_valid_frac=float(np.mean([float(r.get("decoder_valid", False)) for r in results[m]])),
                          decoder_min_domain_classes=int(np.min([r.get("decoder_min_domain_classes", 0) for r in results[m]])),
                          decoder_single_class_frac=float(np.mean([r.get("decoder_single_class_frac", 1.0) for r in results[m]])),
                          leakage_advantage_rw=float(np.mean([r["leakage_advantage_rw"] for r in results[m]])),
                          label_sep=float(np.mean([r["label_sep"] for r in results[m]])),
                          inloop_reg=float(np.mean([r["inloop_reg"] for r in results[m]])),
                          inloop_dec=float(np.mean([r.get("inloop_dec", 0.0) for r in results[m]])),
                          inloop_dec_loss=float(np.mean([r.get("inloop_dec_loss", 0.0) for r in results[m]])),
                          train_dec_margin=float(np.mean([r.get("train_dec_margin", 0.0) for r in results[m]])),
                          stepA_dom_acc=float(np.mean([r["stepA_dom_acc"] for r in results[m]])),
                          per_target=results[m])
        add_decoder_valid_means(summary[m], results[m])
        s = summary[m]
        print(f"{m:16s} {s['balanced_acc_mean']*100:6.1f}±{s['balanced_acc_std']*100:4.1f} "
              f"{s['worst_dataset']*100:6.1f} {s['macro_f1']*100:7.1f} {s['leakage_kl']:7.3f} {s['label_sep']*100:8.1f}")
    if args.out:
        json.dump(dict(config=vars(args), classes=classes, summary=summary), open(args.out, "w"), indent=2)
        print(f"\nsaved -> {args.out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["BNCI2014_001", "Lee2019_MI", "Cho2017"])
    ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--configs", nargs="+", default=["erm:0", "lpc_prior:0.3", "cdann:1"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--n_inner", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--sampler", default="classbal", choices=["classbal", "raw", "domainbal"])
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dec_margin", type=float, default=None,
                    help="decoder gate tau. Default is method-specific: dualpc/dualpc_marginal=0, others=0.02")
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_subj", type=int, default=0)
    ap.add_argument("--out", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
