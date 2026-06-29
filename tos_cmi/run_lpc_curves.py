"""Phase 2.1 -- LPC collapse-mechanism audit runner. Trains global-LPC (lpc_prior) at a given
lambda on one LOSO fold with PER-EPOCH curve logging (train_model(log_curves=True)) and saves the
curves + final metrics. Settles whether the high-lambda collapse is an OPTIMIZATION instability vs
a smooth geometric over-compression. NO change to loss/optimizer/scheduler/sampler/warmup/estimator
-- instrumentation only (default-off flag). GPU (TSMNet); run via scripts/tos_eeg_lpc_curves.sbatch.

  python -m tos_cmi.run_lpc_curves --target-subject 1 --lams 0 0.3 1.0 3.0 --seed 0 --epochs 300
"""
import argparse
import json
import os
import numpy as np
import torch

from cmi.paths import configure_offline_moabb
from cmi.data.moabb_data import load, domain_labels
from cmi.models.backbones import build_backbone
from cmi.train.trainer import train_model
from tos_cmi.eeg.feature_dump import _remap, _git_sha, _forward_dump


def _bacc(logits, y, n_cls):
    pred = logits.argmax(1)
    r = [(pred[y == c] == c).mean() for c in range(n_cls) if (y == c).any()]
    return float(np.mean(r)) if r else float("nan")


def run(dataset, target_subject, lam, seed, out_path, *, backbone="TSMNet", epochs=300, bs=64,
        warmup=40, curve_every=10, device="cuda", tmin=0.5, tmax=3.5, resample=128):
    configure_offline_moabb()
    X, y, meta, classes = load(dataset, tmin=tmin, tmax=tmax, resample=resample)
    n_cls = len(classes); subj = meta["subject"].to_numpy()
    dom_all, _ = domain_labels(meta, "subject")
    te = (subj == target_subject); tr = ~te
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    dtr, n_dom = _remap(dom_all[tr])
    torch.manual_seed(seed); np.random.seed(seed)
    bb = build_backbone(backbone, X.shape[1], X.shape[2], n_cls, device=device)
    # lpc_prior at lam=0 == ERM but logs the (zero-weighted) CMI term + curves; uses_cmi path.
    bb, _post, out = train_model(bb, Xtr, ytr, dtr, n_cls, method="lpc_prior", lam=lam,
                                 epochs=epochs, bs=bs, warmup=warmup, device=device, seed=seed,
                                 log_curves=True, curve_every=curve_every)
    lg_s, _ = _forward_dump(bb, Xtr, device); lg_t, _ = _forward_dump(bb, Xte, device)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rec = {"dataset": dataset, "backbone": backbone, "target_subject": int(target_subject),
           "lam": float(lam), "seed": int(seed), "epochs": epochs, "n_cls": n_cls,
           "final_source_bAcc": _bacc(lg_s, ytr, n_cls), "final_target_bAcc": _bacc(lg_t, yte, n_cls),
           "git_sha": _git_sha(root), "tsmnet_sha": _git_sha(os.path.join(root, "repos/TSMNet")),
           "curves": out.get("curves", [])}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(rec, open(out_path, "w"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="TSMNet")
    ap.add_argument("--target-subject", type=int, required=True)
    ap.add_argument("--lams", nargs="+", type=float, default=[0.0, 0.3, 1.0, 3.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--curve-every", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default="tos_cmi/results/tos_cmi_eeg_frozen/lpc_collapse_curves")
    args = ap.parse_args()
    for lam in args.lams:
        tag = "sub%d_lam%g_seed%d" % (args.target_subject, lam, args.seed)
        out = "%s/%s.json" % (args.out_root, tag)
        r = run(args.dataset, args.target_subject, lam, args.seed, out, backbone=args.backbone,
                epochs=args.epochs, curve_every=args.curve_every, device=args.device)
        c = r["curves"]
        fin = c[-1] if c else {}
        print("[%s] final src=%.3f tgt=%.3f | last eff_rank=%.0f task_CE=%.3f grad_norm=%.2f -> %s"
              % (tag, r["final_source_bAcc"], r["final_target_bAcc"], fin.get("eff_rank", -1),
                 fin.get("train_task_CE", -1), fin.get("grad_total_encoder_norm", -1), out), flush=True)
    print("LPC_CURVES_DONE")


if __name__ == "__main__":
    main()
