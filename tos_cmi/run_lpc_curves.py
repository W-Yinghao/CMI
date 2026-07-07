"""Phase 2.1/2.2 -- LPC collapse-mechanism + objective-scaling sanity runner. Trains global-LPC
(lpc_prior) on one LOSO fold with PER-EPOCH curve logging (train_model(log_curves=True)) under a
chosen VARIANT, and saves curves + final metrics + final-feature npz (for the domain-leakage check).
NO change to loss/optimizer/sampler/estimator -- variants are flag-gated, default-off in train_model.

Variants (Phase 2.2 objective-scaling sanity ablation):
  raw_lpc             : original global LPC (= Phase 2.1).
  lpc_scale_invariant : LPC penalty (and its posterior) see scale-detached normalized z; task head
                        sees raw z. Blocks the Z->0 trivial minimizer. (lpc_pen_normalize=True)
  lpc_warm_ramp       : ERM until ep ramp_start, lambda ramps over ramp_len, then fixed. Tests whether
                        the collapse is a high-lambda cold-start basin. (lam_warm_ramp=True)

  python -m tos_cmi.run_lpc_curves --target-subject 1 --lams 1.0 3.0 --seed 0 --variant lpc_scale_invariant
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

VARIANTS = {
    "raw_lpc": dict(lpc_pen_normalize=False, lam_warm_ramp=False),
    "lpc_scale_invariant": dict(lpc_pen_normalize=True, lam_warm_ramp=False),
    "lpc_warm_ramp": dict(lpc_pen_normalize=False, lam_warm_ramp=True),
}


def _bacc(logits, y, n_cls):
    pred = logits.argmax(1)
    r = [(pred[y == c] == c).mean() for c in range(n_cls) if (y == c).any()]
    return float(np.mean(r)) if r else float("nan")


def _decode(Z, lab, seed=0):
    """MLP decode accuracy of `lab` from features Z, on a 50/50 split (proxy leakage/task probe)."""
    try:
        if len(np.unique(lab)) < 2:
            return float("nan")
        from sklearn.neural_network import MLPClassifier
        rng = np.random.default_rng(seed); perm = rng.permutation(len(lab)); cut = len(lab) // 2
        tr, te = perm[:cut], perm[cut:]
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=seed).fit(Z[tr], lab[tr])
        return float((clf.predict(Z[te]) == lab[te]).mean())
    except Exception:
        return float("nan")


def run(dataset, target_subject, lam, seed, variant, out_path, *, backbone="TSMNet", epochs=300,
        bs=64, warmup=40, curve_every=10, device="cuda", tmin=0.5, tmax=3.5, resample=128):
    configure_offline_moabb()
    X, y, meta, classes = load(dataset, tmin=tmin, tmax=tmax, resample=resample)
    n_cls = len(classes); subj = meta["subject"].to_numpy()
    dom_all, _ = domain_labels(meta, "subject")
    te = (subj == target_subject); tr = ~te
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    subj_tr = subj[tr]; dtr, n_dom = _remap(dom_all[tr])
    torch.manual_seed(seed); np.random.seed(seed)
    bb = build_backbone(backbone, X.shape[1], X.shape[2], n_cls, device=device)
    flags = VARIANTS[variant]
    bb, _post, out = train_model(bb, Xtr, ytr, dtr, n_cls, method="lpc_prior", lam=lam,
                                 epochs=epochs, bs=bs, warmup=warmup, device=device, seed=seed,
                                 log_curves=True, curve_every=curve_every, **flags)
    lg_s, Z_s = _forward_dump(bb, Xtr, device); lg_t, Z_t = _forward_dump(bb, Xte, device)
    # decision-rule-A probes: does the variant preserve task AND reduce domain (subject) leakage?
    subj_id = _remap(subj_tr)[0]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rec = {"dataset": dataset, "backbone": backbone, "variant": variant,
           "target_subject": int(target_subject), "lam": float(lam), "seed": int(seed),
           "epochs": epochs, "n_cls": n_cls,
           "final_source_bAcc": _bacc(lg_s, ytr, n_cls), "final_target_bAcc": _bacc(lg_t, yte, n_cls),
           "final_task_decode": _decode(Z_s, ytr, seed), "final_subject_decode": _decode(Z_s, subj_id, seed),
           "n_subj": int(subj_id.max() + 1), "chance_subj": 1.0 / int(subj_id.max() + 1),
           "git_sha": _git_sha(root), "tsmnet_sha": _git_sha(os.path.join(root, "repos/TSMNet")),
           "curves": out.get("curves", [])}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(rec, open(out_path, "w"))
    np.savez_compressed(out_path.replace(".json", ".npz"), Z_source=Z_s.astype(np.float32),
                        y_source=ytr, subject_source=subj_id, Z_target=Z_t.astype(np.float32), y_target=yte)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="TSMNet")
    ap.add_argument("--target-subject", type=int, required=True)
    ap.add_argument("--lams", nargs="+", type=float, default=[1.0, 3.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=list(VARIANTS), default="raw_lpc")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--curve-every", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default="tos_cmi/results/tos_cmi_eeg_frozen/lpc_collapse_curves")
    args = ap.parse_args()
    for lam in args.lams:
        tag = "%s_sub%d_lam%g_seed%d" % (args.variant, args.target_subject, lam, args.seed)
        out = "%s/%s/%s.json" % (args.out_root, args.backbone, tag)   # backbone-specific (no cross-backbone collision)
        r = run(args.dataset, args.target_subject, lam, args.seed, args.variant, out,
                backbone=args.backbone, epochs=args.epochs, curve_every=args.curve_every, device=args.device)
        fin = r["curves"][-1] if r["curves"] else {}
        print("[%s] src=%.3f tgt=%.3f | task_dec=%.2f subj_dec=%.2f (ch %.2f) | feat_norm->%.4f -> %s"
              % (tag, r["final_source_bAcc"], r["final_target_bAcc"], r["final_task_decode"],
                 r["final_subject_decode"], r["chance_subj"], fin.get("feat_norm_mean", -1), out), flush=True)
    print("LPC_CURVES_DONE")


if __name__ == "__main__":
    main()
