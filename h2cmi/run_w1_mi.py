"""W1-A: standard unseen-subject (LOSO) MI benchmark, SAME-BACKBONE controlled panel (review
W1_W2_FROZEN). One frozen H2 source model per fold = all non-target subjects; target adaptation =
earlier disjoint block (unlabeled), evaluation = later block; unit = target subject. Methods:
identity, EA, pooled, canonical fixed-prior CC, current_joint, SPDIM. (BTTA-DG = W1-B native panel,
separate; blocked until the official repo is on disk.) Target labels eval-only.

  python -m h2cmi.run_w1_mi --dataset Cho2017 --folds 0-6 --bundle-root results/h2cmi/w1_bundles \
      --out results/h2cmi/w1_cho_0.jsonl --device cuda
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from h2cmi.eval.harness import _embed, _predict_generative, _predict_transform
from h2cmi.tta.class_conditional import ClassConditionalTTA, B1A_VARIANTS_BY_NAME
from h2cmi.eval.ea import reference_cov, ea_transform, apply_ea
from h2cmi.eval.spdim import spdim_fit
from h2cmi.data.real_eeg import load_dataset, contiguous_split
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, sha256_file
from h2cmi.run_v2 import build_cfg, get_source

W1_METHODS = ["identity", "euclidean_alignment", "always_pooled", "always_canonical_CC",
              "current_joint", "spdim"]
UNI = np.full(2, 0.5)


def w1_eval_unit(model, tta, pooled_ref, R_src, Xa, Xe, ye, device):
    Ua = _embed(model, Xa, device); Ue = _embed(model, Xe, device)
    V = B1A_VARIANTS_BY_NAME
    out = {}
    p_id = _predict_generative(model, Ue, UNI)
    out["identity"] = (p_id.argmax(1), None)
    M = ea_transform(R_src, reference_cov(Xa))
    out["euclidean_alignment"] = (_predict_generative(model, _embed(model, apply_ea(Xe, M), device), UNI).argmax(1), None)
    fp = tta.fit_variant(Ua, V["pooled_empirical_diag"], pooled_ref=pooled_ref, tta_seed=1)
    out["always_pooled"] = (_predict_transform(model, Ue, fp.transform, UNI).argmax(1), fp)
    fc = tta.fit_variant(Ua, V["gen_oneshot_diag"], tta_seed=1)
    out["always_canonical_CC"] = (_predict_transform(model, Ue, fc.transform, UNI).argmax(1), fc)
    fj = tta.fit_variant(Ua, V["joint_iterative_diag"], tta_seed=1)
    pij = np.asarray(fj.pi_T.cpu().numpy() if torch.is_tensor(fj.pi_T) else fj.pi_T)
    out["current_joint"] = (_predict_transform(model, Ue, fj.transform, pij).argmax(1), fj)
    Ts = spdim_fit(model.head.density, Ua, UNI, device)
    out["spdim"] = (_predict_transform(model, Ue, Ts, UNI).argmax(1), Ts)
    return out, ye


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--folds", default="", help="subject-index range like 0-6 (default all)")
    ap.add_argument("--bundle-root", default="results/h2cmi/w1_bundles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.dirname(args.out) or "."
    commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=[out_dir, args.bundle_root])
    code_sig = source_code_signature()
    if os.path.exists(args.out):
        os.remove(args.out)
    ep = load_dataset(args.dataset, MOABB_CLASS(args.dataset)().subject_list)
    subs = list(np.unique(ep.subject))
    if args.folds:
        a, b = (int(x) for x in args.folds.split("-")); subs = subs[a:b + 1]
    cfg = build_cfg(ep.X.shape[1], args.epochs, args.device, seed=0)
    for tgt in subs:
        m_src = ep.subject != tgt
        if m_src.sum() < cfg.tta.min_target * 2:
            continue
        model, pooled_ref, R_src, pi_star = get_source(
            args.bundle_root, f"W1:{args.dataset}:loso{tgt}", cfg, code_sig,
            lambda ms=m_src: (ep.X[ms], ep.y[ms], ep.subject[ms]))
        tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, 2, args.device)
        sess0 = ep.session[ep.subject == tgt].min()
        a, e = contiguous_split(ep, tgt, sess0)
        if len(a) < cfg.tta.min_target or len(e) < 4:
            continue
        res, ye = w1_eval_unit(model, tta, pooled_ref, R_src, ep.X[a], ep.X[e], ep.y[e], args.device)
        b_id = float(balanced_accuracy_score(ye, res["identity"][0]))
        f_id = float(f1_score(ye, res["identity"][0], average="macro"))
        for method in W1_METHODS:
            pred, fit = res[method]
            bacc = float(balanced_accuracy_score(ye, pred)); mf1 = float(f1_score(ye, pred, average="macro"))
            tnorm = float("nan")
            if fit is not None and hasattr(fit, "transform"):
                A = fit.transform.matrix(); tnorm = float(((A - torch.eye(A.shape[0], device=A.device)) ** 2).sum().sqrt().cpu())
            elif fit is not None and hasattr(fit, "matrix"):
                A = fit.matrix(); tnorm = float(((A - torch.eye(A.shape[0], device=A.device)) ** 2).sum().sqrt().cpu())
            append_row(args.out, dict(panel="W1A", commit=commit, code_sig=code_sig, dataset=args.dataset,
                                      target_subject=int(tgt), method=method, bacc=bacc, macro_f1=mf1,
                                      bacc_identity=b_id, delta=bacc - b_id, delta_f1=mf1 - f_id,
                                      harm=bool(bacc - b_id < -1e-9), n_adapt=int(len(a)), n_eval=int(len(e)),
                                      transform_norm=tnorm))
        print(f"[W1 {args.dataset}] loso target={tgt} done (id bAcc={b_id:.3f})", flush=True)
    if os.path.exists(args.out):
        print(f"[W1 {args.dataset}] -> {args.out} sha={sha256_file(args.out)[:12]}", flush=True)


if __name__ == "__main__":
    main()
