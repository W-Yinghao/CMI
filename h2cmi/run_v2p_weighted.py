"""V2P_WEIGHTED_PREVALENCE_INTERVENTION (REVIEW_P0 section D). Reuses the supported cross-session V2-B
units; source seeds {0,1,2}. Per target unit: ONE fixed adaptation reservoir + ONE fixed evaluation set;
the three class-0-mass ratios q in {0.50,0.75,0.25} reweight EXACTLY the same trials (no resubsetting).
Weighted estimators (pooled / fixed-reference one-shot / fixed-prior iterative / joint EM) receive
embeddings + sample weights, never labels; oracle_label_conditional uses true labels (diagnostic only).
Geometry displacement outputs are computed per unit across ratios. Old V2P artifacts are untouched.

  python -m h2cmi.run_v2p_weighted --pairs BNCI2014_001:0>1 --seeds 0,1,2 --device cuda \
      --out results/h2cmi/p0_v2pw_bnci001.jsonl
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from h2cmi.eval.harness import _embed, _predict_generative, _predict_transform
from h2cmi.eval.p0_eval import _record
from h2cmi.tta.weighted_tta import fit_weighted_pooled, fit_weighted_em, effective_weights, canonical_weights
from h2cmi.p0_source import get_source_p0, ProvenanceError
from h2cmi.run_v2 import build_cfg, PAIRS_B
from h2cmi.data.real_eeg import load_dataset, contiguous_split
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, sha256_file, stable_hash_int

K = 2
QS = [("q0.50", 0.50), ("q0.75", 0.75), ("q0.25", 0.25)]   # class-0 mass; q0.50 = the 1:1 baseline
EST = ["pooled", "fixed_reference_oneshot", "fixed_iterative", "joint", "oracle_label_conditional"]


def _fit(est, density, Ua, w, pi_star, cfg, device, oracle_y, pooled_ref, ts):
    if est == "pooled":
        return fit_weighted_pooled(Ua, w, pooled_ref, device), np.asarray(pi_star)
    kind = {"fixed_reference_oneshot": "oneshot", "fixed_iterative": "iterative",
            "joint": "joint", "oracle_label_conditional": "oracle"}[est]
    T, piT = fit_weighted_em(density, Ua, w, pi_star, cfg.tta, K, device, kind,
                             oracle_labels=(oracle_y if kind == "oracle" else None), tta_seed=ts)
    return T, np.asarray(piT.cpu().numpy() if torch.is_tensor(piT) else piT)


def _geom(T):
    a = np.asarray(T.a.detach().cpu().numpy(), float); b = np.asarray(T.b.detach().cpu().numpy(), float)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seed0-root", default="results/h2cmi/v2_bundles")
    ap.add_argument("--new-root", default="results/h2cmi/p0_v2pw_bundles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.dirname(args.out) or "."
    commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=[out_dir, args.new_root, args.seed0_root])
    code_sig = source_code_signature()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    pairs = []
    for p in (args.pairs.split(",") if args.pairs else []):
        ds, ss = p.split(":"); a, b = ss.split(">"); pairs.append((ds, int(a), int(b)))
    pairs = pairs or PAIRS_B
    if os.path.exists(args.out):
        os.remove(args.out)
    uni = np.full(K, 1.0 / K)
    for ds, s_src, s_tgt in pairs:
        ep = load_dataset(ds, MOABB_CLASS(ds)().subject_list)
        for subj in np.unique(ep.subject):
            m_src = (ep.subject == subj) & (ep.session == s_src)
            cfg0 = build_cfg(ep.X.shape[1], args.epochs, args.device, seed=0)
            if m_src.sum() < cfg0.tta.min_target * 2:
                continue
            ai, ei = contiguous_split(ep, subj, s_tgt)
            if len(ai) < cfg0.tta.min_target or len(ei) < 8:
                continue
            Xa, ya = ep.X[ai], ep.y[ai]; Xe, ye = ep.X[ei], ep.y[ei]
            if (ya == 0).sum() < 4 or (ya == 1).sum() < 4:
                continue
            adapt_ids = [int(i) for i in ai]; eval_ids = [int(i) for i in ei]
            for seed in seeds:
                cfg = build_cfg(ep.X.shape[1], args.epochs, args.device, seed=seed)
                tag = f"B:{ds}:s{int(subj)}:sess{s_src}"
                try:
                    model, pooled_ref, R_src, pi_star, val = get_source_p0(
                        args.seed0_root, args.new_root, tag, cfg, code_sig, K,
                        lambda ms=m_src: (ep.X[ms], ep.y[ms], ep.subject[ms]))
                except ProvenanceError as pe:
                    append_row(args.out, dict(panel="V2PW", pair=f"{ds}:{s_src}>{s_tgt}", subject=int(subj),
                                              seed=int(seed), provenance_fail=str(pe))); print(f"PROV FAIL: {pe}"); continue
                Ua = _embed(model, Xa, args.device); Ue = _embed(model, Xe, args.device)
                density = model.head.density
                base = dict(panel="V2PW", commit=commit, code_sig=code_sig, pair=f"{ds}:{s_src}>{s_tgt}",
                            dataset=ds, subject=int(subj), src_sess=s_src, tgt_sess=s_tgt, seed=int(seed),
                            seed0_validated=bool(val), n_adapt=int(len(ai)), n_eval=int(len(ei)),
                            adapt_ids_hash=stable_hash_int(*adapt_ids), eval_ids_hash=stable_hash_int(*eval_ids))
                # identity (ratio-invariant) recorded once
                p_id = _predict_generative(model, Ue, uni)
                append_row(args.out, dict(base, estimator="identity", ratio="*", **_record(p_id, ye, K)))
                store = {e: {} for e in EST}   # estimator -> ratio -> (a,b,Tz)
                for rname, q in QS:
                    w = effective_weights(ya, q)
                    cmass = [float(canonical_weights(ya, q)[ya == c].sum()) for c in (0, 1)]
                    ts = stable_hash_int("V2PW", ds, int(subj), int(seed), rname)
                    for est in EST:
                        T, piJ = _fit(est, density, Ua, w, pi_star, cfg, args.device, ya, pooled_ref, ts)
                        a, b = _geom(T)
                        p_u = _predict_transform(model, Ue, T, uni)
                        rec = _record(p_u, ye, K, T)
                        row = dict(base, estimator=est, ratio=rname, class0_mass=cmass[0], class1_mass=cmass[1],
                                   weight_sum=float(w.sum()), pi_J=[float(x) for x in piJ], **rec)
                        if est == "joint":
                            p_j = _predict_transform(model, Ue, T, piJ)
                            jr = _record(p_j, ye, K, T)
                            row["acc_joint_prior"] = jr["acc"]; row["nll_joint_prior"] = jr["nll"]; row["bacc_joint_prior"] = jr["bacc"]
                        append_row(args.out, row)
                        store[est][rname] = (a, b, T.apply(Ue.detach()).detach().cpu().numpy())
                # geometry displacement outputs per estimator (vs q0.50)
                for est in EST:
                    a0, b0, z0 = store[est]["q0.50"]; a3, b3, z3 = store[est]["q0.75"]; a1, b1, z1 = store[est]["q0.25"]
                    append_row(args.out, dict(base, estimator=est, ratio="__geom__",
                        logscale_disp_q075=float(np.linalg.norm(a3 - a0)), logscale_disp_q025=float(np.linalg.norm(a1 - a0)),
                        translation_disp_q075=float(np.linalg.norm(b3 - b0)), translation_disp_q025=float(np.linalg.norm(b1 - b0)),
                        embed_disp_q075=float(np.linalg.norm(z3 - z0, axis=1).mean()), embed_disp_q025=float(np.linalg.norm(z1 - z0, axis=1).mean()),
                        symmetric_disp=float(np.linalg.norm(np.concatenate([a3 - a1, b3 - b1]))),
                        slope_vec_norm=float(np.linalg.norm(np.concatenate([a3 - a1, b3 - b1]) / 2.0))))
            print(f"[V2PW {ds}:{s_src}>{s_tgt}] subj={subj} done", flush=True)
    if os.path.exists(args.out):
        print(f"[V2PW] -> {args.out} sha={sha256_file(args.out)[:12]}", flush=True)


if __name__ == "__main__":
    main()
