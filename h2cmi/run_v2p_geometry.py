"""WAVE 1 / W1.geometry — geometry-only latent-diagonal falsification (frozen W1_GEOMETRY_FROZEN.md).

Per V2P unit + source seed: reuse the frozen bundle; for each channel-space perturbation P in
{none, reref, gain, dropout} applied to BOTH Xa and Xe, embed and run eval_unit_p0 (identity + diagonal-
latent operators + EA-sensor) and additionally CORAL-latent (full-covariance latent alignment to the
source latent covariance). Record balanced accuracy per operator, per perturbation. Real-id addressed.

  python -m h2cmi.run_v2p_geometry --pairs BNCI2014_001:0>1 --seeds 0,1,2 --out results/h2cmi/wave1_geom/w1g_bnci001.jsonl
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from h2cmi.eval.harness import _embed, _predict_generative
from h2cmi.eval.p0_eval import eval_unit_p0
from h2cmi.p0_source import get_source_p0, ProvenanceError
from h2cmi.run_v2 import build_cfg, PAIRS_B
from h2cmi.run_v2p_wave0 import _pair_units
from h2cmi.tta.class_conditional import ClassConditionalTTA
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, sha256_file, stable_hash_int

K = 2
PERTS = ["none", "reref", "gain", "dropout"]
DIAG_OPS = ["fixed_reference_oneshot_uniform", "fixed_iterative_geometry_uniform",
            "joint_geometry_uniform", "latent_im_diag_uniform", "pooled_uniform"]
OUT_DIR = "results/h2cmi/wave1_geom"


def _perturbation(kind, C, seed):
    P = np.eye(C, dtype=np.float32)
    if kind == "none":
        return P
    if kind == "reref":
        P[:, 0] -= 1.0                                    # X'_c = X_c - X_0
        return P
    rng = np.random.default_rng(seed)
    if kind == "gain":
        return np.diag(rng.lognormal(0.0, 0.3, C)).astype(np.float32)
    if kind == "dropout":
        mask = np.ones(C, np.float32); k = max(1, int(0.2 * C))
        mask[rng.choice(C, k, replace=False)] = 0.0
        return np.diag(mask)
    raise ValueError(kind)


def _apply(P, X):
    return np.einsum("cd,ndt->nct", P, X).astype(np.float32)


def _coral_latent(model, Us, Ua_p, Ue_p, uni, eps=1e-4):
    """Full-covariance latent alignment: whiten perturbed-adapt latent cov, recolor to source latent cov."""
    D = Us.shape[1]; I = torch.eye(D, device=Us.device, dtype=Us.dtype)
    mu_s = Us.mean(0); mu_t = Ua_p.mean(0)
    Cs = torch.cov((Us - mu_s).T) + eps * I
    Ct = torch.cov((Ua_p - mu_t).T) + eps * I
    def msqrt(Cm, inv):
        w, V = torch.linalg.eigh(Cm); w = w.clamp_min(eps)
        p = w.pow(-0.5) if inv else w.pow(0.5)
        return (V * p) @ V.T
    A = msqrt(Ct, True) @ msqrt(Cs, False)
    Ue_c = (Ue_p - mu_t) @ A + mu_s
    return np.asarray(_predict_generative(model, Ue_c, uni)).argmax(1)   # HARD predictions for _bacc


def _bacc(pred, ye):
    return float(np.mean([np.mean(pred[ye == c] == c) for c in range(K) if (ye == c).sum()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=""); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seed0-root", default="results/h2cmi/v2_bundles")
    ap.add_argument("--new-root", default="results/h2cmi/p0_v2pw_bundles")
    ap.add_argument("--out", required=True); ap.add_argument("--subjects", default="")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    only_subj = set(int(s) for s in args.subjects.split(",") if s != "") if args.subjects else None
    os.makedirs(OUT_DIR, exist_ok=True)
    commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=["results/h2cmi", args.new_root, args.seed0_root])
    code_sig = source_code_signature()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    pairs = []
    for p in (args.pairs.split(",") if args.pairs else []):
        d, ss = p.split(":"); a, b = ss.split(">"); pairs.append((d, int(a), int(b)))
    pairs = pairs or PAIRS_B
    done = set()
    if os.path.exists(args.out):
        import json as _j
        for l in open(args.out):
            if l.strip():
                r = _j.loads(l)
                if r.get("perturbation") == PERTS[-1]:
                    done.add((int(r["subject"]), int(r["seed"])))
    uni = np.full(K, 1.0 / K)
    for ds, s_src, s_tgt, subj, ai, ei, Xa, ya, Xe, ye, ep, m_src in _pair_units(pairs, seeds, args, code_sig, commit, only_subj):
        if only_subj is not None and int(subj) not in only_subj:
            continue
        C = Xa.shape[1]
        for seed in seeds:
            if (int(subj), int(seed)) in done:
                continue
            cfg = build_cfg(ep.X.shape[1], args.epochs, args.device, seed=seed)
            tag = f"B:{ds}:s{int(subj)}:sess{s_src}"
            try:
                model, pooled_ref, R_src, pi_star, val = get_source_p0(
                    args.seed0_root, args.new_root, tag, cfg, code_sig, K,
                    lambda ms=m_src: (ep.X[ms], ep.y[ms], ep.subject[ms]))
            except ProvenanceError as pe:
                append_row(args.out, dict(panel="W1GEOM", pair=f"{ds}:{s_src}>{s_tgt}", subject=int(subj),
                                          seed=int(seed), provenance_fail=str(pe))); continue
            Us = _embed(model, ep.X[m_src], args.device)         # source latent (for CORAL reference)
            base = dict(panel="W1GEOM", commit=commit, code_sig=code_sig, pair=f"{ds}:{s_src}>{s_tgt}",
                        dataset=ds, subject=int(subj), seed=int(seed), seed0_validated=bool(val),
                        n_chans=int(C), n_adapt=int(len(ai)), n_eval=int(len(ei)))
            for kind in PERTS:
                pseed = stable_hash_int(f"W1{kind}", ds)
                P = _perturbation(kind, C, pseed)
                Xa_p, Xe_p = _apply(P, Xa), _apply(P, Xe)
                Ua_p, Ue_p = _embed(model, Xa_p, args.device), _embed(model, Xe_p, args.device)
                tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, K, args.device)
                ts = stable_hash_int("W1geom", ds, int(subj), int(seed), kind)
                branches, _ = eval_unit_p0(model, tta, pooled_ref, R_src, Xa_p, Xe_p, Ua_p, Ue_p, ye,
                                           args.device, K, ts)
                ba = {op: branches[op]["bacc"] for op in (["identity_uniform", "source_recolored_ea"] + DIAG_OPS)}
                ba["coral_latent"] = _bacc(_coral_latent(model, Us, Ua_p, Ue_p, uni), ye)
                ba["P_hash"] = int(stable_hash_int(f"W1{kind}", ds, int(C)))
                append_row(args.out, dict(base, perturbation=kind, **{f"bacc_{k}": v for k, v in ba.items()}))
            print(f"[W1GEOM {ds}:{s_src}>{s_tgt}] subj={subj} seed={seed} done", flush=True)
    if os.path.exists(args.out):
        print(f"[W1GEOM] -> {args.out} sha={sha256_file(args.out)[:12]}", flush=True)


if __name__ == "__main__":
    main()
