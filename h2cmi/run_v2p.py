"""V2P_MECHANISM_AUDIT: controlled real-signal prevalence intervention (review V2P_FROZEN).

Reuses the FROZEN V2-B source checkpoints. Per supported cross-session target unit (subject), holds
the real EEG trials fixed and varies ONLY the unlabeled adaptation-pool class composition
{1:1, 3:1, 1:3} at EQUAL total pool size, on ONE FIXED evaluation set. For each pool, fits
{identity, always_pooled, always_canonical_CC, current_joint} (labels NEVER enter the estimator;
the builder uses labels only to compose the pool) and records the transform (diag log-scale a, bias
b), transform/bias norms, predicted eval occupancy, estimated target prior, and fixed-eval bAcc.
Pool construction trial-IDs + seed are written to a manifest before estimators run. Does NOT change
any Stage V verdict.

  python -m h2cmi.run_v2p --bundle-root results/h2cmi/v2_bundles --out results/h2cmi/v2p.jsonl \
      --manifest results/h2cmi/v2p_pools.jsonl --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from h2cmi.eval.harness import _embed, _predict_generative, _predict_transform
from h2cmi.tta.class_conditional import ClassConditionalTTA, B1A_VARIANTS_BY_NAME
from h2cmi.data.real_eeg import load_dataset, contiguous_split
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, stable_hash_int, sha256_file
from h2cmi.run_v2 import build_cfg, get_source, PAIRS_B

RATIOS = [("1:1", 0.5), ("3:1", 0.75), ("1:3", 0.25)]   # name -> fraction LEFT (class 0)
METHODS = ["identity", "always_pooled", "always_canonical_CC", "current_joint"]
UNI = np.full(2, 0.5)


def _pool_indices(adapt_idx, y, frac_left, M, seed):
    """Pick M trials from adapt_idx with n0=round(M*frac_left) class-0 (left) + n1=M-n0 class-1,
    contiguous-first within class (deterministic). Returns the global trial ids or None if short."""
    ya = y[adapt_idx]
    i0 = adapt_idx[ya == 0]; i1 = adapt_idx[ya == 1]
    n0 = int(round(M * frac_left)); n1 = M - n0
    if len(i0) < n0 or len(i1) < n1:
        return None
    return np.sort(np.concatenate([i0[:n0], i1[:n1]]))


def eval_pool(model, tta, pooled_ref, U_eval, ye, U_pool, device):
    """Fit each method on the (unlabeled) pool embeddings; apply to the fixed eval. Returns dict."""
    V = B1A_VARIANTS_BY_NAME
    res = {}
    p_id = _predict_generative(model, U_eval, UNI)
    res["identity"] = dict(a=[0.0], b=[0.0], tnorm=0.0, bnorm=0.0,
                           occ=float((p_id.argmax(1) == 1).mean()), pi_T=[0.5, 0.5],
                           bacc=float(balanced_accuracy_score(ye, p_id.argmax(1))))
    def rec(fit, pi_for_pred):
        T = fit.transform
        A = T.matrix(); I = torch.eye(A.shape[0], device=A.device)
        pa = _predict_transform(model, U_eval, T, pi_for_pred)
        return dict(a=[float(x) for x in T.a.detach().cpu().numpy()],
                    b=[float(x) for x in T.b.detach().cpu().numpy()],
                    tnorm=float(((A - I) ** 2).sum().sqrt().cpu()),
                    bnorm=float((T.b ** 2).sum().sqrt().cpu()),
                    occ=float((pa.argmax(1) == 1).mean()),
                    pi_T=[float(x) for x in np.asarray(fit.pi_T.cpu().numpy() if torch.is_tensor(fit.pi_T) else fit.pi_T)],
                    bacc=float(balanced_accuracy_score(ye, pa.argmax(1))))
    fp = tta.fit_variant(U_pool, V["pooled_empirical_diag"], pooled_ref=pooled_ref, tta_seed=1)
    res["always_pooled"] = rec(fp, UNI)
    fc = tta.fit_variant(U_pool, V["gen_oneshot_diag"], tta_seed=1)
    res["always_canonical_CC"] = rec(fc, UNI)
    fj = tta.fit_variant(U_pool, V["joint_iterative_diag"], tta_seed=1)
    res["current_joint"] = rec(fj, np.asarray(fj.pi_T.cpu().numpy() if torch.is_tensor(fj.pi_T) else fj.pi_T))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", default="results/h2cmi/v2_bundles")
    ap.add_argument("--out", default="results/h2cmi/v2p.jsonl")
    ap.add_argument("--manifest", default="results/h2cmi/v2p_pools.jsonl")
    ap.add_argument("--pairs", default="", help="subset like 'BNCI2014_004:0>1,BNCI2014_004:2>3' (default all B)")
    ap.add_argument("--epochs-b", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.dirname(args.out) or "."
    commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=[out_dir, args.bundle_root])
    code_sig = source_code_signature()
    for p in (args.out, args.manifest):
        if os.path.exists(p):
            os.remove(p)
    if args.pairs:
        sel = []
        for p in args.pairs.split(","):
            ds, ss = p.split(":"); a, b = ss.split(">"); sel.append((ds, int(a), int(b)))
    else:
        sel = PAIRS_B
    ep_cache = {}

    for ds, s_src, s_tgt in sel:
        ep = ep_cache.get(ds) or load_dataset(ds, MOABB_CLASS(ds)().subject_list)
        ep_cache[ds] = ep
        cfg = build_cfg(ep.X.shape[1], args.epochs_b, args.device, seed=0)
        for subj in np.unique(ep.subject):
            m_src = (ep.subject == subj) & (ep.session == s_src)
            if m_src.sum() < cfg.tta.min_target * 2:
                continue
            # REUSE the frozen V2-B checkpoint (cache hit -> no retrain)
            model, pooled_ref, R_src, pi_star = get_source(
                args.bundle_root, f"B:{ds}:s{subj}:sess{s_src}", cfg, code_sig,
                lambda ms=m_src: (ep.X[ms], ep.y[ms], ep.subject[ms]))
            tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, 2, args.device)
            adapt, evalm = contiguous_split(ep, subj, s_tgt)
            if len(evalm) < 4:
                continue
            ya = ep.y[adapt]
            n0a, n1a = int((ya == 0).sum()), int((ya == 1).sum())
            M = 4 * (min(n0a, n1a) // 3)                        # 3M/4 <= min(n0,n1); divisible by 4
            if M < cfg.tta.min_target:
                continue
            U_eval = _embed(model, ep.X[evalm], args.device)
            ye = ep.y[evalm]
            base = dict(audit="V2P", commit=commit, code_sig=code_sig, dataset=ds,
                        pair=f"{ds}:{s_src}>{s_tgt}", subject=int(subj), session=int(s_tgt),
                        pool_size=int(M), n_eval=int(len(evalm)))
            for rname, frac_left in RATIOS:
                seed = stable_hash_int(ds, int(subj), s_tgt, rname)
                pool = _pool_indices(adapt, ep.y, frac_left, M, seed)
                if pool is None:
                    continue
                yp = ep.y[pool]
                append_row(args.manifest, dict(dataset=ds, subject=int(subj), session=int(s_tgt),
                           ratio=rname, pool_size=int(M), n_left=int((yp == 0).sum()),
                           n_right=int((yp == 1).sum()), seed=int(seed),
                           pool_trial_ids=[int(i) for i in pool],
                           eval_trial_ids=[int(i) for i in evalm]))
                U_pool = _embed(model, ep.X[pool], args.device)
                logodds = math.log(((yp == 1).sum() + 1e-9) / ((yp == 0).sum() + 1e-9))  # right log-odds
                res = eval_pool(model, tta, pooled_ref, U_eval, ye, U_pool, args.device)
                for method in METHODS:
                    r = res[method]
                    append_row(args.out, dict(base, ratio=rname, pool_logodds=float(logodds),
                                              method=method, **r))
        print(f"[V2P] {ds}:{s_src}>{s_tgt} done", flush=True)
    if os.path.exists(args.out):
        print(f"[V2P] rows -> {args.out} sha={sha256_file(args.out)[:12]} manifest={args.manifest}", flush=True)


if __name__ == "__main__":
    main()
