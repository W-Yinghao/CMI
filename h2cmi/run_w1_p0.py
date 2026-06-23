"""W1 corrected runner (REVIEW_P0 section B): unseen-subject MI LOSO, source seeds {0,1,2}, corrected
section-A evaluation (fit joint once -> four geometry x decision-prior branches + comparators + G/P/
Interaction decomposition). Reuses the frozen seed-0 bundles after STRICT provenance validation; trains
seeds 1,2 into a new root; a provenance failure STOPS that unit (recorded, never relaxed).

  python -m h2cmi.run_w1_p0 --dataset Cho2017 --folds 0-6 --device cuda \
      --out results/h2cmi/p0_w1_Cho2017_0.jsonl
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from h2cmi.eval.harness import _embed
from h2cmi.tta.class_conditional import ClassConditionalTTA
from h2cmi.eval.p0_eval import eval_unit_p0
from h2cmi.p0_source import get_source_p0, ProvenanceError
from h2cmi.data.real_eeg import load_dataset, contiguous_split
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.run_v2 import build_cfg
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, sha256_file, stable_hash_int

K = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--folds", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seed0-root", default="results/h2cmi/w1_bundles")
    ap.add_argument("--new-root", default="results/h2cmi/p0_w1_bundles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.dirname(args.out) or "."
    commit = require_clean_git(allow_dirty=args.allow_dirty,
                               ignore_prefixes=[out_dir, args.new_root, args.seed0_root])
    code_sig = source_code_signature()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    if os.path.exists(args.out):
        os.remove(args.out)
    ep = load_dataset(args.dataset, MOABB_CLASS(args.dataset)().subject_list)
    subs = list(np.unique(ep.subject))
    if args.folds:
        a, b = (int(x) for x in args.folds.split("-")); subs = subs[a:b + 1]
    for tgt in subs:
        m_src = ep.subject != tgt
        Xs, ys, subj_s = ep.X[m_src], ep.y[m_src], ep.subject[m_src]
        sess0 = ep.session[ep.subject == tgt].min()
        ai, ei = contiguous_split(ep, tgt, sess0)
        if len(ai) < 16 or len(ei) < 4:
            continue
        Xa, Xe, ye = ep.X[ai], ep.X[ei], ep.y[ei]
        for seed in seeds:
            cfg = build_cfg(ep.X.shape[1], args.epochs, args.device, seed=seed)
            tag = f"W1:{args.dataset}:loso{tgt}"
            try:
                model, pooled_ref, R_src, pi_star, val = get_source_p0(
                    args.seed0_root, args.new_root, tag, cfg, code_sig, K, lambda: (Xs, ys, subj_s))
            except ProvenanceError as pe:
                append_row(args.out, dict(panel="W1_P0", dataset=args.dataset, target_subject=int(tgt),
                                          seed=int(seed), provenance_fail=str(pe)))
                print(f"PROVENANCE FAIL -> stop unit: {pe}", flush=True); continue
            tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, K, args.device)
            Ua, Ue = _embed(model, Xa, args.device), _embed(model, Xe, args.device)
            ts = stable_hash_int(args.dataset, int(tgt), int(seed))
            branches, decomp = eval_unit_p0(model, tta, pooled_ref, R_src, Xa, Xe, Ua, Ue, ye,
                                            args.device, K, ts)
            base = dict(panel="W1_P0", commit=commit, code_sig=code_sig, dataset=args.dataset,
                        target_subject=int(tgt), seed=int(seed), seed0_validated=bool(val),
                        n_adapt=int(len(ai)), n_eval=int(len(ei)))
            for name, rec in branches.items():
                append_row(args.out, dict(base, branch=name, **{k: v for k, v in rec.items() if k != "probs"}))
            append_row(args.out, dict(base, branch="__decomposition__", **decomp))
        print(f"[W1_P0 {args.dataset}] target={tgt} seeds={seeds} done", flush=True)
    if os.path.exists(args.out):
        print(f"[W1_P0 {args.dataset}] -> {args.out} sha={sha256_file(args.out)[:12]}", flush=True)


if __name__ == "__main__":
    main()
