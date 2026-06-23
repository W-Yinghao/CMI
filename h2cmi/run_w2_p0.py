"""W2 corrected runner (REVIEW_P0 section C): Sleep-EDF Sleep-Cassette, ALL valid paired-night subjects
(audited, not hard-coded), source seeds {0,1,2}, two protocols, corrected section-A evaluation + the
required W2 recordings (rho_source / rho_adapt / rho_eval, four JS divergences, per-stage recall,
per-subject + aggregate confusion). Frozen preprocessing reused. No reuse of the nb=20 seed-0 bundles
(the all-subjects LOSO source set differs); all seeds train into a new root.

  python -m h2cmi.run_w2_p0 --mode audit  --cache results/h2cmi/p0_sleep_cache
  python -m h2cmi.run_w2_p0 --mode cache  --cache results/h2cmi/p0_sleep_cache --subjects 0-9
  python -m h2cmi.run_w2_p0 --mode loso   --targets 0-4 --seeds 0,1,2 --protocol primary \
      --cache results/h2cmi/p0_sleep_cache --new-root results/h2cmi/p0_w2_bundles \
      --out results/h2cmi/p0_w2_primary_0.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from h2cmi.eval.harness import _embed
from h2cmi.tta.class_conditional import ClassConditionalTTA
from h2cmi.eval.p0_eval import eval_unit_p0
from h2cmi.p0_source import get_source_p0, ProvenanceError
from h2cmi.run_w2_sleep import sleep_cfg, NC
from h2cmi.data.sleep_eeg import load_subjects, _pair_files, STAGE_NAMES
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, sha256_file, stable_hash_int

CONF_BRANCHES = ("identity_uniform", "joint_geometry_uniform")   # keep preds for confusion


def paired_subjects():
    pairs = _pair_files()
    return [s for s in sorted(pairs) if len(pairs[s]) >= 2]


def _cache_path(cache, s):
    return os.path.join(cache, f"subj{s:02d}.npz")


def _rho(y, K=NC):
    return (np.bincount(y, minlength=K) / max(1, len(y))).tolist()


def _js(p, q):
    p = np.asarray(p, float) + 1e-12; q = np.asarray(q, float) + 1e-12
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: float((a * np.log(a / b)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _confusion(y, pred, K=NC):
    C = np.zeros((K, K), int)
    for t, p in zip(y, pred):
        C[t, p] += 1
    recall = [float(C[c, c] / C[c].sum()) if C[c].sum() else float("nan") for c in range(K)]
    return C.tolist(), recall


def _load_cached(cache, subs):
    Xs, ys, ss, ns = [], [], [], []
    for s in subs:
        d = np.load(_cache_path(cache, s))
        Xs.append(d["X"]); ys.append(d["y"]); ss.append(d["subject"]); ns.append(d["night"])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss), np.concatenate(ns)


def mode_audit(args):
    bench = paired_subjects()
    rec = dict(marker="W2_P0_PAIRED_NIGHT_SUBJECTS", n=len(bench), subject_ids=bench,
               stage_names=STAGE_NAMES)
    os.makedirs(args.cache, exist_ok=True)
    json.dump(rec, open(os.path.join(args.cache, "p0_benchmark.json"), "w"), indent=2)
    print(f"[W2_P0 audit] {len(bench)} valid paired-night subjects -> {args.cache}/p0_benchmark.json")
    print(f"  ids: {bench}")


def mode_cache(args):
    os.makedirs(args.cache, exist_ok=True)
    bench = json.load(open(os.path.join(args.cache, "p0_benchmark.json")))["subject_ids"]
    sel = bench
    if args.subjects:
        a, b = (int(x) for x in args.subjects.split("-")); sel = bench[a:b + 1]
    for s in sel:
        cp = _cache_path(args.cache, s)
        if os.path.exists(cp):
            continue
        ep = load_subjects([s]); np.savez(cp, X=ep.X, y=ep.y, subject=ep.subject, night=ep.night)
        print(f"[cache] subj {s}: X{ep.X.shape} stages={np.bincount(ep.y, minlength=NC).tolist()}", flush=True)


def mode_loso(args, code_sig, commit):
    bench = json.load(open(os.path.join(args.cache, "p0_benchmark.json")))["subject_ids"]
    N = len(bench)
    targets = bench
    if args.targets:
        a, b = (int(x) for x in args.targets.split("-")); targets = bench[a:b + 1]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    for tgt in targets:
        others = [s for s in bench if s != tgt]
        Xt, yt, st, nt = _load_cached(args.cache, [tgt])
        if args.protocol == "primary":
            am, em = (nt == 1), (nt == 2)
        else:                                            # secondary: night2 first half adapt / second half eval
            idx2 = np.where(nt == 2)[0]; h = len(idx2) // 2
            am = np.zeros(len(nt), bool); em = np.zeros(len(nt), bool)
            am[idx2[:h]] = True; em[idx2[h:]] = True
        if am.sum() < 16 or em.sum() < 8:
            print(f"[W2_P0] target {tgt} protocol {args.protocol}: insufficient -> skip", flush=True); continue
        Xa, ya, Xe, ye = Xt[am], yt[am], Xt[em], yt[em]
        for seed in seeds:
            cfg = sleep_cfg(args.epochs, args.device, seed=seed)
            tag = f"W2P0:sleep:loso{tgt}:nb{N}"
            srcX = srcY = None
            def _data_fn():
                nonlocal srcX, srcY
                X, y, subj, _n = _load_cached(args.cache, others); srcX, srcY = X, y
                return X, y, subj
            try:
                model, pooled_ref, R_src, pi_star, val = get_source_p0(
                    args.seed0_root, args.new_root, tag, cfg, code_sig, NC, _data_fn)
            except ProvenanceError as pe:
                append_row(args.out, dict(panel="W2_P0", protocol=args.protocol, target_subject=int(tgt),
                                          seed=int(seed), provenance_fail=str(pe))); print(f"PROV FAIL: {pe}"); continue
            rho_source = _rho(srcY) if srcY is not None else None
            tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, NC, args.device)
            Ua, Ue = _embed(model, Xa, args.device), _embed(model, Xe, args.device)
            ts = stable_hash_int("W2", int(tgt), int(seed), args.protocol)
            branches, decomp = eval_unit_p0(model, tta, pooled_ref, R_src, Xa, Xe, Ua, Ue, ye,
                                            args.device, NC, ts, keep_preds=True)
            rho_eval = _rho(ye); rho_adapt = _rho(ya); piJ = decomp["pi_J"]
            js = dict(js_source_adapt=_js(rho_source, rho_adapt), js_adapt_eval=_js(rho_adapt, rho_eval),
                      js_source_eval=_js(rho_source, rho_eval), js_piJ_eval=_js(piJ, rho_eval))
            base = dict(panel="W2_P0", commit=commit, code_sig=code_sig, protocol=args.protocol,
                        target_subject=int(tgt), seed=int(seed), seed0_validated=bool(val),
                        n_adapt=int(am.sum()), n_eval=int(em.sum()), rho_source=rho_source,
                        rho_adaptation_night=rho_adapt, rho_evaluation_night=rho_eval, **js)
            for name, rec in branches.items():
                preds = rec.pop("preds", None)
                row = dict(base, branch=name, **rec)
                if name in CONF_BRANCHES and preds is not None:
                    C, recall = _confusion(np.asarray(ye), np.asarray(preds))
                    row["confusion"] = C; row["per_stage_recall"] = recall
                append_row(args.out, row)
            append_row(args.out, dict(base, branch="__decomposition__", **decomp))
        print(f"[W2_P0 {args.protocol}] target={tgt} seeds={seeds} done", flush=True)
    if os.path.exists(args.out):
        print(f"[W2_P0 {args.protocol}] -> {args.out} sha={sha256_file(args.out)[:12]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["audit", "cache", "loso"])
    ap.add_argument("--targets", default=""); ap.add_argument("--subjects", default="")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--protocol", default="primary", choices=["primary", "secondary"])
    ap.add_argument("--cache", default="results/h2cmi/p0_sleep_cache")
    ap.add_argument("--seed0-root", default="results/h2cmi/w2_bundles")
    ap.add_argument("--new-root", default="results/h2cmi/p0_w2_bundles")
    ap.add_argument("--out", default="results/h2cmi/p0_w2.jsonl")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.mode == "audit":
        mode_audit(args); return
    if args.mode == "cache":
        mode_cache(args); return
    out_dir = os.path.dirname(args.out) or "."
    commit = require_clean_git(allow_dirty=args.allow_dirty,
                               ignore_prefixes=[out_dir, args.new_root, args.seed0_root, args.cache])
    code_sig = source_code_signature()
    if os.path.exists(args.out):
        os.remove(args.out)
    mode_loso(args, code_sig, commit)


if __name__ == "__main__":
    main()
