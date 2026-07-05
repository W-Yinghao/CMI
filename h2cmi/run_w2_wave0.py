"""WAVE 0 / W0.1 — W2 DETERMINISTIC rerun (closes the REVIEW_P0 confusion reproducibility hole).

Reuses the existing p0_sleep_cache; trains all seeds DETERMINISTICALLY into a NEW root
(p0_w2_det_bundles) so the terminal 278fc85 bundles are untouched. Full per-(subject,seed,branch)
logging: pred_hash, logit_hash, confusion, per_stage_recall, pi_J, T_J params/norm + a per-fold
provenance manifest (GPU type, cuda/torch/lib versions, adapt/eval split hash, source-bundle SHA, seed).

Append-only, one file per subject (results/h2cmi/wave0_w2det/p0w2det_<subj>.jsonl); a (seed) already
present is SKIPPED -> no completed fold is ever recomputed or lost on a walltime kill.

  python -m h2cmi.run_w2_wave0 --targets 12-12 --seeds 0,1,2 --protocol primary
  python -m h2cmi.run_w2_wave0 --self-replay 12   # re-run subj 12 into a scratch file + compare hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform

import numpy as np
import torch

from h2cmi.eval.harness import _embed
from h2cmi.tta.class_conditional import ClassConditionalTTA
from h2cmi.eval.p0_eval import eval_unit_p0
from h2cmi.p0_source import get_source_p0, ProvenanceError
from h2cmi.run_w2_sleep import sleep_cfg, NC
from h2cmi.run_w2_p0 import paired_subjects, _load_cached, _rho, _js, _confusion
from h2cmi.grid_io import (require_clean_git, source_code_signature, append_row, sha256_file,
                           stable_hash_int)

ALL_BRANCHES = ("identity_uniform", "identity_joint_prior", "joint_geometry_uniform",
                "joint_geometry_joint_prior", "fixed_iterative_geometry_uniform",
                "fixed_reference_oneshot_uniform", "pooled_uniform", "latent_im_diag_uniform",
                "source_recolored_ea")
DET_ROOT = "results/h2cmi/p0_w2_det_bundles"
OUT_DIR = "results/h2cmi/wave0_w2det"


def determinism_setup(seed):
    """Full determinism. CUBLAS_WORKSPACE_CONFIG must also be exported in the SLURM env (before cuda init)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _gpu_manifest():
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return dict(gpu_name=dev, cuda=torch.version.cuda, torch=torch.__version__,
                cublas_ws=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                slurm_partition=os.environ.get("SLURM_JOB_PARTITION"),
                slurm_node=os.environ.get("SLURMD_NODENAME"), python=platform.python_version())


def _hash_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _seeds_present(out_path):
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            if r.get("branch") == "__decomposition__":
                done.add(int(r["seed"]))
    return done


def run_subject(tgt, seeds, protocol, cache, out_path, code_sig, commit, gpu_manifest, device, epochs):
    bench = json.load(open(os.path.join(cache, "p0_benchmark.json")))["subject_ids"]
    N = len(bench)
    others = [s for s in bench if s != tgt]
    Xt, yt, st, nt = _load_cached(cache, [tgt])
    if protocol == "primary":
        am, em = (nt == 1), (nt == 2)
    else:
        idx2 = np.where(nt == 2)[0]; h = len(idx2) // 2
        am = np.zeros(len(nt), bool); em = np.zeros(len(nt), bool)
        am[idx2[:h]] = True; em[idx2[h:]] = True
    if am.sum() < 16 or em.sum() < 8:
        print(f"[W0.1] target {tgt} protocol {protocol}: insufficient -> skip", flush=True); return
    Xa, ya, Xe, ye = Xt[am], yt[am], Xt[em], yt[em]
    split_hash = _hash_arr(np.concatenate([am.astype(np.int8), em.astype(np.int8)]))
    already = _seeds_present(out_path)
    for seed in seeds:
        if seed in already:
            print(f"[W0.1] subj {tgt} seed {seed} already recorded -> skip", flush=True); continue
        determinism_setup(seed)
        cfg = sleep_cfg(epochs, device, seed=seed)
        tag = f"W2W0:sleep:loso{tgt}:nb{N}"
        srcX = srcY = None
        def _data_fn():
            nonlocal srcX, srcY
            X, y, subj, _n = _load_cached(cache, others); srcX, srcY = X, y
            return X, y, subj
        det_fail = None
        try:
            model, pooled_ref, R_src, pi_star, val = get_source_p0(DET_ROOT, DET_ROOT, tag, cfg, code_sig, NC, _data_fn)
        except ProvenanceError as pe:
            append_row(out_path, dict(panel="W2_W0", protocol=protocol, target_subject=int(tgt),
                                      seed=int(seed), provenance_fail=str(pe))); print(f"PROV FAIL: {pe}"); continue
        except RuntimeError as re:
            if "deterministic" in str(re).lower():
                append_row(out_path, dict(panel="W2_W0", protocol=protocol, target_subject=int(tgt),
                                          seed=int(seed), branch="__decomposition__", determinism_fail=str(re)[:300],
                                          gpu=gpu_manifest))
                print(f"DETERMINISM FAIL subj {tgt} seed {seed}: {str(re)[:160]}", flush=True); continue
            raise
        src_sha = None
        try:
            from h2cmi.p0_source import source_sig
            src_sha = sha256_file(os.path.join(DET_ROOT, f"{source_sig(tag, code_sig, cfg)}.pt"))[:16]
        except Exception:
            pass
        rho_source = _rho(srcY) if srcY is not None else None
        tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, NC, device)
        Ua, Ue = _embed(model, Xa, device), _embed(model, Xe, device)
        ts = stable_hash_int("W2", int(tgt), int(seed), protocol)
        branches, decomp = eval_unit_p0(model, tta, pooled_ref, R_src, Xa, Xe, Ua, Ue, ye,
                                        device, NC, ts, keep_probs=True, keep_preds=True)
        rho_eval = _rho(ye); rho_adapt = _rho(ya); piJ = decomp["pi_J"]
        js = dict(js_source_adapt=_js(rho_source, rho_adapt), js_adapt_eval=_js(rho_adapt, rho_eval),
                  js_source_eval=_js(rho_source, rho_eval), js_piJ_eval=_js(piJ, rho_eval))
        base = dict(panel="W2_W0", commit=commit, code_sig=code_sig, protocol=protocol,
                    target_subject=int(tgt), seed=int(seed), seed0_validated=bool(val),
                    n_adapt=int(am.sum()), n_eval=int(em.sum()), split_hash=split_hash,
                    source_bundle_sha=src_sha, gpu=gpu_manifest, rho_source=rho_source,
                    rho_adaptation_night=rho_adapt, rho_evaluation_night=rho_eval, **js)
        for name in ALL_BRANCHES:
            rec = branches.get(name)
            if rec is None:
                continue
            probs = rec.pop("probs", None); preds = rec.pop("preds", None)
            row = dict(base, branch=name, **rec)
            if preds is not None:
                C, recall = _confusion(np.asarray(ye), np.asarray(preds))
                row["confusion"] = C; row["per_stage_recall"] = recall
                row["pred_hash_full"] = _hash_arr(np.asarray(preds, np.int64))
            if probs is not None:
                row["logit_hash"] = _hash_arr(np.asarray(probs, np.float32).round(6))
            append_row(out_path, row)
        append_row(out_path, dict(base, branch="__decomposition__", **decomp))
        print(f"[W0.1] subj {tgt} seed {seed} DONE (det, gpu={gpu_manifest['gpu_name']})", flush=True)
    if os.path.exists(out_path):
        print(f"[W0.1] subj {tgt} -> {out_path} sha={sha256_file(out_path)[:12]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--protocol", default="primary", choices=["primary", "secondary"])
    ap.add_argument("--cache", default="results/h2cmi/p0_sleep_cache")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--self-replay", type=int, default=-1, help="subject id: re-run into scratch + compare hashes")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(DET_ROOT, exist_ok=True)
    commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=[OUT_DIR, DET_ROOT, args.cache])
    code_sig = source_code_signature()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    gpu = _gpu_manifest()
    bench = json.load(open(os.path.join(args.cache, "p0_benchmark.json")))["subject_ids"]
    if args.self_replay >= 0:
        tgt = args.self_replay
        scratch = os.path.join(OUT_DIR, f"replay_scratch_{tgt}.jsonl")
        if os.path.exists(scratch):
            os.remove(scratch)
        run_subject(tgt, seeds, args.protocol, args.cache, scratch, code_sig, commit, gpu, args.device, args.epochs)
        orig = os.path.join(OUT_DIR, f"p0w2det_{args.protocol}_{tgt}.jsonl")
        _compare_replay(orig, scratch)
        return
    targets = bench
    if args.targets:
        a, b = (int(x) for x in args.targets.split("-")); targets = bench[a:b + 1]
    for tgt in targets:
        out_path = os.path.join(OUT_DIR, f"p0w2det_{args.protocol}_{tgt}.jsonl")
        run_subject(tgt, seeds, args.protocol, args.cache, out_path, code_sig, commit, gpu, args.device, args.epochs)


def _compare_replay(orig, scratch):
    def hashes(p):
        h = {}
        for l in open(p):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            if r.get("branch") and r.get("branch") != "__decomposition__" and "pred_hash_full" in r:
                h[(r["seed"], r["branch"])] = (r["pred_hash_full"], r.get("logit_hash"))
        return h
    if not os.path.exists(orig):
        print(f"[self-replay] no original {orig} -> cannot compare"); return
    a, b = hashes(orig), hashes(scratch)
    keys = set(a) & set(b)
    match = sum(1 for k in keys if a[k] == b[k])
    print(f"[self-replay] {match}/{len(keys)} (seed,branch) predictions BIT-IDENTICAL on re-run "
          f"({'PASS deterministic' if match == len(keys) and keys else 'FAIL nondeterministic'})")


if __name__ == "__main__":
    main()
