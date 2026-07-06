"""WAVE 0 / W0.3 — same-session CONSISTENCY control (within-session fake-split, bidirectional).

Deterministic eval-only reuse of the frozen terminal W2 bundles (p0_w2_bundles, code_sig 763bf49d): for
each subject we take a SINGLE night, split it STRATIFIED-by-stage into halves A,B (equal prevalence), and
evaluate BOTH directions (adapt A -> eval B, and adapt B -> eval A). Because prevalence is held equal,
this is NOT an "identity-geometry null" -- a real subject-geometry gap may remain. Primary readouts
(computed in the analyzer): directional asymmetry Delta(A->B)-Delta(B->A) ~ 0; G/P/I_int distribution;
minority-stage recall collapse expected WEAKER than the W0.1 cross-night case (since pi_J ~ rho_eval).
Addressing by real subject id (no bench index). Terminal bundles read-only.

  python -m h2cmi.run_w2_wave0_null --subjects 12,17 --seeds 0,1,2 \
      --out results/h2cmi/wave0_w2null/W0.3_s12.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from h2cmi.eval.harness import _embed
from h2cmi.tta.class_conditional import ClassConditionalTTA
from h2cmi.eval.p0_eval import eval_unit_p0
from h2cmi.p0_source import get_source_p0, ProvenanceError, source_sig
from h2cmi.run_w2_sleep import sleep_cfg, NC
from h2cmi.run_w2_p0 import paired_subjects, _load_cached, _rho, _js, _confusion
from h2cmi.run_w2_wave0 import determinism_setup, _gpu_manifest, ALL_BRANCHES, _hash_arr
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, sha256_file, stable_hash_int

TERMINAL_ROOT = "results/h2cmi/p0_w2_bundles"     # frozen terminal bundles, REUSED read-only
OUT_DIR = "results/h2cmi/wave0_w2null"


def stratified_halves(y, seed):
    """Split indices into A,B with ~equal per-class counts (equal prevalence). Deterministic given seed."""
    rng = np.random.default_rng(seed)
    A = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        A[idx[:len(idx) // 2]] = True
    return A, ~A


def _seeds_present(out_path):
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            if l.strip():
                r = json.loads(l)
                if r.get("branch") == "__decomposition__" and r.get("direction") == "BA":
                    done.add(int(r["seed"]))
    return done


def run_subject(tgt, seeds, cache, out_path, code_sig, commit, gpu, device, epochs):
    bench = json.load(open(os.path.join(cache, "p0_benchmark.json")))["subject_ids"]
    N = len(bench); others = [s for s in bench if s != tgt]
    Xt, yt, st, nt = _load_cached(cache, [tgt])
    # single session: pick the night with more trials
    n1, n2 = int((nt == 1).sum()), int((nt == 2).sum())
    night = 1 if n1 >= n2 else 2
    sess = (nt == night)
    Xs, ys = Xt[sess], yt[sess]
    if len(ys) < 48 or (ys == np.bincount(ys, minlength=NC).argmax()).sum() == len(ys):
        print(f"[W0.3] subj {tgt}: night{night} insufficient/degenerate -> skip", flush=True); return
    split_seed = stable_hash_int("W0.3split", int(tgt))
    A, B = stratified_halves(ys, split_seed)
    if A.sum() < 16 or B.sum() < 16 or (ys[A] == 0).sum() < 2 or (ys[B] == 0).sum() < 2:
        print(f"[W0.3] subj {tgt}: halves too small -> skip", flush=True); return
    already = _seeds_present(out_path)
    for seed in seeds:
        if seed in already:
            print(f"[W0.3] subj {tgt} seed {seed} recorded -> skip", flush=True); continue
        determinism_setup(seed)
        cfg = sleep_cfg(epochs, device, seed=seed)
        tag = f"W2P0:sleep:loso{tgt}:nb{N}"           # SAME tag -> reuse terminal LOSO bundle
        srcX = srcY = None
        def _data_fn():
            nonlocal srcX, srcY
            X, y, subj, _n = _load_cached(cache, others); srcX, srcY = X, y
            return X, y, subj
        try:
            model, pooled_ref, R_src, pi_star, val = get_source_p0(TERMINAL_ROOT, TERMINAL_ROOT, tag, cfg, code_sig, NC, _data_fn)
        except ProvenanceError as pe:
            append_row(out_path, dict(panel="W2_W0NULL", target_subject=int(tgt), seed=int(seed), provenance_fail=str(pe)))
            print(f"PROV FAIL: {pe}"); continue
        except RuntimeError as re:
            if "deterministic" in str(re).lower():
                append_row(out_path, dict(panel="W2_W0NULL", target_subject=int(tgt), seed=int(seed),
                                          branch="__decomposition__", direction="BA", determinism_fail=str(re)[:300]))
                print(f"DETERMINISM FAIL subj {tgt} seed {seed}"); continue
            raise
        try:
            src_sha = sha256_file(os.path.join(TERMINAL_ROOT, f"{source_sig(tag, code_sig, cfg)}.pt"))[:16]
        except Exception:
            src_sha = None
        rho_source = _rho(srcY) if srcY is not None else None
        tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, NC, device)
        base0 = dict(panel="W2_W0NULL", commit=commit, code_sig=code_sig, target_subject=int(tgt),
                     seed=int(seed), seed0_validated=bool(val), night=night, split_seed=int(split_seed),
                     source_bundle_sha=src_sha, gpu=gpu, rho_source=rho_source)
        for direction, am, em in (("AB", A, B), ("BA", B, A)):
            Xa, ya, Xe, ye = Xs[am], ys[am], Xs[em], ys[em]
            Ua, Ue = _embed(model, Xa, device), _embed(model, Xe, device)
            ts = stable_hash_int("W0.3", int(tgt), int(seed), direction)
            branches, decomp = eval_unit_p0(model, tta, pooled_ref, R_src, Xa, Xe, Ua, Ue, ye,
                                            device, NC, ts, keep_probs=False, keep_preds=True)
            rho_adapt, rho_eval = _rho(ya), _rho(ye); piJ = decomp["pi_J"]
            base = dict(base0, direction=direction, n_adapt=int(am.sum()), n_eval=int(em.sum()),
                        rho_adaptation=rho_adapt, rho_evaluation=rho_eval,
                        js_adapt_eval=_js(rho_adapt, rho_eval), js_piJ_eval=_js(piJ, rho_eval))
            for name in ALL_BRANCHES:
                rec = branches.get(name)
                if rec is None:
                    continue
                preds = rec.pop("preds", None); rec.pop("probs", None)
                row = dict(base, branch=name, **rec)
                if preds is not None:
                    C, recall = _confusion(np.asarray(ye), np.asarray(preds))
                    row["confusion"] = C; row["per_stage_recall"] = recall
                    row["pred_hash_full"] = _hash_arr(np.asarray(preds, np.int64))
                append_row(out_path, row)
            append_row(out_path, dict(base, branch="__decomposition__", **decomp))
        print(f"[W0.3] subj {tgt} seed {seed} DONE (night{night}, both directions)", flush=True)
    if os.path.exists(out_path):
        print(f"[W0.3] subj {tgt} -> {out_path} sha={sha256_file(out_path)[:12]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="", help="comma-separated REAL subject ids (NOT indices)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--cache", default="results/h2cmi/p0_sleep_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=["results/h2cmi", OUT_DIR, TERMINAL_ROOT, args.cache])
    code_sig = source_code_signature()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    bench = json.load(open(os.path.join(args.cache, "p0_benchmark.json")))["subject_ids"]
    targets = list(bench) if not args.subjects else [int(s) for s in args.subjects.split(",") if s != ""]
    bad = [t for t in targets if t not in bench]
    if bad:
        raise SystemExit(f"subjects {bad} not in benchmark (REAL ids required)")
    gpu = _gpu_manifest()
    from h2cmi.wave0_fanout import output_path
    for tgt in targets:
        u = dict(wave="W0.3", protocol="samesession", real_subject_id=int(tgt))
        run_subject(tgt, seeds, args.cache, output_path(u, OUT_DIR, prefix="w0p3"), code_sig, commit, gpu, args.device, args.epochs)


if __name__ == "__main__":
    main()
