"""WAVE 0 / W0.4 — batch-size / weak-identification audit (frozen in W0.4_MECH_APPENDUM.md).

For each subject: reuse the frozen terminal W2 bundle, embed the full adaptation night + a FIXED evaluation
set. For n in {16,32,64,128,256} and draws d in 0..4, draw n trials NATURALLY (preserving the non-uniform
stage prevalence) from the adaptation night, fit the joint-EM prior pi_J(n,d) on the draw, and decode the
identity-geometry decision under {Unif, rho_E, rho_A_draw, pi_J} on the fixed eval set. Emit the exact
3-part P_J split + convergence diagnostics. Deterministic eval-only reuse; real-subject-id addressed.

  python -m h2cmi.run_w2_wave0_batchsweep --subjects 12 --seeds 0,1,2 --out results/h2cmi/wave0_batchsweep/w0p4_s12.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from h2cmi.eval.harness import _embed
from h2cmi.p0_source import get_source_p0, ProvenanceError, source_sig
from h2cmi.run_w2_sleep import sleep_cfg, NC
from h2cmi.run_w2_p0 import _load_cached, _rho
from h2cmi.run_w2_wave0 import determinism_setup, _gpu_manifest, _hash_arr
from h2cmi.run_prior_decomp import _bacc_recall, _tv
from h2cmi.tta.weighted_tta import fit_weighted_em
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, sha256_file, stable_hash_int

TERMINAL_ROOT = "results/h2cmi/p0_w2_bundles"
OUT_DIR = "results/h2cmi/wave0_batchsweep"
NS = [16, 32, 64, 128, 256]
DRAWS = 5
PRIOR_FLOOR = 1e-12                                    # frozen implementation prior transform: log(clamp_min)


def _entropy(pi):
    p = np.asarray(pi, float); p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _natural_draw(ya, n, seed):
    """Random n indices WITHOUT stratification (preserves the natural non-uniform prevalence)."""
    rng = np.random.default_rng(seed)
    if n >= len(ya):
        return np.arange(len(ya))
    return rng.choice(len(ya), size=n, replace=False)


def run_subject(tgt, seeds, cache, out_path, code_sig, commit, gpu, device, epochs):
    bench = json.load(open(os.path.join(cache, "p0_benchmark.json")))["subject_ids"]
    N = len(bench); others = [s for s in bench if s != tgt]
    Xt, yt, st, nt = _load_cached(cache, [tgt])
    am, em = (nt == 1), (nt == 2)                     # primary protocol: night1 adapt, night2 eval (fixed)
    if am.sum() < 16 or em.sum() < 8:
        print(f"[W0.4] subj {tgt}: insufficient -> skip", flush=True); return
    Xa, ya, Xe, ye = Xt[am], yt[am], Xt[em], yt[em]
    eval_hash = _hash_arr(np.asarray(ye, np.int64))
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            if l.strip():
                r = json.loads(l)
                if r.get("marker") == "BATCHSWEEP":
                    done.add((int(r["seed"]), int(r["n"]), int(r["draw"])))
    for seed in seeds:
        determinism_setup(seed)
        cfg = sleep_cfg(epochs, device, seed=seed)
        tag = f"W2P0:sleep:loso{tgt}:nb{N}"
        def _data_fn():
            X, y, subj, _n = _load_cached(cache, others); return X, y, subj
        try:
            model, pooled_ref, R_src, pi_star, val = get_source_p0(TERMINAL_ROOT, TERMINAL_ROOT, tag, cfg, code_sig, NC, _data_fn)
        except ProvenanceError as pe:
            append_row(out_path, dict(marker="BATCHSWEEP", target_subject=int(tgt), seed=int(seed), provenance_fail=str(pe))); continue
        except RuntimeError as re:
            if "deterministic" in str(re).lower():
                append_row(out_path, dict(marker="BATCHSWEEP", target_subject=int(tgt), seed=int(seed), n=-1, draw=-1, determinism_fail=str(re)[:200])); continue
            raise
        density = model.head.density
        Ua = _embed(model, Xa, device); Ue = _embed(model, Xe, device)
        with torch.no_grad():
            logp = density.log_prob_all(Ue)
        rhoE = np.asarray(_rho(ye), float); rhoA_full = np.asarray(_rho(ya), float); unif = np.full(NC, 1.0 / NC)
        # invariant anchors (do not depend on n/draw)
        B_unif, _ = _bacc_recall(logp, ye, unif)
        B_rhoE, _ = _bacc_recall(logp, ye, rhoE)
        try:
            src_sha = sha256_file(os.path.join(TERMINAL_ROOT, f"{source_sig(tag, code_sig, cfg)}.pt"))[:16]
        except Exception:
            src_sha = None
        base = dict(marker="BATCHSWEEP", panel="W2_W0BATCH", commit=commit, code_sig=code_sig,
                    target_subject=int(tgt), seed=int(seed), seed0_validated=bool(val), eval_hash=eval_hash,
                    source_bundle_sha=src_sha, gpu=gpu, prior_floor=PRIOR_FLOOR, n_adapt_full=int(am.sum()),
                    n_eval=int(em.sum()), rho_E=[float(x) for x in rhoE], rho_A_full=[float(x) for x in rhoA_full],
                    B_unif=B_unif, B_rhoE=B_rhoE, metric_mismatch=float(B_rhoE - B_unif))
        for n in NS:
            for d in range(DRAWS):
                if (seed, n, d) in done:
                    continue
                idx = _natural_draw(ya, n, stable_hash_int("W0.4", int(tgt), int(seed), n, d))
                Ua_d = Ua[idx]; ya_d = ya[idx]
                rhoA_draw = np.asarray(_rho(ya_d), float)
                miss = int((np.bincount(ya_d, minlength=NC) == 0).sum())
                ts = stable_hash_int("W0.4fit", int(tgt), int(seed), n, d)
                _, piJt = fit_weighted_em(density, Ua_d, np.ones(len(Ua_d)), pi_star, cfg.tta, NC, device, "joint", tta_seed=ts)
                piJ = np.asarray(piJt.cpu().numpy() if torch.is_tensor(piJt) else piJt, float)
                B_rhoA, _ = _bacc_recall(logp, ye, rhoA_draw)
                B_piJ, _ = _bacc_recall(logp, ye, piJ)
                P_J = B_piJ - B_unif
                transfer = B_rhoA - B_rhoE; deviation = B_piJ - B_rhoA
                append_row(out_path, dict(base, n=int(n), draw=int(d), n_actual=int(len(idx)),
                    rho_A_draw=[float(x) for x in rhoA_draw], missing_classes=miss,
                    P_J=P_J, transfer=transfer, prior_estimate_deviation=deviation,
                    residual=float(P_J - ((B_rhoE - B_unif) + transfer + deviation)),
                    B_rhoA_draw=B_rhoA, B_piJ=B_piJ, pi_J=[float(x) for x in piJ],
                    TV_piJ_rhoA_draw=_tv(piJ, rhoA_draw), TV_piJ_rhoA_full=_tv(piJ, rhoA_full),
                    min_piJ=float(piJ.min()), H_piJ=_entropy(piJ), min_rhoA_draw=float(rhoA_draw.min())))
        print(f"[W0.4] subj {tgt} seed {seed} done ({len(NS)}x{DRAWS})", flush=True)
    if os.path.exists(out_path):
        print(f"[W0.4] subj {tgt} -> sha={sha256_file(out_path)[:12]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default=""); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--cache", default="results/h2cmi/p0_sleep_cache")
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=["results/h2cmi", OUT_DIR, TERMINAL_ROOT, args.cache])
    code_sig = source_code_signature()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    bench = json.load(open(os.path.join(args.cache, "p0_benchmark.json")))["subject_ids"]
    targets = list(bench) if not args.subjects else [int(s) for s in args.subjects.split(",") if s != ""]
    bad = [t for t in targets if t not in bench]
    if bad:
        raise SystemExit(f"subjects {bad} not in benchmark")
    gpu = _gpu_manifest()
    from h2cmi.wave0_fanout import output_path
    for tgt in targets:
        u = dict(wave="W0.4", protocol="batchsweep", real_subject_id=int(tgt))
        run_subject(tgt, seeds, args.cache, output_path(u, OUT_DIR, prefix="w0p4"), code_sig, commit, gpu, args.device, args.epochs)


if __name__ == "__main__":
    main()
