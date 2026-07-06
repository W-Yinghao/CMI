"""WAVE 0 mechanistic SECONDARY (frozen in W0.3_MECH_APPENDUM.md): exact three-part decomposition of the
decision-prior harm P_J = B_E(pi_J) - B_E(Unif) into
  metric-prior mismatch  [B_E(rho_E) - B_E(Unif)]
  + adapt->eval transfer [B_E(rho_A) - B_E(rho_E)]     (== 0 for same-session)
  + pi_J estimation error[B_E(pi_J) - B_E(rho_A)].
Reuses the frozen terminal W2 bundles (p0_w2_bundles) and the SAME splits/seeds as the main runs, so
B_E(pi_J) equals the main identity_joint_prior bAcc (built-in consistency check). Deterministic eval-only.
Protocols: crossnight (night1 adapt -> night2 eval, matches W0.1) and samesession (stratified halves,
both directions, matches W0.3). Real-id addressed; skip-done.

  python -m h2cmi.run_prior_decomp --protocol crossnight  --subjects 12 --out results/h2cmi/wave0_priordecomp/pd_crossnight_12.jsonl
  python -m h2cmi.run_prior_decomp --protocol samesession --subjects 12 --out results/h2cmi/wave0_priordecomp/pd_samesession_12.jsonl
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
from h2cmi.run_w2_p0 import _load_cached, _rho
from h2cmi.run_w2_wave0 import determinism_setup, _gpu_manifest
from h2cmi.run_w2_wave0_null import stratified_halves
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, sha256_file, stable_hash_int

TERMINAL_ROOT = "results/h2cmi/p0_w2_bundles"
OUT_DIR = "results/h2cmi/wave0_priordecomp"
STAGES = ["W", "N1", "N2", "N3", "REM"]


def _bacc_recall(logp, ye, pi):
    """balanced accuracy + per-stage recall of the identity decoder argmax(logp + log pi)."""
    lp = logp + torch.log(torch.tensor(pi, dtype=logp.dtype, device=logp.device).clamp_min(1e-12)).view(1, -1)
    pred = lp.argmax(1).cpu().numpy()
    rec = [float((pred[ye == c] == c).mean()) if (ye == c).sum() else float("nan") for c in range(NC)]
    return float(np.nanmean(rec)), rec


def _tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def _kl(p, q):
    p = np.asarray(p, float) + 1e-12; q = np.asarray(q, float) + 1e-12
    return float((p * np.log(p / q)).sum())


def _units(protocol, Xt, yt, nt, tgt):
    """yield (direction, Xa, ya, Xe, ye)."""
    if protocol == "crossnight":
        am, em = (nt == 1), (nt == 2)
        if am.sum() >= 16 and em.sum() >= 8:
            yield "N1N2", Xt[am], yt[am], Xt[em], yt[em]
    else:                                                 # samesession: stratified halves, both directions
        n1, n2 = int((nt == 1).sum()), int((nt == 2).sum())
        night = 1 if n1 >= n2 else 2
        sess = (nt == night); Xs, ys = Xt[sess], yt[sess]
        if len(ys) < 48:
            return
        A, B = stratified_halves(ys, stable_hash_int("W0.3split", int(tgt)))
        if A.sum() >= 16 and B.sum() >= 16:
            yield "AB", Xs[A], ys[A], Xs[B], ys[B]
            yield "BA", Xs[B], ys[B], Xs[A], ys[A]


def run_subject(tgt, seeds, protocol, cache, out_path, code_sig, commit, gpu, device, epochs):
    bench = json.load(open(os.path.join(cache, "p0_benchmark.json")))["subject_ids"]
    N = len(bench); others = [s for s in bench if s != tgt]
    Xt, yt, st, nt = _load_cached(cache, [tgt])
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            if l.strip():
                r = json.loads(l)
                if r.get("marker") == "PRIORDECOMP":
                    done.add((int(r["seed"]), r["direction"]))
    for seed in seeds:
        determinism_setup(seed)
        cfg = sleep_cfg(epochs, device, seed=seed)
        tag = f"W2P0:sleep:loso{tgt}:nb{N}"
        def _data_fn():
            X, y, subj, _n = _load_cached(cache, others)
            return X, y, subj
        try:
            model, pooled_ref, R_src, pi_star, val = get_source_p0(TERMINAL_ROOT, TERMINAL_ROOT, tag, cfg, code_sig, NC, _data_fn)
        except ProvenanceError as pe:
            append_row(out_path, dict(marker="PRIORDECOMP", target_subject=int(tgt), seed=int(seed), provenance_fail=str(pe))); continue
        except RuntimeError as re:
            if "deterministic" in str(re).lower():
                append_row(out_path, dict(marker="PRIORDECOMP", target_subject=int(tgt), seed=int(seed), direction="NA", determinism_fail=str(re)[:200])); continue
            raise
        tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, NC, device)
        density = model.head.density
        for direction, Xa, ya, Xe, ye in _units(protocol, Xt, yt, nt, tgt):
            if (seed, direction) in done:
                continue
            Ua, Ue = _embed(model, Xa, device), _embed(model, Xe, device)
            ts_key = ("W2", int(tgt), int(seed), "primary") if protocol == "crossnight" else ("W0.3", int(tgt), int(seed), direction)
            ts = stable_hash_int(*ts_key)
            branches, decomp = eval_unit_p0(model, tta, pooled_ref, R_src, Xa, Xe, Ua, Ue, ye, device, NC, ts)
            piJ = np.asarray(decomp["pi_J"], float)
            rhoE = np.asarray(_rho(ye), float); rhoA = np.asarray(_rho(ya), float); unif = np.full(NC, 1.0 / NC)
            with torch.no_grad():
                logp = density.log_prob_all(Ue)
            B_unif, rec_unif = _bacc_recall(logp, ye, unif)
            B_rhoE, rec_rhoE = _bacc_recall(logp, ye, rhoE)
            B_rhoA, rec_rhoA = _bacc_recall(logp, ye, rhoA)
            B_piJ, rec_piJ = _bacc_recall(logp, ye, piJ)
            metric_mismatch = B_rhoE - B_unif; transfer = B_rhoA - B_rhoE; estimation = B_piJ - B_rhoA
            P_J = B_piJ - B_unif
            append_row(out_path, dict(marker="PRIORDECOMP", panel="PRIORDECOMP", protocol=protocol, commit=commit,
                code_sig=code_sig, target_subject=int(tgt), seed=int(seed), direction=direction, night_free=(protocol == "samesession"),
                B_unif=B_unif, B_rhoE=B_rhoE, B_rhoA=B_rhoA, B_piJ=B_piJ,
                P_J=P_J, metric_mismatch=metric_mismatch, transfer=transfer, estimation=estimation,
                residual=float(P_J - (metric_mismatch + transfer + estimation)),
                consistency_B_piJ_vs_main=float(B_piJ - branches["identity_joint_prior"]["bacc"]),
                consistency_B_unif_vs_main=float(B_unif - branches["identity_uniform"]["bacc"]),
                recall_unif=rec_unif, recall_rhoE=rec_rhoE, recall_rhoA=rec_rhoA, recall_piJ=rec_piJ,
                pi_J=[float(x) for x in piJ], rho_A=[float(x) for x in rhoA], rho_E=[float(x) for x in rhoE],
                TV_piJ_rhoA=_tv(piJ, rhoA), TV_piJ_rhoE=_tv(piJ, rhoE), KL_rhoA_piJ=_kl(rhoA, piJ),
                min_piJ=float(piJ.min()), piJ_minus_rhoA=[float(x) for x in (piJ - rhoA)], js_adapt_eval=_tv(rhoA, rhoE)))
        print(f"[PD {protocol}] subj {tgt} seed {seed} done", flush=True)
    if os.path.exists(out_path):
        print(f"[PD {protocol}] subj {tgt} -> sha={sha256_file(out_path)[:12]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["crossnight", "samesession"])
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
    for tgt in targets:
        run_subject(tgt, seeds, args.protocol, args.cache, args.out if args.subjects and len(targets) == 1 else
                    os.path.join(OUT_DIR, f"pd_{args.protocol}_{tgt}.jsonl"), code_sig, commit, gpu, args.device, args.epochs)


if __name__ == "__main__":
    main()
