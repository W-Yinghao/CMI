"""W2: Sleep-EDF Sleep-Cassette cross-subject natural-prevalence-shift staging (review W1_W2_FROZEN).

Per target subject (2 nights): source = ALL nights of all OTHER benchmark subjects; target adaptation
= target NIGHT 1 (unlabeled); target evaluation = target NIGHT 2; unit = target subject. Methods:
identity, pooled, canonical fixed-prior CC, current_joint, EA, SPDIM, metadata_only. Frozen route =
DIAG_COMPATIBLE x DIFFERENT -> CC (first real-data trigger), so metadata_only == canonical CC. Target
labels eval-only (the adapt night's labels are used ONLY post-hoc for the JS mechanism analysis).

Modes:
  cache  load benchmark subjects -> per-subject npz (run ONCE; LOSO jobs then read the cache fast).
  loso   per target subject: train LOSO source (cached) + eval night1->night2.

Benchmark set (tractable standard): the first N Sleep-Cassette subjects WITH 2 valid nights
(Sleep-EDF-20 style); scaling to all 78 is a compute multiplier, not a method change.

  python -m h2cmi.run_w2_sleep --mode cache --n-bench 20 --cache results/h2cmi/sleep_cache
  python -m h2cmi.run_w2_sleep --mode loso --targets 0-9 --cache results/h2cmi/sleep_cache \
      --bundle-root results/h2cmi/w2_bundles --out results/h2cmi/w2_sleep_0.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from h2cmi.config import core_config, H2Config
from h2cmi.domains import DomainDAG, DomainFactor, DomainLabels
from h2cmi.train.trainer import train_h2, reference_prior, H2Model
from h2cmi.eval.harness import _embed, _predict_generative, _predict_transform
from h2cmi.tta.class_conditional import (ClassConditionalTTA, B1A_VARIANTS_BY_NAME,
                                         reference_weighted_source_moments)
from h2cmi.eval.ea import reference_cov, ea_transform, apply_ea
from h2cmi.eval.spdim import spdim_fit
from h2cmi.data.sleep_eeg import (load_subjects, subject_list, _pair_files,
                                  SLEEP_N_TIMES, SLEEP_FS, STAGE_NAMES)
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, sha256_file

W2_METHODS = ["identity", "always_pooled", "always_canonical_CC", "current_joint",
              "euclidean_alignment", "spdim", "metadata_only"]
NC = 5
UNI = np.full(NC, 1.0 / NC)


def sleep_cfg(epochs, device, seed=0):
    cfg = core_config(H2Config(n_classes=NC))
    cfg.encoder.n_chans = 2; cfg.encoder.n_times = SLEEP_N_TIMES; cfg.encoder.fs = SLEEP_FS
    cfg.encoder.use_spd = False; cfg.encoder.use_graph = False; cfg.encoder.use_temporal = True
    cfg.train.epochs = epochs; cfg.train.device = device; cfg.train.seed = seed
    cfg.cmi.enabled = False
    return cfg


def _domains(subject):
    subs = np.unique(subject); smap = {int(s): i for i, s in enumerate(subs)}
    site = np.array([smap[int(s)] for s in subject], np.int64)
    dag = DomainDAG([DomainFactor("site", max(1, len(subs)), (), "invariant", 0.02)])
    return dag, DomainLabels(dag, site.reshape(-1, 1))


def benchmark_subjects(n_bench):
    """First n_bench Sleep-Cassette subjects that have BOTH nights."""
    pairs = _pair_files()
    two = [s for s in sorted(pairs) if len(pairs[s]) >= 2]
    return two[:n_bench]


def _cache_path(cache, s):
    return os.path.join(cache, f"subj{s:02d}.npz")


def mode_cache(args):
    os.makedirs(args.cache, exist_ok=True)
    bench = benchmark_subjects(args.n_bench)
    json.dump({"benchmark_subjects": bench, "n_bench": args.n_bench},
              open(os.path.join(args.cache, "benchmark.json"), "w"), indent=2)
    for s in bench:
        cp = _cache_path(args.cache, s)
        if os.path.exists(cp):
            continue
        ep = load_subjects([s])
        np.savez(cp, X=ep.X, y=ep.y, subject=ep.subject, night=ep.night)
        print(f"[cache] subj {s}: X{ep.X.shape} stages={np.bincount(ep.y, minlength=5).tolist()}", flush=True)
    print(f"[cache] {len(bench)} benchmark subjects -> {args.cache}", flush=True)


def _load_cached(cache, subs):
    Xs, ys, ss, ns = [], [], [], []
    for s in subs:
        d = np.load(_cache_path(cache, s))
        Xs.append(d["X"]); ys.append(d["y"]); ss.append(d["subject"]); ns.append(d["night"])
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss), np.concatenate(ns))


def get_sleep_source(bundle_root, tag, cfg, code_sig, data_fn):
    sig = hashlib.sha256(f"{tag}|{code_sig}|ep{cfg.train.epochs}|t{cfg.encoder.n_times}".encode()).hexdigest()[:16]
    os.makedirs(bundle_root, exist_ok=True)
    pt = os.path.join(bundle_root, f"{sig}.pt"); js = os.path.join(bundle_root, f"{sig}.json")
    npz = os.path.join(bundle_root, f"{sig}.moments.npz")
    pi_unif = UNI.copy()
    if os.path.exists(pt) and os.path.exists(js) and os.path.exists(npz):
        model = H2Model(cfg, pi_unif).to(cfg.train.device)
        model.load_state_dict(torch.load(pt, map_location=cfg.train.device)); model.eval()
        m = np.load(npz)
        return model, (torch.tensor(m["mu"]), torch.tensor(m["sd"])), m["R_src"], pi_unif, m["rho_S"]
    X, y, subject = data_fn()
    pi_star = reference_prior(y, NC, "uniform")
    dag, dom = _domains(subject)
    model, *_ = train_h2(X, y, dom, dag, cfg, align_factor="site")
    torch.save(model.state_dict(), pt)
    Us = _embed(model, X, cfg.train.device)
    mu, sd = reference_weighted_source_moments(Us, y, pi_star)
    R_src = reference_cov(X); rho_S = np.bincount(y, minlength=NC) / len(y)
    np.savez(npz, mu=np.asarray(mu), sd=np.asarray(sd), R_src=R_src, rho_S=rho_S)
    json.dump(dict(tag=tag, code_sig=code_sig, sig=sig, n_train=int(len(y))), open(js, "w"), indent=2)
    return model, (mu, sd), R_src, pi_star, rho_S


def _js(p, q):
    p = np.asarray(p, float) + 1e-9; q = np.asarray(q, float) + 1e-9
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    def kl(a, b): return float((a * np.log(a / b)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mode_loso(args, code_sig, commit):
    cache = args.cache
    bench = json.load(open(os.path.join(cache, "benchmark.json")))["benchmark_subjects"]
    targets = bench
    if args.targets:
        a, b = (int(x) for x in args.targets.split("-")); targets = bench[a:b + 1]
    cfg = sleep_cfg(args.epochs, args.device)
    for tgt in targets:
        others = [s for s in bench if s != tgt]
        model, pooled_ref, R_src, pi_star, rho_S = get_sleep_source(
            args.bundle_root, f"W2:sleep:loso{tgt}:nb{len(bench)}", cfg, code_sig,
            lambda o=others: _load_cached(cache, o))
        tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, NC, args.device)
        Xt, yt, st, nt = _load_cached(cache, [tgt])
        a_m = nt == 1; e_m = nt == 2
        if a_m.sum() < cfg.tta.min_target or e_m.sum() < 8:
            print(f"[W2] target {tgt}: insufficient nights -> skip", flush=True); continue
        Xa, ya = Xt[a_m], yt[a_m]; Xe, ye = Xt[e_m], yt[e_m]
        rho_T = np.bincount(ya, minlength=NC) / len(ya)            # post-hoc only (JS mechanism)
        js_TS = _js(rho_T, rho_S)
        Ua = _embed(model, Xa, args.device); Ue = _embed(model, Xe, args.device)
        V = B1A_VARIANTS_BY_NAME
        preds, fits = {}, {}
        preds["identity"] = _predict_generative(model, Ue, UNI).argmax(1); fits["identity"] = None
        fp = tta.fit_variant(Ua, V["pooled_empirical_diag"], pooled_ref=pooled_ref, tta_seed=1)
        preds["always_pooled"] = _predict_transform(model, Ue, fp.transform, UNI).argmax(1); fits["always_pooled"] = fp
        fc = tta.fit_variant(Ua, V["gen_oneshot_diag"], tta_seed=1)
        preds["always_canonical_CC"] = _predict_transform(model, Ue, fc.transform, UNI).argmax(1); fits["always_canonical_CC"] = fc
        fj = tta.fit_variant(Ua, V["joint_iterative_diag"], tta_seed=1)
        pij = np.asarray(fj.pi_T.cpu().numpy() if torch.is_tensor(fj.pi_T) else fj.pi_T)
        preds["current_joint"] = _predict_transform(model, Ue, fj.transform, pij).argmax(1); fits["current_joint"] = fj
        M = ea_transform(R_src, reference_cov(Xa))
        preds["euclidean_alignment"] = _predict_generative(model, _embed(model, apply_ea(Xe, M), args.device), UNI).argmax(1); fits["euclidean_alignment"] = None
        Ts = spdim_fit(model.head.density, Ua, UNI, args.device)
        preds["spdim"] = _predict_transform(model, Ue, Ts, UNI).argmax(1); fits["spdim"] = Ts
        preds["metadata_only"] = preds["always_canonical_CC"]; fits["metadata_only"] = fc   # DIAG x DIFFERENT -> CC
        b_id = float(balanced_accuracy_score(ye, preds["identity"])); f_id = float(f1_score(ye, preds["identity"], average="macro"))
        for method in W2_METHODS:
            pred = preds[method]; fit = fits[method]
            bacc = float(balanced_accuracy_score(ye, pred)); mf1 = float(f1_score(ye, pred, average="macro"))
            tnorm = float("nan")
            if fit is not None:
                A = fit.transform.matrix() if hasattr(fit, "transform") else fit.matrix()
                tnorm = float(((A - torch.eye(A.shape[0], device=A.device)) ** 2).sum().sqrt().cpu())
            occ = (np.bincount(pred, minlength=NC) / len(pred)).tolist()
            append_row(args.out, dict(panel="W2", commit=commit, code_sig=code_sig, dataset="Sleep-EDF-cassette",
                                      target_subject=int(tgt), method=method, bacc=bacc, macro_f1=mf1,
                                      bacc_identity=b_id, delta=bacc - b_id, delta_f1=mf1 - f_id,
                                      harm=bool(bacc - b_id < -1e-9), n_adapt=int(len(ya)), n_eval=int(len(ye)),
                                      transform_norm=tnorm, pred_occupancy=occ, rho_T=rho_T.tolist(),
                                      rho_S=rho_S.tolist(), js_target_source=float(js_TS)))
        print(f"[W2] target {tgt} done: id bAcc={b_id:.3f} JS(T,S)={js_TS:.3f} "
              f"pooled Δ={float(balanced_accuracy_score(ye,preds['always_pooled']))-b_id:+.3f} "
              f"CC Δ={float(balanced_accuracy_score(ye,preds['always_canonical_CC']))-b_id:+.3f}", flush=True)
    if os.path.exists(args.out):
        print(f"[W2] -> {args.out} sha={sha256_file(args.out)[:12]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cache", "loso"])
    ap.add_argument("--n-bench", type=int, default=20)
    ap.add_argument("--targets", default="")
    ap.add_argument("--cache", default="results/h2cmi/sleep_cache")
    ap.add_argument("--bundle-root", default="results/h2cmi/w2_bundles")
    ap.add_argument("--out", default="results/h2cmi/w2_sleep.jsonl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.mode == "cache":
        mode_cache(args); return
    out_dir = os.path.dirname(args.out) or "."
    commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=[out_dir, args.bundle_root, args.cache])
    code_sig = source_code_signature()
    if os.path.exists(args.out):
        os.remove(args.out)
    mode_loso(args, code_sig, commit)


if __name__ == "__main__":
    main()
