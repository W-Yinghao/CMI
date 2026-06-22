"""V2 real-EEG external validation runner (review V2_FROZEN): A out-of-support abstention audit +
B supported-regime utility. Binary left/right; offline datalake; target labels EVAL-ONLY.

Modes:
  grid      compute + freeze the dense common-channel grid (BNCI2014_001/Cho2017/Lee2019_MI) -> manifest.
  source-A  train + cache ONE source model per A dataset (all its subjects, on the frozen grid).
  A         eval the cross-dataset pairs (load cached source models). metadata -> UNSUPPORTED -> identity.
  B         per-subject cross-session: train source on the subject's earlier session, eval the later.
  A-severe  descriptive BNCI2014_001-LR -> BNCI2014_004 on C3/Cz/C4 (NOT in any verdict).

Every method fits its operator on the UNLABELED adapt split and applies it to the held-out eval split
(mutually exclusive, contiguous). Methods: identity, always_pooled, always_canonical_CC, metadata_only,
euclidean_alignment, current_joint(diagnostic). Split manifest is written BEFORE eval.

  python -m h2cmi.run_v2 --mode grid --out results/h2cmi/v2_grid.json
  python -m h2cmi.run_v2 --mode source-A --source Cho2017 --grid results/h2cmi/v2_grid.json \
      --bundle-root results/h2cmi/v2_bundles
  python -m h2cmi.run_v2 --mode A --pairs Cho2017>Lee2019_MI --grid ... --bundle-root ... --out ...
  python -m h2cmi.run_v2 --mode B --pairs BNCI2014_001:0>1 --bundle-root ... --out ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from h2cmi.config import core_config, H2Config
from h2cmi.domains import DomainDAG, DomainFactor, DomainLabels
from h2cmi.train.trainer import train_h2, reference_prior, H2Model
from h2cmi.eval.harness import _embed, _predict_generative, _predict_transform
from h2cmi.tta.class_conditional import (ClassConditionalTTA, B1A_VARIANTS_BY_NAME,
                                         reference_weighted_source_moments)
from h2cmi.eval.ea import reference_cov, ea_transform, apply_ea
from h2cmi.data.real_eeg import load_dataset, contiguous_split, common_channel_grid, N_TIMES, FS
from h2cmi.data.real_metadata import MOABB_CLASS, v2_operator, ACQ
from h2cmi.data.metadata_substrate import CANONICAL_CC
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, sha256_file

A_DATASETS = ["BNCI2014_001", "Cho2017", "Lee2019_MI"]
PAIRS_A = [(s, t) for s in A_DATASETS for t in A_DATASETS if s != t]
PAIRS_B = [("BNCI2014_001", 0, 1), ("Lee2019_MI", 0, 1),
           ("BNCI2014_004", 0, 1), ("BNCI2014_004", 2, 3), ("BNCI2014_004", 2, 4)]
METHODS = ["identity", "always_pooled", "always_canonical_CC", "metadata_only",
           "euclidean_alignment", "current_joint"]
UNI = np.full(2, 0.5)


def build_cfg(n_chans, epochs, device, seed=0):
    cfg = core_config(H2Config(n_classes=2))
    cfg.encoder.n_chans = int(n_chans); cfg.encoder.n_times = N_TIMES; cfg.encoder.fs = FS
    cfg.train.epochs = int(epochs); cfg.train.device = device; cfg.train.seed = seed
    cfg.cmi.enabled = False
    return cfg


def _site_domains(subject):
    subs = np.unique(subject); smap = {int(s): i for i, s in enumerate(subs)}
    site = np.array([smap[int(s)] for s in subject], np.int64)
    dag = DomainDAG([DomainFactor("site", max(1, len(subs)), (), "invariant", 0.02)])
    return dag, DomainLabels(dag, site.reshape(-1, 1))


def _data_hash(X, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _source_sig(tag, code_sig, cfg):
    s = f"{tag}|{code_sig}|ep{cfg.train.epochs}|ch{cfg.encoder.n_chans}|t{cfg.encoder.n_times}|sd{cfg.train.seed}"
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def get_source(bundle_root, tag, cfg, code_sig, data_fn):
    """Frozen source model + cached pooled latent moments (mu,sd) + raw reference cov R_src.
    data_fn() -> (X, y, subject) is called ONLY on cache miss (so A eval-shards never reload the
    huge source datasets once the source-A job has populated the cache)."""
    sig = _source_sig(tag, code_sig, cfg)
    os.makedirs(bundle_root, exist_ok=True)
    pt = os.path.join(bundle_root, f"{sig}.pt"); js = os.path.join(bundle_root, f"{sig}.json")
    npz = os.path.join(bundle_root, f"{sig}.moments.npz")
    pi_unif = np.full(2, 0.5)
    if os.path.exists(pt) and os.path.exists(js) and os.path.exists(npz):
        meta = json.load(open(js))
        if meta.get("code_sig") != code_sig:
            raise RuntimeError(f"{js}: code_sig mismatch")
        model = H2Model(cfg, pi_unif).to(cfg.train.device)
        model.load_state_dict(torch.load(pt, map_location=cfg.train.device)); model.eval()
        m = np.load(npz)
        pooled_ref = (torch.tensor(m["mu"], dtype=torch.float32), torch.tensor(m["sd"], dtype=torch.float32))
        return model, pooled_ref, m["R_src"], pi_unif
    X, y, subject = data_fn()
    dhash = _data_hash(X, y)
    pi_star = reference_prior(y, 2, "uniform")
    dag, dom = _site_domains(subject)
    model, *_ = train_h2(X, y, dom, dag, cfg, align_factor="site")
    torch.save(model.state_dict(), pt)
    Us = _embed(model, X, cfg.train.device)
    mu, sd = reference_weighted_source_moments(Us, y, pi_star)
    R_src = reference_cov(X)
    np.savez(npz, mu=np.asarray(mu), sd=np.asarray(sd), R_src=R_src)
    json.dump(dict(tag=tag, code_sig=code_sig, data_hash=dhash, sig=sig,
                   n_train=int(len(y)), epochs=cfg.train.epochs, n_chans=cfg.encoder.n_chans),
              open(js, "w"), indent=2)
    return model, (mu, sd), R_src, pi_star


@torch.no_grad()
def _bacc(model, U, y, prior):
    return float(balanced_accuracy_score(y, _predict_generative(model, U, prior).argmax(1)))


def eval_unit(model, tta, pooled_ref, R_src, Xa, Xe, ye, device, meta_op):
    """Return dict method -> (bacc, pred_argmax) for one target unit (adapt Xa / eval Xe)."""
    Ua = _embed(model, Xa, device); Ue = _embed(model, Xe, device)
    V = B1A_VARIANTS_BY_NAME
    out = {}
    p_id = _predict_generative(model, Ue, UNI)
    out["identity"] = (float(balanced_accuracy_score(ye, p_id.argmax(1))), p_id.argmax(1))
    # pooled (classless empirical diag)
    fp = tta.fit_variant(Ua, V["pooled_empirical_diag"], pooled_ref=pooled_ref, tta_seed=1)
    pp = _predict_transform(model, Ue, fp.transform, UNI)
    out["always_pooled"] = (float(balanced_accuracy_score(ye, pp.argmax(1))), pp.argmax(1), fp)
    # canonical class-conditional (gen_oneshot, prior fixed)
    fc = tta.fit_variant(Ua, V["gen_oneshot_diag"], tta_seed=1)
    pc = _predict_transform(model, Ue, fc.transform, UNI)
    out["always_canonical_CC"] = (float(balanced_accuracy_score(ye, pc.argmax(1))), pc.argmax(1), fc)
    # Euclidean Alignment (raw-space, frozen transform applied to eval only)
    M = ea_transform(R_src, reference_cov(Xa))
    Ue_ea = _embed(model, apply_ea(Xe, M), device)
    pe = _predict_generative(model, Ue_ea, UNI)
    out["euclidean_alignment"] = (float(balanced_accuracy_score(ye, pe.argmax(1))), pe.argmax(1))
    # current_joint (diagnostic; transform + prior M-step)
    fj = tta.fit_variant(Ua, V["joint_iterative_diag"], tta_seed=1)
    pj = _predict_transform(model, Ue, fj.transform,
                            torch.as_tensor(fj.pi_T, dtype=torch.float32).cpu().numpy()
                            if not isinstance(fj.pi_T, np.ndarray) else fj.pi_T)
    out["current_joint"] = (float(balanced_accuracy_score(ye, pj.argmax(1))), pj.argmax(1), fj)
    # metadata_only -> frozen rule operator
    if meta_op == "identity":
        out["metadata_only"] = (out["identity"][0], out["identity"][1])
    elif meta_op == "pooled_empirical_diag":
        out["metadata_only"] = (out["always_pooled"][0], out["always_pooled"][1])
    else:
        out["metadata_only"] = (out["always_canonical_CC"][0], out["always_canonical_CC"][1])
    return out


def _emit(out_path, base, results, meta_op, geom, prev, n_a, n_e, split_hash):
    b_id = results["identity"][0]
    id_pred = results["identity"][1]
    meta_pred = results["metadata_only"][1]
    for method in METHODS:
        r = results[method]
        bacc, pred = r[0], r[1]
        tnorm = float("nan")
        if len(r) > 2:                                  # has a VariantFit
            A = r[2].transform.matrix()
            tnorm = float(((A - torch.eye(A.shape[0], device=A.device)) ** 2).sum().sqrt().cpu())
        row = dict(base, method=method, bacc=bacc, bacc_identity=b_id, delta=bacc - b_id,
                   harm=bool(bacc - b_id < -1e-9), metadata_action=meta_op,
                   geometry=geom, prevalence=prev, n_adapt=int(n_a), n_eval=int(n_e),
                   transform_norm=tnorm, split_hash=split_hash,
                   pred_equiv_identity=bool(np.array_equal(pred, id_pred)) if method == "metadata_only" else None)
        append_row(out_path, row)


def _split_hash(idx_a, idx_e):
    h = hashlib.sha256(); h.update(np.asarray(idx_a).tobytes()); h.update(np.asarray(idx_e).tobytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- modes
def mode_grid(args):
    names = {}
    for ds in A_DATASETS:
        ep = load_dataset(ds, [MOABB_CLASS(ds)().subject_list[0]])
        names[ds] = ep.channels
        print(f"  {ds}: {len(ep.channels)} chans", flush=True)
    keys, h = common_channel_grid(names)
    rec = dict(marker="V2_COMMON_GRID", datasets=A_DATASETS, ordered_keys=keys, n=len(keys),
               sha256=h, per_dataset_nch={k: len(v) for k, v in names.items()})
    json.dump(rec, open(args.out, "w"), indent=2)
    print(f"V2 common grid: {len(keys)} channels sha={h[:12]} -> {args.out}\n  {keys}", flush=True)


def _load_A_source(src, grid):
    eps = load_dataset(src, MOABB_CLASS(src)().subject_list, channels=grid)
    return eps.X, eps.y, eps.subject


def mode_source_A(args, code_sig):
    grid = json.load(open(args.grid))["ordered_keys"]
    cfg = build_cfg(len(grid), args.epochs, args.device, seed=0)
    print(f"[source-A] {args.source} on {len(grid)}-ch grid", flush=True)
    get_source(args.bundle_root, f"A:{args.source}:grid", cfg, code_sig,
               lambda: _load_A_source(args.source, grid))
    print(f"[source-A] {args.source} cached", flush=True)


def mode_A(args, code_sig, commit):
    grid = json.load(open(args.grid))["ordered_keys"]
    pairs = [tuple(p.split(">")) for p in args.pairs.split(",")] if args.pairs else PAIRS_A
    cfg = build_cfg(len(grid), args.epochs, args.device, seed=0)
    src_cache = {}
    for src, tgt in pairs:
        if src not in src_cache:
            src_cache[src] = get_source(args.bundle_root, f"A:{src}:grid", cfg, code_sig,
                                        lambda s=src: _load_A_source(s, grid))
        model, pooled_ref, R_src, pi_star = src_cache[src]
        tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, 2, args.device)
        ept = load_dataset(tgt, MOABB_CLASS(tgt)().subject_list, channels=grid)
        meta_op, d = v2_operator(src, 0, tgt, 0)
        for subj in np.unique(ept.subject):
            sess0 = ept.session[ept.subject == subj].min()
            a, e = contiguous_split(ept, subj, sess0)
            if len(a) < cfg.tta.min_target or len(e) < 4:
                continue
            sh = _split_hash(a, e)
            res = eval_unit(model, tta, pooled_ref, R_src, ept.X[a], ept.X[e], ept.y[e], args.device, meta_op)
            base = dict(mode="A", commit=commit, code_sig=code_sig, pair=f"{src}>{tgt}",
                        source=src, target=tgt, subject=int(subj), session=int(sess0))
            _emit(args.out, base, res, meta_op, d.geometry_compatibility, d.prevalence_risk, len(a), len(e), sh)
        print(f"[A] {src}>{tgt} ({meta_op}) done", flush=True)


def mode_B(args, code_sig, commit):
    pairs = []
    for p in (args.pairs.split(",") if args.pairs else []):
        ds, ss = p.split(":"); a, b = ss.split(">"); pairs.append((ds, int(a), int(b)))
    pairs = pairs or PAIRS_B
    for ds, s_src, s_tgt in pairs:
        ep = load_dataset(ds, MOABB_CLASS(ds)().subject_list)
        cfg = build_cfg(ep.X.shape[1], args.epochs_b, args.device, seed=0)
        meta_op, d = v2_operator(ds, s_src, ds, s_tgt)
        for subj in np.unique(ep.subject):
            m_src = (ep.subject == subj) & (ep.session == s_src)
            if m_src.sum() < cfg.tta.min_target * 2:
                continue
            model, pooled_ref, R_src, pi_star = get_source(
                args.bundle_root, f"B:{ds}:s{subj}:sess{s_src}", cfg, code_sig,
                lambda ms=m_src: (ep.X[ms], ep.y[ms], ep.subject[ms]))
            tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, 2, args.device)
            a, e = contiguous_split(ep, subj, s_tgt)
            if len(a) < cfg.tta.min_target or len(e) < 4:
                continue
            sh = _split_hash(a, e)
            res = eval_unit(model, tta, pooled_ref, R_src, ep.X[a], ep.X[e], ep.y[e], args.device, meta_op)
            base = dict(mode="B", commit=commit, code_sig=code_sig, pair=f"{ds}:{s_src}>{s_tgt}",
                        source=ds, target=ds, subject=int(subj), session=int(s_tgt))
            _emit(args.out, base, res, meta_op, d.geometry_compatibility, d.prevalence_risk, len(a), len(e), sh)
        print(f"[B] {ds}:{s_src}>{s_tgt} ({meta_op}) done", flush=True)


def mode_A_severe(args, code_sig, commit):
    """Descriptive: BNCI2014_001-LR -> BNCI2014_004 on C3/Cz/C4. NOT in any verdict."""
    grid = ["C3", "CZ", "C4"]
    cfg = build_cfg(3, args.epochs, args.device, seed=0)
    model, pooled_ref, R_src, pi_star = get_source(
        args.bundle_root, "Asevere:BNCI2014_001:C3CzC4", cfg, code_sig,
        lambda: _load_A_source("BNCI2014_001", grid))
    tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, 2, args.device)
    ept = load_dataset("BNCI2014_004", MOABB_CLASS("BNCI2014_004")().subject_list, channels=grid)
    meta_op, d = v2_operator("BNCI2014_001", 0, "BNCI2014_004", 0)
    for subj in np.unique(ept.subject):
        sess0 = ept.session[ept.subject == subj].min()
        a, e = contiguous_split(ept, subj, sess0)
        if len(a) < cfg.tta.min_target or len(e) < 4:
            continue
        res = eval_unit(model, tta, pooled_ref, R_src, ept.X[a], ept.X[e], ept.y[e], args.device, meta_op)
        base = dict(mode="A_severe", commit=commit, code_sig=code_sig, pair="BNCI2014_001>BNCI2014_004@C3CzC4",
                    source="BNCI2014_001", target="BNCI2014_004", subject=int(subj), session=int(sess0))
        _emit(args.out, base, res, meta_op, d.geometry_compatibility, d.prevalence_risk, len(a), len(e), _split_hash(a, e))
    print("[A-severe] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["grid", "source-A", "A", "B", "A-severe"])
    ap.add_argument("--source", default="")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--grid", default="results/h2cmi/v2_grid.json")
    ap.add_argument("--bundle-root", default="results/h2cmi/v2_bundles")
    ap.add_argument("--out", default="results/h2cmi/v2.jsonl")
    ap.add_argument("--epochs", type=int, default=40)       # A source (many subjects)
    ap.add_argument("--epochs-b", type=int, default=80)     # B source (single subject/session)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.mode == "grid":
        mode_grid(args); return
    out_dir = os.path.dirname(args.out) or "."
    commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=[out_dir, args.bundle_root, args.grid])
    code_sig = source_code_signature()
    if args.mode != "source-A" and os.path.exists(args.out):
        os.remove(args.out)
    if args.mode == "source-A":
        mode_source_A(args, code_sig)
    elif args.mode == "A":
        mode_A(args, code_sig, commit)
    elif args.mode == "B":
        mode_B(args, code_sig, commit)
    elif args.mode == "A-severe":
        mode_A_severe(args, code_sig, commit)
    if args.mode in ("A", "B", "A-severe") and os.path.exists(args.out):
        print(f"[{args.mode}] rows -> {args.out} sha={sha256_file(args.out)[:12]}", flush=True)


if __name__ == "__main__":
    main()
