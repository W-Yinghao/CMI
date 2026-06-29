#!/usr/bin/env python
"""CIGL Phase 2-real — exploratory source-only leakage probe for GraphCMINet-ERM.

Still Gate 2, still DIAGNOSTIC. Trains a GraphCMINet with PLAIN ERM (no graphcmi regularization,
lambda_g=lambda_node=lambda_edge=0) on SOURCE EEG only, freezes it, and audits whether the learned
graph objects carry label-conditional source-domain (subject) leakage:

    graph: I(Z_g;D|Y)    node: (1/C) Σ_v I(Z_v;D|Y)    edge: I(A(X);D|Y)

The held-out target subject is NEVER used (training or audit). The domain probe uses a support-aware
(Y,D) trial-level split so each source subject appears in both probe-train and probe-val; the
permutation null permutes D within-label over the probe-train split only and retrains the probe.
Across seeds we also measure node/edge map stability vs a random-map null.

This is EXPLORATORY evidence, never a benchmark number. Real-EEG evidence needs n_perm>=50 and several
seeds; the synthetic dry-run is the binding engineering check.

    # always works (no data, no training):
    python scripts/run_cigl_phase2_real_probe.py --dry_run_synthetic --device cpu --seeds 0 1 2 --n_perm 10
    # real (needs the offline MOABB datalake cache; heavy on CPU -> prefer GPU/sbatch):
    python scripts/run_cigl_phase2_real_probe.py --dataset BNCI2014_001 --device cpu --seeds 0 1 2 --n_perm 50 --max_folds 1

See docs/CIGL_10_PHASE2_REAL_PROBES.md.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cmi.eval.graph_leakage import audit_graph_objects                  # noqa: E402
from cmi.eval.probe_splits import stratified_trial_split_by_y_d          # noqa: E402
from cmi.eval.graph_map_stability import (                               # noqa: E402
    flatten_node_map, flatten_edge_map, spearman_or_pearson_stability, random_map_stability_null)

OUT_DIR = REPO / "results" / "cigl" / "phase2_real"


# ----------------------------------------------------------------------------- provenance helpers
def _git_commit_hash():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _remap_contiguous(d):
    uniq = {v: i for i, v in enumerate(sorted(np.unique(d)))}
    return np.array([uniq[v] for v in d], dtype=np.int64), len(uniq)


@torch.no_grad()
def _extract_graph_features(net, X, device, bs=256):
    """Frozen GraphCMINet -> (graph_z[N,Dg], node_z[N,C,Dn], edge_logits[N,C,C])."""
    net.eval()
    gz, nz, el = [], [], []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)   # robust to float64 inputs
        _, g, n, e = net.forward_graph(xb)
        gz.append(g.cpu().numpy()); nz.append(n.cpu().numpy()); el.append(e.cpu().numpy())
    return np.concatenate(gz), np.concatenate(nz), np.concatenate(el)


# ----------------------------------------------------------------------------- feature sources
def _synthetic_features(seed, n=240, C=8, Dg=6, Dn=4, n_cls=2, n_dom=4, strength=2.5):
    """Dry-run DGP: leakage injected into graph_z, a FIXED subset of channels, and a FIXED edge, so
    the maps SHOULD be seed-stable; per-seed noise differs. Y ⟂ D (conditional leakage)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_cls, n).astype("int64")
    d = rng.integers(0, n_dom, n).astype("int64")
    # guarantee enough per-(Y,D)-cell support for the stratified split (fill cells round-robin)
    fill = np.arange(n) % (n_cls * n_dom)
    y = (fill % n_cls).astype("int64")
    d = (fill // n_cls).astype("int64")
    graph_z = rng.standard_normal((n, Dg)).astype("float32"); graph_z[:, 0] += strength * d
    node_z = rng.standard_normal((n, C, Dn)).astype("float32")
    leak_chans = [1, 2, 3]                                   # FIXED across seeds
    for c in leak_chans:
        node_z[:, c, 0] += strength * d
    base = rng.standard_normal((n, C, C)).astype("float32")
    edge = 0.5 * (base + base.transpose(0, 2, 1))
    edge[:, 0, 4] += strength * d; edge[:, 4, 0] = edge[:, 0, 4]   # FIXED leaking edge (0,4)
    for k in range(C):
        edge[:, k, k] = 0.0
    return graph_z, node_z, edge, y, d, n_cls, n_dom, dict(leak_channels=leak_chans, leak_edge=[0, 4])


def _train_and_extract_real(data, fold_idx, seed, args, device):
    """Train GraphCMINet-ERM on SOURCE (all-but-target) enc-split; extract frozen features on a
    held-out SOURCE probe pool. Target subject is never touched. Returns features + (y,d) for the pool."""
    from cmi.models.gnn import GraphCMINet
    from cmi.train.trainer import train_model
    X, y, meta, classes, dom_all, splits = data
    tgt, tr_mask, te_mask = splits[fold_idx]
    n_cls = len(classes)
    Xs, ys = X[tr_mask], y[tr_mask]                          # SOURCE only (target excluded entirely)
    ds, n_dom = _remap_contiguous(dom_all[tr_mask])
    # split SOURCE into encoder-train and a held-out probe pool (support-aware so encoder sees all subjects)
    enc_idx, pool_idx, enc_split_diag = stratified_trial_split_by_y_d(
        ys, ds, train_frac=args.enc_train_frac, seed=seed, min_per_cell=args.min_per_cell)
    net = GraphCMINet(X.shape[1], X.shape[2], n_cls).to(device)
    net, _post, _diag = train_model(net, Xs[enc_idx], ys[enc_idx], ds[enc_idx], n_cls,
                                    method="erm", epochs=args.epochs, bs=args.bs,
                                    warmup=max(1, args.epochs // 5), device=device, seed=seed)
    gz, nz, el = _extract_graph_features(net, Xs[pool_idx], device)
    info = dict(heldout_subject=str(tgt), n_source_subjects=int(n_dom),
                source_subjects=sorted(set(meta["subject"][tr_mask].astype(str))),
                n_enc_train=int(len(enc_idx)), n_probe_pool=int(len(pool_idx)),
                enc_split_diagnostics=enc_split_diag, epochs=int(args.epochs))
    return gz, nz, el, ys[pool_idx], ds[pool_idx], n_cls, n_dom, info


def _load_real(args):
    """Load a MOABB dataset offline (datalake cache via cmi.paths). Fails clearly if unavailable."""
    try:
        from cmi.data import moabb_data
    except Exception as e:
        raise SystemExit(f"[phase2-real] cannot import data loader ({type(e).__name__}: {e}). "
                         f"Need moabb/mne in this env, or use --dry_run_synthetic.")
    try:
        X, y, meta, classes = moabb_data.load(args.dataset, tmin=args.tmin, tmax=args.tmax,
                                              resample=args.resample)
    except Exception as e:
        raise SystemExit(f"[phase2-real] dataset '{args.dataset}' could not be loaded offline "
                         f"({type(e).__name__}: {e}). Ensure the MOABB datalake cache is present "
                         f"(see cmi/paths.py); this script will NOT download. Use --dry_run_synthetic "
                         f"to validate the pipeline without data.")
    if args.max_subjects:
        keep = sorted(meta["subject"].unique())[:args.max_subjects]
        m = meta["subject"].isin(keep).to_numpy()
        X, y, meta = X[m], y[m], meta[m].reset_index(drop=True)
    dom_all, _ = moabb_data.domain_labels(meta, "subject")
    splits = list(moabb_data.loso_splits(meta))
    return X, y, meta, classes, dom_all, splits


# ----------------------------------------------------------------------------- audit + persistence
# ----------------------------------------------------------------------------- Gate-2 verdicts
def _positive_excess(block):
    """Directional signal only: observed leakage above the permutation-null MEAN. NOT significance."""
    return bool(block["kl_mean"] > block["permutation_mean"])


def _clears_null(block, alpha):
    """Gate-2 'clears null': positive excess AND the within-train permutation p-value <= alpha.
    A dry-run with tiny n_perm cannot satisfy this (min p = 1/(n_perm+1)) — that is intended."""
    return bool(block["kl_mean"] > block["permutation_mean"] and block["permutation_p"] <= alpha)


def _obj_summary(block, alpha):
    return dict(kl_mean=block["kl_mean"], permutation_mean=block["permutation_mean"],
                permutation_p=block["permutation_p"],
                positive_excess=_positive_excess(block),
                clears_null=_clears_null(block, alpha),
                gate_alpha=float(alpha))


def _edge_top_k(edge_map, k=8):
    m = np.asarray(edge_map); C = m.shape[0]; iu = np.triu_indices(C, 1)
    vals = m[iu]
    order = np.argsort(vals)[::-1][:k]
    return [dict(i=int(iu[0][o]), j=int(iu[1][o]), cmi=float(vals[o])) for o in order]


def _audit_one(features, args, device, seed, commit, cfg_hash, fold_tag, extra_meta):
    gz, nz, el, y, d, n_cls, n_dom, src_info = features
    tr_idx, va_idx, split_diag = stratified_trial_split_by_y_d(
        y, d, train_frac=args.train_frac, seed=seed, min_per_cell=args.min_per_cell)
    audit = audit_graph_objects(gz, nz, el, y, d, n_cls, n_dom, n_perm=args.n_perm, seed=seed,
                                device=device, epochs=args.probe_epochs,
                                train_idx=tr_idx, val_idx=va_idx)
    node_map = np.asarray(audit["node"].pop("node_leakage_map"))
    edge_map = np.asarray(audit["edge"].pop("edge_leakage_map"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    node_path = OUT_DIR / f"{fold_tag}_seed{seed}_node_map.npy"
    edge_path = OUT_DIR / f"{fold_tag}_seed{seed}_edge_map.npy"
    np.save(node_path, node_map); np.save(edge_path, edge_map)
    audit["node"]["node_leakage_map"] = node_map.tolist()                      # compact inline (length C)
    audit["node"]["node_leakage_map_path"] = str(node_path.relative_to(REPO))
    audit["edge"]["edge_leakage_top_k"] = _edge_top_k(edge_map)                 # compact inline summary
    audit["edge"]["edge_leakage_map_path"] = str(edge_path.relative_to(REPO))
    rec = dict(
        meta=dict(exploratory=True, setting="strict_source_only_DG",
                  used_target_labels=False, used_target_covariates=False,
                  dataset=args.dataset if not args.dry_run_synthetic else "synthetic",
                  fold=fold_tag, seed=int(seed), n_perm=int(args.n_perm), epochs=int(args.epochs),
                  probe_epochs=int(args.probe_epochs), n_classes=int(n_cls), n_domains=int(n_dom),
                  commit_hash=commit, config_hash=cfg_hash, **extra_meta),
        source_info=src_info,
        probe_split_diagnostics=split_diag,
        graph=audit["graph"], node=audit["node"], edge=audit["edge"])
    out_path = OUT_DIR / f"{fold_tag}_seed{seed}.json"
    json.dump(rec, open(out_path, "w"), indent=2)
    return rec, node_map, edge_map, out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run_synthetic", action="store_true", help="run the full pipeline on synthetic features (no data/training)")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_perm", type=int, default=50, help="permutation-null repetitions (>=50 for real probing)")
    ap.add_argument("--max_folds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=80, help="GraphCMINet-ERM encoder training epochs (real mode)")
    ap.add_argument("--probe_epochs", type=int, default=100, help="conditional domain-probe fit epochs")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--train_frac", type=float, default=0.7, help="probe-train fraction of the probe pool")
    ap.add_argument("--enc_train_frac", type=float, default=0.7, help="encoder-train fraction of source")
    ap.add_argument("--min_per_cell", type=int, default=2)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--max_subjects", type=int, default=0)
    ap.add_argument("--map_stability_perms", type=int, default=200)
    ap.add_argument("--gate_alpha", type=float, default=0.05,
                    help="Gate-2 significance level: clears_null requires permutation_p <= gate_alpha")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[phase2-real] --device cuda requested but CUDA unavailable")
    if not args.dry_run_synthetic and args.n_perm < 50:
        print(f"[phase2-real] WARNING: n_perm={args.n_perm} < 50 for a REAL probe — exploratory only, "
              f"not statistical evidence.", flush=True)

    commit = _git_commit_hash()
    cfg_hash = _config_hash(vars(args))
    data = None if args.dry_run_synthetic else _load_real(args)
    folds = [("synthetic", None)] if args.dry_run_synthetic else \
        [(f"{args.dataset}_fold{f}", f) for f in range(min(args.max_folds, len(data[5])))]

    fold_summaries = []
    for fold_tag, fold_idx in folds:
        node_maps, edge_maps, seed_rows, fold_records = [], [], [], []
        for seed in args.seeds:
            if args.dry_run_synthetic:
                gz, nz, el, y, d, n_cls, n_dom, inj = _synthetic_features(seed)
                feats = (gz, nz, el, y, d, n_cls, n_dom, dict(injected=inj))
                extra = dict(injected=inj)
            else:
                feats = _train_and_extract_real(data, fold_idx, seed, args, args.device)
                extra = dict(heldout_subject=feats[7]["heldout_subject"])
            rec, node_map, edge_map, out_path = _audit_one(
                feats, args, args.device, seed, commit, cfg_hash, fold_tag, extra)
            fold_records.append(str(out_path.relative_to(REPO)))
            node_maps.append(flatten_node_map(node_map))
            edge_maps.append(flatten_edge_map(edge_map))
            g, nn_, e = rec["graph"], rec["node"], rec["edge"]
            seed_rows.append((seed, g, nn_, e))
            print(f"[{fold_tag} seed{seed}] "
                  f"graph kl={g['kl_mean']:.4f}(null {g['permutation_mean']:.4f},p={g['permutation_p']:.3f}) "
                  f"node kl={nn_['kl_mean']:.4f}(p={nn_['permutation_p']:.3f}) "
                  f"edge kl={e['kl_mean']:.4f}(p={e['permutation_p']:.3f})", flush=True)

        # seed-stability of the maps (needs >=2 seeds)
        stability = {}
        if len(args.seeds) >= 2:
            stability = dict(
                node=dict(stability=spearman_or_pearson_stability(node_maps),
                          null=random_map_stability_null(node_maps, n_perm=args.map_stability_perms)),
                edge=dict(stability=spearman_or_pearson_stability(edge_maps),
                          null=random_map_stability_null(edge_maps, n_perm=args.map_stability_perms)))
        fold_summary = dict(
            meta=dict(exploratory=True, setting="strict_source_only_DG",
                      dataset="synthetic" if args.dry_run_synthetic else args.dataset,
                      fold=fold_tag, seeds=list(args.seeds), n_perm=int(args.n_perm),
                      gate_alpha=float(args.gate_alpha),
                      commit_hash=commit, config_hash=cfg_hash,
                      used_target_labels=False, used_target_covariates=False),
            per_seed=[dict(seed=s,
                           graph=_obj_summary(g, args.gate_alpha),
                           node=_obj_summary(nn_, args.gate_alpha),
                           edge=_obj_summary(e, args.gate_alpha))
                      for (s, g, nn_, e) in seed_rows],
            map_stability=stability,
            record_files=fold_records)
        sum_path = OUT_DIR / f"{fold_tag}_summary.json"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        json.dump(fold_summary, open(sum_path, "w"), indent=2)
        fold_summaries.append(fold_summary)
        print(f"[{fold_tag}] wrote {sum_path}")
        if stability:
            print(f"[{fold_tag}] node-map stability mean_corr={stability['node']['stability']['mean_corr']:.3f} "
                  f"(above_random={stability['node']['null']['above_random']}); "
                  f"edge-map mean_corr={stability['edge']['stability']['mean_corr']:.3f} "
                  f"(above_random={stability['edge']['null']['above_random']})", flush=True)

    # ---- Gate-2 directional read (per fold): positive excess vs (the binding) p<=alpha clears-null ----
    print(f"\n=== Gate-2 directional read (exploratory; NOT a decision; gate_alpha={args.gate_alpha}) ===")
    for fs in fold_summaries:
        n = len(fs["per_seed"])
        excess = {obj: sum(r[obj]["positive_excess"] for r in fs["per_seed"]) for obj in ("graph", "node", "edge")}
        clears = {obj: sum(r[obj]["clears_null"] for r in fs["per_seed"]) for obj in ("graph", "node", "edge")}
        print(f"  {fs['meta']['fold']} (of {n} seeds): positive_excess={excess}  |  "
              f"clears_null(p<=alpha)={clears}")
    print("positive_excess is a DIRECTIONAL signal only; Gate-2 'clears null' requires permutation_p<=gate_alpha. "
          "The reviewer decides paths A/B/C/D per docs/CIGL_10 — this run only reports evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
