#!/usr/bin/env python
"""CIGL Phase 3A-S — known-good MI decoder sanity check (source-only, BNCI2014_001 fold-0).

Phase 3A-R found that GraphCMINet-ERM cannot learn BNCI2014_001 4-class MI (all repairs stay near
chance), so the baseline is the blocker. Phase 3A-S asks the prerequisite question:

  Under the SAME strict source-only fold-0 protocol, can known-good MI decoders (EEGNet, ShallowConvNet,
  DeepConvNet) reach a non-degenerate source baseline that GraphCMINet cannot?

This decides whether the problem is GraphCMINet-specific (a known-good decoder succeeds) or the
protocol/preprocessing/data itself (everything fails). NO CMI regularization here — ERM only. Strict
source-only: target subject excluded from training and source_probe; target labels are used ONLY for
after-the-fact target_eval; success is judged on source_probe ONLY. Non-graph CNNs emit NO graph/node/
edge leakage (only the GraphCMINet reference, which exposes forward_graph, gets a light leakage audit).

    python scripts/run_cigl_phase3a_backbone_sanity.py --dry_run_synthetic --device cpu --seeds 0 1 --epochs 3
    python scripts/run_cigl_phase3a_backbone_sanity.py --dataset BNCI2014_001 --device cuda --fold 0 --seeds 0 1 2 --epochs 80

See docs/CIGL_20_PHASE3A_S_BACKBONE_SANITY.md.
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

OUT_DIR = REPO / "results" / "cigl" / "phase3a_backbone_sanity"
PHASE = "Phase3A_S_backbone_sanity"
CANDIDATES = ["graphcmi_current_ref", "eegnet", "shallow_convnet", "deep_convnet", "dgcnn"]
GRAPH_BACKBONES = {"graphcmi_current_ref"}   # only these expose forward_graph -> get a leakage audit


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


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _per_seed_meta(dataset, fold_tag, commit, cfg_hash):
    return dict(exploratory=True, phase=PHASE, setting="strict_source_only_DG",
                used_target_labels_for_training=False, used_target_labels_for_selection=False,
                used_target_covariates=False, target_eval_is_evaluation_only=True,
                target_labels_used_for="evaluation_only metrics",
                dataset=dataset, fold=fold_tag, commit_hash=commit, config_hash=cfg_hash)


@torch.no_grad()
def _extract_graph_features(net, X, device, bs=256):
    net.eval(); gz, nz, el = [], [], []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
        _, g, n, e = net.forward_graph(xb)
        gz.append(g.cpu().numpy()); nz.append(n.cpu().numpy()); el.append(e.cpu().numpy())
    return np.concatenate(gz), np.concatenate(nz), np.concatenate(el)


def _train_eval(name, fold, seed, args, device):
    """Train candidate `name` with ERM on SOURCE enc-train; report train / source_probe / target_eval
    task metrics. For graph backbones only, add a light leakage audit on the source probe-pool. Target
    labels touch ONLY target_eval."""
    from cmi.models.sanity_backbones import build_sanity_backbone
    from cmi.train.trainer import train_model, predict
    from cmi.eval.metrics import classification_metrics
    X, y, dom_all, tr_mask, te_mask, n_cls, heldout = fold
    Xs, ys = X[tr_mask], y[tr_mask]
    ds, n_dom = _remap_contiguous(dom_all[tr_mask])
    enc_idx, pool_idx, _ = stratified_trial_split_by_y_d(ys, ds, train_frac=args.enc_train_frac,
                                                         seed=seed, min_per_cell=args.min_per_cell)
    torch.manual_seed(int(seed)); np.random.seed(int(seed))      # seed before construction (reproducible)
    net = build_sanity_backbone(name, X.shape[1], X.shape[2], n_cls).to(device)
    net, _post, _diag = train_model(net, Xs[enc_idx], ys[enc_idx], ds[enc_idx], n_cls, method="erm",
                                    epochs=args.epochs, bs=args.bs, warmup=max(1, args.epochs // 5),
                                    device=device, seed=seed)
    train_m = classification_metrics(predict(net, Xs[enc_idx], device), ys[enc_idx])
    src = classification_metrics(predict(net, Xs[pool_idx], device), ys[pool_idx])
    tgt = classification_metrics(predict(net, X[te_mask], device), y[te_mask])
    rec = dict(candidate=name, seed=int(seed), n_classes=int(n_cls), is_graph_backbone=name in GRAPH_BACKBONES,
               train=dict(balanced_acc=train_m["balanced_acc"], macro_f1=train_m["macro_f1"]),
               source_probe=dict(balanced_acc=src["balanced_acc"], macro_f1=src["macro_f1"]),
               target_eval=dict(balanced_acc=tgt["balanced_acc"], macro_f1=tgt["macro_f1"], evaluation_only=True),
               train_minus_source_gap=float(train_m["balanced_acc"] - src["balanced_acc"]),
               n_enc_train=int(len(enc_idx)), n_probe_pool=int(len(pool_idx)), heldout_subject=heldout)
    if name in GRAPH_BACKBONES and hasattr(net, "forward_graph"):    # light leakage audit (graph only)
        gz, nz, el = _extract_graph_features(net, Xs[pool_idx], device)
        yp, dp = ys[pool_idx], ds[pool_idx]
        tr_i, va_i, _ = stratified_trial_split_by_y_d(yp, dp, train_frac=args.train_frac, seed=seed,
                                                      min_per_cell=args.min_per_cell)
        au = audit_graph_objects(gz, nz, el, yp, dp, n_cls, n_dom, n_perm=args.leak_n_perm, seed=seed,
                                 device=device, epochs=args.probe_epochs, train_idx=tr_i, val_idx=va_i)
        rec["leakage"] = {o: {"kl_mean": au[o]["kl_mean"], "permutation_p": au[o]["permutation_p"]}
                          for o in ("graph", "node", "edge")}
    return rec


def _synthetic_fold(seed, n_per_subj=60, C=8, T=48, n_cls=3, n_subj=4):
    """Learnable 4-ish-class fold (strong per-class temporal pattern) + a subject-encoding channel subset
    (so a good decoder beats chance and the success path is exercised). One subject is held-out target."""
    rng = np.random.default_rng(seed)
    proto = 2.5 * rng.standard_normal((n_cls, C, T)).astype("float32")
    Xs, ys, ds = [], [], []
    for s in range(n_subj):
        for _ in range(n_per_subj):
            yy = rng.integers(0, n_cls)
            x = proto[yy] + 0.5 * rng.standard_normal((C, T)).astype("float32"); x[1:4] += 0.8 * s
            Xs.append(x); ys.append(yy); ds.append(s)
    X = np.stack(Xs).astype("float32"); y = np.array(ys, "int64"); d = np.array(ds, "int64")
    target = n_subj - 1
    return X, y, d, d != target, d == target, n_cls, str(target)


def _load_real_fold(args):
    from cmi.data import moabb_data
    try:
        X, y, meta, classes = moabb_data.load(args.dataset, tmin=args.tmin, tmax=args.tmax, resample=args.resample)
    except Exception as e:
        raise SystemExit(f"[phase3a-S] dataset '{args.dataset}' not loadable offline ({type(e).__name__}: {e}); "
                         f"ensure the MOABB datalake cache is present. Use --dry_run_synthetic to validate.")
    dom_all, _ = moabb_data.domain_labels(meta, "subject")
    tgt, tr_mask, te_mask = list(moabb_data.loso_splits(meta))[args.fold]
    return X, y, dom_all, tr_mask, te_mask, len(classes), str(tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run_synthetic", action="store_true")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--probe_epochs", type=int, default=100)
    ap.add_argument("--leak_n_perm", type=int, default=10, help="light leakage audit for the graph reference")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--enc_train_frac", type=float, default=0.7)
    ap.add_argument("--min_per_cell", type=int, default=2)
    ap.add_argument("--success_bacc_floor", type=float, default=0.45, help="source_probe bAcc to count as a credible baseline")
    ap.add_argument("--candidates", nargs="+", default=CANDIDATES)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[phase3a-S] --device cuda requested but CUDA unavailable")
    if args.device == "cpu":
        torch.set_num_threads(1)

    commit = _git_commit_hash(); cfg_hash = _config_hash(vars(args))
    fold = _synthetic_fold(seed=0) if args.dry_run_synthetic else _load_real_fold(args)
    dataset = "synthetic" if args.dry_run_synthetic else args.dataset
    fold_tag = f"{dataset}_fold{args.fold}"
    n_cls = fold[5]; chance = 1.0 / n_cls
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Phase 3A-S backbone sanity ({fold_tag}; chance={chance:.3f}, floor={args.success_bacc_floor}) ===")
    per_cand = {}
    for name in args.candidates:
        recs = []
        for seed in args.seeds:
            rec = _train_eval(name, fold, seed, args, args.device)
            rec["meta"] = _per_seed_meta(dataset, fold_tag, commit, cfg_hash)
            json.dump(rec, open(OUT_DIR / f"{fold_tag}_{name}_seed{seed}.json", "w"), indent=2)
            recs.append(rec)
        a = dict(candidate=name, is_graph_backbone=name in GRAPH_BACKBONES,
                 train_bacc=_mean([r["train"]["balanced_acc"] for r in recs]),
                 source_probe_bacc=_mean([r["source_probe"]["balanced_acc"] for r in recs]),
                 target_eval_bacc=_mean([r["target_eval"]["balanced_acc"] for r in recs]),
                 train_minus_source_gap=_mean([r["train_minus_source_gap"] for r in recs]),
                 target_eval_is_evaluation_only=True)
        if name in GRAPH_BACKBONES and "leakage" in recs[0]:
            a["leakage_kl"] = {o: _mean([r["leakage"][o]["kl_mean"] for r in recs]) for o in ("graph", "node", "edge")}
        per_cand[name] = a
        lk = (f" leak g/n/e={a['leakage_kl']['graph']:.2f}/{a['leakage_kl']['node']:.2f}/{a['leakage_kl']['edge']:.2f}"
              if "leakage_kl" in a else "")
        print(f"  {name:22s} train={a['train_bacc']:.3f} src={a['source_probe_bacc']:.3f} "
              f"tgt={a['target_eval_bacc']:.3f} gap={a['train_minus_source_gap']:+.3f}{lk}", flush=True)

    # SUCCESS decision is SOURCE-ONLY (target_eval never consulted)
    selected_successful_models = [c for c in args.candidates
                                  if (per_cand[c]["source_probe_bacc"] or 0.0) >= args.success_bacc_floor]
    known_good = [c for c in selected_successful_models if c != "graphcmi_current_ref"]
    graphcmi_succeeds = "graphcmi_current_ref" in selected_successful_models
    summary = dict(
        meta=dict(exploratory=True, phase=PHASE, setting="strict_source_only_DG", dataset=dataset, fold=fold_tag,
                  heldout_subject=fold[6], seeds=list(args.seeds), n_classes=int(n_cls), chance=float(chance),
                  success_bacc_floor=float(args.success_bacc_floor), epochs=int(args.epochs),
                  commit_hash=commit, config_hash=cfg_hash,
                  used_target_labels_for_training=False, used_target_labels_for_selection=False,
                  used_target_covariates=False, target_eval_is_evaluation_only=True),
        candidates=per_cand, selected_successful_models=selected_successful_models,
        known_good_decoders_succeed=bool(known_good), graphcmi_succeeds=graphcmi_succeeds,
        success_selection_uses_target_eval=False)
    json.dump(summary, open(OUT_DIR / f"{fold_tag}_backbone_sanity_summary.json", "w"), indent=2)
    print(f"\n[phase3a-S] wrote {OUT_DIR / f'{fold_tag}_backbone_sanity_summary.json'}")

    # exploratory decision read (reviewer decides)
    if known_good and not graphcmi_succeeds:
        verdict = "A: protocol USABLE, GraphCMINet is the bottleneck (redesign graph backbone)"
    elif not selected_successful_models:
        verdict = "B: NO decoder learns -> protocol/preprocessing/data diagnosis"
    elif graphcmi_succeeds:
        verdict = "graphcmi also succeeds -> revisit baseline/repair settings"
    else:
        verdict = "mixed -> reviewer inspects"
    print(f"\n=== Phase 3A-S read (exploratory; reviewer decides): {verdict} ===")
    print(f"  selected_successful_models (source-only, bAcc>={args.success_bacc_floor}): {selected_successful_models}")
    print("  target_eval is EVALUATION-ONLY (never used for success selection).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
