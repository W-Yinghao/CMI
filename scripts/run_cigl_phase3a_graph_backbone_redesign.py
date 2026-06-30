#!/usr/bin/env python
"""CIGL Phase 3A-G — task-capable graph backbone redesign (source-only, BNCI2014_001 fold-0, ERM only).

Phase 3A-S (Decision A) showed the strict source-only fold-0 is learnable by known-good MI decoders while
GraphCMINet stays ~0.33 -> GraphCMINet is the bottleneck. Phase 3A-G tests redesigned graph-COMPATIBLE
backbones (a known-good temporal stem feeding the graph/node/edge structure) that must learn the task
*through the graph path* (no CNN bypass). NO CMI regularization here — ERM only.

A candidate PASSES iff (source-only): source_probe bAcc >= floor on average AND in >=2/3 seeds, train
bAcc clearly above chance, forward_graph returns valid + non-degenerate graph_z/node_z, and a graph-usage
check shows logits depend on the graph readout (zeroing it collapses source_probe toward chance). Target
labels are evaluation-only; success/selection never use target_eval.

    python scripts/run_cigl_phase3a_graph_backbone_redesign.py --dry_run_synthetic --device cpu --seeds 0 1 --epochs 3 --leak_n_perm 5
    python scripts/run_cigl_phase3a_graph_backbone_redesign.py --dataset BNCI2014_001 --device cuda --fold 0 --seeds 0 1 2 --epochs 80 --leak_n_perm 10

See docs/CIGL_22_PHASE3A_G_GRAPH_BACKBONE_REDESIGN.md.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cmi.eval.graph_leakage import audit_graph_objects                       # noqa: E402
from cmi.eval.probe_splits import stratified_trial_split_by_y_d               # noqa: E402
from cmi.models.graph_task_backbones import build_graph_task_backbone, GRAPH_TASK_BACKBONES  # noqa: E402
# reuse the EXACT Phase 3A-S source-only fold construction + provenance helpers
from scripts.run_cigl_phase3a_backbone_sanity import (                        # noqa: E402
    _synthetic_fold, _load_real_fold, _remap_contiguous, _git_commit_hash, _config_hash, _mean)

OUT_DIR = REPO / "results" / "cigl" / "phase3a_graph_backbone_redesign"
PHASE = "Phase3A_G_graph_backbone_redesign"


@torch.no_grad()
def _forward_graph_feats(net, X, device, bs=256):
    """Run forward_graph over X; return (graph_z, node_z, edge_logits_or_None) as numpy."""
    net.eval()
    gz, nz, el = [], [], []
    edge_dynamic = net.meta.get("edge_logits_dynamic", False)
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
        _, g, n, e = net.forward_graph(xb)
        gz.append(g.cpu().numpy()); nz.append(n.cpu().numpy())
        if edge_dynamic and e is not None:
            el.append(e.cpu().numpy())
    edge = np.concatenate(el) if el else None
    return np.concatenate(gz), np.concatenate(nz), edge


@torch.no_grad()
def _ablation_bacc(net, X, y, mode, device, bs=256):
    from cmi.eval.metrics import classification_metrics
    net.eval()
    logits = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
        logits.append(net.ablate(xb, mode).cpu().numpy())
    return classification_metrics(np.concatenate(logits), y)["balanced_acc"]


def _train_eval(name, fold, seed, args, device):
    from cmi.train.trainer import train_model, predict
    from cmi.eval.metrics import classification_metrics
    X, y, dom_all, tr_mask, te_mask, n_cls, heldout = fold
    Xs, ys = X[tr_mask], y[tr_mask]
    ds, n_dom = _remap_contiguous(dom_all[tr_mask])
    enc_idx, pool_idx, _ = stratified_trial_split_by_y_d(ys, ds, train_frac=args.enc_train_frac,
                                                         seed=seed, min_per_cell=args.min_per_cell)
    torch.manual_seed(int(seed)); np.random.seed(int(seed))      # seed BEFORE construction (reproducible)
    net = build_graph_task_backbone(name, X.shape[1], X.shape[2], n_cls).to(device)
    net, _post, _diag = train_model(net, Xs[enc_idx], ys[enc_idx], ds[enc_idx], n_cls, method="erm",
                                    epochs=args.epochs, bs=args.bs, warmup=max(1, args.epochs // 5),
                                    device=device, seed=seed)
    train_m = classification_metrics(predict(net, Xs[enc_idx], device), ys[enc_idx])
    src = classification_metrics(predict(net, Xs[pool_idx], device), ys[pool_idx])
    tgt = classification_metrics(predict(net, X[te_mask], device), y[te_mask])

    # forward_graph validity + non-degeneracy on the source probe-pool
    gz, nz, el = _forward_graph_feats(net, Xs[pool_idx], device)
    fg = dict(graph_z_shape=list(gz.shape), node_z_shape=list(nz.shape),
              edge_logits=("none" if el is None else list(el.shape)),
              graph_z_finite=bool(np.isfinite(gz).all()), node_z_finite=bool(np.isfinite(nz).all()),
              graph_z_std=float(gz.std()), node_z_std=float(nz.std()))
    fg["valid"] = bool(fg["graph_z_finite"] and fg["node_z_finite"] and nz.shape[1] == X.shape[1])
    fg["nondegenerate"] = bool(fg["graph_z_std"] > args.nondegen_tol and fg["node_z_std"] > args.nondegen_tol)

    # graph-usage check on the SAME source probe-pool (source-only)
    yp = ys[pool_idx]
    gu = dict(source_probe_bacc=src["balanced_acc"],
              zero_graph_bacc=_ablation_bacc(net, Xs[pool_idx], yp, "zero_graph", device),
              permute_nodes_bacc=_ablation_bacc(net, Xs[pool_idx], yp, "permute_nodes", device))
    gu["zero_graph_drop"] = float(src["balanced_acc"] - gu["zero_graph_bacc"])
    gu["permute_nodes_drop"] = float(src["balanced_acc"] - gu["permute_nodes_bacc"])
    gu["graph_path_used"] = bool(gu["zero_graph_drop"] >= args.graph_usage_min_drop)

    rec = dict(candidate=name, seed=int(seed), n_classes=int(n_cls), meta_arch=net.meta,
               train=dict(balanced_acc=train_m["balanced_acc"], macro_f1=train_m["macro_f1"]),
               source_probe=dict(balanced_acc=src["balanced_acc"], macro_f1=src["macro_f1"]),
               target_eval=dict(balanced_acc=tgt["balanced_acc"], macro_f1=tgt["macro_f1"], evaluation_only=True),
               train_minus_source_gap=float(train_m["balanced_acc"] - src["balanced_acc"]),
               forward_graph=fg, graph_usage=gu, heldout_subject=heldout,
               n_enc_train=int(len(enc_idx)), n_probe_pool=int(len(pool_idx)))

    # OPTIONAL light leakage audit (magnitude sanity only) — dynamic-edge candidates only
    if net.meta.get("edge_logits_dynamic", False) and el is not None and args.leak_n_perm > 0:
        dp = ds[pool_idx]
        tr_i, va_i, _ = stratified_trial_split_by_y_d(yp, dp, train_frac=args.train_frac, seed=seed,
                                                      min_per_cell=args.min_per_cell)
        au = audit_graph_objects(gz, nz, el, yp, dp, n_cls, n_dom, n_perm=args.leak_n_perm, seed=seed,
                                 device=device, epochs=args.probe_epochs, train_idx=tr_i, val_idx=va_i)
        rec["leakage"] = {o: {"kl_mean": au[o]["kl_mean"], "permutation_p": au[o]["permutation_p"]}
                          for o in ("graph", "node", "edge")}
        rec["leakage_note"] = "light n_perm audit (magnitude sanity only; p NOT required <0.05)"
    else:
        rec["leakage"] = None
        rec["leakage_skipped_reason"] = ("static/shared adjacency: edge is not a per-sample object; "
                                         "graph/node leakage deferred to the repaired-backbone Gate-2 (n_perm=50)")
    return rec


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
    ap.add_argument("--leak_n_perm", type=int, default=10, help="light leakage audit (0 disables); dynamic-edge only")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--enc_train_frac", type=float, default=0.7)
    ap.add_argument("--min_per_cell", type=int, default=2)
    ap.add_argument("--success_bacc_floor", type=float, default=0.45)
    ap.add_argument("--min_seeds_pass", type=int, default=2)
    ap.add_argument("--train_above_chance_margin", type=float, default=0.10)
    ap.add_argument("--graph_usage_min_drop", type=float, default=0.10,
                    help="min source_probe drop when zeroing the graph readout (proves the graph path is used)")
    ap.add_argument("--nondegen_tol", type=float, default=1e-4)
    ap.add_argument("--candidates", nargs="+", default=GRAPH_TASK_BACKBONES)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[phase3a-G] --device cuda requested but CUDA unavailable")
    if args.device == "cpu":
        torch.set_num_threads(1)

    commit = _git_commit_hash(); cfg_hash = _config_hash(vars(args))
    fold = _synthetic_fold(seed=0) if args.dry_run_synthetic else _load_real_fold(args)
    dataset = "synthetic" if args.dry_run_synthetic else args.dataset
    fold_tag = f"{dataset}_fold{args.fold}"
    n_cls = fold[5]; chance = 1.0 / n_cls
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_flags = dict(exploratory=True, phase=PHASE, setting="strict_source_only_DG",
                      used_target_labels_for_training=False, used_target_labels_for_selection=False,
                      used_target_covariates=False, target_eval_is_evaluation_only=True,
                      graph_backbone_selection_uses_target_eval=False, cmi_regularization_used=False)

    print(f"\n=== Phase 3A-G graph backbone redesign ({fold_tag}; chance={chance:.3f}, floor={args.success_bacc_floor}) ===")
    per_cand = {}
    for name in args.candidates:
        recs = []
        for seed in args.seeds:
            rec = _train_eval(name, fold, seed, args, args.device)
            rec["meta"] = dict(meta_flags, dataset=dataset, fold=fold_tag, commit_hash=commit, config_hash=cfg_hash)
            json.dump(rec, open(OUT_DIR / f"{fold_tag}_{name}_seed{seed}.json", "w"), indent=2)
            recs.append(rec)
        src_seeds = [r["source_probe"]["balanced_acc"] for r in recs]
        n_pass = int(sum(s >= args.success_bacc_floor for s in src_seeds))
        a = dict(candidate=name, meta_arch=recs[0]["meta_arch"],
                 train_bacc=_mean([r["train"]["balanced_acc"] for r in recs]),
                 source_probe_bacc=_mean(src_seeds), source_probe_per_seed=[round(s, 3) for s in src_seeds],
                 target_eval_bacc=_mean([r["target_eval"]["balanced_acc"] for r in recs]),
                 train_minus_source_gap=_mean([r["train_minus_source_gap"] for r in recs]),
                 n_seeds_source_pass=n_pass,
                 forward_graph_valid=all(r["forward_graph"]["valid"] for r in recs),
                 forward_graph_nondegenerate=all(r["forward_graph"]["nondegenerate"] for r in recs),
                 zero_graph_bacc=_mean([r["graph_usage"]["zero_graph_bacc"] for r in recs]),
                 zero_graph_drop=_mean([r["graph_usage"]["zero_graph_drop"] for r in recs]),
                 permute_nodes_drop=_mean([r["graph_usage"]["permute_nodes_drop"] for r in recs]),
                 graph_path_used=all(r["graph_usage"]["graph_path_used"] for r in recs),
                 target_eval_is_evaluation_only=True)
        if recs[0].get("leakage"):
            a["leakage_kl"] = {o: _mean([r["leakage"][o]["kl_mean"] for r in recs]) for o in ("graph", "node", "edge")}
        # PASS = source-only task + valid/non-degenerate graph objects + graph path genuinely used
        a["passes"] = bool((a["source_probe_bacc"] or 0.0) >= args.success_bacc_floor
                           and n_pass >= args.min_seeds_pass
                           and (a["train_bacc"] or 0.0) >= chance + args.train_above_chance_margin
                           and a["forward_graph_valid"] and a["forward_graph_nondegenerate"]
                           and a["graph_path_used"])
        per_cand[name] = a
        lk = (f" leak g/n/e={a['leakage_kl']['graph']:.2f}/{a['leakage_kl']['node']:.2f}/{a['leakage_kl']['edge']:.2f}"
              if "leakage_kl" in a else "")
        print(f"  {name:28s} train={a['train_bacc']:.3f} src={a['source_probe_bacc']:.3f} "
              f"({n_pass}/{len(args.seeds)} seeds) tgt={a['target_eval_bacc']:.3f} "
              f"zeroG_drop={a['zero_graph_drop']:+.3f} graph_used={a['graph_path_used']} PASS={a['passes']}{lk}",
              flush=True)

    selected = [c for c in args.candidates if per_cand[c]["passes"]]                  # source-only
    dynamic_ok = [c for c in selected if per_cand[c]["meta_arch"]["edge_logits_dynamic"]]
    summary = dict(
        meta=dict(meta_flags, dataset=dataset, fold=fold_tag, heldout_subject=fold[6], seeds=list(args.seeds),
                  n_classes=int(n_cls), chance=float(chance), success_bacc_floor=float(args.success_bacc_floor),
                  min_seeds_pass=int(args.min_seeds_pass), graph_usage_min_drop=float(args.graph_usage_min_drop),
                  epochs=int(args.epochs), commit_hash=commit, config_hash=cfg_hash),
        candidates=per_cand, selected_successful_graph_backbones=selected,
        any_graph_backbone_succeeds=bool(selected),
        dynamic_edge_backbone_succeeds=bool(dynamic_ok),
        only_static_adapter_succeeds=bool(selected and not dynamic_ok))
    json.dump(summary, open(OUT_DIR / f"{fold_tag}_graph_backbone_redesign_summary.json", "w"), indent=2)
    print(f"\n[phase3a-G] wrote {OUT_DIR / f'{fold_tag}_graph_backbone_redesign_summary.json'}")

    if dynamic_ok:
        verdict = "A: a DYNAMIC-edge graph backbone learns the task via the graph path -> repaired-backbone Gate-2 next"
    elif selected:
        verdict = "C: only the static DGCNN adapter passes -> graph/node CIGL path (not edge-CMI) on a repaired backbone"
    else:
        verdict = "B: no graph-compatible backbone passes -> pause CIGL method path; keep diagnostic framework"
    print(f"\n=== Phase 3A-G read (exploratory; reviewer decides): {verdict} ===")
    print(f"  selected_successful_graph_backbones (source-only): {selected}")
    print("  target_eval is EVALUATION-ONLY; selection never uses it. No CMI regularization used.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
