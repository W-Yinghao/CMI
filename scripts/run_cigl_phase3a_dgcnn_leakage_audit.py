#!/usr/bin/env python
"""CIGL Phase 3A-H — graph/node leakage audit of the task-capable DGCNN adapter (source-only, ERM).

Phase 3A-G (Decision B): only `dgcnn_forward_graph_adapter` learns BNCI2014_001 fold-0 as a graph
backbone; the dynamic-edge stems overfit. The next diagnostic question is whether the SUCCESSFUL static
DGCNN adapter's `graph_z` and `node_z` carry label-conditional source-domain leakage I(Z;D|Y).

This is DIAGNOSTIC ONLY — NO CMI regularization. DGCNN's adjacency is shared/static, so there is no
per-sample edge object: the EDGE audit is SKIPPED (never faked). We audit graph_z and node_z with fresh
held-out conditional-domain probes and within-label retrained permutation nulls (n_perm), and report a
node-map stability check across seeds. Strict source-only: target labels/covariates never touch training,
selection, normalization, probe fitting, or the audit; target_eval is evaluation-only.

    python scripts/run_cigl_phase3a_dgcnn_leakage_audit.py --dry_run_synthetic --device cpu --seeds 0 1 --epochs 3 --probe_epochs 5 --n_perm 5
    python scripts/run_cigl_phase3a_dgcnn_leakage_audit.py --dataset BNCI2014_001 --device cuda --fold 0 --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 50 --gate_alpha 0.05

See docs/CIGL_24_PHASE3A_H_DGCNN_LEAKAGE_AUDIT.md.
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
from cmi.eval.graph_leakage import audit_graph_node_objects                  # noqa: E402
from cmi.eval.probe_splits import stratified_trial_split_by_y_d              # noqa: E402
from cmi.models.graph_task_backbones import build_graph_task_backbone        # noqa: E402
# reuse the EXACT Phase 3A-G/3A-S source-only fold + forward_graph/ablation helpers
from scripts.run_cigl_phase3a_graph_backbone_redesign import (               # noqa: E402
    _forward_graph_feats, _ablation_bacc)
from scripts.run_cigl_phase3a_backbone_sanity import (                       # noqa: E402
    _synthetic_fold, _load_real_fold, _remap_contiguous, _git_commit_hash, _config_hash, _mean)

OUT_DIR = REPO / "results" / "cigl" / "phase3a_dgcnn_leakage_audit"
PHASE = "Phase3A_H_dgcnn_leakage_audit"
CANDIDATE = "dgcnn_forward_graph_adapter"


def _node_map_stability(maps, seed=0, n_null=200):
    """Mean pairwise Pearson correlation of per-seed node_leakage_maps vs a channel-permutation null."""
    M = np.asarray(maps, dtype=np.float64)                       # [n_seeds, C]
    if M.shape[0] < 2:
        return dict(mean_corr=None, null_q95=None, above_random=None, degenerate=bool(M.std() < 1e-9),
                    note="need >=2 seeds for stability")
    degenerate = bool(M.std(axis=1).min() < 1e-9)

    def mean_pairwise(mat):
        cs = []
        for i in range(mat.shape[0]):
            for j in range(i + 1, mat.shape[0]):
                a, b = mat[i], mat[j]
                if a.std() < 1e-12 or b.std() < 1e-12:
                    cs.append(0.0)
                else:
                    cs.append(float(np.corrcoef(a, b)[0, 1]))
        return float(np.mean(cs)) if cs else 0.0

    mean_corr = mean_pairwise(M)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(n_null)):
        permuted = np.stack([row[rng.permutation(M.shape[1])] for row in M])    # break channel alignment
        null.append(mean_pairwise(permuted))
    null = np.asarray(null)
    q95 = float(np.quantile(null, 0.95))
    return dict(mean_corr=mean_corr, null_q95=q95, above_random=bool(mean_corr > q95), degenerate=degenerate)


def decide_leakage(graph_clears, node_clears, node_stab, task_ok, graph_path_used, min_seeds_pass):
    """Source-only leakage verdict. A NODE claim requires both a per-seed null-clearance signal AND a
    stable, NON-DEGENERATE node_leakage_map (CIGL_24 criterion 5) — a partially-degenerate map that
    happens to score above_random must NOT clear the node claim. Graph claim is independent of node."""
    graph_leakage_exists = bool(graph_clears >= min_seeds_pass)
    node_leakage_signal = bool(node_clears >= min_seeds_pass)                       # per-seed null clearance
    node_map_stable = bool(node_stab.get("above_random")) and not bool(node_stab.get("degenerate", False))
    node_leakage_claimed = bool(node_leakage_signal and node_map_stable)            # signal AND non-degenerate stability
    leakage_exists = bool(graph_leakage_exists or node_leakage_claimed)
    audit_passes = bool(task_ok and graph_path_used and leakage_exists)
    return dict(graph_leakage_exists=graph_leakage_exists, node_leakage_signal=node_leakage_signal,
                node_map_stable=node_map_stable, node_leakage_claimed=node_leakage_claimed,
                leakage_exists=leakage_exists, audit_passes=audit_passes)


def _train_eval(fold, seed, args, device):
    from cmi.train.trainer import train_model, predict
    from cmi.eval.metrics import classification_metrics
    X, y, dom_all, tr_mask, te_mask, n_cls, heldout = fold
    Xs, ys = X[tr_mask], y[tr_mask]
    ds, n_dom = _remap_contiguous(dom_all[tr_mask])
    enc_idx, pool_idx, _ = stratified_trial_split_by_y_d(ys, ds, train_frac=args.enc_train_frac,
                                                         seed=seed, min_per_cell=args.min_per_cell)
    torch.manual_seed(int(seed)); np.random.seed(int(seed))      # seed BEFORE construction (reproducible)
    net = build_graph_task_backbone(CANDIDATE, X.shape[1], X.shape[2], n_cls).to(device)
    assert net.meta.get("edge_logits_dynamic", None) is False, "DGCNN adapter must be static-edge"
    net, _post, _diag = train_model(net, Xs[enc_idx], ys[enc_idx], ds[enc_idx], n_cls, method="erm",
                                    epochs=args.epochs, bs=args.bs, warmup=max(1, args.epochs // 5),
                                    device=device, seed=seed)
    train_m = classification_metrics(predict(net, Xs[enc_idx], device), ys[enc_idx])
    src = classification_metrics(predict(net, Xs[pool_idx], device), ys[pool_idx])
    tgt = classification_metrics(predict(net, X[te_mask], device), y[te_mask])

    yp, dp = ys[pool_idx], ds[pool_idx]
    gz, nz, el = _forward_graph_feats(net, Xs[pool_idx], device)
    assert el is None, "DGCNN adapter must expose edge_logits=None (no per-sample edge object)"

    # graph-usage anti-bypass on the source probe-pool
    gu = dict(source_probe_bacc=src["balanced_acc"],
              zero_graph_bacc=_ablation_bacc(net, Xs[pool_idx], yp, "zero_graph", device),
              permute_nodes_bacc=_ablation_bacc(net, Xs[pool_idx], yp, "permute_nodes", device))
    gu["zero_graph_drop"] = float(src["balanced_acc"] - gu["zero_graph_bacc"])
    gu["permute_nodes_drop"] = float(src["balanced_acc"] - gu["permute_nodes_bacc"])
    gu["graph_path_used"] = bool(gu["zero_graph_drop"] >= args.graph_usage_min_drop)

    # graph + node leakage audit (NO edge); within-label retrained permutation null
    tr_i, va_i, _ = stratified_trial_split_by_y_d(yp, dp, train_frac=args.train_frac, seed=seed,
                                                  min_per_cell=args.min_per_cell)
    au = audit_graph_node_objects(gz, nz, yp, dp, n_cls, n_dom, n_perm=args.n_perm, seed=seed,
                                  device=device, epochs=args.probe_epochs, train_idx=tr_i, val_idx=va_i)

    def _block(o):
        b = au[o]
        clears = bool(b["kl_mean"] > b["permutation_mean"] and b["permutation_p"] <= args.gate_alpha)
        return dict(kl_mean=b["kl_mean"], permutation_mean=b["permutation_mean"],
                    permutation_p=b["permutation_p"], clears_null=clears,
                    **({"node_leakage_map": b["node_leakage_map"]} if o == "node" else {}))

    return dict(candidate=CANDIDATE, seed=int(seed), n_classes=int(n_cls),
                meta_arch=net.meta, edge_audit_skipped=True, edge_skip_reason=au["edge_skip_reason"],
                train=dict(balanced_acc=train_m["balanced_acc"], macro_f1=train_m["macro_f1"]),
                source_probe=dict(balanced_acc=src["balanced_acc"], macro_f1=src["macro_f1"]),
                target_eval=dict(balanced_acc=tgt["balanced_acc"], macro_f1=tgt["macro_f1"], evaluation_only=True),
                train_minus_source_gap=float(train_m["balanced_acc"] - src["balanced_acc"]),
                graph_usage=gu, leakage=dict(graph=_block("graph"), node=_block("node")),
                heldout_subject=heldout, n_enc_train=int(len(enc_idx)), n_probe_pool=int(len(pool_idx)))


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
    ap.add_argument("--n_perm", type=int, default=50)
    ap.add_argument("--gate_alpha", type=float, default=0.05)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--enc_train_frac", type=float, default=0.7)
    ap.add_argument("--min_per_cell", type=int, default=2)
    ap.add_argument("--success_bacc_floor", type=float, default=0.45)
    ap.add_argument("--min_seeds_pass", type=int, default=2)
    ap.add_argument("--graph_usage_min_drop", type=float, default=0.10)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[phase3a-H] --device cuda requested but CUDA unavailable")
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
                      cmi_regularization_used=False, edge_logits_dynamic=False, edge_audit_skipped=True)

    print(f"\n=== Phase 3A-H DGCNN leakage audit ({fold_tag}; chance={chance:.3f}, floor={args.success_bacc_floor}, "
          f"n_perm={args.n_perm}, alpha={args.gate_alpha}) ===")
    recs = []
    for seed in args.seeds:
        rec = _train_eval(fold, seed, args, args.device)
        rec["meta"] = dict(meta_flags, dataset=dataset, fold=fold_tag, commit_hash=commit, config_hash=cfg_hash)
        json.dump(rec, open(OUT_DIR / f"{fold_tag}_{CANDIDATE}_seed{seed}.json", "w"), indent=2)
        recs.append(rec)
        L = rec["leakage"]
        print(f"  seed{seed}: src={rec['source_probe']['balanced_acc']:.3f} train={rec['train']['balanced_acc']:.3f} "
              f"zeroG_drop={rec['graph_usage']['zero_graph_drop']:+.3f} | "
              f"graph kl={L['graph']['kl_mean']:.3f} perm={L['graph']['permutation_mean']:.3f} "
              f"p={L['graph']['permutation_p']:.3f} clears={L['graph']['clears_null']} | "
              f"node kl={L['node']['kl_mean']:.3f} p={L['node']['permutation_p']:.3f} clears={L['node']['clears_null']}",
              flush=True)

    src_seeds = [r["source_probe"]["balanced_acc"] for r in recs]
    n_src_pass = int(sum(s >= args.success_bacc_floor for s in src_seeds))
    graph_clears = int(sum(r["leakage"]["graph"]["clears_null"] for r in recs))
    node_clears = int(sum(r["leakage"]["node"]["clears_null"] for r in recs))
    node_stab = _node_map_stability([r["leakage"]["node"]["node_leakage_map"] for r in recs], seed=0)
    task_ok = bool((_mean(src_seeds) or 0.0) >= args.success_bacc_floor and n_src_pass >= args.min_seeds_pass)
    graph_path_used = all(r["graph_usage"]["graph_path_used"] for r in recs)
    dec = decide_leakage(graph_clears, node_clears, node_stab, task_ok, graph_path_used, args.min_seeds_pass)
    leakage_exists = dec["leakage_exists"]; audit_passes = dec["audit_passes"]

    summary = dict(
        meta=dict(meta_flags, dataset=dataset, fold=fold_tag, heldout_subject=fold[6], seeds=list(args.seeds),
                  n_classes=int(n_cls), chance=float(chance), success_bacc_floor=float(args.success_bacc_floor),
                  min_seeds_pass=int(args.min_seeds_pass), n_perm=int(args.n_perm), gate_alpha=float(args.gate_alpha),
                  epochs=int(args.epochs), commit_hash=commit, config_hash=cfg_hash),
        edge_skip_reason=recs[0]["edge_skip_reason"],
        task=dict(source_probe_bacc=_mean(src_seeds), source_probe_per_seed=[round(s, 3) for s in src_seeds],
                  train_bacc=_mean([r["train"]["balanced_acc"] for r in recs]),
                  target_eval_bacc=_mean([r["target_eval"]["balanced_acc"] for r in recs]),
                  n_seeds_source_pass=n_src_pass, task_ok=task_ok),
        graph_usage=dict(zero_graph_drop=_mean([r["graph_usage"]["zero_graph_drop"] for r in recs]),
                         permute_nodes_drop=_mean([r["graph_usage"]["permute_nodes_drop"] for r in recs]),
                         graph_path_used=graph_path_used),
        leakage=dict(graph=dict(kl_mean=_mean([r["leakage"]["graph"]["kl_mean"] for r in recs]),
                                clears_null_seeds=graph_clears),
                     node=dict(kl_mean=_mean([r["leakage"]["node"]["kl_mean"] for r in recs]),
                               clears_null_seeds=node_clears)),
        node_map_stability=node_stab,
        graph_leakage_exists=dec["graph_leakage_exists"],
        node_leakage_signal=dec["node_leakage_signal"], node_map_stable=dec["node_map_stable"],
        node_leakage_claimed=dec["node_leakage_claimed"],
        leakage_exists=leakage_exists, audit_passes=audit_passes)
    json.dump(summary, open(OUT_DIR / f"{fold_tag}_dgcnn_leakage_audit_summary.json", "w"), indent=2)
    print(f"\n[phase3a-H] wrote {OUT_DIR / f'{fold_tag}_dgcnn_leakage_audit_summary.json'}")

    if not task_ok:
        verdict = "C: DGCNN task unstable on rerun -> return to backbone diagnosis"
    elif leakage_exists:
        verdict = "A: graph/node leakage EXISTS on the task-capable DGCNN -> a graph/node regularizer pilot may be considered"
    else:
        verdict = "B: NO graph/node leakage clears null -> pause CIGL method path (diagnostic story only)"
    print(f"\n=== Phase 3A-H read (exploratory; reviewer decides): {verdict} ===")
    print(f"  task_ok={task_ok} graph_path_used={graph_path_used} graph_clears={graph_clears}/{len(recs)} "
          f"node_clears={node_clears}/{len(recs)} node_map_above_random={node_stab.get('above_random')} "
          f"node_map_degenerate={node_stab.get('degenerate')} node_leakage_claimed={dec['node_leakage_claimed']}")
    print("  EDGE audit SKIPPED (static/shared adjacency; no per-sample edge object; not faked).")
    print("  target labels/covariates NOT used for training/selection/probe/audit; target_eval evaluation-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
