#!/usr/bin/env python
"""CIGL Phase 3A-I — graph/node CMI regularizer pilot on the task-capable DGCNN adapter (source-only).

Phase 3A-H: the static DGCNN adapter is task-capable (source ~0.458) AND carries strong, significant,
stable graph/node leakage. Phase 3A-I asks the controllability question:

  Can graph/node CMI regularization REDUCE the verified graph/node leakage WITHOUT destroying the (modest)
  DGCNN task baseline?

GRAPH/NODE ONLY — no edge term (DGCNN adjacency is static; edge_logits=None; edge audit skipped, never
faked). Strict source-only: target labels/covariates never touch training, selection, confirmation-label
choice, normalization, probe fitting, or the audit; target_eval is evaluation-only and enters only a
final REPORTED retention verdict (never selection). Training-time posterior heads drive the regularizer;
evidence uses FRESH held-out audit probes with retrained within-label permutation nulls.

    python scripts/run_cigl_phase3a_dgcnn_gn_regularizer_pilot.py --dry_run_synthetic --device cpu --seeds 0 1 --epochs 3 --probe_epochs 5 --n_perm 5 --n_perm_confirm 5
    python scripts/run_cigl_phase3a_dgcnn_gn_regularizer_pilot.py --dataset BNCI2014_001 --device cuda --fold 0 --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 20 --n_perm_confirm 50 --gate_alpha 0.05

See docs/CIGL_26_PHASE3A_I_DGCNN_GN_REGULARIZER_PILOT.md.
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
from scripts.run_cigl_phase3a_graph_backbone_redesign import (               # noqa: E402
    _forward_graph_feats, _ablation_bacc)
from scripts.run_cigl_phase3a_backbone_sanity import (                       # noqa: E402
    _synthetic_fold, _load_real_fold, _remap_contiguous, _git_commit_hash, _config_hash, _mean)

OUT_DIR = REPO / "results" / "cigl" / "phase3a_dgcnn_gn_regularizer_pilot"
PHASE = "Phase3A_I_dgcnn_gn_regularizer_pilot"
CANDIDATE = "dgcnn_forward_graph_adapter"
ERM_LABEL = "erm_fixed"
# graph/node only (lambda_g, lambda_node); NO edge term anywhere.
CONFIGS = [
    ("erm_fixed", 0.000, 0.000),
    ("graph_001", 0.001, 0.000),
    ("node_001", 0.000, 0.001),
    ("graph_node_001", 0.001, 0.001),
    ("graph_003", 0.003, 0.000),
    ("node_003", 0.000, 0.003),
    ("graph_node_003", 0.003, 0.003),
    ("graph_node_010", 0.010, 0.010),
]


def _reduction(erm_kls, cfg_kls):
    """Per-seed fractional reduction (erm - cfg)/erm, paired by seed index; clamps erm<=0 to None."""
    out = []
    for e, c in zip(erm_kls, cfg_kls):
        out.append((e - c) / e if e and e > 1e-9 else None)
    return out


def decide_pilot_selection(agg, erm_label=ERM_LABEL, source_drop_max=0.02, reduce_min=0.30,
                           reduce_min_seeds=2, target_drop_max=0.05, src_floor=0.45):
    """SOURCE-ONLY selection firewall (Phase 3A-R pattern). Leakage reduction + source retention drive
    ALL selection and the confirmation labels; target_eval enters ONLY a final reported retention verdict.
    Returns reductions, source_only_reducers, best_pareto, best_graph_node, confirmation_labels (source-
    only), and final_target_retaining_reducers (reported)."""
    erm = agg[erm_label]
    erm_src = erm["source_probe_bacc"] or 0.0
    erm_g = erm["graph_kl_per_seed"]; erm_n = erm["node_kl_per_seed"]
    erm_reproduces = bool(erm_src >= src_floor and erm["n_seeds_source_pass"] >= reduce_min_seeds)

    red = {}
    for label, a in agg.items():
        if label == erm_label:
            continue
        gr = _reduction(erm_g, a["graph_kl_per_seed"]); nr = _reduction(erm_n, a["node_kl_per_seed"])
        g30 = int(sum((r is not None and r >= reduce_min) for r in gr))
        n30 = int(sum((r is not None and r >= reduce_min) for r in nr))
        src_drop = float(erm_src - (a["source_probe_bacc"] or 0.0))
        tgt_drop = float((erm["target_eval_bacc"] or 0.0) - (a["target_eval_bacc"] or 0.0))
        red[label] = dict(graph_reduction_vs_erm=_mean([r for r in gr if r is not None]),
                          node_reduction_vs_erm=_mean([r for r in nr if r is not None]),
                          graph_reduce30_seeds=g30, node_reduce30_seeds=n30,
                          source_drop_vs_erm=src_drop, target_drop_vs_erm=tgt_drop,
                          source_probe_bacc=a["source_probe_bacc"])

    leakage_reducers = [l for l in red if red[l]["graph_reduce30_seeds"] >= reduce_min_seeds
                        or red[l]["node_reduce30_seeds"] >= reduce_min_seeds]
    # SOURCE-ONLY: keep source task (drop<=2pt AND still >=floor). target_eval NOT consulted here.
    source_only_reducers = [l for l in leakage_reducers
                            if red[l]["source_drop_vs_erm"] <= source_drop_max
                            and (red[l]["source_probe_bacc"] or 0.0) >= src_floor]

    def _best(cands):
        return max(cands, key=lambda l: max(red[l]["graph_reduction_vs_erm"] or 0.0,
                                            red[l]["node_reduction_vs_erm"] or 0.0)) if cands else None
    best_pareto = _best(source_only_reducers)
    best_graph_node = _best([l for l in source_only_reducers if l.startswith("graph_node")])
    confirmation_labels = sorted({erm_label} | ({best_pareto} if best_pareto else set())
                                 | ({best_graph_node} if best_graph_node else set()))
    # REPORTED verdict only (adds target retention AFTER confirmation labels are fixed):
    final_target_retaining = [l for l in source_only_reducers if red[l]["target_drop_vs_erm"] <= target_drop_max]
    return dict(reductions=red, erm_reproduces=erm_reproduces, leakage_reducers=leakage_reducers,
                source_only_reducers=source_only_reducers, best_pareto=best_pareto,
                best_graph_node=best_graph_node, confirmation_labels=confirmation_labels,
                final_target_retaining_reducers=final_target_retaining,
                pilot_pass_source_only=bool(erm_reproduces and source_only_reducers),
                pilot_pass_with_target_retention=bool(erm_reproduces and final_target_retaining))


def _train_eval(label, lam_g, lam_node, fold, seed, args, device, n_perm):
    from cmi.train.trainer import train_model, predict
    from cmi.eval.metrics import classification_metrics
    X, y, dom_all, tr_mask, te_mask, n_cls, heldout = fold
    Xs, ys = X[tr_mask], y[tr_mask]
    ds, n_dom = _remap_contiguous(dom_all[tr_mask])
    enc_idx, pool_idx, _ = stratified_trial_split_by_y_d(ys, ds, train_frac=args.enc_train_frac,
                                                         seed=seed, min_per_cell=args.min_per_cell)
    method = "erm" if (lam_g == 0.0 and lam_node == 0.0) else "graphcmi"      # graph/node CMI; NO edge term
    torch.manual_seed(int(seed)); np.random.seed(int(seed))                  # seed BEFORE construction
    net = build_graph_task_backbone(CANDIDATE, X.shape[1], X.shape[2], n_cls).to(device)
    assert net.meta.get("edge_logits_dynamic") is False
    net, _post, _diag = train_model(net, Xs[enc_idx], ys[enc_idx], ds[enc_idx], n_cls, method=method,
                                    lam=lam_g, gamma=lam_node, lam_edge=0.0, epochs=args.epochs, bs=args.bs,
                                    warmup=max(1, args.epochs // 5), device=device, seed=seed)
    train_m = classification_metrics(predict(net, Xs[enc_idx], device), ys[enc_idx])
    src = classification_metrics(predict(net, Xs[pool_idx], device), ys[pool_idx])
    tgt = classification_metrics(predict(net, X[te_mask], device), y[te_mask])
    yp, dp = ys[pool_idx], ds[pool_idx]
    gz, nz, el = _forward_graph_feats(net, Xs[pool_idx], device)
    assert el is None, "DGCNN adapter must expose edge_logits=None"
    gu = dict(zero_graph_drop=float(src["balanced_acc"] - _ablation_bacc(net, Xs[pool_idx], yp, "zero_graph", device)),
              permute_nodes_drop=float(src["balanced_acc"] - _ablation_bacc(net, Xs[pool_idx], yp, "permute_nodes", device)))
    gu["graph_path_used"] = bool(gu["zero_graph_drop"] >= args.graph_usage_min_drop)
    # FRESH held-out audit probes (Step-A training heads are discarded; this refits new probes)
    tr_i, va_i, _ = stratified_trial_split_by_y_d(yp, dp, train_frac=args.train_frac, seed=seed,
                                                  min_per_cell=args.min_per_cell)
    au = audit_graph_node_objects(gz, nz, yp, dp, n_cls, n_dom, n_perm=n_perm, seed=seed,
                                  device=device, epochs=args.probe_epochs, train_idx=tr_i, val_idx=va_i)

    def _blk(o):
        b = au[o]
        return dict(kl_mean=b["kl_mean"], permutation_mean=b["permutation_mean"], permutation_p=b["permutation_p"],
                    clears_null=bool(b["kl_mean"] > b["permutation_mean"] and b["permutation_p"] <= args.gate_alpha))
    return dict(config=label, lambda_g=lam_g, lambda_node=lam_node, seed=int(seed), n_classes=int(n_cls),
                method=method, edge_audit_skipped=True, edge_skip_reason=au["edge_skip_reason"],
                train=dict(balanced_acc=train_m["balanced_acc"], macro_f1=train_m["macro_f1"]),
                source_probe=dict(balanced_acc=src["balanced_acc"], macro_f1=src["macro_f1"]),
                target_eval=dict(balanced_acc=tgt["balanced_acc"], macro_f1=tgt["macro_f1"], evaluation_only=True),
                train_minus_source_gap=float(train_m["balanced_acc"] - src["balanced_acc"]),
                graph_usage=gu, leakage=dict(graph=_blk("graph"), node=_blk("node")),
                heldout_subject=heldout, n_enc_train=int(len(enc_idx)), n_probe_pool=int(len(pool_idx)))


def _aggregate(label, recs, args):
    src_seeds = [r["source_probe"]["balanced_acc"] for r in recs]
    return dict(config=label, lambda_g=recs[0]["lambda_g"], lambda_node=recs[0]["lambda_node"],
                train_bacc=_mean([r["train"]["balanced_acc"] for r in recs]),
                source_probe_bacc=_mean(src_seeds), source_probe_per_seed=[round(s, 3) for s in src_seeds],
                n_seeds_source_pass=int(sum(s >= args.success_bacc_floor for s in src_seeds)),
                target_eval_bacc=_mean([r["target_eval"]["balanced_acc"] for r in recs]),
                graph_kl_per_seed=[r["leakage"]["graph"]["kl_mean"] for r in recs],
                node_kl_per_seed=[r["leakage"]["node"]["kl_mean"] for r in recs],
                graph_kl_mean=_mean([r["leakage"]["graph"]["kl_mean"] for r in recs]),
                node_kl_mean=_mean([r["leakage"]["node"]["kl_mean"] for r in recs]),
                graph_clears_seeds=int(sum(r["leakage"]["graph"]["clears_null"] for r in recs)),
                node_clears_seeds=int(sum(r["leakage"]["node"]["clears_null"] for r in recs)),
                zero_graph_drop=_mean([r["graph_usage"]["zero_graph_drop"] for r in recs]),
                permute_nodes_drop=_mean([r["graph_usage"]["permute_nodes_drop"] for r in recs]),
                graph_path_used=all(r["graph_usage"]["graph_path_used"] for r in recs))


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
    ap.add_argument("--n_perm", type=int, default=20)
    ap.add_argument("--n_perm_confirm", type=int, default=50)
    ap.add_argument("--gate_alpha", type=float, default=0.05)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--enc_train_frac", type=float, default=0.7)
    ap.add_argument("--min_per_cell", type=int, default=2)
    ap.add_argument("--success_bacc_floor", type=float, default=0.45)
    ap.add_argument("--source_drop_max", type=float, default=0.02)
    ap.add_argument("--reduce_min", type=float, default=0.30)
    ap.add_argument("--target_drop_max", type=float, default=0.05)
    ap.add_argument("--graph_usage_min_drop", type=float, default=0.10)
    ap.add_argument("--configs", nargs="+", default=[c[0] for c in CONFIGS])
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[phase3a-I] --device cuda requested but CUDA unavailable")
    if args.device == "cpu":
        torch.set_num_threads(1)

    lam_map = {name: (lg, ln) for name, lg, ln in CONFIGS}
    run_configs = [(n, lam_map[n][0], lam_map[n][1]) for n in args.configs]
    commit = _git_commit_hash(); cfg_hash = _config_hash(vars(args))
    fold = _synthetic_fold(seed=0) if args.dry_run_synthetic else _load_real_fold(args)
    dataset = "synthetic" if args.dry_run_synthetic else args.dataset
    fold_tag = f"{dataset}_fold{args.fold}"
    n_cls = fold[5]; chance = 1.0 / n_cls
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def meta_for(label, lam_g, lam_node):
        return dict(exploratory=True, phase=PHASE, setting="strict_source_only_DG",
                    cmi_regularization_used=bool(lam_g != 0.0 or lam_node != 0.0), edge_regularization_used=False,
                    edge_logits_dynamic=False, edge_audit_skipped=True,
                    used_target_labels_for_training=False, used_target_labels_for_selection=False,
                    used_target_covariates=False, target_eval_is_evaluation_only=True,
                    selection_uses_target_eval=False, confirmation_label_selection_uses_target_eval=False,
                    dataset=dataset, fold=fold_tag, commit_hash=commit, config_hash=cfg_hash)

    print(f"\n=== Phase 3A-I DGCNN graph/node CMI pilot ({fold_tag}; chance={chance:.3f}, floor={args.success_bacc_floor}) ===")
    agg = {}
    for label, lam_g, lam_node in run_configs:
        recs = []
        for seed in args.seeds:
            rec = _train_eval(label, lam_g, lam_node, fold, seed, args, args.device, args.n_perm)
            rec["meta"] = meta_for(label, lam_g, lam_node)
            json.dump(rec, open(OUT_DIR / f"{fold_tag}_{label}_seed{seed}.json", "w"), indent=2)
            recs.append(rec)
        agg[label] = _aggregate(label, recs, args)
        a = agg[label]
        print(f"  {label:16s} (lg={lam_g},ln={lam_node}) src={a['source_probe_bacc']:.3f}"
              f"({a['n_seeds_source_pass']}/{len(args.seeds)}) tgt={a['target_eval_bacc']:.3f} "
              f"gKL={a['graph_kl_mean']:.3f}({a['graph_clears_seeds']}) nKL={a['node_kl_mean']:.3f}"
              f"({a['node_clears_seeds']}) graph_used={a['graph_path_used']}", flush=True)

    sel = decide_pilot_selection(agg, ERM_LABEL, args.source_drop_max, args.reduce_min, 2,
                                 args.target_drop_max, args.success_bacc_floor)

    # CONFIRMATION (n_perm_confirm) for the SOURCE-ONLY-selected confirmation labels
    confirmation = {}
    for label in sel["confirmation_labels"]:
        lam_g, lam_node = lam_map[label]
        per_seed = []
        for seed in args.seeds:
            r = _train_eval(label, lam_g, lam_node, fold, seed, args, args.device, args.n_perm_confirm)
            per_seed.append(dict(seed=int(seed),
                                 graph=r["leakage"]["graph"], node=r["leakage"]["node"],
                                 source_probe_bacc=r["source_probe"]["balanced_acc"]))
        confirmation[label] = dict(
            graph_clears_seeds=int(sum(p["graph"]["clears_null"] for p in per_seed)),
            node_clears_seeds=int(sum(p["node"]["clears_null"] for p in per_seed)),
            graph_kl_mean=_mean([p["graph"]["kl_mean"] for p in per_seed]),
            node_kl_mean=_mean([p["node"]["kl_mean"] for p in per_seed]), per_seed=per_seed)

    summary = dict(
        meta=dict(meta_for("_run", 0.0, 0.0),
                  cmi_regularization_used=any(lg != 0.0 or ln != 0.0 for _, lg, ln in run_configs),
                  heldout_subject=fold[6], seeds=list(args.seeds),
                  n_classes=int(n_cls), chance=float(chance), success_bacc_floor=float(args.success_bacc_floor),
                  source_drop_max=float(args.source_drop_max), reduce_min=float(args.reduce_min),
                  target_drop_max=float(args.target_drop_max), n_perm=int(args.n_perm),
                  n_perm_confirm=int(args.n_perm_confirm), gate_alpha=float(args.gate_alpha),
                  epochs=int(args.epochs), candidate=CANDIDATE),
        configs={n: (lg, ln) for n, lg, ln in CONFIGS},
        per_config=agg, selection=sel, confirmation=confirmation,
        edge_skip_reason="static/shared adjacency: edge_logits=None; no per-sample edge object")
    json.dump(summary, open(OUT_DIR / f"{fold_tag}_dgcnn_gn_pilot_summary.json", "w"), indent=2)
    print(f"\n[phase3a-I] wrote {OUT_DIR / f'{fold_tag}_dgcnn_gn_pilot_summary.json'}")

    if not sel["erm_reproduces"]:
        verdict = "D: ERM_fixed does NOT reproduce the DGCNN baseline -> return to DGCNN stability diagnosis"
    elif sel["pilot_pass_with_target_retention"]:
        verdict = "A: graph/node regularizer reduces leakage AND retains source+target -> candidate for multi-fold confirmation"
    elif sel["pilot_pass_source_only"]:
        verdict = "B(tradeoff): leakage reduced with source retained, but target retention/headroom thin -> tradeoff signal, not a method win"
    else:
        verdict = "C: no config reduces graph/node leakage without source loss -> diagnostic/redesign"
    print(f"\n=== Phase 3A-I read (exploratory; reviewer decides): {verdict} ===")
    print(f"  source_only_reducers={sel['source_only_reducers']} best_pareto={sel['best_pareto']} "
          f"best_graph_node={sel['best_graph_node']}")
    print(f"  confirmation_labels (SOURCE-ONLY)={sel['confirmation_labels']} "
          f"final_target_retaining={sel['final_target_retaining_reducers']}")
    print("  EDGE term/audit absent (static adjacency). target_eval evaluation-only; selection source-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
