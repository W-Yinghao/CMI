#!/usr/bin/env python
"""CIGL Phase 3A-J — fixed-config multi-fold confirmation of DGCNN graph/node CMI (BNCI2014_001).

Phase 3A-I passed as a single-fold (fold-0) pilot: the FIXED candidate `graph_node_010` (λ_g=λ_node=0.010,
no edge) reduced DGCNN graph/node leakage ~42-48% without source/target task collapse. Phase 3A-J tests
whether that SAME fixed candidate replicates across BNCI2014_001 LOSO folds. NO λ grid, NO new configs,
NO edge term, NO SOTA — a replication test only.

fold-0 SELECTED graph_node_010, so it is a DEVELOPMENT fold; the primary confirmation set is folds 1-8.
The main decision is based on folds 1-8 (not all folds pooled). Strict source-only firewall: configs are
FIXED (no selection); target labels/covariates never touch training/normalization/probe-fit/audit;
target_eval is an evaluation-only guardrail.

    python scripts/run_cigl_phase3a_dgcnn_gn_multifold_confirmation.py --dry_run_synthetic --device cpu --folds 0 1 2 --seeds 0 1 --epochs 3 --probe_epochs 5 --n_perm 5
    python scripts/run_cigl_phase3a_dgcnn_gn_multifold_confirmation.py --dataset BNCI2014_001 --device cuda --folds 0 1 2 3 4 5 6 7 8 --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 50 --gate_alpha 0.05

See docs/CIGL_28_PHASE3A_J_MULTIFOLD_CONFIRMATION.md.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_cigl_phase3a_dgcnn_gn_regularizer_pilot import _train_eval, _reduction   # noqa: E402
from scripts.run_cigl_phase3a_backbone_sanity import _git_commit_hash, _config_hash, _mean  # noqa: E402

OUT_DIR = REPO / "results" / "cigl" / "phase3a_dgcnn_gn_multifold_confirmation"
PHASE = "Phase3A_J_multifold_confirmation"
FIXED_CONFIGS = [("erm_fixed", 0.000, 0.000), ("graph_node_010", 0.010, 0.010)]   # graph/node only; no edge
DEV_FOLD = 0


def _synthetic_folds(folds, n_per_subj=60, C=8, T=48, n_cls=3, n_subj=4):
    """Multi-fold learnable synthetic: build once; fold f holds out subject (f % n_subj)."""
    rng = np.random.default_rng(0)
    proto = 2.5 * rng.standard_normal((n_cls, C, T)).astype("float32")
    Xs, ys, ds = [], [], []
    for s in range(n_subj):
        for _ in range(n_per_subj):
            yy = rng.integers(0, n_cls)
            x = proto[yy] + 0.5 * rng.standard_normal((C, T)).astype("float32"); x[1:4] += 0.8 * s
            Xs.append(x); ys.append(yy); ds.append(s)
    X = np.stack(Xs).astype("float32"); y = np.array(ys, "int64"); d = np.array(ds, "int64")
    out = {}
    for f in folds:
        tgt = f % n_subj
        out[f] = (X, y, d, d != tgt, d == tgt, n_cls, str(tgt))
    return out


def _load_real_folds(args, folds):
    from cmi.data import moabb_data
    try:
        X, y, meta, classes = moabb_data.load(args.dataset, tmin=args.tmin, tmax=args.tmax, resample=args.resample)
    except Exception as e:
        raise SystemExit(f"[phase3a-J] dataset '{args.dataset}' not loadable offline ({type(e).__name__}: {e}); "
                         f"use --dry_run_synthetic to validate.")
    dom_all, _ = moabb_data.domain_labels(meta, "subject")
    splits = list(moabb_data.loso_splits(meta))
    out = {}
    for f in folds:
        if f >= len(splits):
            raise SystemExit(f"[phase3a-J] fold {f} out of range (dataset has {len(splits)} LOSO folds)")
        tgt, tr_mask, te_mask = splits[f]
        out[f] = (X, y, dom_all, tr_mask, te_mask, len(classes), str(tgt))
    return out


def _fold_flags(erm, reg, args):
    """Per-fold pass flags from the two fixed configs' per-seed aggregates (SOURCE-ONLY + reported target).
    The per-seed ">=2/3 seeds" rule is generalized to ceil(2/3 * n_seeds) (== 2 for the 3-seed real run)."""
    floor = args.success_bacc_floor
    erm_src = erm["source_probe_per_seed"]; reg_src = reg["source_probe_per_seed"]
    min_seeds = max(1, math.ceil(2 / 3 * len(erm_src)))                   # 2/3 of seeds (==2 for 3 seeds)
    erm_adequate = bool((_mean(erm_src) or 0.0) >= floor and sum(s >= floor for s in erm_src) >= min_seeds)
    erm_leakage_exists = bool(erm["graph_clears_seeds"] >= min_seeds or erm["node_clears_seeds"] >= min_seeds)
    gred = _reduction(erm["graph_kl_per_seed"], reg["graph_kl_per_seed"])
    nred = _reduction(erm["node_kl_per_seed"], reg["node_kl_per_seed"])
    g30 = int(sum((r is not None and r >= args.reduce_min) for r in gred))
    n30 = int(sum((r is not None and r >= args.reduce_min) for r in nred))
    reg_reduces = bool(g30 >= min_seeds or n30 >= min_seeds)
    src_drop = float((_mean(erm_src) or 0.0) - (_mean(reg_src) or 0.0))
    source_retained = bool((_mean(reg_src) or 0.0) >= floor and src_drop <= args.source_drop_max)
    tgt_drop = float((erm["target_eval_bacc"] or 0.0) - (reg["target_eval_bacc"] or 0.0))
    target_guardrail = bool(tgt_drop <= args.target_drop_max)
    return dict(erm_adequate=erm_adequate, erm_leakage_exists=erm_leakage_exists, reg_reduces=reg_reduces,
                source_retained=source_retained, target_guardrail=target_guardrail,
                graph_reduction=_mean([r for r in gred if r is not None]),
                node_reduction=_mean([r for r in nred if r is not None]),
                graph_reduce30_seeds=g30, node_reduce30_seeds=n30,
                source_drop_vs_erm=src_drop, target_drop_vs_erm=tgt_drop,
                fold_pass=bool(erm_adequate and erm_leakage_exists and reg_reduces and source_retained))


def decide_multifold(per_fold_flags, confirmation_folds):
    """Aggregate the PRIMARY decision over the confirmation folds (NOT all folds pooled). Thresholds match
    the reviewer's 6/8 (adequacy, leakage) and 5/8 (reduction, retention) at n=8, generalized by fraction."""
    cf = [f for f in confirmation_folds if f in per_fold_flags]
    n = len(cf)
    need_strong = math.ceil(0.75 * n) if n else 1          # 6/8
    need_majority = math.ceil(0.625 * n) if n else 1       # 5/8
    c1 = sum(per_fold_flags[f]["erm_adequate"] for f in cf)
    c2 = sum(per_fold_flags[f]["erm_leakage_exists"] for f in cf)
    c3 = sum(per_fold_flags[f]["reg_reduces"] for f in cf)
    c4 = sum(per_fold_flags[f]["source_retained"] for f in cf)
    c5 = sum(per_fold_flags[f]["target_guardrail"] for f in cf)
    crit1 = c1 >= need_strong; crit2 = c2 >= need_strong
    crit3 = c3 >= need_majority; crit4 = c4 >= need_majority
    confirmed = bool(crit1 and crit2 and crit3 and crit4)
    if not crit1:
        decision = "D"      # ERM baseline unstable across folds
    elif confirmed:
        decision = "A"      # confirmed
    elif crit2 and (crit3 or crit4):
        decision = "B"      # partial signal
    else:
        decision = "C"      # not confirmed
    return dict(n_confirmation_folds=n, need_strong=need_strong, need_majority=need_majority,
                erm_adequate_folds=c1, erm_leakage_folds=c2, reg_reduces_folds=c3,
                source_retained_folds=c4, target_guardrail_folds=c5,
                crit1_erm_adequate=crit1, crit2_erm_leakage=crit2, crit3_reg_reduces=crit3,
                crit4_source_retained=crit4, confirmed=confirmed, decision=decision)


def _meta(commit, cfg_hash, dataset):
    return dict(exploratory=True, phase=PHASE, setting="strict_source_only_DG",
                fixed_candidate="graph_node_010", fold0_is_dev=True,
                primary_confirmation_folds=[1, 2, 3, 4, 5, 6, 7, 8],
                cmi_regularization_used=True, edge_regularization_used=False,
                edge_logits_dynamic=False, edge_audit_skipped=True,
                used_target_labels_for_training=False, used_target_labels_for_selection=False,
                used_target_covariates=False, target_eval_is_evaluation_only=True,
                selection_uses_target_eval=False, confirmation_label_selection_uses_target_eval=False,
                dataset=dataset, commit_hash=commit, config_hash=cfg_hash)


def _aggregate(recs, args):
    src = [r["source_probe"]["balanced_acc"] for r in recs]
    return dict(source_probe_bacc=_mean(src), source_probe_per_seed=[round(s, 3) for s in src],
                train_bacc=_mean([r["train"]["balanced_acc"] for r in recs]),
                target_eval_bacc=_mean([r["target_eval"]["balanced_acc"] for r in recs]),
                graph_kl_per_seed=[r["leakage"]["graph"]["kl_mean"] for r in recs],
                node_kl_per_seed=[r["leakage"]["node"]["kl_mean"] for r in recs],
                graph_kl_mean=_mean([r["leakage"]["graph"]["kl_mean"] for r in recs]),
                node_kl_mean=_mean([r["leakage"]["node"]["kl_mean"] for r in recs]),
                graph_clears_seeds=int(sum(r["leakage"]["graph"]["clears_null"] for r in recs)),
                node_clears_seeds=int(sum(r["leakage"]["node"]["clears_null"] for r in recs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run_synthetic", action="store_true")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8])
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
    ap.add_argument("--source_drop_max", type=float, default=0.02)
    ap.add_argument("--reduce_min", type=float, default=0.30)
    ap.add_argument("--target_drop_max", type=float, default=0.05)
    ap.add_argument("--graph_usage_min_drop", type=float, default=0.10)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[phase3a-J] --device cuda requested but CUDA unavailable")
    if args.device == "cpu":
        torch.set_num_threads(1)

    commit = _git_commit_hash(); cfg_hash = _config_hash(vars(args))
    dataset = "synthetic" if args.dry_run_synthetic else args.dataset
    folds = _synthetic_folds(args.folds) if args.dry_run_synthetic else _load_real_folds(args, args.folds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Phase 3A-J multi-fold confirmation ({dataset}; folds={args.folds}; dev_fold={DEV_FOLD}; "
          f"fixed graph_node_010) ===")
    per_fold = {}
    per_fold_flags = {}
    for f in args.folds:
        fold = folds[f]
        cfg_aggs = {}
        for label, lam_g, lam_node in FIXED_CONFIGS:
            recs = []
            for seed in args.seeds:
                rec = _train_eval(label, lam_g, lam_node, fold, seed, args, args.device, args.n_perm)
                rec["meta"] = dict(_meta(commit, cfg_hash, dataset), fold=f, is_dev_fold=(f == DEV_FOLD),
                                   cmi_regularization_used=bool(lam_g != 0.0 or lam_node != 0.0))   # per-config
                json.dump(rec, open(OUT_DIR / f"{dataset}_fold{f}_{label}_seed{seed}.json", "w"), indent=2)
                recs.append(rec)
            cfg_aggs[label] = _aggregate(recs, args)
        flags = _fold_flags(cfg_aggs["erm_fixed"], cfg_aggs["graph_node_010"], args)
        per_fold[f] = dict(heldout_subject=fold[6], is_dev_fold=(f == DEV_FOLD),
                           erm_fixed=cfg_aggs["erm_fixed"], graph_node_010=cfg_aggs["graph_node_010"], flags=flags)
        per_fold_flags[f] = flags
        e, r = cfg_aggs["erm_fixed"], cfg_aggs["graph_node_010"]
        print(f"  fold{f}{'(dev)' if f == DEV_FOLD else '':5s} ermSrc={e['source_probe_bacc']:.3f} regSrc={r['source_probe_bacc']:.3f} "
              f"(drop={flags['source_drop_vs_erm']:+.3f}) gKL {e['graph_kl_mean']:.2f}->{r['graph_kl_mean']:.2f}"
              f"(g{flags['graph_reduce30_seeds']}/n{flags['node_reduce30_seeds']} @>=30%) "
              f"ermLeak={flags['erm_leakage_exists']} regReduces={flags['reg_reduces']} srcRetain={flags['source_retained']} "
              f"PASS={flags['fold_pass']}", flush=True)

    confirmation_folds = [f for f in args.folds if f != DEV_FOLD]
    primary = decide_multifold(per_fold_flags, confirmation_folds)
    fold0_dev = per_fold.get(DEV_FOLD, {}).get("flags") if DEV_FOLD in args.folds else None
    all_folds = decide_multifold(per_fold_flags, list(args.folds))   # descriptive only (includes dev)

    summary = dict(
        meta=dict(_meta(commit, cfg_hash, dataset), folds=list(args.folds), seeds=list(args.seeds),
                  n_perm=int(args.n_perm), gate_alpha=float(args.gate_alpha), epochs=int(args.epochs),
                  success_bacc_floor=float(args.success_bacc_floor), source_drop_max=float(args.source_drop_max),
                  reduce_min=float(args.reduce_min), target_drop_max=float(args.target_drop_max)),
        configs={n: (lg, ln) for n, lg, ln in FIXED_CONFIGS}, per_fold=per_fold,
        fold0_dev=fold0_dev, folds1_8_confirmation=primary, all_folds_descriptive=all_folds,
        edge_skip_reason="static/shared adjacency: edge_logits=None; no per-sample edge object")
    json.dump(summary, open(OUT_DIR / f"{dataset}_dgcnn_gn_multifold_summary.json", "w"), indent=2)
    print(f"\n[phase3a-J] wrote {OUT_DIR / f'{dataset}_dgcnn_gn_multifold_summary.json'}")

    dmap = {"A": "A: CONFIRMED on folds1-8 -> proceed to method framing / second-dataset confirmation",
            "B": "B: PARTIAL on folds1-8 -> bounded finding / refine; not a full method claim",
            "C": "C: NOT confirmed on folds1-8 -> pilot-only, no method claim",
            "D": "D: ERM baseline unstable across folds -> return to DGCNN stability diagnosis"}
    print(f"\n=== Phase 3A-J PRIMARY read (folds1-8; exploratory; reviewer decides): {dmap[primary['decision']]} ===")
    print(f"  folds1-8: erm_adequate {primary['erm_adequate_folds']}/{primary['n_confirmation_folds']} "
          f"(need {primary['need_strong']}), erm_leakage {primary['erm_leakage_folds']} (need {primary['need_strong']}), "
          f"reg_reduces {primary['reg_reduces_folds']} (need {primary['need_majority']}), "
          f"source_retained {primary['source_retained_folds']} (need {primary['need_majority']})")
    print(f"  fold0(dev) pass={fold0_dev['fold_pass'] if fold0_dev else 'n/a'} (dev fold: NOT counted in primary)")
    print("  EDGE absent (static adjacency). target_eval evaluation-only guardrail; configs FIXED (no selection).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
