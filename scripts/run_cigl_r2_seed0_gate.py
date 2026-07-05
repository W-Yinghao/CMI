#!/usr/bin/env python
"""CIGL R2 seed0 gate — the FIVE PM-approved methods on the SAME static-adjacency DGCNN adapter, full LOSO,
seed 0 only. Scientific SCREENING (not method-level judgment; that needs seeds 0/1/2).

Methods (all via `_train_eval` on `dgcnn_forward_graph_adapter` -> identical backbone path, satisfying gate
stop-condition #5):
    erm              method=erm      lg=0.000 ln=0.000
    cigl_graph_node  method=graphcmi lg=0.010 ln=0.010   (pre-registered fixed strength; NO lambda sweep)
    dann             method=dann     lam=1.0
    cond_dann        method=cdann    lam=1.0
    cdan             method=cdan     lam=1.0

Per (fold, method) writes: metrics JSON, a Pareto row (cmi.eval.pareto_report.ROW_SCHEMA), a verified
head-replay .audit.npz (via _train_eval --audit_dir), and firewall metadata. Resumable: an existing metrics
JSON is skipped unless --overwrite. Strict source-only firewall: target labels/covariates never touch training,
model selection, probe fitting, subspace fitting, or k; target_eval is evaluation-only.

    python scripts/run_cigl_r2_seed0_gate.py --dry_run_synthetic --device cpu --folds 0 1 --epochs 2 --probe_epochs 3 --n_perm 3
    python scripts/run_cigl_r2_seed0_gate.py --dataset BNCI2014_001 --device cuda --epochs 80 --probe_epochs 100 --n_perm 50
    python scripts/run_cigl_r2_seed0_gate.py --dataset BNCI2015_001 --device cuda --epochs 80 --probe_epochs 100 --n_perm 50
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path


def _atomic_dump(obj, path):
    """Write JSON to a temp sibling then os.replace() into place, so a preemption/OOM mid-write can never leave a
    truncated checkpoint that poisons every subsequent resume."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_cigl_phase3a_dgcnn_gn_regularizer_pilot import _train_eval, CANDIDATE   # noqa: E402
from scripts.run_cigl_phase3a_backbone_sanity import _synthetic_fold, _git_commit_hash, _config_hash  # noqa: E402
from cmi.eval.baseline_registry import BACKBONE, SAME_BACKBONE_CONTRACT                  # noqa: E402
from cmi.eval.pareto_report import ROW_SCHEMA                                            # noqa: E402

OUT_DIR = REPO / "results" / "cigl" / "r2_seed0_gate"
PHASE = "R2_seed0_gate"
# (label, trainer method, lambda_g / adv-strength, lambda_node)
GATE_METHODS = [
    ("erm",             "erm",      0.000, 0.000),
    ("cigl_graph_node", "graphcmi", 0.010, 0.010),
    ("dann",            "dann",     1.000, 0.000),
    ("cond_dann",       "cdann",    1.000, 0.000),
    ("cdan",            "cdan",     1.000, 0.000),
]


def _load_all_folds(args):
    """Load the dataset ONCE; return the list of LOSO folds as (X, y, dom_all, tr_mask, te_mask, n_cls, heldout)."""
    if args.dry_run_synthetic:
        return [_synthetic_fold(seed=fi) for fi in range(4)]              # 4 synthetic subjects -> 4 folds
    from cmi.data import moabb_data
    X, y, meta, classes = moabb_data.load(args.dataset, tmin=args.tmin, tmax=args.tmax, resample=args.resample)
    dom_all, _ = moabb_data.domain_labels(meta, "subject")
    return [(X, y, dom_all, tr_mask, te_mask, len(classes), str(tgt))
            for tgt, tr_mask, te_mask in moabb_data.loso_splits(meta)]


def _pareto_row(rec, dataset, seed):
    g, n = rec["leakage"]["graph"], rec["leakage"]["node"]
    row = {k: None for k in ROW_SCHEMA}
    row.update(dataset=dataset, fold=int(rec.get("fold_index", -1)), target_subject=str(rec["heldout_subject"]),
               seed=int(seed), method=rec["config"], lambda_g=float(rec["lambda_g"]), lambda_node=float(rec["lambda_node"]),
               source_bacc=float(rec["source_probe"]["balanced_acc"]),
               target_bacc=float(rec["target_eval"]["balanced_acc"]),
               graph_kl_proxy=float(g["kl_mean"]), node_kl_proxy=float(n["kl_mean"]),
               graph_perm_p=float(g["permutation_p"]), node_perm_p=float(n["permutation_p"]))
    # graph_fdr_q / node_fdr_q / multiprobe_detect_count / task_retention_delta / leakage_reduction_delta
    # are cross-method/cross-fold aggregates -> filled by the post-gate Pareto+R1 analysis, not per-row.
    return row


def _firewall_meta(dataset, fold_index, heldout, commit, cfg_hash):
    # HONEST protocol record: this proven audit line trains FIXED epochs + warmup (no early stopping) — the
    # same protocol the PM accepted for the confirmation runs. The source-probe pool (pool_idx) is held out of
    # ENCODER training but is NOT used for epoch selection; there is no target-based selection anywhere.
    return dict(phase=PHASE, backbone=BACKBONE,
                same_backbone_builder=SAME_BACKBONE_CONTRACT["builder"],
                target_firewall=SAME_BACKBONE_CONTRACT["target_firewall"],
                audit_objects=SAME_BACKBONE_CONTRACT["audit_objects"], null=SAME_BACKBONE_CONTRACT["null"],
                setting="strict_source_only_DG", scientific_level="seed0_screening_not_method_judgment",
                model_selection="fixed_epochs_plus_warmup", early_stopping_implemented=False,
                used_target_labels_for_training=False, used_target_labels_for_selection=False,
                used_target_labels_for_probe_fit=False, used_target_labels_for_subspace_fit=False,
                used_target_covariates=False, target_eval_is_evaluation_only=True,
                dataset=dataset, fold_index=int(fold_index), heldout_subject=str(heldout),
                commit_hash=commit, config_hash=cfg_hash)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run_synthetic", action="store_true")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="default: all LOSO folds")
    ap.add_argument("--seed", type=int, default=0)               # seed0 gate: single seed
    ap.add_argument("--methods", nargs="+", default=[m[0] for m in GATE_METHODS])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--probe_epochs", type=int, default=100)
    ap.add_argument("--n_perm", type=int, default=50)
    ap.add_argument("--gate_alpha", type=float, default=0.05)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--enc_train_frac", type=float, default=0.7)
    ap.add_argument("--min_per_cell", type=int, default=2)
    ap.add_argument("--success_bacc_floor", type=float, default=0.45)
    ap.add_argument("--graph_usage_min_drop", type=float, default=0.10)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[r2-gate] --device cuda requested but CUDA unavailable (fail closed; no CPU fallback)")
    if args.device == "cpu":
        torch.set_num_threads(max(1, args.bs // 16))

    out = Path(args.out_dir) if args.out_dir else OUT_DIR
    dataset = "synthetic" if args.dry_run_synthetic else args.dataset
    audit_dir = out / dataset / "audit"
    (out / dataset).mkdir(parents=True, exist_ok=True); audit_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir = str(audit_dir)                              # -> _train_eval writes verified .audit.npz
    args.dataset = dataset
    commit = _git_commit_hash(); cfg_hash = _config_hash(vars(args))
    method_map = {m[0]: m for m in GATE_METHODS}
    run_methods = [method_map[m] for m in args.methods]

    all_folds = _load_all_folds(args)
    folds = args.folds if args.folds is not None else list(range(len(all_folds)))
    print(f"\n=== R2 seed0 gate ({dataset}; {len(all_folds)} LOSO folds; seed={args.seed}; "
          f"methods={[m[0] for m in run_methods]}; backbone={CANDIDATE}) ===", flush=True)

    rows = []
    for f in folds:
        if f >= len(all_folds):
            raise SystemExit(f"[r2-gate] fold {f} out of range ({len(all_folds)} LOSO folds)")
        fold = all_folds[f]
        args.fold = int(f)                                      # _train_eval reads args.fold for meta + audit name
        for label, method, lam_g, lam_node in run_methods:
            jpath = out / dataset / f"{dataset}_fold{f}_{label}_seed{args.seed}.json"
            if jpath.exists() and not args.overwrite:
                try:                                            # a truncated checkpoint (preempt mid-write) -> rerun
                    prev = json.load(open(jpath)).get("pareto_row")
                    print(f"  [skip] {jpath.name} exists", flush=True)
                    rows.append(prev)
                    continue
                except (json.JSONDecodeError, ValueError):
                    print(f"  [warn] {jpath.name} corrupt -> rerun", flush=True)
            print(f"  [run] fold{f} sub{fold[6]} {label} ({method}, lg={lam_g}, ln={lam_node})", flush=True)
            rec = _train_eval(label, lam_g, lam_node, fold, args.seed, args, args.device, args.n_perm,
                              method_override=method)
            rec["fold_index"] = int(f); rec["gate_label"] = label
            row = _pareto_row(rec, dataset, args.seed)
            rec["pareto_row"] = row
            rec["meta"] = _firewall_meta(dataset, f, fold[6], commit, cfg_hash)
            _atomic_dump(rec, jpath)                             # temp + os.replace: preempt can't truncate a checkpoint
            rows.append(row)
            print(f"    src={row['source_bacc']:.3f} tgt={row['target_bacc']:.3f} "
                  f"gKL={row['graph_kl_proxy']:.3f}(p={row['graph_perm_p']:.3g}) "
                  f"nKL={row['node_kl_proxy']:.3f}(p={row['node_perm_p']:.3g})", flush=True)

    rows = [r for r in rows if r is not None]
    rows_path = out / dataset / f"{dataset}_seed{args.seed}_pareto_rows.json"
    _atomic_dump({"schema": list(ROW_SCHEMA), "rows": rows, "commit": commit, "config_hash": cfg_hash,
                  "phase": PHASE, "dataset": dataset, "seed": int(args.seed),
                  "n_folds": len(all_folds), "methods": [m[0] for m in run_methods]}, rows_path)
    print(f"\n[r2-gate] wrote {len(rows)} rows -> {rows_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
