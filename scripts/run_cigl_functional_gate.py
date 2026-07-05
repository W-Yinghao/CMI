#!/usr/bin/env python
"""CIGL_67 functional-CMI seed0 gate — the 4 PM-approved functional variants on the SAME DGCNN adapter, full
LOSO, seed 0. Direct GPU verification of CIGL_66 Outcome A: penalize the intersection of the label-conditional
subject subspace with the task path, and test whether that reduces R3 reliance + head/subject alignment while
retaining task.

Methods (all via `_train_eval` on `dgcnn_forward_graph_adapter`; same backbone path as the frozen comparators):
    fcigl_align_eta0.01      method=fcigl_align        lg=ln=0.010  eta=0.01
    fcigl_align_eta0.05      method=fcigl_align        lg=ln=0.010  eta=0.05
    fcigl_removal_aug_a0.5   method=fcigl_removal_aug  lg=ln=0.010  alpha=0.5
    fcigl_removal_aug_a1.0   method=fcigl_removal_aug  lg=ln=0.010  alpha=1.0

Reference comparators (ERM, CIGL graph+node, DANN, cond-DANN, CDAN) are NOT rerun — use the frozen CIGL_65
tables. Per (fold, method) writes metrics JSON + Pareto row + verified head-replay .audit.npz + firewall meta,
so R3 task_drop + task_head_alignment are computable post-hoc (analyze_r2_multiseed / analyze_cigl_gap).

    python scripts/run_cigl_functional_gate.py --dry_run_synthetic --device cpu --folds 0 --epochs 2 --probe_epochs 3 --n_perm 3
    python scripts/run_cigl_functional_gate.py --dataset BNCI2014_001 --device cuda --epochs 80 --probe_epochs 100 --n_perm 50
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_cigl_phase3a_dgcnn_gn_regularizer_pilot import _train_eval, CANDIDATE   # noqa: E402
from scripts.run_cigl_r2_seed0_gate import (_load_all_folds, _pareto_row, _firewall_meta,   # noqa: E402
                                            _atomic_dump)
from scripts.run_cigl_phase3a_backbone_sanity import _git_commit_hash, _config_hash          # noqa: E402
from cmi.eval.pareto_report import ROW_SCHEMA                                                # noqa: E402

OUT_DIR = REPO / "results" / "cigl" / "functional_gate"
PHASE = "CIGL_67_functional_gate"
# (label, trainer method, lambda_g, lambda_node, fcigl_strength)
GATE_METHODS = [
    ("fcigl_align_eta0.01",    "fcigl_align",       0.010, 0.010, 0.01),
    ("fcigl_align_eta0.05",    "fcigl_align",       0.010, 0.010, 0.05),
    ("fcigl_removal_aug_a0.5", "fcigl_removal_aug", 0.010, 0.010, 0.50),
    ("fcigl_removal_aug_a1.0", "fcigl_removal_aug", 0.010, 0.010, 1.00),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run_synthetic", action="store_true")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
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
    ap.add_argument("--fcigl_k", type=int, default=2)                 # k=2 primary (CIGL_66/R3)
    ap.add_argument("--fcigl_update_every", type=int, default=10)     # refresh source-only projector every K epochs
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[fcigl-gate] --device cuda requested but CUDA unavailable (fail closed; no CPU fallback)")
    if args.device == "cpu":
        torch.set_num_threads(max(1, args.bs // 16))

    out = Path(args.out_dir) if args.out_dir else OUT_DIR
    dataset = "synthetic" if args.dry_run_synthetic else args.dataset
    audit_dir = out / dataset / "audit"
    (out / dataset).mkdir(parents=True, exist_ok=True); audit_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir = str(audit_dir); args.dataset = dataset
    commit = _git_commit_hash(); cfg_hash = _config_hash(vars(args))
    method_map = {m[0]: m for m in GATE_METHODS}
    run_methods = [method_map[m] for m in args.methods]

    all_folds = _load_all_folds(args)
    folds = args.folds if args.folds is not None else list(range(len(all_folds)))
    print(f"\n=== CIGL_67 functional gate ({dataset}; {len(all_folds)} LOSO folds; seed={args.seed}; "
          f"methods={[m[0] for m in run_methods]}; k={args.fcigl_k}; backbone={CANDIDATE}) ===", flush=True)

    rows = []
    for f in folds:
        if f >= len(all_folds):
            raise SystemExit(f"[fcigl-gate] fold {f} out of range ({len(all_folds)} LOSO folds)")
        fold = all_folds[f]; args.fold = int(f)
        for label, method, lam_g, lam_node, strength in run_methods:
            jpath = out / dataset / f"{dataset}_fold{f}_{label}_seed{args.seed}.json"
            if jpath.exists() and not args.overwrite:
                try:
                    rows.append(json.load(open(jpath)).get("pareto_row"))
                    print(f"  [skip] {jpath.name} exists", flush=True); continue
                except (json.JSONDecodeError, ValueError):
                    print(f"  [warn] {jpath.name} corrupt -> rerun", flush=True)
            print(f"  [run] fold{f} sub{fold[6]} {label} ({method}, lg={lam_g}, ln={lam_node}, s={strength})", flush=True)
            rec = _train_eval(label, lam_g, lam_node, fold, args.seed, args, args.device, args.n_perm,
                              method_override=method, fcigl_strength=strength)
            rec["fold_index"] = int(f); rec["gate_label"] = label; rec["fcigl_strength"] = strength
            row = _pareto_row(rec, dataset, args.seed); rec["pareto_row"] = row
            rec["meta"] = {**_firewall_meta(dataset, f, fold[6], commit, cfg_hash),
                           "phase": PHASE, "fcigl_strength": strength, "fcigl_k": args.fcigl_k,
                           "fcigl_update_every": args.fcigl_update_every,
                           "functional_projector": "source_only_label_conditional_subject_subspace_graph_z_detached"}
            _atomic_dump(rec, jpath); rows.append(row)
            print(f"    src={row['source_bacc']:.3f} tgt={row['target_bacc']:.3f} "
                  f"gKL={row['graph_kl_proxy']:.3f} nKL={row['node_kl_proxy']:.3f}", flush=True)

    rows = [r for r in rows if r is not None]
    rows_path = out / dataset / f"{dataset}_seed{args.seed}_functional_rows.json"
    _atomic_dump({"schema": list(ROW_SCHEMA), "rows": rows, "commit": commit, "config_hash": cfg_hash,
                  "phase": PHASE, "dataset": dataset, "seed": int(args.seed), "n_folds": len(all_folds),
                  "methods": [m[0] for m in run_methods], "fcigl_k": args.fcigl_k}, rows_path)
    print(f"\n[fcigl-gate] wrote {len(rows)} rows -> {rows_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
