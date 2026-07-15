#!/usr/bin/env python
"""CMI-Trace P0.1/P0.2 — same-backbone objective comparison + unified objective->effect audit.

Runs the CMI-Trace objective method set (ERM / encoder-CMI fixed-anchor / encoder-CMI nested / source-CORAL /
conditional-CORAL / IRMv1 / V-REx / conditional-adversarial) on the SAME static-adjacency DGCNN adapter under
the SAME strict source-only LOSO protocol, then computes the SAME objective->effect audit for every model:
moment gaps, graph/node encoder-CMI + null, per-domain risk variance, IRMv1 diagnostic, feature
norm/top-sv/effective-rank, exact-head reliance R_rel(k=2), and same-rank random-subspace control.

Reuses the proven `_train_eval` (verified head-replay `.audit.npz` + firewall metadata) and reads the audit
sidecar with `cmi.eval.objective_effect_report` — artifact-driven, NEVER retrains for the audit.

Firewall (strict): target labels/covariates never touch training, nested selection, probe fitting, subspace
fitting, or k. `select: True` rows use SOURCE-ONLY nested leave-one-source-domain validation over the frozen
grid in configs/cmi_trace_p0p1.yaml (lambda=0=ERM always a candidate). Target enters only final scoring.

Smoke (CPU, synthetic):
  python scripts/run_cmi_trace_objective_comparison.py --dry_run_synthetic --device cpu \
      --folds 0 1 --seeds 0 --epochs 2 --probe_epochs 3 --n_perm 3 --select_epochs 2 --select_inner_folds 1 \
      --methods erm coral label_coral irm vrex cigl_graph_node cigl_nested
Full (GPU):
  python scripts/run_cmi_trace_objective_comparison.py --dataset BNCI2014_001 --device cuda \
      --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 50
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_cigl_phase3a_dgcnn_gn_regularizer_pilot import _train_eval, CANDIDATE   # noqa: E402
from scripts.run_cigl_phase3a_backbone_sanity import (_synthetic_fold, _remap_contiguous,   # noqa: E402
                                                      _git_commit_hash, _config_hash)
from cmi.eval.baseline_registry import OBJECTIVE_METHODS, BACKBONE, SAME_BACKBONE_CONTRACT   # noqa: E402
from cmi.eval.objective_effect_report import objective_effect_row                            # noqa: E402

OUT_DIR = REPO / "results" / "cmi_trace_p0p1" / "objective_comparison"

# Frozen nested-selection grids (mirror configs/cmi_trace_p0p1.yaml: selection.grids). lambda=0=ERM always in.
SELECT_GRIDS = {
    "cigl_nested":        {"method": "graphcmi",    "tied": True,  "lams": [0.0, 0.003, 0.010, 0.030]},
    "coral_nested":       {"method": "coral",       "tied": True,  "lams": [0.0, 0.010, 0.100, 1.000]},
    "label_coral_nested": {"method": "label_coral", "tied": True,  "lams": [0.0, 0.010, 0.100, 1.000]},
    "irm_nested":         {"method": "irm",         "tied": False, "lams": [0.0, 0.100, 1.000, 10.000]},
    "vrex_nested":        {"method": "vrex",        "tied": False, "lams": [0.0, 0.100, 1.000, 10.000]},
}


def _atomic_dump(obj, path):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def _parse_fixed(config):
    """Fixed-config row -> (method, lam_g, lam_node). erm:0 -> (erm,0,0); graphcmi:lg:ln:le; coral/lc:lg:ln;
    irm/vrex/cdann/cdan:lam (lam on lam_g, lam_node 0)."""
    parts = config.split(":"); method = parts[0]
    nums = [float(x) for x in parts[1:]] if len(parts) > 1 else [0.0]
    if method == "graphcmi":
        return "graphcmi", nums[0], (nums[1] if len(nums) > 1 else 0.0)
    if method in ("coral", "label_coral"):
        return method, nums[0], (nums[1] if len(nums) > 1 else 0.0)
    return method, (nums[0] if nums else 0.0), 0.0


def _inner_source_select(grid, fold, seed, args, device):
    """SOURCE-ONLY nested leave-one-source-domain selection. For each candidate lambda, hold out inner source
    domains (up to args.select_inner_folds), train on the rest of source with a shorter budget, score inner
    source-val balanced accuracy + CE; pick best by (bAcc desc, CE asc, |lambda| asc). NO target labels."""
    from cmi.models.graph_task_backbones import build_graph_task_backbone
    from cmi.train.trainer import train_model, predict
    from cmi.eval.metrics import classification_metrics
    from sklearn.metrics import balanced_accuracy_score, log_loss

    X, y, dom_all, tr_mask, te_mask, n_cls, heldout = fold
    Xs, ys = X[tr_mask], y[tr_mask]
    ds, n_dom = _remap_contiguous(dom_all[tr_mask])
    method, tied, lams = grid["method"], grid["tied"], grid["lams"]
    inner_doms = list(np.unique(ds))
    rng = np.random.default_rng(20000 + seed)
    rng.shuffle(inner_doms)
    inner_doms = inner_doms[: max(1, args.select_inner_folds)]
    sel_epochs = max(1, args.select_epochs or args.epochs // 2)

    scored = []
    for lam in lams:
        baccs, ces = [], []
        for vd in inner_doms:
            vmask = ds == vd
            if vmask.sum() == 0 or (~vmask).sum() == 0 or len(np.unique(ds[~vmask])) < 2:
                continue
            Xtr, ytr, dtr = Xs[~vmask], ys[~vmask], ds[~vmask]
            dtr, _ = _remap_contiguous(dtr)
            torch.manual_seed(int(seed)); np.random.seed(int(seed))
            net = build_graph_task_backbone(CANDIDATE, X.shape[1], X.shape[2], n_cls).to(device)
            m = "erm" if lam == 0.0 else method
            lg = lam
            ln = lam if (tied and m in ("graphcmi", "coral", "label_coral")) else 0.0
            net, _, _ = train_model(net, Xtr, ytr, dtr, n_cls, method=m, lam=lg, gamma=ln,
                                    epochs=sel_epochs, bs=args.bs, warmup=max(1, sel_epochs // 5),
                                    device=device, seed=seed)
            prob = predict(net, Xs[vmask], device)
            baccs.append(balanced_accuracy_score(ys[vmask], prob.argmax(1)))
            try:
                ces.append(log_loss(ys[vmask], prob, labels=list(range(n_cls))))
            except ValueError:
                ces.append(float("nan"))
        if baccs:
            scored.append(dict(lam=float(lam), bacc=float(np.mean(baccs)), ce=float(np.nanmean(ces))))
    if not scored:                                           # degenerate (too few source domains) -> ERM
        return 0.0, [{"lam": 0.0, "bacc": float("nan"), "ce": float("nan")}]
    # tie-break: higher bAcc, then lower CE, then smaller |lambda|
    best = min(scored, key=lambda s: (-s["bacc"], s["ce"] if np.isfinite(s["ce"]) else 1e9, abs(s["lam"])))
    return best["lam"], scored


def _load_all_folds(args):
    if args.dry_run_synthetic:
        return [_synthetic_fold(seed=fi) for fi in range(4)]
    from cmi.data import moabb_data
    X, y, meta, classes = moabb_data.load(args.dataset, tmin=args.tmin, tmax=args.tmax, resample=args.resample)
    dom_all, _ = moabb_data.domain_labels(meta, "subject")
    return [(X, y, dom_all, tr_mask, te_mask, len(classes), str(tgt))
            for tgt, tr_mask, te_mask in moabb_data.loso_splits(meta)]


def _firewall_meta(dataset, fold_index, heldout, commit, cfg_hash, selected_lambda, selected_via):
    return dict(phase="CMI_TRACE_P0P1_objective_comparison", backbone=BACKBONE,
                same_backbone_builder=SAME_BACKBONE_CONTRACT["builder"],
                target_firewall=SAME_BACKBONE_CONTRACT["target_firewall"],
                audit_objects=list(SAME_BACKBONE_CONTRACT["audit_objects"]), null=SAME_BACKBONE_CONTRACT["null"],
                setting="strict_source_only_DG", model_selection=selected_via,
                used_target_labels_for_training=False, used_target_labels_for_selection=False,
                used_target_labels_for_probe_fit=False, used_target_labels_for_subspace_fit=False,
                used_target_covariates=False, target_eval_is_evaluation_only=True,
                selected_lambda=selected_lambda, dataset=dataset, fold_index=int(fold_index),
                heldout_subject=str(heldout), commit_hash=commit, config_hash=cfg_hash)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run_synthetic", action="store_true")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=list(OBJECTIVE_METHODS.keys()))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--probe_epochs", type=int, default=100)
    ap.add_argument("--n_perm", type=int, default=50)
    ap.add_argument("--gate_alpha", type=float, default=0.05)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--enc_train_frac", type=float, default=0.7)
    ap.add_argument("--min_per_cell", type=int, default=2)
    ap.add_argument("--graph_usage_min_drop", type=float, default=0.10)
    ap.add_argument("--select_epochs", type=int, default=0, help="0 -> epochs//2 for nested selection")
    ap.add_argument("--select_inner_folds", type=int, default=3, help="inner source-domain holdouts per candidate")
    ap.add_argument("--primary_k", type=int, default=2)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[obj-cmp] --device cuda requested but CUDA unavailable (fail closed)")
    if args.device == "cpu":
        torch.set_num_threads(max(1, args.bs // 16))

    out = Path(args.out_dir) if args.out_dir else OUT_DIR
    dataset = "synthetic" if args.dry_run_synthetic else args.dataset
    audit_dir = out / dataset / "audit"
    (out / dataset).mkdir(parents=True, exist_ok=True); audit_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir = str(audit_dir); args.dataset = dataset
    commit = _git_commit_hash(); cfg_hash = _config_hash(vars(args))

    all_folds = _load_all_folds(args)
    folds = args.folds if args.folds is not None else list(range(len(all_folds)))
    methods = [m for m in args.methods if m in OBJECTIVE_METHODS]
    print(f"\n=== CMI-Trace objective comparison ({dataset}; {len(all_folds)} folds; seeds={args.seeds}; "
          f"methods={methods}; backbone={CANDIDATE}) ===", flush=True)

    raw_rows_path = out / dataset / "raw_rows.jsonl"
    rows_csv_path = out / dataset / "raw_rows.csv"
    rows = []
    t0 = time.time()
    for f in folds:
        fold = all_folds[f]; args.fold = int(f); heldout = fold[6]
        for seed in args.seeds:
            for label in methods:
                spec = OBJECTIVE_METHODS[label]
                jpath = out / dataset / f"{dataset}_fold{f}_{label}_seed{seed}.json"
                if jpath.exists() and not args.overwrite:
                    try:
                        prev = json.load(open(jpath)).get("row")
                        if prev:
                            rows.append(prev); print(f"  [skip] {jpath.name}", flush=True); continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                # resolve method + lambda (nested source-only selection for select rows)
                selected_via = "fixed_epochs_plus_warmup"; sel_scored = None
                if spec["select"]:
                    grid = SELECT_GRIDS[label]
                    lam, sel_scored = _inner_source_select(grid, fold, seed, args, args.device)
                    method, lam_g = grid["method"], lam
                    lam_node = lam if (grid["tied"] and method in ("graphcmi", "coral", "label_coral")) else 0.0
                    if lam == 0.0:
                        method = "erm"
                    selected_via = "nested_source_domain_val_bacc"
                else:
                    method, lam_g, lam_node = _parse_fixed(spec["config"])
                    lam = lam_g
                print(f"  [run] fold{f} sub{heldout} {label} ({method}, lg={lam_g}, ln={lam_node}, "
                      f"seed={seed}, sel={selected_via})", flush=True)
                rec = _train_eval(label, lam_g, lam_node, fold, seed, args, args.device, args.n_perm,
                                  method_override=method)
                # objective->effect audit row from the verified .audit.npz
                npz = str(audit_dir / f"{dataset}_fold{f}_sub{heldout}_{label}_seed{seed}.audit.npz")
                leakage = {"graph_kl": rec["leakage"]["graph"]["kl_mean"],
                           "graph_null": rec["leakage"]["graph"]["permutation_mean"],
                           "graph_perm_p": rec["leakage"]["graph"]["permutation_p"],
                           "node_kl": rec["leakage"]["node"]["kl_mean"],
                           "node_null": rec["leakage"]["node"]["permutation_mean"],
                           "node_perm_p": rec["leakage"]["node"]["permutation_p"]}
                row = objective_effect_row(npz, leakage=leakage, primary_k=args.primary_k, seed=seed)
                row.update(method=label, trainer_method=method, dataset=dataset, fold=int(f), seed=int(seed),
                           lambda_g=float(lam_g), lambda_node=float(lam_node), selected_lambda=float(lam),
                           family=spec["family"], select_row=bool(spec["select"]),
                           target_bacc=rec["target_eval"]["balanced_acc"],
                           source_bacc=rec["source_probe"]["balanced_acc"])
                meta = _firewall_meta(dataset, f, heldout, commit, cfg_hash, float(lam), selected_via)
                _atomic_dump({"row": row, "meta": meta, "selection_scored": sel_scored,
                              "leakage_block": rec["leakage"]}, jpath)
                rows.append(row)
                print(f"    tgtBAcc={row['target_bacc']:.3f} srcBAcc={row['source_bacc']:.3f} "
                      f"gKL={row['graph_kl']:.3f} R_rel_k2={row['R_rel_k2']:.3f} "
                      f"rand={row['R_rel_k2_random_control']:.3f} margGap={row['marginal_moment_gap']:.3f} "
                      f"ccGap={row['class_conditional_moment_gap']:.3f} ({time.time()-t0:.0f}s)", flush=True)
                # incremental persistence
                with open(raw_rows_path, "w") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")

    # CSV mirror
    if rows:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(rows_csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\n[obj-cmp] wrote {len(rows)} rows -> {raw_rows_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
