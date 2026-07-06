#!/usr/bin/env python
"""CIGL_69 MetaCMI gate — source-only LOSO on strong NON-graph EEG decoders (EEGNetMini anchor + EEGConformerMini
high-capacity arm), CMI/audit defined on feature_z (pre-classifier), not graph_z. Phase 1 = ERM audit preflight:
does Conformer beat EEGNetMini under strict source-only LOSO, and is its feature_z auditable (R3 + head-replay)?
Phase 2 adds metace / metacmi_direct (source-episodic).

Per (backbone, fold, method): metrics JSON + Pareto row + verified feature_z .audit.npz (head-replay) + firewall
metadata. Firewall: training on source_TRAIN only (enc_idx); the source hold-out pool (pool_idx) feeds source
metrics + leakage/R3 audit ONLY (fixed epochs, NO early stopping / selection); outer target eval-only; the
projector / meta objective (Phase 2) fit on source-train subjects only (meta_train subset).

    python scripts/run_metacmi_gate.py --dry_run_synthetic --device cpu --backbones eegnet conformer --methods erm --folds 0 --epochs 2 --probe_epochs 3 --n_perm 3
    python scripts/run_metacmi_gate.py --dataset BNCI2014_001 --device cuda --backbones eegnet conformer --methods erm --epochs 80 --probe_epochs 100 --n_perm 50
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from scripts.run_cigl_r2_seed0_gate import _load_all_folds, _atomic_dump                     # noqa: E402
from scripts.run_cigl_phase3a_backbone_sanity import _remap_contiguous, _git_commit_hash, _config_hash, _synthetic_fold  # noqa: E402
from cmi.eval.probe_splits import stratified_trial_split_by_y_d                              # noqa: E402
from cmi.models.sanity_backbones import build_sanity_backbone                                # noqa: E402
from cmi.eval.pareto_report import ROW_SCHEMA                                                # noqa: E402

OUT_DIR = REPO / "results" / "cigl" / "metacmi_gate"
PHASE = "CIGL_69_metacmi_gate"
BACKBONES = ["eegnet", "conformer"]
# (label, trainer method, beta[=fcigl_strength for metacmi_direct])
GATE_METHODS = {
    "erm":                  ("erm", 0.0),
    "metace":               ("metace", 0.0),
    "metacmi_direct_beta0.1": ("metacmi_direct", 0.1),
    "metacmi_direct_beta0.5": ("metacmi_direct", 0.5),
}


def _feat_leakage(net, X, y, d, n_cls, n_dom, device, args, seed):
    """feature_z label-conditional domain-leakage proxy (the 'graph' object on feature_z; node object is a dummy
    and ignored). Returns (feature_kl_mean, permutation_p, feature_z, node_dummy, logits)."""
    from cmi.eval.head_export import forward_feature_capture
    from cmi.eval.graph_leakage import audit_graph_node_objects
    logits, fz, nz = forward_feature_capture(net, X, device)
    tr_i, va_i, _ = stratified_trial_split_by_y_d(y, d, train_frac=args.train_frac, seed=seed, min_per_cell=args.min_per_cell)
    au = audit_graph_node_objects(fz, nz, y, d, n_cls, n_dom, n_perm=args.n_perm, seed=seed, device=device,
                                  epochs=args.probe_epochs, train_idx=tr_i, val_idx=va_i)
    g = au["graph"]
    return float(g["kl_mean"]), float(g["permutation_p"]), fz, nz, logits


def _train_eval_feature(backbone, label, method, beta, fold, seed, args, device):
    from cmi.train.trainer import train_model, predict
    from cmi.eval.metrics import classification_metrics
    from cmi.eval.head_export import save_fold_audit
    X, y, dom_all, tr_mask, te_mask, n_cls, heldout = fold
    Xs, ys = X[tr_mask], y[tr_mask]
    ds, n_dom = _remap_contiguous(dom_all[tr_mask])
    enc_idx, pool_idx, _ = stratified_trial_split_by_y_d(ys, ds, train_frac=args.enc_train_frac, seed=seed, min_per_cell=args.min_per_cell)
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    net = build_sanity_backbone(backbone, X.shape[1], X.shape[2], n_cls).to(device)
    net, _post, _diag = train_model(net, Xs[enc_idx], ys[enc_idx], ds[enc_idx], n_cls, method=method,
                                    lam=0.0, gamma=0.0, epochs=args.epochs, bs=args.bs,
                                    warmup=max(1, args.epochs // 5), device=device, seed=seed,
                                    fcigl_strength=float(beta), fcigl_k=int(args.fcigl_k),
                                    fcigl_update_every=int(args.fcigl_update_every), dcigl_gamma=float(args.dcigl_gamma),
                                    meta_rho=float(args.meta_rho), meta_train_frac=float(args.meta_train_frac))
    train_m = classification_metrics(predict(net, Xs[enc_idx], device), ys[enc_idx])
    src = classification_metrics(predict(net, Xs[pool_idx], device), ys[pool_idx])
    tgt = classification_metrics(predict(net, X[te_mask], device), y[te_mask])
    yp, dp = ys[pool_idx], ds[pool_idx]
    fkl, fp, _, _, _ = _feat_leakage(net, Xs[pool_idx], yp, dp, n_cls, n_dom, device, args, seed)
    # feature_z audit export (source pool + eval-only target under a distinct domain id)
    audit_dir = Path(args.audit_dir); audit_dir.mkdir(parents=True, exist_ok=True)
    ap = str(audit_dir / f"{args.dataset}_{backbone}_fold{args.fold}_sub{heldout}_{label}_seed{int(seed)}")
    _, replay_ok, mad = save_fold_audit(ap, model=net, X_source=Xs[pool_idx], y_source=yp, d_source=dp, device=device,
                                        fold=int(args.fold), seed=int(seed), target_subject=heldout,
                                        method=f"{backbone}:{label}", dataset=args.dataset,
                                        X_target=X[te_mask], y_target=y[te_mask], target_domain=int(n_dom),
                                        source_indices=np.arange(len(pool_idx)),
                                        target_indices=np.arange(len(pool_idx), len(pool_idx) + int(te_mask.sum())))
    return dict(backbone=backbone, gate_label=label, method=method, beta=beta, heldout_subject=str(heldout),
                fold_index=int(args.fold), n_classes=int(n_cls),
                train_bacc=train_m["balanced_acc"], source_bacc=src["balanced_acc"], target_bacc=tgt["balanced_acc"],
                feature_kl=fkl, feature_perm_p=fp, head_replay_ok=bool(replay_ok), head_replay_max_abs_diff=float(mad))


def _row(rec, dataset, seed):
    row = {k: None for k in ROW_SCHEMA}
    row.update(dataset=dataset, fold=int(rec["fold_index"]), target_subject=str(rec["heldout_subject"]), seed=int(seed),
               method=f"{rec['backbone']}:{rec['gate_label']}", lambda_g=0.0, lambda_node=0.0,
               source_bacc=float(rec["source_bacc"]), target_bacc=float(rec["target_bacc"]),
               graph_kl_proxy=float(rec["feature_kl"]), node_kl_proxy=float("nan"),
               graph_perm_p=float(rec["feature_perm_p"]), node_perm_p=float("nan"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run_synthetic", action="store_true")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--methods", nargs="+", default=["erm"])
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--probe_epochs", type=int, default=100); ap.add_argument("--n_perm", type=int, default=50)
    ap.add_argument("--train_frac", type=float, default=0.7); ap.add_argument("--enc_train_frac", type=float, default=0.7)
    ap.add_argument("--min_per_cell", type=int, default=2)
    ap.add_argument("--fcigl_k", type=int, default=2); ap.add_argument("--fcigl_update_every", type=int, default=10)
    ap.add_argument("--dcigl_gamma", type=float, default=0.5)
    ap.add_argument("--meta_rho", type=float, default=1.0)           # meta_heldout CE weight
    ap.add_argument("--meta_train_frac", type=float, default=0.7)    # fraction of source subjects used as meta_train
    ap.add_argument("--tmin", type=float, default=0.5); ap.add_argument("--tmax", type=float, default=3.5); ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--out_dir", default=None); ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[metacmi] --device cuda requested but CUDA unavailable (fail closed; no CPU fallback)")

    out = Path(args.out_dir) if args.out_dir else OUT_DIR
    dataset = "synthetic" if args.dry_run_synthetic else args.dataset
    args.audit_dir = str(out / dataset / "audit"); args.dataset = dataset
    (out / dataset).mkdir(parents=True, exist_ok=True)
    commit = _git_commit_hash(); cfg_hash = _config_hash(vars(args))
    all_folds = [_synthetic_fold(seed=fi) for fi in range(4)] if args.dry_run_synthetic else _load_all_folds(args)
    folds = args.folds if args.folds is not None else list(range(len(all_folds)))
    print(f"\n=== {PHASE} ({dataset}; {len(all_folds)} folds; seed={args.seed}; backbones={args.backbones}; methods={args.methods}) ===", flush=True)

    rows = []
    for f in folds:
        fold = all_folds[f]; args.fold = int(f)
        for bb in args.backbones:
            for label in args.methods:
                method, beta = GATE_METHODS[label]
                jp = out / dataset / f"{dataset}_{bb}_fold{f}_{label}_seed{args.seed}.json"
                if jp.exists() and not args.overwrite:
                    try:
                        rows.append(json.load(open(jp)).get("row")); print(f"  [skip] {jp.name}", flush=True); continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                print(f"  [run] fold{f} sub{fold[6]} {bb}:{label}", flush=True)
                rec = _train_eval_feature(bb, label, method, beta, fold, args.seed, args, args.device)
                row = _row(rec, dataset, args.seed); rec["row"] = row
                is_meta = method in ("metace", "metacmi_direct")
                rec["meta"] = dict(phase=PHASE, backbone=bb, setting="strict_source_only_DG", commit_hash=commit,
                                   config_hash=cfg_hash, used_target_labels_for_training=False,
                                   used_target_labels_for_selection=False, target_eval_is_evaluation_only=True,
                                   representation="feature_z", projector_source_train_only=True,
                                   # training uses source_TRAIN only (enc_idx); the source hold-out pool (pool_idx)
                                   # is used for source metrics + leakage/R3 audit ONLY (fixed-epoch training, NO
                                   # early stopping / model selection -> no data-dependent selection leakage).
                                   source_holdout_pool_used_for="source_metrics+leakage_R3_audit",
                                   early_stopping=False, fixed_epochs=int(args.epochs),
                                   is_meta_method=is_meta,
                                   meta_firewall=(dict(
                                       outer_target_eval_only=True,
                                       training_on_source_train_only=True,
                                       meta_split_over_source_train_subjects_only=True,
                                       meta_heldout_are_source_subjects_not_target=True,
                                       projector_fit_on_meta_train_subjects_only=True,
                                       projector_excludes_meta_heldout_pool_and_target=True,
                                       projector_detached_before_applied_to_meta_heldout=True,
                                       meta_rho=float(args.meta_rho), meta_train_frac=float(args.meta_train_frac),
                                       subspace_k=int(args.fcigl_k),
                                   ) if is_meta else None))
                _atomic_dump(rec, jp); rows.append(row)
                print(f"    src={row['source_bacc']:.3f} tgt={row['target_bacc']:.3f} featKL={row['graph_kl_proxy']:.3f} "
                      f"replay_ok={rec['head_replay_ok']}({rec['head_replay_max_abs_diff']:.1e})", flush=True)

    rows = [r for r in rows if r is not None]
    _atomic_dump({"schema": list(ROW_SCHEMA), "rows": rows, "commit": commit, "phase": PHASE, "dataset": dataset,
                  "seed": int(args.seed), "backbones": args.backbones, "methods": args.methods},
                 out / dataset / f"{dataset}_seed{args.seed}_metacmi_rows.json")
    print(f"\n[metacmi] wrote {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
