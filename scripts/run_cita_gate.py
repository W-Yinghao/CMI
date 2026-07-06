#!/usr/bin/env python
"""CITA_01 gate — target-unlabeled offline transductive CMI adaptation on EEGNetMini + EEGConformerMini
(internal/faithful-minimal/audit-compatible, NOT official). NEW information regime vs the closed source-only CMI
line (docs/CIGL_70): the held-out LOSO target subject's UNLABELED X is used during an offline adaptation phase;
target y is FORBIDDEN everywhere except the final reported metric.

Per (fold, backbone): train ONE source-ERM model M0, then adapt a fresh COPY of M0 for each method
{erm_no_adapt, tta_control, cita_cmi}. So ERM is reused within-fold and all three share the same M0 -> the
attribution ERM->TTA_Control->CITA_CMI is clean. Both backbones keep a single linear head -> R3 head-replay is
exact (classifier-level). Emits per method: metrics JSON + feature_z .audit.npz + strict target-y firewall
metadata.

    python scripts/run_cita_gate.py --dry_run_synthetic --device cpu --backbones eegnet conformer \
        --methods erm_no_adapt tta_control cita_cmi --folds 0 --epochs 4 --adapt_steps 6 --probe_epochs 3 --n_perm 3
    python scripts/run_cita_gate.py --dataset BNCI2014_001 --device cuda --backbones eegnet conformer \
        --methods erm_no_adapt tta_control cita_cmi --epochs 80 --adapt_steps 50 --probe_epochs 100 --n_perm 50
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from scripts.run_cigl_r2_seed0_gate import _load_all_folds, _atomic_dump                     # noqa: E402
from scripts.run_cigl_phase3a_backbone_sanity import _remap_contiguous, _git_commit_hash, _config_hash, _synthetic_fold  # noqa: E402
from cmi.eval.probe_splits import stratified_trial_split_by_y_d                              # noqa: E402
from cmi.models.sanity_backbones import build_sanity_backbone                                # noqa: E402
from cmi.eval.pareto_report import ROW_SCHEMA                                                # noqa: E402
from cmi.adaptation.cita import adapt, CITA_METHODS                                          # noqa: E402

OUT_DIR = REPO / "results" / "cita" / "gate"
PHASE = "CITA_01_target_unlabeled_gate"
BACKBONES = ["eegnet", "conformer"]


def _feat_leakage(net, X, y, d, n_cls, n_dom, device, args, seed):
    from cmi.eval.head_export import forward_feature_capture
    from cmi.eval.graph_leakage import audit_graph_node_objects
    logits, fz, nz = forward_feature_capture(net, X, device)
    tr_i, va_i, _ = stratified_trial_split_by_y_d(y, d, train_frac=args.train_frac, seed=seed, min_per_cell=args.min_per_cell)
    au = audit_graph_node_objects(fz, nz, y, d, n_cls, n_dom, n_perm=args.n_perm, seed=seed, device=device,
                                  epochs=args.probe_epochs, train_idx=tr_i, val_idx=va_i)
    g = au["graph"]
    return float(g["kl_mean"]), float(g["permutation_p"])


def _run_fold(backbone, fold, seed, args, device):
    from cmi.train.trainer import train_model, predict
    from cmi.eval.metrics import classification_metrics
    from cmi.eval.head_export import save_fold_audit
    X, y, dom_all, tr_mask, te_mask, n_cls, heldout = fold
    Xs, ys = X[tr_mask], y[tr_mask]
    ds, n_dom = _remap_contiguous(dom_all[tr_mask])
    enc_idx, pool_idx, _ = stratified_trial_split_by_y_d(ys, ds, train_frac=args.enc_train_frac, seed=seed, min_per_cell=args.min_per_cell)
    Xt = X[te_mask]                                                # TARGET X (unlabeled during adaptation)
    # ---- source-ERM model M0 (trained once; source_train only) ----
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    m0 = build_sanity_backbone(backbone, X.shape[1], X.shape[2], n_cls).to(device)
    m0, _post, _diag = train_model(m0, Xs[enc_idx], ys[enc_idx], ds[enc_idx], n_cls, method="erm",
                                   lam=0.0, gamma=0.0, epochs=args.epochs, bs=args.bs,
                                   warmup=max(1, args.epochs // 5), device=device, seed=seed)
    recs = []
    for method in args.methods:
        model = copy.deepcopy(m0)                                  # every method starts from the SAME M0
        model, adiag = adapt(model, Xs[enc_idx], ys[enc_idx], Xt, method, steps=args.adapt_steps, bs=args.bs,
                             lr=args.adapt_lr, tau=args.tau, mu=args.mu, lam_cita=args.lam_cita,
                             cond_inner=args.cond_inner, device=device, seed=seed)
        tgt = classification_metrics(predict(model, Xt, device), y[te_mask])         # y[te_mask] used ONLY here
        src = classification_metrics(predict(model, Xs[pool_idx], device), ys[pool_idx])
        yp, dp = ys[pool_idx], ds[pool_idx]
        fkl, fp = _feat_leakage(model, Xs[pool_idx], yp, dp, n_cls, n_dom, device, args, seed)
        audit_dir = Path(args.audit_dir); audit_dir.mkdir(parents=True, exist_ok=True)
        ap = str(audit_dir / f"{args.dataset}_{backbone}_fold{args.fold}_sub{heldout}_{method}_seed{int(seed)}")
        _, replay_ok, mad = save_fold_audit(ap, model=model, X_source=Xs[pool_idx], y_source=yp, d_source=dp,
                                            device=device, fold=int(args.fold), seed=int(seed), target_subject=heldout,
                                            method=f"{backbone}:{method}", dataset=args.dataset,
                                            X_target=Xt, y_target=y[te_mask], target_domain=int(n_dom),
                                            source_indices=np.arange(len(pool_idx)),
                                            target_indices=np.arange(len(pool_idx), len(pool_idx) + int(te_mask.sum())))
        recs.append(dict(backbone=backbone, method=method, heldout_subject=str(heldout), fold_index=int(args.fold),
                         n_classes=int(n_cls), source_bacc=src["balanced_acc"], target_bacc=tgt["balanced_acc"],
                         feature_kl=fkl, feature_perm_p=fp, head_replay_ok=bool(replay_ok),
                         head_replay_max_abs_diff=float(mad), adapt_diag=adiag))
    # STOP-CONDITION: TTA-Control and CITA-CMI must share the EXACT adaptation budget (except lambda_cita) and the
    # EXACT model-mode policy, else the CITA-vs-TTA attribution is polluted. Fail closed if they diverge.
    by_m = {r["method"]: r["adapt_diag"] for r in recs}
    if "tta_control" in by_m and "cita_cmi" in by_m:
        bt, bc = dict(by_m["tta_control"]["budget"]), dict(by_m["cita_cmi"]["budget"])
        bt.pop("lambda_cita"); bc.pop("lambda_cita")               # lambda is the ONLY allowed difference
        if bt != bc:
            raise RuntimeError(f"CITA firewall: TTA vs CITA adaptation budget differs (attribution polluted): {bt} != {bc}")
        if by_m["tta_control"]["mode_policy"] != by_m["cita_cmi"]["mode_policy"]:
            raise RuntimeError(f"CITA firewall: TTA vs CITA model-mode policy differs: "
                               f"{by_m['tta_control']['mode_policy']} != {by_m['cita_cmi']['mode_policy']}")
    return recs


def _row(rec, dataset, seed):
    row = {k: None for k in ROW_SCHEMA}
    row.update(dataset=dataset, fold=int(rec["fold_index"]), target_subject=str(rec["heldout_subject"]), seed=int(seed),
               method=f"{rec['backbone']}:{rec['method']}", lambda_g=0.0, lambda_node=0.0,
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
    ap.add_argument("--methods", nargs="+", default=list(CITA_METHODS))
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--adapt_steps", type=int, default=50); ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=1.0); ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--lam_cita", type=float, default=0.010); ap.add_argument("--cond_inner", type=int, default=1)
    ap.add_argument("--probe_epochs", type=int, default=100); ap.add_argument("--n_perm", type=int, default=50)
    ap.add_argument("--train_frac", type=float, default=0.7); ap.add_argument("--enc_train_frac", type=float, default=0.7)
    ap.add_argument("--min_per_cell", type=int, default=2)
    ap.add_argument("--tmin", type=float, default=0.5); ap.add_argument("--tmax", type=float, default=3.5); ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--out_dir", default=None); ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[cita] --device cuda requested but CUDA unavailable (fail closed; no CPU fallback)")
    for m in args.methods:
        if m not in CITA_METHODS:
            raise SystemExit(f"[cita] unknown method '{m}' (allowed: {CITA_METHODS})")

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
            # skip only if ALL requested methods for this (fold,bb) already exist
            jps = {m: out / dataset / f"{dataset}_{bb}_fold{f}_{m}_seed{args.seed}.json" for m in args.methods}
            if all(p.exists() for p in jps.values()) and not args.overwrite:
                for m, p in jps.items():
                    try:
                        rows.append(json.load(open(p)).get("row")); print(f"  [skip] {p.name}", flush=True)
                    except (json.JSONDecodeError, ValueError):
                        pass
                continue
            print(f"  [run] fold{f} sub{fold[6]} {bb}: {args.methods}", flush=True)
            for rec in _run_fold(bb, fold, args.seed, args, args.device):
                m = rec["method"]; row = _row(rec, dataset, args.seed); rec["row"] = row
                ad = rec["adapt_diag"]
                rec["meta"] = dict(phase=PHASE, backbone=bb, method=m,
                                   setting="offline_transductive_target_unlabeled",
                                   commit_hash=commit, config_hash=cfg_hash, representation="feature_z",
                                   cita_firewall=dict(
                                       target_X_allowed=True,
                                       target_y_used_for_training=False, target_y_used_for_adaptation=False,
                                       target_y_used_for_model_selection=False,
                                       target_y_used_for_hyperparameter_selection=False,
                                       adaptation_uses_target_X=bool(ad["adaptation_uses_all_target_X"]),
                                       target_eval_same_X_as_adapt=True, early_stopping=False,
                                       source_replay_uses_source_labels_only=True,
                                       cond_domain_active=bool(ad["cond_domain_active"])),
                                   adaptation_budget=ad["budget"], model_mode_policy=ad["mode_policy"],
                                   shared_budget_and_mode_across_tta_and_cita=True,
                                   adaptation_diagnostics=dict(
                                       source_replay_ce_before=ad.get("source_replay_ce_before"),
                                       source_replay_ce_after=ad.get("source_replay_ce_after"),
                                       target_entropy_before=ad.get("target_entropy_before"),
                                       target_entropy_after=ad.get("target_entropy_after"),
                                       target_label_balance_before=ad.get("target_label_balance_before"),
                                       target_label_balance_after=ad.get("target_label_balance_after"),
                                       final_source_ce=ad.get("final_source_ce"),
                                       final_target_entropy=ad.get("final_target_entropy"),
                                       final_label_balance_kl=ad.get("final_label_balance_kl"),
                                       final_cond_domain=ad.get("final_cond_domain"),
                                       lambda_times_cond_domain=ad.get("lambda_times_cond_domain"),
                                       final_total_loss=ad.get("final_total_loss"),
                                       cond_domain_fraction_of_total_loss=ad.get("cond_domain_fraction_of_total_loss"),
                                       cond_domain_gradient_norm=ad.get("cond_domain_gradient_norm"),
                                       lambda_cita_used=ad.get("lambda_cita_used")))
                _atomic_dump(rec, jps[m]); rows.append(row)
                print(f"    {m:14s} src={row['source_bacc']:.3f} tgt={row['target_bacc']:.3f} "
                      f"featKL={row['graph_kl_proxy']:.3f} replay_ok={rec['head_replay_ok']}", flush=True)

    rows = [r for r in rows if r is not None]
    _atomic_dump({"schema": list(ROW_SCHEMA), "rows": rows, "commit": commit, "phase": PHASE, "dataset": dataset,
                  "seed": int(args.seed), "backbones": args.backbones, "methods": args.methods},
                 out / dataset / f"{dataset}_seed{args.seed}_cita_rows.json")
    print(f"\n[cita] wrote {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
