#!/usr/bin/env python
"""CIGL Phase 3A-K — fixed-config SECOND-DATASET confirmation of DGCNN graph/node CMI (BNCI2015_001).

Phase 3A-J confirmed the FIXED candidate `graph_node_010` (λ_g=λ_node=0.010, no edge) across all
BNCI2014_001 LOSO folds — a single-dataset method signal. Phase 3A-K tests externality: does the SAME
fixed candidate replicate on a SECOND MI dataset (BNCI2015_001, binary) under the same source-only,
graph/node-only, edge-skipped protocol? NO λ grid, NO new configs, NO edge term, NO SOTA.

BNCI2015_001 is binary (chance bAcc = 0.50), so the adequacy/retention floors differ from BNCI2014_001:
ERM mean source ≥ 0.60 and ≥2/3 seeds ≥ 0.55. graph_node_010 was selected on BNCI2014_001, NOT here, so
there is NO dev fold — ALL BNCI2015_001 LOSO folds are confirmation folds. If the loaded dataset is not
binary (n_classes != 2), the runner FAILS before training and asks for reviewer approval (no silent
threshold mismatch).

    python scripts/run_cigl_phase3a_dgcnn_gn_second_dataset_confirmation.py --dry_run_synthetic --device cpu --seeds 0 1 --epochs 3 --probe_epochs 5 --n_perm 5
    python scripts/run_cigl_phase3a_dgcnn_gn_second_dataset_confirmation.py --dataset BNCI2015_001 --device cuda --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 50 --gate_alpha 0.05

See docs/CIGL_30_PHASE3A_K_SECOND_DATASET_CONFIRMATION.md.
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

OUT_DIR = REPO / "results" / "cigl" / "phase3a_dgcnn_gn_second_dataset_confirmation"
PHASE = "Phase3A_K_second_dataset_confirmation"
DEFAULT_DATASET = "BNCI2015_001"
FIXED_CONFIGS = [("erm_fixed", 0.000, 0.000), ("graph_node_010", 0.010, 0.010)]   # graph/node only; no edge


def _synthetic_folds(folds, n_per_subj=80, C=8, T=48, n_cls=2, n_subj=4):
    """Binary multi-fold learnable synthetic; fold f holds out subject (f % n_subj). n_cls=2 to match
    BNCI2015_001 and exercise the binary-threshold logic."""
    rng = np.random.default_rng(0)
    proto = 2.5 * rng.standard_normal((n_cls, C, T)).astype("float32")
    Xs, ys, ds = [], [], []
    for s in range(n_subj):
        for _ in range(n_per_subj):
            yy = rng.integers(0, n_cls)
            x = proto[yy] + 0.5 * rng.standard_normal((C, T)).astype("float32"); x[1:4] += 0.8 * s
            Xs.append(x); ys.append(yy); ds.append(s)
    X = np.stack(Xs).astype("float32"); y = np.array(ys, "int64"); d = np.array(ds, "int64")
    out = {f: (X, y, d, d != (f % n_subj), d == (f % n_subj), n_cls, str(f % n_subj)) for f in folds}
    return out, n_cls


def _load_real_all(args):
    """Load the dataset ONCE; return ({fold_idx: fold_tuple} over ALL subjects, n_classes)."""
    from cmi.data import moabb_data
    try:
        X, y, meta, classes = moabb_data.load(args.dataset, tmin=args.tmin, tmax=args.tmax, resample=args.resample)
    except Exception as e:
        raise SystemExit(f"[phase3a-K] dataset '{args.dataset}' not loadable offline "
                         f"({type(e).__name__}: {e}); use --dry_run_synthetic to validate the pipeline.")
    dom_all, _ = moabb_data.domain_labels(meta, "subject")
    splits = list(moabb_data.loso_splits(meta))
    n_cls = len(classes)
    sel = args.folds if args.folds is not None else list(range(len(splits)))
    out = {}
    for f in sel:
        if f >= len(splits):
            raise SystemExit(f"[phase3a-K] fold {f} out of range ({len(splits)} LOSO folds)")
        tgt, tr_mask, te_mask = splits[f]
        out[f] = (X, y, dom_all, tr_mask, te_mask, n_cls, str(tgt))
    return out, n_cls


def _fold_flags(erm, reg, args):
    """Per-fold flags with BINARY-aware floors (distinct mean vs per-seed floor):
      erm_adequate    = ERM mean source >= source_mean_floor AND >=ceil(2/3 seeds) >= source_seed_floor
      source_retained = reg mean source >= source_mean_floor AND source drop <= source_drop_max
    SOURCE-ONLY; target_eval only a reported guardrail. (No dev fold on a 2nd dataset — every fold is a
    confirmation fold.)"""
    mf, sf = args.source_mean_floor, args.source_seed_floor
    erm_src = erm["source_probe_per_seed"]; reg_src = reg["source_probe_per_seed"]
    min_seeds = max(1, math.ceil(2 / 3 * len(erm_src)))
    erm_adequate = bool((_mean(erm_src) or 0.0) >= mf and sum(s >= sf for s in erm_src) >= min_seeds)
    erm_leakage_exists = bool(erm["graph_clears_seeds"] >= min_seeds or erm["node_clears_seeds"] >= min_seeds)
    gred = _reduction(erm["graph_kl_per_seed"], reg["graph_kl_per_seed"])
    nred = _reduction(erm["node_kl_per_seed"], reg["node_kl_per_seed"])
    g30 = int(sum((r is not None and r >= args.reduce_min) for r in gred))
    n30 = int(sum((r is not None and r >= args.reduce_min) for r in nred))
    reg_reduces = bool(g30 >= min_seeds or n30 >= min_seeds)
    src_drop = float((_mean(erm_src) or 0.0) - (_mean(reg_src) or 0.0))
    source_retained = bool((_mean(reg_src) or 0.0) >= mf and src_drop <= args.source_drop_max)
    tgt_drop = float((erm["target_eval_bacc"] or 0.0) - (reg["target_eval_bacc"] or 0.0))
    target_guardrail = bool(tgt_drop <= args.target_drop_max)
    return dict(erm_adequate=erm_adequate, erm_leakage_exists=erm_leakage_exists, reg_reduces=reg_reduces,
                source_retained=source_retained, target_guardrail=target_guardrail,
                graph_reduction=_mean([r for r in gred if r is not None]),
                node_reduction=_mean([r for r in nred if r is not None]),
                graph_reduce30_seeds=g30, node_reduce30_seeds=n30,
                source_drop_vs_erm=src_drop, target_drop_vs_erm=tgt_drop,
                fold_pass=bool(erm_adequate and erm_leakage_exists and reg_reduces and source_retained))


def decide_second_dataset(per_fold_flags, folds):
    """Aggregate over ALL confirmation folds (no dev fold on a 2nd dataset). Three-layer verdict:
      source_only_confirmed             = criteria 1-4 (ERM adequacy, ERM leakage, reg reduces, source
                                          retained) — the source-only replication result.
      target_guardrail_pass            = target_eval drop <= limit in >= need_target_guardrail folds
                                          (EVALUATION-ONLY; never used for training/selection/configs).
      confirmed_with_target_guardrail  = source_only_confirmed AND target_guardrail_pass — the final
                                          reviewer-facing Decision-A condition.
    Thresholds: 8/12 (adequacy, leakage) and 7/12 (reduction, retention, target guardrail) at n=12,
    generalized by fraction."""
    cf = [f for f in folds if f in per_fold_flags]
    n = len(cf)
    need_strong = math.ceil(8 / 12 * n) if n else 1
    need_majority = math.ceil(7 / 12 * n) if n else 1
    need_target_guardrail = math.ceil(7 / 12 * n) if n else 1
    c1 = sum(per_fold_flags[f]["erm_adequate"] for f in cf)
    c2 = sum(per_fold_flags[f]["erm_leakage_exists"] for f in cf)
    c3 = sum(per_fold_flags[f]["reg_reduces"] for f in cf)
    c4 = sum(per_fold_flags[f]["source_retained"] for f in cf)
    c5 = sum(per_fold_flags[f]["target_guardrail"] for f in cf)
    crit1 = c1 >= need_strong; crit2 = c2 >= need_strong
    crit3 = c3 >= need_majority; crit4 = c4 >= need_majority
    source_only_confirmed = bool(crit1 and crit2 and crit3 and crit4)
    target_guardrail_pass = bool(c5 >= need_target_guardrail)
    confirmed_with_target_guardrail = bool(source_only_confirmed and target_guardrail_pass)
    if not crit1:
        decision = "D"                              # ERM baseline inadequate
    elif confirmed_with_target_guardrail:
        decision = "A"                              # source-only confirmed AND target guardrail held
    elif source_only_confirmed or (crit2 and (crit3 or crit4)):
        decision = "B"                              # source-only confirmed but guardrail fails, OR strong partial
    else:
        decision = "C"                              # source-only confirmation fails (reduction/retention)
    return dict(n_folds=n, need_strong=need_strong, need_majority=need_majority,
                need_target_guardrail=need_target_guardrail,
                erm_adequacy_folds=c1, erm_leakage_folds=c2, reg_reduces_folds=c3,
                source_retained_folds=c4, target_guardrail_folds=c5,
                crit1_erm_adequate=crit1, crit2_erm_leakage=crit2, crit3_reg_reduces=crit3,
                crit4_source_retained=crit4,
                source_only_confirmed=source_only_confirmed, target_guardrail_pass=target_guardrail_pass,
                confirmed_with_target_guardrail=confirmed_with_target_guardrail,
                confirmed=confirmed_with_target_guardrail, decision=decision)


def _meta(commit, cfg_hash, dataset, n_cls, chance, args):
    return dict(exploratory=True, phase=PHASE, setting="strict_source_only_DG", dataset=dataset,
                fixed_candidate="graph_node_010", second_dataset_confirmation=True,
                n_classes=int(n_cls), chance_bacc=float(chance),
                source_mean_floor=float(args.source_mean_floor), source_seed_floor=float(args.source_seed_floor),
                allow_non_binary=bool(args.allow_non_binary),
                cmi_regularization_used=True, edge_regularization_used=False,
                edge_logits_dynamic=False, edge_audit_skipped=True,
                used_target_labels_for_training=False, used_target_labels_for_selection=False,
                used_target_covariates=False, target_eval_is_evaluation_only=True,
                selection_uses_target_eval=False, confirmation_label_selection_uses_target_eval=False,
                commit_hash=commit, config_hash=cfg_hash)


def _aggregate(recs):
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


def _dataset_interval(name):
    """MOABB dataset.interval (metadata-only; no data load/download). None if unavailable."""
    try:
        import moabb.datasets as D
        iv = getattr(getattr(D, name)(), "interval", None)
        return list(iv) if iv else None
    except Exception:
        return None


def _preprocessing_meta(args, dataset_name):
    """Record the exact preprocessing for the run (paradigm/events/resample/window/interval)."""
    if args.dry_run_synthetic:
        return dict(moabb_paradigm="synthetic", events=None, resample=int(args.resample),
                    tmin=float(args.tmin), tmax=float(args.tmax))
    from cmi.data import moabb_data
    p = moabb_data.paradigm_info(dataset_name)
    interval = _dataset_interval(dataset_name)
    inside = bool(interval and len(interval) == 2 and args.tmin >= 0 and args.tmax <= (interval[1] - interval[0]))
    return dict(moabb_paradigm=p["moabb_paradigm"], events=p["events"], resample=int(args.resample),
                resample_rationale="match BNCI2014_001 confirmation protocol",
                known_preprocessing_note=("notes/preprocessing_decision.md mentions 250 Hz; not changed "
                                          "here to avoid introducing a new variable in the confirmation"),
                tmin=float(args.tmin), tmax=float(args.tmax), dataset_interval=interval,
                effective_window_relative_to_interval=[float(args.tmin), float(args.tmax)],
                window_inside_declared_interval=inside)


def _preflight(args, commit, cfg_hash):
    """Verify dataset/paradigm/classes/shape + record preprocessing metadata. NO training, NO probes,
    NO silent downloads (loads ONE subject from the local cache to check classes/channels/time-shape)."""
    from cmi.data import moabb_data
    import moabb.datasets as D
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = getattr(D, args.dataset)()
    subs = list(ds.subject_list)
    pinfo = moabb_data.paradigm_info(args.dataset)
    try:
        X, y, meta, classes = moabb_data.load(args.dataset, subjects=[subs[0]],
                                              tmin=args.tmin, tmax=args.tmax, resample=args.resample)
    except Exception as e:
        raise SystemExit(f"[phase3a-K preflight] '{args.dataset}' not loadable from local cache "
                         f"({type(e).__name__}: {e}); check the datalake path. No download attempted here.")
    n_cls = len(classes); chance = 1.0 / n_cls
    pp = _preprocessing_meta(args, args.dataset)
    pf = dict(mode="preflight_only", trained=False, probes_run=False,
              dataset=args.dataset, commit_hash=commit, config_hash=cfg_hash,
              n_subjects=len(subs), subject_list=subs, classes=classes, n_classes=n_cls,
              chance_bacc=chance, classes_are_right_hand_feet=bool(set(classes) == {"right_hand", "feet"}),
              one_subject_X_shape=list(X.shape), n_channels=int(X.shape[1]), n_times=int(X.shape[2]),
              binary_ok=bool(n_cls == 2), preprocessing=pp)
    json.dump(pf, open(OUT_DIR / f"{args.dataset}_preflight.json", "w"), indent=2)
    print("\n=== Phase 3A-K PREFLIGHT (no training, no probes) ===")
    for k in ("dataset", "n_subjects", "classes", "n_classes", "chance_bacc", "n_channels", "n_times"):
        print(f"  {k}: {pf[k]}")
    print(f"  paradigm: {pp['moabb_paradigm']}  events: {pp['events']}  resample: {pp['resample']}")
    print(f"  dataset_interval: {pp.get('dataset_interval')}  window: [{args.tmin},{args.tmax}]  "
          f"inside_interval: {pp.get('window_inside_declared_interval')}")
    if n_cls != 2:
        print(f"  STOP: n_classes={n_cls} != 2 -> binary thresholds do NOT apply; reviewer re-authorization needed.")
    elif set(classes) != {"right_hand", "feet"}:
        print(f"  STOP: classes {classes} != right_hand/feet -> report for reviewer.")
    else:
        print("  OK: binary right_hand/feet via MotorImagery; preflight passed (no training ran).")
    print(f"[phase3a-K] wrote preflight {OUT_DIR / f'{args.dataset}_preflight.json'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run_synthetic", action="store_true")
    ap.add_argument("--preflight_only", action="store_true",
                    help="verify dataset/paradigm/classes/shape + record preprocessing metadata; NO training/probes")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--allow_non_default_dataset", action="store_true")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="default: all LOSO folds for the dataset")
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
    # binary-dataset floors (BNCI2015_001, chance 0.50)
    ap.add_argument("--source_mean_floor", type=float, default=0.60)
    ap.add_argument("--source_seed_floor", type=float, default=0.55)
    ap.add_argument("--source_drop_max", type=float, default=0.02)
    ap.add_argument("--reduce_min", type=float, default=0.30)
    ap.add_argument("--target_drop_max", type=float, default=0.05)
    ap.add_argument("--graph_usage_min_drop", type=float, default=0.10)
    ap.add_argument("--allow_non_binary", action="store_true",
                    help="explicit re-authorization to run a NON-binary dataset; the binary floors "
                         "(0.60/0.55) would not apply, so thresholds must be re-set for that n_classes")
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--success_bacc_floor", type=float, default=0.55)   # used by reused _train_eval graph-usage only
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[phase3a-K] --device cuda requested but CUDA unavailable")
    if args.device == "cpu":
        torch.set_num_threads(1)
    if (not args.dry_run_synthetic) and args.dataset != DEFAULT_DATASET and not args.allow_non_default_dataset:
        raise SystemExit(f"[phase3a-K] default 2nd dataset is {DEFAULT_DATASET}; to run '{args.dataset}' pass "
                         f"--allow_non_default_dataset (and document why). No silent dataset switch.")

    commit = _git_commit_hash(); cfg_hash = _config_hash(vars(args))
    if args.preflight_only:                                   # verify load/paradigm/shape only; NO training
        return _preflight(args, commit, cfg_hash)
    dataset = "synthetic" if args.dry_run_synthetic else args.dataset
    folds_req = args.folds if args.folds is not None else list(range(4))   # synthetic default 4 folds
    if args.dry_run_synthetic:
        folds, n_cls = _synthetic_folds(folds_req)
    else:
        folds, n_cls = _load_real_all(args)
    chance = 1.0 / n_cls

    # FAIL CLOSED on a class-count mismatch: the binary floors apply ONLY to binary (n_classes==2). This
    # guard keys on the CONSTANT 2 (not a settable arg). Running a non-binary set requires the explicit
    # --allow_non_binary re-authorization (deliberate, logged) AND thresholds matched to that n_classes.
    if (not args.dry_run_synthetic) and n_cls != 2 and not args.allow_non_binary:
        raise SystemExit(f"[phase3a-K] STOP for reviewer approval: '{dataset}' has n_classes={n_cls} "
                         f"(chance_bacc={chance:.3f}), not binary (2). The binary floors "
                         f"(mean {args.source_mean_floor}/seed {args.source_seed_floor}) do NOT apply. "
                         f"Re-authorize explicitly with --allow_non_binary AND thresholds matched to "
                         f"n_classes={n_cls} before the real run.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Phase 3A-K second-dataset confirmation ({dataset}; n_classes={n_cls}, chance={chance:.3f}; "
          f"folds={sorted(folds)}; fixed graph_node_010; mean_floor={args.source_mean_floor}) ===")
    per_fold, per_fold_flags = {}, {}
    for f in sorted(folds):
        fold = folds[f]
        cfg_aggs = {}
        for label, lam_g, lam_node in FIXED_CONFIGS:
            recs = []
            for seed in args.seeds:
                rec = _train_eval(label, lam_g, lam_node, fold, seed, args, args.device, args.n_perm)
                rec["meta"] = dict(_meta(commit, cfg_hash, dataset, n_cls, chance, args), fold=f,
                                   cmi_regularization_used=bool(lam_g != 0.0 or lam_node != 0.0))
                json.dump(rec, open(OUT_DIR / f"{dataset}_fold{f}_{label}_seed{seed}.json", "w"), indent=2)
                recs.append(rec)
            cfg_aggs[label] = _aggregate(recs)
        flags = _fold_flags(cfg_aggs["erm_fixed"], cfg_aggs["graph_node_010"], args)
        per_fold[f] = dict(heldout_subject=fold[6], erm_fixed=cfg_aggs["erm_fixed"],
                           graph_node_010=cfg_aggs["graph_node_010"], flags=flags)
        per_fold_flags[f] = flags
        e, r = cfg_aggs["erm_fixed"], cfg_aggs["graph_node_010"]
        print(f"  fold{f} sub{fold[6]}: ermSrc={e['source_probe_bacc']:.3f} regSrc={r['source_probe_bacc']:.3f}"
              f"(d{flags['source_drop_vs_erm']:+.3f}) gKL {e['graph_kl_mean']:.2f}->{r['graph_kl_mean']:.2f}"
              f"(g{flags['graph_reduce30_seeds']}/n{flags['node_reduce30_seeds']}) ermLeak={flags['erm_leakage_exists']} "
              f"regReduces={flags['reg_reduces']} srcRetain={flags['source_retained']} PASS={flags['fold_pass']}", flush=True)

    primary = decide_second_dataset(per_fold_flags, sorted(folds))
    summary = dict(
        meta=dict(_meta(commit, cfg_hash, dataset, n_cls, chance, args), folds=sorted(folds), seeds=list(args.seeds),
                  n_perm=int(args.n_perm), gate_alpha=float(args.gate_alpha), epochs=int(args.epochs),
                  source_drop_max=float(args.source_drop_max), reduce_min=float(args.reduce_min),
                  target_drop_max=float(args.target_drop_max), preprocessing=_preprocessing_meta(args, dataset)),
        configs={n: (lg, ln) for n, lg, ln in FIXED_CONFIGS}, per_fold=per_fold,
        second_dataset_confirmation=primary, all_folds_descriptive=primary,   # no dev fold -> identical
        edge_skip_reason="static/shared adjacency: edge_logits=None; no per-sample edge object")
    json.dump(summary, open(OUT_DIR / f"{dataset}_dgcnn_gn_2nd_dataset_summary.json", "w"), indent=2)
    print(f"\n[phase3a-K] wrote {OUT_DIR / f'{dataset}_dgcnn_gn_2nd_dataset_summary.json'}")

    dmap = {"A": "A: CONFIRMED on the 2nd dataset (source-only AND target guardrail) -> method framing may begin",
            "B": "B: PARTIAL -> source-only confirmed but target guardrail fails, or strong partial; bounded signal",
            "C": "C: NOT confirmed (reduction/retention fails) -> BNCI2014_001-only finding",
            "D": "D: ERM baseline unstable on the 2nd dataset -> dataset/backbone diagnosis"}
    print(f"\n=== Phase 3A-K read ({dataset}; exploratory; reviewer decides): {dmap[primary['decision']]} ===")
    print(f"  {dataset}: erm_adequacy {primary['erm_adequacy_folds']}/{primary['n_folds']} (need {primary['need_strong']}), "
          f"erm_leakage {primary['erm_leakage_folds']} (need {primary['need_strong']}), "
          f"reg_reduces {primary['reg_reduces_folds']} (need {primary['need_majority']}), "
          f"source_retained {primary['source_retained_folds']} (need {primary['need_majority']}), "
          f"target_guardrail {primary['target_guardrail_folds']} (need {primary['need_target_guardrail']})")
    print(f"  source_only_confirmed={primary['source_only_confirmed']} target_guardrail_pass={primary['target_guardrail_pass']} "
          f"confirmed_with_target_guardrail={primary['confirmed_with_target_guardrail']}")
    print("  EDGE absent (static adjacency). target_eval evaluation-only (final guardrail ONLY; never "
          "training/selection/configs). configs FIXED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
