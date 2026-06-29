#!/usr/bin/env python
"""CIGL Phase 3A-R — baseline repair + gentle-λ re-pilot (source-only, BNCI2014_001 fold-0).

Phase 3A showed strong leakage CONTROLLABILITY but failed the task-preserving gate, and the
GraphCMINet-ERM baseline was near chance (src bAcc ~0.33 on a 4-class task). The baseline is now the
bottleneck, not leakage. Phase 3A-R asks, in order:

  PART A (baseline adequacy gate): can a small named set of GraphCMINet-ERM variants reach a
    NON-DEGENERATE source-only baseline (source_probe bAcc >= 0.45, or >= current_default + 0.10),
    with train bAcc clearly above chance and a label-shuffle control near chance?
  PART B (gentle re-pilot, ONLY if A passes): on the selected baseline, does a GENTLE CMI micro-ladder
    (λ <= ~0.05) buy a task-preserving leakage reduction (>=30% graph or node KL drop, <=3 pt source
    drop, <=5 pt target drop)?

If Part A fails, STOP and recommend architecture/preprocessing diagnosis — do NOT run Part B, do NOT
claim CIGL as a method. Strict source-only throughout: target labels are used ONLY for after-the-fact
target_eval metrics (evaluation_only), never for training/selection/normalization/probe/audit.

    # always works (CPU, synthetic, tiny):
    python scripts/run_cigl_phase3a_baseline_repair.py --dry_run_synthetic --device cpu \
        --seeds 0 1 --epochs 3 --probe_epochs 5 --n_perm 5
    # real (GPU/sbatch):
    python scripts/run_cigl_phase3a_baseline_repair.py --dataset BNCI2014_001 --device cuda \
        --fold 0 --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 20 --n_perm_confirm 50

See docs/CIGL_17_PHASE3A_R_BASELINE_REPAIR.md.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cmi.eval.graph_leakage import audit_graph_objects                  # noqa: E402
from cmi.eval.probe_splits import stratified_trial_split_by_y_d          # noqa: E402

OUT_DIR = REPO / "results" / "cigl" / "phase3a_baseline_repair"
PHASE = "Phase3A_R_baseline_repair"

# Part A — small NAMED candidate list (each changes ONE coherent axis vs current_default; NOT a
# Cartesian grid). Two candidates bundle naturally-paired knobs: stronger_graphcmi_backbone (feat+hidden
# together = a bigger net) and lower_lr_longer (lr+epochs together = a slower, longer schedule).
BASELINE_CANDIDATES = [
    dict(name="current_default",            feat=16, hidden=32, hops=2, lr=1e-3, epochs=None, sampler="classbal", balance=False, chan_zscore=False),
    dict(name="source_channel_zscore",      feat=16, hidden=32, hops=2, lr=1e-3, epochs=None, sampler="classbal", balance=False, chan_zscore=True),
    dict(name="stronger_graphcmi_backbone", feat=32, hidden=64, hops=2, lr=1e-3, epochs=None, sampler="classbal", balance=False, chan_zscore=False),
    dict(name="lower_lr_longer",            feat=16, hidden=32, hops=2, lr=3e-4, epochs=150,  sampler="classbal", balance=False, chan_zscore=False),
    dict(name="no_classbal_sampler",        feat=16, hidden=32, hops=2, lr=1e-3, epochs=None, sampler="raw",      balance=False, chan_zscore=False),
    dict(name="ce_balance_check",           feat=16, hidden=32, hops=2, lr=1e-3, epochs=None, sampler="classbal", balance=True,  chan_zscore=False),
]

# Part B — gentle CMI MICRO-LADDER (NOT a grid). (label, lambda_g, lambda_node, lambda_edge)
GENTLE_CONFIGS = [
    ("erm_fixed",      0.0,   0.0,   0.0),
    ("graph_node_003", 0.003, 0.003, 0.0),
    ("graph_node_01",  0.01,  0.01,  0.0),
    ("graph_node_03",  0.03,  0.03,  0.0),
    ("graph_only_01",  0.01,  0.0,   0.0),
    ("node_only_01",   0.0,   0.01,  0.0),
    ("edge_only_03",   0.0,   0.0,   0.03),
    ("edge_only_10",   0.0,   0.0,   0.10),
    ("full_01",        0.01,  0.01,  0.003),
    ("full_03",        0.03,  0.03,  0.01),
]


# ----------------------------------------------------------------------------- helpers
def _git_commit_hash():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _remap_contiguous(d):
    uniq = {v: i for i, v in enumerate(sorted(np.unique(d)))}
    return np.array([uniq[v] for v in d], dtype=np.int64), len(uniq)


def _channel_zscore(X_ref, *arrays):
    """Per-channel mean/std fitted on X_ref (over trials AND time); applied to every array. Source-only:
    X_ref is always the source enc-train, so target stats never enter."""
    mu = X_ref.mean(axis=(0, 2), keepdims=True)
    sd = X_ref.std(axis=(0, 2), keepdims=True) + 1e-7
    return [((a - mu) / sd).astype("float32") for a in arrays]


@torch.no_grad()
def _extract_graph_features(net, X, device, bs=256):
    net.eval(); gz, nz, el = [], [], []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
        _, g, n, e = net.forward_graph(xb)
        gz.append(g.cpu().numpy()); nz.append(n.cpu().numpy()); el.append(e.cpu().numpy())
    return np.concatenate(gz), np.concatenate(nz), np.concatenate(el)


def _per_seed_meta(dataset, fold_tag, commit, cfg_hash):
    return dict(exploratory=True, phase=PHASE, setting="strict_source_only_DG",
                used_target_labels_for_training=False, used_target_labels_for_selection=False,
                used_target_covariates=False, target_eval_is_evaluation_only=True,
                target_labels_used_for="evaluation_only metrics",
                dataset=dataset, fold=fold_tag, commit_hash=commit, config_hash=cfg_hash)


def _obj_block(b, alpha):
    pe = bool(b["kl_mean"] > b["permutation_mean"])
    return dict(kl_mean=b["kl_mean"], permutation_mean=b["permutation_mean"], permutation_p=b["permutation_p"],
                positive_excess=pe, clears_null=bool(pe and b["permutation_p"] <= alpha), gate_alpha=float(alpha))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def controls_ok(overfit, shuffle_src, chance):
    """Sanity controls PASS iff the architecture CAN overfit a tiny subset (overfit train bAcc >>
    chance) AND a label-shuffle baseline does NOT predict above chance (the probe isn't cheating).
    The shuffle check is one-sided: at/below chance both pass; only ABOVE chance fails."""
    return bool(overfit is not None and overfit > chance + 0.15) and \
           bool(shuffle_src is not None and shuffle_src < chance + 0.10)


def decide_baseline_gate(cand_agg, control_ok, floor, gain, force_fail=False, force_pass=False):
    """Source-only baseline-adequacy gate. A candidate PASSES if its `source_probe_bacc` >= `floor` OR
    >= current_default + `gain`. The gate passes iff some candidate passes AND the controls behave.
    `best_baseline` is the highest-`source_probe` passing candidate — NEVER `target_eval`. `force_fail`/
    `force_pass` are TESTING hooks that bypass the real decision. Returns (passing_names, best, gate_pass)."""
    cd_src = (cand_agg["current_default"]["source_probe_bacc"] or 0.0) if "current_default" in cand_agg else 0.0

    def _ok(a):
        s = a["source_probe_bacc"] or 0.0
        return (s >= floor) or (s >= cd_src + gain)
    passing = {} if force_fail else {k: a for k, a in cand_agg.items() if _ok(a)}
    best = max(passing, key=lambda k: passing[k]["source_probe_bacc"]) if passing else None
    if force_pass and not force_fail and best is None:   # force_pass needs a baseline to run Part B on
        best = max(cand_agg, key=lambda k: cand_agg[k]["source_probe_bacc"] or 0.0)
    gate_pass = (not force_fail) and ((bool(passing) and control_ok) or force_pass)
    return list(passing), best, gate_pass


def decide_gentle_selection(gentle_agg, source_drop_max=0.03, target_drop_max=0.05, reduce30_min_seeds=2):
    """SOURCE-ONLY selection firewall for Part B. ALL selection (which configs get the n_perm_confirm
    re-audit, the best reducer) uses ONLY source-side evidence (graph/node reduction + source_drop).
    target_eval enters ONLY `final_task_preserving_reducers`, a REPORTED verdict computed AFTER the
    source-only decisions — it must never change `confirmation_labels`/`source_only_reducers`/
    `best_reducer`. Returns a dict with both layers."""
    gn = [l for l in gentle_agg if l != "erm_fixed"
          and (gentle_agg[l].get("graph_reduce30_seeds", 0) >= reduce30_min_seeds
               or gentle_agg[l].get("node_reduce30_seeds", 0) >= reduce30_min_seeds)]
    source_only_reducers = [l for l in gn if gentle_agg[l]["source_drop_vs_erm"] <= source_drop_max]
    best_reducer = max(gn, key=lambda l: max(gentle_agg[l].get("graph_reduction_vs_erm") or 0.0,
                                             gentle_agg[l].get("node_reduction_vs_erm") or 0.0)) if gn else None
    confirmation_labels = sorted({"erm_fixed"} | set(source_only_reducers)
                                 | ({best_reducer} if best_reducer else set()))
    final_task_preserving_reducers = [l for l in source_only_reducers
                                      if gentle_agg[l]["target_drop_vs_erm"] <= target_drop_max]
    return dict(source_only_reducers=source_only_reducers, best_reducer=best_reducer,
                confirmation_labels=confirmation_labels,
                final_task_preserving_reducers=final_task_preserving_reducers,
                gentle_gate_pass_source_only=bool(source_only_reducers),
                gentle_gate_pass_with_target_retention=bool(final_task_preserving_reducers))


# ----------------------------------------------------------------------------- core train+eval
def _train_eval(candidate, lambdas, fold, seed, args, device, n_perm, label_shuffle=False, subset=None,
                epochs_override=None):
    """Train GraphCMINet on SOURCE enc-train with the candidate's backbone/training config and `lambdas`
    (all-zero -> ERM via method='erm'); extract frozen features on a held-out source probe-pool; audit
    leakage with fresh probes; report train / source_probe / target_eval task metrics. Target labels
    touch ONLY target_eval. `label_shuffle` permutes SOURCE labels (a control). `subset` (int) restricts
    the source enc-train to a tiny balanced subset (overfit control)."""
    from cmi.models.gnn import GraphCMINet
    from cmi.train.trainer import train_model, predict
    from cmi.eval.metrics import classification_metrics
    lg, ln, le = lambdas
    X, y, dom_all, tr_mask, te_mask, n_cls, heldout = fold
    Xs, ys = X[tr_mask].copy(), y[tr_mask].copy()
    ds, n_dom = _remap_contiguous(dom_all[tr_mask])
    cand_epochs = None if args.dry_run_synthetic else candidate["epochs"]   # dry-run: ignore long overrides
    epochs = epochs_override or cand_epochs or args.epochs

    enc_idx, pool_idx, _ = stratified_trial_split_by_y_d(ys, ds, train_frac=args.enc_train_frac,
                                                         seed=seed, min_per_cell=args.min_per_cell)
    if subset:                                              # overfit control: tiny balanced enc subset
        rng = np.random.default_rng(seed); keep = []
        for c in range(n_cls):
            idx = enc_idx[ys[enc_idx] == c]
            if len(idx):
                keep.append(rng.choice(idx, min(subset, len(idx)), replace=False))
        enc_idx = np.sort(np.concatenate(keep)) if keep else enc_idx

    Xe, ye, de = Xs[enc_idx], ys[enc_idx].copy(), ds[enc_idx]
    if label_shuffle:                                      # control: source labels carry no signal
        ye = np.random.default_rng(seed + 7).permutation(ye)
    Xp = Xs[pool_idx]
    Xt = X[te_mask]
    if candidate["chan_zscore"]:
        Xe, Xp, Xt = _channel_zscore(Xs[enc_idx], Xe, Xp, Xt)

    torch.manual_seed(int(seed)); np.random.seed(int(seed))   # anchor encoder init to seed (order-independent)
    net = GraphCMINet(X.shape[1], X.shape[2], n_cls, feat=candidate["feat"],
                      hidden=candidate["hidden"], hops=candidate["hops"]).to(device)
    method = "graphcmi" if (lg or ln or le) else "erm"
    net, _post, diag = train_model(net, Xe, ye, de, n_cls, method=method, lam=lg, gamma=ln, lam_edge=le,
                                   epochs=epochs, bs=args.bs, warmup=max(1, epochs // 5), lr=candidate["lr"],
                                   sampler=candidate["sampler"], balance=candidate["balance"],
                                   device=device, seed=seed)
    gz, nz, el = _extract_graph_features(net, Xp, device)
    y_pool, d_pool = ys[pool_idx], ds[pool_idx]
    tr_idx, va_idx, split_diag = stratified_trial_split_by_y_d(y_pool, d_pool, train_frac=args.train_frac,
                                                              seed=seed, min_per_cell=args.min_per_cell)
    audit = audit_graph_objects(gz, nz, el, y_pool, d_pool, n_cls, n_dom, n_perm=n_perm, seed=seed,
                                device=device, epochs=args.probe_epochs, train_idx=tr_idx, val_idx=va_idx)
    for o in ("graph", "node", "edge"):
        audit[o].pop("node_leakage_map", None); audit[o].pop("edge_leakage_map", None)

    train_m = classification_metrics(predict(net, Xe, device), ye)
    src = classification_metrics(predict(net, Xp, device), y_pool)
    tgt = classification_metrics(predict(net, Xt, device), y[te_mask])
    rec = dict(
        candidate=candidate["name"], lambda_g=lg, lambda_node=ln, lambda_edge=le, seed=int(seed),
        n_perm=int(n_perm), gate_alpha=float(args.gate_alpha), n_classes=int(n_cls), n_domains=int(n_dom),
        train=dict(balanced_acc=train_m["balanced_acc"], macro_f1=train_m["macro_f1"]),
        source_probe=dict(balanced_acc=src["balanced_acc"], macro_f1=src["macro_f1"]),
        target_eval=dict(balanced_acc=tgt["balanced_acc"], macro_f1=tgt["macro_f1"], evaluation_only=True),
        train_minus_source_gap=float(train_m["balanced_acc"] - src["balanced_acc"]),
        graph=_obj_block(audit["graph"], args.gate_alpha), node=_obj_block(audit["node"], args.gate_alpha),
        edge=_obj_block(audit["edge"], args.gate_alpha),
        n_enc_train=int(len(enc_idx)), n_probe_pool=int(len(pool_idx)), heldout_subject=heldout)
    return rec


# ----------------------------------------------------------------------------- data
def _synthetic_fold(seed, n_per_subj=60, C=8, T=48, n_cls=3, n_subj=4):
    """Tiny EEG-like fold: STRONG label signal (so a few epochs reach > chance, exercising Part A's
    gate) + a subject-encoding subset of channels (leakage). One subject is the held-out target."""
    rng = np.random.default_rng(seed)
    proto = 2.5 * rng.standard_normal((n_cls, C, T)).astype("float32")     # strong, separable classes
    Xs, ys, ds = [], [], []
    for s in range(n_subj):
        for _ in range(n_per_subj):
            yy = rng.integers(0, n_cls)
            x = proto[yy] + 0.5 * rng.standard_normal((C, T)).astype("float32")
            x[1:4] += 0.8 * s
            Xs.append(x); ys.append(yy); ds.append(s)
    X = np.stack(Xs).astype("float32"); y = np.array(ys, "int64"); d = np.array(ds, "int64")
    target = n_subj - 1
    return X, y, d, d != target, d == target, n_cls, str(target)


def _load_real_fold(args):
    from cmi.data import moabb_data
    try:
        X, y, meta, classes = moabb_data.load(args.dataset, tmin=args.tmin, tmax=args.tmax, resample=args.resample)
    except Exception as e:
        raise SystemExit(f"[phase3a-R] dataset '{args.dataset}' not loadable offline ({type(e).__name__}: {e}); "
                         f"ensure the MOABB datalake cache is present. Use --dry_run_synthetic to validate.")
    dom_all, _ = moabb_data.domain_labels(meta, "subject")
    tgt, tr_mask, te_mask = list(moabb_data.loso_splits(meta))[args.fold]
    return X, y, dom_all, tr_mask, te_mask, len(classes), str(tgt)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run_synthetic", action="store_true")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--probe_epochs", type=int, default=100)
    ap.add_argument("--n_perm", type=int, default=20)
    ap.add_argument("--n_perm_confirm", type=int, default=50)
    ap.add_argument("--baseline_n_perm", type=int, default=10, help="light audit during baseline characterization")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--enc_train_frac", type=float, default=0.7)
    ap.add_argument("--min_per_cell", type=int, default=2)
    ap.add_argument("--gate_alpha", type=float, default=0.05)
    ap.add_argument("--baseline_bacc_floor", type=float, default=0.45, help="absolute source_probe bAcc gate")
    ap.add_argument("--baseline_bacc_gain", type=float, default=0.10, help="OR >= current_default + this")
    ap.add_argument("--overfit_subset", type=int, default=8, help="per-class trials for the overfit control")
    ap.add_argument("--overfit_epochs", type=int, default=80, help="epochs for the overfit/shuffle controls (must overfit)")
    ap.add_argument("--force_baseline_fail", action="store_true", help="(testing) force Part-A gate to FAIL")
    ap.add_argument("--force_baseline_pass", action="store_true", help="(testing) bypass Part-A gate so Part B runs")
    ap.add_argument("--skip_part_b", action="store_true", help="(testing) run Part A + gate only; skip Part B")
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[phase3a-R] --device cuda requested but CUDA unavailable")
    if args.device == "cpu":
        torch.set_num_threads(1)   # tiny EEG ops: avoid many-core dispatch overhead. No effect on GPU.

    commit = _git_commit_hash(); cfg_hash = _config_hash(vars(args))
    if args.dry_run_synthetic and args.n_perm_confirm > args.n_perm:
        args.n_perm_confirm = args.n_perm   # dry-run validates the pipeline, not significance
    fold = _synthetic_fold(seed=0) if args.dry_run_synthetic else _load_real_fold(args)
    dataset = "synthetic" if args.dry_run_synthetic else args.dataset
    fold_tag = f"{dataset}_fold{args.fold}"
    n_cls = fold[5]
    chance = 1.0 / n_cls
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _save(rec, fname):
        rec["meta"] = _per_seed_meta(dataset, fold_tag, commit, cfg_hash)
        json.dump(rec, open(OUT_DIR / fname, "w"), indent=2)

    # ---- PART A: baseline adequacy gate (ERM only, source-only selection) ----------------------
    print(f"\n=== Phase 3A-R PART A: baseline adequacy (chance={chance:.3f}, floor={args.baseline_bacc_floor}) ===")
    cand_agg = {}
    for cand in BASELINE_CANDIDATES:
        recs = []
        for seed in args.seeds:
            rec = _train_eval(cand, (0.0, 0.0, 0.0), fold, seed, args, args.device, n_perm=args.baseline_n_perm)
            _save(rec, f"{fold_tag}_baseline_{cand['name']}_seed{seed}.json")
            recs.append(rec)
        a = dict(candidate=cand["name"],
                 train_bacc=_mean([r["train"]["balanced_acc"] for r in recs]),
                 source_probe_bacc=_mean([r["source_probe"]["balanced_acc"] for r in recs]),
                 target_eval_bacc=_mean([r["target_eval"]["balanced_acc"] for r in recs]),
                 train_minus_source_gap=_mean([r["train_minus_source_gap"] for r in recs]),
                 graph_kl=_mean([r["graph"]["kl_mean"] for r in recs]),
                 node_kl=_mean([r["node"]["kl_mean"] for r in recs]),
                 edge_kl=_mean([r["edge"]["kl_mean"] for r in recs]), target_eval_is_evaluation_only=True)
        cand_agg[cand["name"]] = a
        print(f"  {cand['name']:26s} train={a['train_bacc']:.3f} src={a['source_probe_bacc']:.3f} "
              f"tgt={a['target_eval_bacc']:.3f} gap={a['train_minus_source_gap']:+.3f} "
              f"g/n/e_kl={a['graph_kl']:.3f}/{a['node_kl']:.3f}/{a['edge_kl']:.3f}", flush=True)

    # sanity controls (use current_default config)
    cd = BASELINE_CANDIDATES[0]
    overfit = _mean([_train_eval(cd, (0., 0., 0.), fold, s, args, args.device, n_perm=2,
                                 subset=args.overfit_subset, epochs_override=args.overfit_epochs
                                 )["train"]["balanced_acc"] for s in args.seeds[:1]])
    shuffle_src = _mean([_train_eval(cd, (0., 0., 0.), fold, s, args, args.device, n_perm=2,
                                     label_shuffle=True, epochs_override=args.overfit_epochs
                                     )["source_probe"]["balanced_acc"] for s in args.seeds[:1]])
    print(f"  [control] overfit_small_source train_bAcc={overfit:.3f} (want >> chance {chance:.3f})  "
          f"label_shuffle_control src_bAcc={shuffle_src:.3f} (want NOT above chance, i.e. < {chance + 0.10:.3f})",
          flush=True)

    control_ok = controls_ok(overfit, shuffle_src, chance)
    passing, best_baseline, baseline_gate_pass = decide_baseline_gate(
        cand_agg, control_ok, args.baseline_bacc_floor, args.baseline_bacc_gain,
        force_fail=args.force_baseline_fail, force_pass=args.force_baseline_pass)
    print(f"\n  baseline_gate_pass={baseline_gate_pass}  best_baseline={best_baseline}  "
          f"(controls_ok={control_ok}; forced_fail={args.force_baseline_fail})", flush=True)

    summary = dict(
        meta=dict(exploratory=True, phase=PHASE, setting="strict_source_only_DG", dataset=dataset, fold=fold_tag,
                  heldout_subject=fold[6], seeds=list(args.seeds), n_classes=int(n_cls), chance=float(chance),
                  gate_alpha=float(args.gate_alpha), epochs=int(args.epochs), commit_hash=commit, config_hash=cfg_hash,
                  used_target_labels_for_training=False, used_target_labels_for_selection=False,
                  used_target_covariates=False, target_eval_is_evaluation_only=True),
        part_a=dict(candidates=cand_agg,
                    controls=dict(overfit_small_source_train_bacc=overfit, label_shuffle_control_src_bacc=shuffle_src,
                                  controls_ok=control_ok),
                    baseline_bacc_floor=args.baseline_bacc_floor, baseline_gate_pass=baseline_gate_pass,
                    best_baseline=best_baseline),
        part_b=None)

    # ---- PART B: gentle micro-ladder (ONLY if Part A passes) ------------------------------------
    if not baseline_gate_pass or args.skip_part_b:
        why = "baseline gate FAILED -> recommend architecture/preprocessing diagnosis (do NOT claim CIGL " \
              "as a method)" if not baseline_gate_pass else "--skip_part_b set (Part A + gate only)"
        print(f"\n=== PART B SKIPPED: {why} ===", flush=True)
    else:
        cand = next(c for c in BASELINE_CANDIDATES if c["name"] == best_baseline)
        print(f"\n=== Phase 3A-R PART B: gentle micro-ladder on baseline '{best_baseline}' ===")
        per_cfg = {}
        for (lbl, lg, ln, le) in GENTLE_CONFIGS:
            recs = []
            for seed in args.seeds:
                rec = _train_eval(cand, (lg, ln, le), fold, seed, args, args.device, n_perm=args.n_perm)
                _save(rec, f"{fold_tag}_gentle_{lbl}_seed{seed}_nperm{args.n_perm}.json")
                recs.append(rec)
            per_cfg[lbl] = recs
            r0 = recs[0]
            print(f"  {lbl:16s} src={_mean([r['source_probe']['balanced_acc'] for r in recs]):.3f} "
                  f"g/n/e_kl={_mean([r['graph']['kl_mean'] for r in recs]):.3f}/"
                  f"{_mean([r['node']['kl_mean'] for r in recs]):.3f}/"
                  f"{_mean([r['edge']['kl_mean'] for r in recs]):.3f}", flush=True)
        erm_kl = {o: _mean([r[o]["kl_mean"] for r in per_cfg["erm_fixed"]]) for o in ("graph", "node", "edge")}
        erm_src = _mean([r["source_probe"]["balanced_acc"] for r in per_cfg["erm_fixed"]]) or 0.0
        erm_tgt = _mean([r["target_eval"]["balanced_acc"] for r in per_cfg["erm_fixed"]]) or 0.0
        gentle_agg = {}
        for lbl, recs in per_cfg.items():
            src_b = _mean([r["source_probe"]["balanced_acc"] for r in recs])
            tgt_b = _mean([r["target_eval"]["balanced_acc"] for r in recs])
            a = dict(label=lbl, source_probe_bacc=src_b, target_eval_bacc=tgt_b,
                     source_drop_vs_erm=float(erm_src - (src_b or 0.0)),
                     target_drop_vs_erm=float(erm_tgt - (tgt_b or 0.0)),
                     target_eval_is_evaluation_only=True)
            for o in ("graph", "node", "edge"):
                kl = _mean([r[o]["kl_mean"] for r in recs]); a[f"{o}_kl_mean"] = kl
                a[f"{o}_reduction_vs_erm"] = (float((erm_kl[o] - kl) / erm_kl[o]) if (erm_kl[o] and erm_kl[o] > 0 and kl is not None) else None)
                a[f"{o}_reduce30_seeds"] = int(sum(
                    1 for r in recs if erm_kl[o] and erm_kl[o] > 0 and (erm_kl[o] - r[o]["kl_mean"]) / erm_kl[o] >= 0.30))
            gentle_agg[lbl] = a
        # SOURCE-ONLY selection firewall (target_eval cannot influence which configs get confirmed):
        sel = decide_gentle_selection(gentle_agg)
        source_only_reducers = sel["source_only_reducers"]
        best_reducer = sel["best_reducer"]
        confirm_labels = sel["confirmation_labels"]
        final_task_preserving_reducers = sel["final_task_preserving_reducers"]
        gentle_gate_pass_source_only = sel["gentle_gate_pass_source_only"]
        gentle_gate_pass_with_target_retention = sel["gentle_gate_pass_with_target_retention"]
        gentle_gate_pass = gentle_gate_pass_with_target_retention   # back-compat alias = final verdict
        confirmation, confirmation_per_seed = {}, {}
        for lbl in confirm_labels:
            _, lg, ln, le = next(c for c in GENTLE_CONFIGS if c[0] == lbl)
            recs = []
            for seed in args.seeds:
                rec = _train_eval(cand, (lg, ln, le), fold, seed, args, args.device, n_perm=args.n_perm_confirm)
                _save(rec, f"{fold_tag}_confirm_{lbl}_seed{seed}_nperm{args.n_perm_confirm}.json")
                recs.append(rec)
            confirmation_per_seed[lbl] = {int(r["seed"]): {o: {k: r[o][k] for k in
                ("kl_mean", "permutation_mean", "permutation_p", "positive_excess", "clears_null")}
                for o in ("graph", "node", "edge")} for r in recs}
            confirmation[lbl] = {f"{o}_clears_null_count": int(sum(r[o]["clears_null"] for r in recs))
                                 for o in ("graph", "node", "edge")}
        summary["part_b"] = dict(
            baseline=best_baseline, erm_reference_kl=erm_kl, gentle=gentle_agg, best_reducer=best_reducer,
            source_only_reducers=source_only_reducers,
            final_task_preserving_reducers=final_task_preserving_reducers,
            task_preserving_reducers=final_task_preserving_reducers,        # back-compat alias
            gentle_gate_pass_source_only=gentle_gate_pass_source_only,
            gentle_gate_pass_with_target_retention=gentle_gate_pass_with_target_retention,
            gentle_gate_pass=gentle_gate_pass,                              # = final target-retention verdict
            confirmation_labels=confirm_labels,
            confirmation_label_selection_uses_target_eval=False,
            target_eval_used_for_verdict_only=True,
            n_perm_confirm=int(args.n_perm_confirm),
            confirmation=confirmation, confirmation_per_seed=confirmation_per_seed)
        print(f"\n  source_only_reducers={source_only_reducers}  confirm_labels={confirm_labels} "
              f"(SOURCE-ONLY) @n_perm={args.n_perm_confirm}\n  final_task_preserving_reducers="
              f"{final_task_preserving_reducers} (verdict only; target_eval never selects)", flush=True)

    json.dump(summary, open(OUT_DIR / f"{fold_tag}_baseline_repair_summary.json", "w"), indent=2)
    print(f"\n[phase3a-R] wrote {OUT_DIR / f'{fold_tag}_baseline_repair_summary.json'}")
    print("Reviewer decides next path; this run only reports evidence (target_eval is evaluation-only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
