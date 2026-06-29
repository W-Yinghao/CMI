#!/usr/bin/env python
"""CIGL Phase 3A — regularizer-effect PILOT (source-only, BNCI2014_001 fold-0).

Gate-2 established that GraphCMINet-ERM graph objects carry label-conditional subject leakage. Phase 3A
asks the next question — CONTROLLABILITY:

    Can graph/node/edge CMI regularization REDUCE that leakage WITHOUT destroying task performance?

This is still a PILOT (one dataset, one fold, a fixed small set of lambda configs) — NOT full LOSO, NOT
SEED/DEAP, NOT a benchmark, NOT a SOTA claim. For each config we train GraphCMINet with the graphcmi
regularizer (ERM = graphcmi:0:0:0), freeze it, and audit leakage with FRESHLY-trained held-out probes
(cmi.eval.graph_leakage.audit_graph_objects) — NOT the Step-A training heads.

TARGET-LABEL RULE (strict): target labels are used ONLY for after-the-fact target_eval metrics
(flagged evaluation_only). They are NEVER used for training, early stopping, config selection,
normalization, probe fitting, or the leakage audit. The held-out target subject is excluded from all
source-side computation.

    # always works (CPU, synthetic, tiny):
    python scripts/run_cigl_phase3a_regularizer_pilot.py --dry_run_synthetic --device cpu \
        --seeds 0 1 --n_perm 5 --epochs 3 --probe_epochs 5
    # real (GPU/sbatch):
    python scripts/run_cigl_phase3a_regularizer_pilot.py --dataset BNCI2014_001 --device cuda \
        --seeds 0 1 2 --n_perm 20 --n_perm_confirm 50 --epochs 80 --probe_epochs 100 --gate_alpha 0.05

See docs/CIGL_14_PHASE3A_REGULARIZER_PILOT.md.
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

OUT_DIR = REPO / "results" / "cigl" / "phase3a_pilot"
PHASE = "Phase3A_regularizer_effect_pilot"

# (label, lambda_g, lambda_node, lambda_edge) — ERM is graphcmi:0:0:0
CONFIGS = [
    ("erm",           0.0, 0.0, 0.0),
    ("graph_only",    0.3, 0.0, 0.0),
    ("node_only",     0.0, 0.3, 0.0),
    ("edge_only",     0.0, 0.0, 0.1),
    ("graph_node",    0.3, 0.3, 0.0),
    ("full_cigl",     0.3, 0.3, 0.1),
    ("low_full_cigl", 0.1, 0.1, 0.03),
]
CONFIRM_LABELS = {"erm", "full_cigl"}   # always re-audited at n_perm_confirm; + the best Pareto config


def parse_configs():
    """Return the canonical Phase-3A config list as (label, graphcmi_string, lg, ln, le)."""
    return [(lbl, f"graphcmi:{lg}:{ln}:{le}", lg, ln, le) for (lbl, lg, ln, le) in CONFIGS]


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


@torch.no_grad()
def _extract_graph_features(net, X, device, bs=256):
    net.eval()
    gz, nz, el = [], [], []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
        _, g, n, e = net.forward_graph(xb)
        gz.append(g.cpu().numpy()); nz.append(n.cpu().numpy()); el.append(e.cpu().numpy())
    return np.concatenate(gz), np.concatenate(nz), np.concatenate(el)


def _synthetic_fold(seed, n_per_subj=40, C=6, T=48, n_cls=3, n_subj=4):
    """Tiny EEG-like DGP for the dry-run: label drives a global pattern (task learnable); a subset of
    channels additionally encode subject (leakage). One subject is the held-out target."""
    rng = np.random.default_rng(seed)
    Xs, ys, ds = [], [], []
    label_proto = rng.standard_normal((n_cls, C, T)).astype("float32")
    for s in range(n_subj):
        for _ in range(n_per_subj):
            y = rng.integers(0, n_cls)
            x = 0.6 * label_proto[y] + rng.standard_normal((C, T)).astype("float32")
            x[1:4] += 0.8 * s                       # subject signal on a subset of channels (leakage)
            Xs.append(x); ys.append(y); ds.append(s)
    X = np.stack(Xs).astype("float32"); y = np.array(ys, "int64"); d = np.array(ds, "int64")
    target = n_subj - 1
    tr_mask = d != target; te_mask = d == target
    return X, y, d, tr_mask, te_mask, n_cls, str(target)


def _load_real_fold(args):
    from cmi.data import moabb_data
    try:
        X, y, meta, classes = moabb_data.load(args.dataset, tmin=args.tmin, tmax=args.tmax, resample=args.resample)
    except Exception as e:
        raise SystemExit(f"[phase3a] dataset '{args.dataset}' not loadable offline ({type(e).__name__}: {e}); "
                         f"ensure the MOABB datalake cache is present. Use --dry_run_synthetic to validate.")
    dom_all, _ = moabb_data.domain_labels(meta, "subject")
    splits = list(moabb_data.loso_splits(meta))
    tgt, tr_mask, te_mask = splits[args.fold]
    return X, y, dom_all, tr_mask, te_mask, len(classes), str(tgt)


# ----------------------------------------------------------------------------- one (config, seed)
def _train_extract_audit(cfg, seed, fold, args, device, n_perm):
    """Train GraphCMINet with the config's lambda triple on SOURCE enc-train; extract frozen features on
    a held-out source probe-pool; audit leakage with fresh probes; measure source/target task metrics.
    Target labels touch ONLY target_eval (evaluation_only)."""
    from cmi.models.gnn import GraphCMINet
    from cmi.train.trainer import train_model, predict
    from cmi.eval.metrics import classification_metrics
    label, gstr, lg, ln, le = cfg
    X, y, dom_all, tr_mask, te_mask, n_cls, heldout = fold
    Xs, ys = X[tr_mask], y[tr_mask]
    ds, n_dom = _remap_contiguous(dom_all[tr_mask])
    enc_idx, pool_idx, enc_split = stratified_trial_split_by_y_d(
        ys, ds, train_frac=args.enc_train_frac, seed=seed, min_per_cell=args.min_per_cell)

    # Anchor the ENCODER INIT to `seed` BEFORE construction. train_model seeds only AFTER the backbone
    # exists, so otherwise the init weights are drawn from the ambient global RNG (which differs by loop
    # position) -> Pass-1 and the Pass-2 confirmation would train DIFFERENT models at the same seed, and
    # ERM's leakage_reduction_vs_erm would not be exactly 0. Re-seeding here makes every (config,seed)
    # reproducible and order-independent, and makes the confirmation a faithful same-model re-audit.
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    net = GraphCMINet(X.shape[1], X.shape[2], n_cls).to(device)
    net, _post, diag = train_model(net, Xs[enc_idx], ys[enc_idx], ds[enc_idx], n_cls,
                                   method="graphcmi", lam=lg, gamma=ln, lam_edge=le,
                                   epochs=args.epochs, bs=args.bs, warmup=max(1, args.epochs // 5),
                                   device=device, seed=seed)
    gz, nz, el = _extract_graph_features(net, Xs[pool_idx], device)
    y_pool, d_pool = ys[pool_idx], ds[pool_idx]
    tr_idx, va_idx, split_diag = stratified_trial_split_by_y_d(
        y_pool, d_pool, train_frac=args.train_frac, seed=seed, min_per_cell=args.min_per_cell)
    audit = audit_graph_objects(gz, nz, el, y_pool, d_pool, n_cls, n_dom, n_perm=n_perm, seed=seed,
                                device=device, epochs=args.probe_epochs, train_idx=tr_idx, val_idx=va_idx)
    for obj in ("graph", "node", "edge"):
        audit[obj].pop("node_leakage_map", None); audit[obj].pop("edge_leakage_map", None)

    # task metrics — source held-out (probe pool) and target (EVAL ONLY)
    src = classification_metrics(predict(net, Xs[pool_idx], device), y_pool)
    tgt = classification_metrics(predict(net, X[te_mask], device), y[te_mask])
    rec = dict(
        config=label, graphcmi=gstr, lambda_g=lg, lambda_node=ln, lambda_edge=le, seed=int(seed),
        n_perm=int(n_perm), gate_alpha=float(args.gate_alpha),
        source_probe=dict(balanced_acc=src["balanced_acc"], macro_f1=src["macro_f1"]),
        target_eval=dict(balanced_acc=tgt["balanced_acc"], macro_f1=tgt["macro_f1"], evaluation_only=True),
        graph=_obj_block(audit["graph"], args.gate_alpha),
        node=_obj_block(audit["node"], args.gate_alpha),
        edge=_obj_block(audit["edge"], args.gate_alpha),
        stepA=dict(graph_dom_acc=diag.get("stepA_graph_dom_acc"), node_dom_acc=diag.get("stepA_node_dom_acc"),
                   edge_dom_acc=diag.get("stepA_edge_dom_acc"), graph_loss=diag.get("stepA_graph_loss"),
                   node_loss=diag.get("stepA_node_loss"), edge_loss=diag.get("stepA_edge_loss"),
                   reg_graph=diag.get("reg_graph"), reg_node=diag.get("reg_node"), reg_edge=diag.get("reg_edge")),
        probe_split_diagnostics=split_diag, n_enc_train=int(len(enc_idx)), n_probe_pool=int(len(pool_idx)),
        heldout_subject=heldout)
    return rec


def _obj_block(b, alpha):
    pe = bool(b["kl_mean"] > b["permutation_mean"])
    return dict(kl_mean=b["kl_mean"], permutation_mean=b["permutation_mean"], permutation_p=b["permutation_p"],
                positive_excess=pe, clears_null=bool(pe and b["permutation_p"] <= alpha), gate_alpha=float(alpha))


# ----------------------------------------------------------------------------- aggregation
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _per_seed_meta(dataset, fold_tag, commit, cfg_hash):
    """Provenance + strict-target-label flags stamped on EVERY per-seed record (pass-1 and confirmation)."""
    return dict(exploratory=True, phase=PHASE, setting="strict_source_only_DG",
                used_target_labels_for_training=False, used_target_labels_for_selection=False,
                used_target_covariates=False, target_eval_is_evaluation_only=True,
                target_labels_used_for="evaluation_only metrics",
                dataset=dataset, fold=fold_tag, commit_hash=commit, config_hash=cfg_hash)


def _aggregate(per_seed_recs, ref_kl, reduction_key):
    """Per-config means over seeds + leakage reduction vs `ref_kl` (named `{obj}_{reduction_key}`) +
    clears_null counts. ref_kl is the ERM KL of the SAME pass (pass-1 ERM for pass-1, confirmation ERM
    for the confirmation pass) so reductions are never compared across permutation-power regimes."""
    objs = ("graph", "node", "edge")
    agg = dict(
        config=per_seed_recs[0]["config"], graphcmi=per_seed_recs[0]["graphcmi"],
        lambda_g=per_seed_recs[0]["lambda_g"], lambda_node=per_seed_recs[0]["lambda_node"],
        lambda_edge=per_seed_recs[0]["lambda_edge"], n_seeds=len(per_seed_recs),
        source_probe_bacc=_mean([r["source_probe"]["balanced_acc"] for r in per_seed_recs]),
        source_probe_f1=_mean([r["source_probe"]["macro_f1"] for r in per_seed_recs]),
        target_eval_bacc=_mean([r["target_eval"]["balanced_acc"] for r in per_seed_recs]),
        target_eval_f1=_mean([r["target_eval"]["macro_f1"] for r in per_seed_recs]),
        target_eval_is_evaluation_only=True)
    for o in objs:
        kl = _mean([r[o]["kl_mean"] for r in per_seed_recs])
        agg[f"{o}_kl_mean"] = kl
        agg[f"{o}_clears_null_count"] = int(sum(r[o]["clears_null"] for r in per_seed_recs))
        agg[f"{o}_{reduction_key}"] = (
            float((ref_kl[o] - kl) / ref_kl[o]) if (ref_kl[o] and ref_kl[o] > 0 and kl is not None) else None)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run_synthetic", action="store_true")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_perm", type=int, default=20, help="permutation null for the all-config pass")
    ap.add_argument("--n_perm_confirm", type=int, default=50, help="null for ERM/full_cigl/best-Pareto re-audit")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--probe_epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--enc_train_frac", type=float, default=0.7)
    ap.add_argument("--min_per_cell", type=int, default=2)
    ap.add_argument("--gate_alpha", type=float, default=0.05)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[phase3a] --device cuda requested but CUDA unavailable")
    if args.device == "cpu":
        # tiny per-trial EEG ops: a single thread avoids the many-core dispatch overhead that makes the
        # CPU dry-run ~8x slower (compute is trivial; threads just thrash). No effect on the GPU run.
        torch.set_num_threads(1)

    commit = _git_commit_hash(); cfg_hash = _config_hash(vars(args))
    if args.dry_run_synthetic:
        fold = _synthetic_fold(seed=0)                 # (X,y,dom,tr_mask,te_mask,n_cls,heldout); seeds vary training
        dataset = "synthetic"
        # the dry-run validates the PIPELINE, not significance — a 50-perm confirmation pass is wasteful
        # (and clears_null can't be significant at tiny n_perm anyway). Cap it to keep the dry-run quick.
        if args.n_perm_confirm > args.n_perm:
            print(f"[phase3a] dry-run: capping n_perm_confirm {args.n_perm_confirm}->{args.n_perm}", flush=True)
            args.n_perm_confirm = args.n_perm
    else:
        fold = _load_real_fold(args); dataset = args.dataset
    configs = parse_configs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_tag = f"{dataset}_fold{args.fold}"
    objs = ("graph", "node", "edge")

    def _save_rec(rec, fname):                          # stamp full meta + write per-seed JSON
        rec["meta"] = _per_seed_meta(dataset, fold_tag, commit, cfg_hash)
        json.dump(rec, open(OUT_DIR / fname, "w"), indent=2)

    # ---- Pass 1: all 7 configs x seeds at n_perm ----------------------------------------------
    per_config = {}
    for cfg in configs:
        label = cfg[0]; recs = []
        for seed in args.seeds:
            rec = _train_extract_audit(cfg, seed, fold, args, args.device, n_perm=args.n_perm)
            _save_rec(rec, f"{fold_tag}_{label}_seed{seed}_nperm{args.n_perm}.json")
            recs.append(rec)
            g, n, e = rec["graph"], rec["node"], rec["edge"]
            print(f"[{label:13s} seed{seed}] src_bAcc={rec['source_probe']['balanced_acc']:.3f} "
                  f"tgt_bAcc={rec['target_eval']['balanced_acc']:.3f} | "
                  f"graph_kl={g['kl_mean']:.3f}(p={g['permutation_p']:.3f},clr={int(g['clears_null'])}) "
                  f"node_kl={n['kl_mean']:.3f}(clr={int(n['clears_null'])}) "
                  f"edge_kl={e['kl_mean']:.3f}(clr={int(e['clears_null'])})", flush=True)
        per_config[label] = recs

    erm_kl = {o: _mean([r[o]["kl_mean"] for r in per_config["erm"]]) for o in objs}
    agg = {lbl: _aggregate(per_config[lbl], erm_kl, "pass1_leakage_reduction_vs_erm") for lbl in per_config}

    # ---- best Pareto config: SOURCE-ONLY selection (never target_eval); full_cigl IS eligible ----
    erm_src = agg["erm"]["source_probe_bacc"] or 0.0
    cand = [l for l in agg if l != "erm"]              # only ERM is excluded; full_cigl can be best

    def _score(l):
        gn = [agg[l][f"{o}_pass1_leakage_reduction_vs_erm"] for o in ("graph", "node")]
        gn = [x for x in gn if x is not None]
        red = float(np.mean(gn)) if gn else 0.0
        drop = max(0.0, erm_src - (agg[l]["source_probe_bacc"] or 0.0))
        return red - 1.0 * drop                       # leakage reduction penalized by source-task drop
    best_pareto = max(cand, key=_score) if cand else None

    # ---- Pass 2: confirmation re-audit at n_perm_confirm for ERM, full_cigl, best_pareto ---------
    confirm_labels = sorted(CONFIRM_LABELS | ({best_pareto} if best_pareto else set()))
    confirm_recs, confirmation_per_seed = {}, {}
    for label in confirm_labels:
        cfg = next(c for c in configs if c[0] == label); recs = []
        for seed in args.seeds:
            # same (config, seed) -> with the pre-construction seed, this is the SAME frozen model as
            # pass-1, re-audited only at higher permutation power.
            rec = _train_extract_audit(cfg, seed, fold, args, args.device, n_perm=args.n_perm_confirm)
            _save_rec(rec, f"{fold_tag}_confirm_{label}_seed{seed}_nperm{args.n_perm_confirm}.json")
            recs.append(rec)
        confirm_recs[label] = recs
        confirmation_per_seed[label] = {
            int(r["seed"]): {o: {k: r[o][k] for k in
                ("kl_mean", "permutation_mean", "permutation_p", "positive_excess", "clears_null", "gate_alpha")}
                for o in objs} for r in recs}

    # confirmation reductions use the CONFIRMATION ERM as reference (NOT the pass-1 ERM)
    confirm_erm_kl = {o: _mean([r[o]["kl_mean"] for r in confirm_recs["erm"]]) for o in objs}
    confirm = {lbl: _aggregate(confirm_recs[lbl], confirm_erm_kl, "confirm_leakage_reduction_vs_confirm_erm")
               for lbl in confirm_labels}
    for label in confirm_labels:
        c = confirm[label]
        print(f"[confirm {label:13s} n_perm={args.n_perm_confirm}] clr g/n/e = "
              f"{c['graph_clears_null_count']}/{c['node_clears_null_count']}/{c['edge_clears_null_count']}"
              f" of {len(args.seeds)}", flush=True)

    summary = dict(
        meta=dict(exploratory=True, phase=PHASE, setting="strict_source_only_DG", dataset=dataset,
                  fold=fold_tag, heldout_subject=fold[6], seeds=list(args.seeds),
                  n_perm=int(args.n_perm), n_perm_confirm=int(args.n_perm_confirm),
                  gate_alpha=float(args.gate_alpha), epochs=int(args.epochs), probe_epochs=int(args.probe_epochs),
                  commit_hash=commit, config_hash=cfg_hash,
                  used_target_labels_for_training=False, used_target_covariates=False,
                  used_target_labels_for_selection=False, target_eval_is_evaluation_only=True,
                  target_labels_used_for="evaluation_only metrics"),
        pass1_reference_kl=erm_kl, confirm_reference_kl=confirm_erm_kl,
        best_pareto_config=best_pareto, per_config=agg, confirmation=confirm,
        confirmation_per_seed=confirmation_per_seed)
    json.dump(summary, open(OUT_DIR / f"{fold_tag}_phase3a_summary.json", "w"), indent=2)
    print(f"\n[phase3a] wrote {OUT_DIR / f'{fold_tag}_phase3a_summary.json'}")

    # ---- exploratory Pareto read (no decision) -------------------------------------------------
    print(f"\n=== Phase 3A pilot read (exploratory; reviewer decides A/B/C/D) — best Pareto: {best_pareto} ===")
    print(f"{'config':14s} {'srcBAcc':>8s} {'tgtBAcc':>8s} {'g_red%':>7s} {'n_red%':>7s} {'e_red%':>7s} "
          f"{'clr g/n/e':>10s}")
    for l in [c[0] for c in configs]:
        a = agg[l]
        pr = lambda v: f"{100*v:6.1f}" if v is not None else "   n/a"
        print(f"{l:14s} {a['source_probe_bacc']:8.3f} {a['target_eval_bacc']:8.3f} "
              f"{pr(a['graph_pass1_leakage_reduction_vs_erm'])} {pr(a['node_pass1_leakage_reduction_vs_erm'])} "
              f"{pr(a['edge_pass1_leakage_reduction_vs_erm'])} "
              f"{a['graph_clears_null_count']}/{a['node_clears_null_count']}/{a['edge_clears_null_count']}")
    print("target_eval is EVALUATION-ONLY (never used for training or config selection).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
