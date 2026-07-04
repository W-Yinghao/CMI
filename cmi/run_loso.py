"""Leave-one-subject-out (Protocol A) runner for the Tri-CMI harness.

For each target subject: train on the remaining (source) subjects, evaluate balanced
accuracy / macro-F1 / ECE / NLL on the held-out target, plus the source-only
conditional-domain-leakage probe and label separability. Aggregates worst-subject and
std across targets. One invocation sweeps several frameworks (methods) on one
(dataset, backbone) so they share identical splits.

Frameworks are selected via --configs "method:lam[:gamma]" (NOT --methods).
Example (GPU via slurm):
  python -m cmi.run_loso --dataset BNCI2014_001 --backbone EEGNet \
      --configs erm:0 lpc_prior:0.3 cdann:1 --epochs 300 --out results/2a_eegnet.json
Smoke test (CPU, tiny):
  python -m cmi.run_loso --dataset BNCI2014_001 --backbone EEGNet \
      --configs erm:0 lpc_prior:0.3 --epochs 4 --max_subjects 3 --resample 100 --device cpu

For SCPS datasets (e.g. ADFTD) the per-target balanced accuracy is degenerate (one class
per subject); read the **pooled** and **subject-level** balanced accuracy in the summary.
"""
from __future__ import annotations
import argparse, json, os, subprocess, time
import numpy as np


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()[:12]
    except Exception:
        return "unknown"


def _ablate_bacc(bb, Xte, yte, mode, device, bs=256):
    """Target balanced accuracy under a graph-usage ablation (zero_graph / permute_nodes). Requires
    backbone.ablate(x, mode). zero_graph ~ chance proves no head/prior bypass; permute_nodes << normal
    bAcc proves trial-specific node content is used."""
    import torch
    from cmi.eval.metrics import classification_metrics
    bb.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xte), bs):
            xb = torch.as_tensor(Xte[i:i + bs], dtype=torch.float32, device=device)
            probs.append(torch.softmax(bb.ablate(xb, mode), 1).cpu().numpy())
    return float(classification_metrics(np.concatenate(probs), yte)["balanced_acc"])


# 10-20 montage channel names for the MI datasets (MOABB order), so FBLGGGraph does LOCAL-GLOBAL grouping
# on real electrode regions instead of a contiguous index partition. Only used when --backbone FBLGGGraph.
_DATASET_CH_NAMES = {
    "BNCI2014_001": ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                     "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"],
    "BNCI2015_001": ["FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CPz", "CP4"],
}


def _infer_ch_names(dataset, n_ch):
    """Return (ch_names, source_tag) for FBLGGGraph. Never SILENTLY misapplies a preset: a channel-count
    mismatch loudly warns and falls back to index-partition grouping (ch_names=None)."""
    names = _DATASET_CH_NAMES.get(dataset)
    if names is not None and len(names) == n_ch:
        return names, f"preset:{dataset}"
    if names is not None:                                # preset exists but n_ch disagrees -> do NOT use it
        print(f"WARNING: ch_names preset for {dataset} has {len(names)} channels but the data has {n_ch}; "
              f"falling back to index-partition grouping.", flush=True)
        return None, f"index_fallback(mismatch:{len(names)}!={n_ch})"
    return None, "index_fallback(no_preset)"


def parse_config(c, default_beta=0.0):
    """Parse a --configs entry 'method[:a[:b[:c[:d]]]]' -> (label, method, lam, gamma, lam_edge, z_margin,
    dec_scale, node_w). node_w is lambda_node for graphdualpc and default_beta (the VIB beta) otherwise, so
    existing methods are unaffected.

      graphcmi:<lambda_g>:<lambda_node>:<lambda_edge>            (unchanged: gamma == lambda_node)
      graphdualpc:<lambda_g>:<lambda_node>:<lambda_edge>:<gamma_dec>[:<dec_scale>]
                                        (lam=lg, node_w=lnode, lam_edge=le, gamma=gdec; dec_scale default 1.0)
    """
    parts = c.split(":"); method = parts[0]
    nums = [float(x) for x in parts[1:]]
    lam_edge, z_margin, dec_scale, node_w, lam_spatial = 0.0, 0.0, 1.0, default_beta, 0.0
    if method == "supcon":                       # supcon:<gamma>
        lam, gamma = 0.0, nums[0]
    elif method in ("lpc_supcon", "lpc_simclr", "lpc_byol"):   # <method>:<lam=CMI>:<gamma=SSL>
        lam, gamma = nums[0], nums[1]
    elif method == "graphcmi":                   # graphcmi:<lam=global>:<lam_node>:<lam_edge>  (gamma == lam_node)
        lam, gamma, lam_edge = nums[0], (nums[1] if len(nums) > 1 else 0.0), (nums[2] if len(nums) > 2 else 0.0)
    elif method == "graphdualpc":                # graphdualpc:<lambda_g>:<lambda_node>:<lambda_edge>:<gamma_dec>[:<dec_scale>]
        lam = nums[0] if nums else 0.0
        node_w = nums[1] if len(nums) > 1 else 0.0     # -> train_model(beta=node_w) == lambda_node
        lam_edge = nums[2] if len(nums) > 2 else 0.0
        gamma = nums[3] if len(nums) > 3 else 0.0       # -> gamma_dec
        dec_scale = nums[4] if len(nums) > 4 else 1.0   # -> train_model(dec_scale=...); default 1.0 keeps 4-field configs unchanged
    elif method == "fbdualpc":                   # fbdualpc:<lambda_g>:<lambda_node>:<lambda_spatial>:<lambda_edge>:<gamma_dec>[:<dec_scale>]
        lam = nums[0] if nums else 0.0                  # lambda_g   (graph encoder CMI)
        node_w = nums[1] if len(nums) > 1 else 0.0      # lambda_node (-> train_model beta)
        lam_spatial = nums[2] if len(nums) > 2 else 0.0  # lambda_spatial (P6-A: FBCSP spatial encoder CMI)
        lam_edge = nums[3] if len(nums) > 3 else 0.0     # lambda_edge
        gamma = nums[4] if len(nums) > 4 else 0.0        # gamma_dec
        dec_scale = nums[5] if len(nums) > 5 else 1.0
    elif method in ("dual", "dualc", "dualpc", "dualpc_hinge", "dualpc_marginal"):
        lam, gamma = nums[0], (nums[1] if len(nums) > 1 else nums[0])
        if method == "dualpc_hinge":
            z_margin = nums[2] if len(nums) > 2 else 0.0
            dec_scale = nums[3] if len(nums) > 3 else 1.0
    else:                                        # erm | {marginal,chain,lpc_uniform,lpc_prior}:<lam>
        lam, gamma = (nums[0] if nums else 0.0), 0.0
    return (c, method, lam, gamma, lam_edge, z_margin, dec_scale, node_w, lam_spatial)
import torch

from cmi.data import moabb_data, emotion_data, diagnosis_data, processed_data
from cmi.data.moabb_data import domain_labels, loso_splits, leave_one_session_splits

EMOTION = {"SEED", "SEED_IV", "DEAP", "DEAP_valence", "DEAP_arousal", "DEAP_quadrant"}
DIAGNOSIS = {"ADFTD", "ADFTD_bin", "MUMTAZ", "TUAB"}
PROCESSED = {"Stieger2021", "PhysionetMI"}            # lab-preprocessed datalake, epoched MI (scale)


def load(name, **kw):
    if name in EMOTION:
        return emotion_data.load(name, **{k: v for k, v in kw.items() if k != "resample"})
    if name in DIAGNOSIS:
        dname = name if name in ("MUMTAZ", "TUAB") else "ADFTD"
        return diagnosis_data.load(dname, binary=name.endswith("_bin"),
                                   **{k: v for k, v in kw.items() if k not in ("tmin", "tmax")})
    if name in PROCESSED:
        return processed_data.load(name, **{k: v for k, v in kw.items() if k not in ("tmin", "tmax")})
    return moabb_data.load(name, **kw)
from cmi.models.backbones import build_backbone
from cmi.train.trainer import train_model, predict, resolve_dec_margin
from cmi.eval.metrics import (classification_metrics, leakage_probe, marginal_leakage_probe,
                              decoder_leakage_probe, label_separability, add_decoder_valid_means)
from cmi.methods.regularizers import METHODS


def _remap(d):
    """Make domain ids contiguous 0..K-1 after a subject is removed."""
    uniq = {v: i for i, v in enumerate(sorted(np.unique(d)))}
    return np.array([uniq[v] for v in d], dtype=np.int64), len(uniq)


def _global_metrics(folds):
    """Pool predictions across all LOSO folds -> valid global metrics for ANY task structure
    (the only valid ones for SCPS, where a single held-out subject has one class).
      pooled_*   : window-level over all held-out windows.
      subject_*  : one majority-vote prediction per held-out target -> across-target metric."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    Y = np.concatenate([y for y, _, _ in folds]); P = np.concatenate([p for _, p, _ in folds])
    subj_true, subj_pred = [], []
    for y, p, _ in folds:
        subj_true.append(int(np.bincount(y).argmax()))      # dominant true label of this target
        subj_pred.append(int(np.bincount(p).argmax()))      # majority-vote prediction
    return dict(
        pooled_balanced_acc=float(balanced_accuracy_score(Y, P)),
        pooled_macro_f1=float(f1_score(Y, P, average="macro")),
        subject_balanced_acc=float(balanced_accuracy_score(subj_true, subj_pred)),
        subject_macro_f1=float(f1_score(subj_true, subj_pred, average="macro")),
    )


def _imbalance_subsample(X, y, d, rho, n_cls, seed):
    """Induce label-domain imbalance in the SOURCE: each domain g keeps all of its
    'preferred' class (g mod n_cls) but only a fraction (1-rho) of the other classes.
    Makes p(D|Y) far from uniform -> the regime where the pi_y(D) correction matters.
    Target is left untouched (still balanced)."""
    rng = np.random.default_rng(seed)
    keep = []
    for g in np.unique(d):
        pref = int(g) % n_cls
        for c in range(n_cls):
            idx = np.where((d == g) & (y == c))[0]
            if len(idx) == 0:
                continue
            if c == pref:
                keep.append(idx)
            else:
                k = max(1, int(round(len(idx) * (1 - rho))))
                keep.append(rng.choice(idx, k, replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep], d[keep]


def _summarize(results, pooled):
    """Build the per-config summary over WHATEVER folds are done so far (valid for partial runs)."""
    summary = {}
    for m in results:
        if not results[m]:
            continue
        ba = np.array([r["balanced_acc"] for r in results[m]])
        acc = np.array([r["accuracy"] for r in results[m]])
        summary[m] = dict(
            per_target_balanced_acc_mean=float(ba.mean()), per_target_balanced_acc_std=float(ba.std()),
            worst_target_balanced_acc=float(ba.min()), worst_target_acc=float(acc.min()),
            macro_f1=float(np.mean([r["macro_f1"] for r in results[m]])),
            **_global_metrics(pooled[m]),
            leakage_kl=float(np.mean([r["leakage_kl"] for r in results[m]])),
            leakage_kl_rw=float(np.mean([r["leakage_kl_rw"] for r in results[m]])),
            marginal_leakage_kl=float(np.mean([r["marginal_leakage_kl"] for r in results[m]])),
            marginal_leakage_kl_rw=float(np.mean([r["marginal_leakage_kl_rw"] for r in results[m]])),
            marginal_leakage_advantage=float(np.mean([r["marginal_leakage_advantage"] for r in results[m]])),
            marginal_leakage_advantage_rw=float(np.mean([r["marginal_leakage_advantage_rw"] for r in results[m]])),
            decoder_cmi=float(np.mean([r["decoder_cmi"] for r in results[m]])),
            decoder_cmi_rw=float(np.mean([r["decoder_cmi_rw"] for r in results[m]])),
            decoder_cmi_res=float(np.mean([r["decoder_cmi_res"] for r in results[m]])),
            decoder_cmi_res_rw=float(np.mean([r["decoder_cmi_res_rw"] for r in results[m]])),
            decoder_js_res=float(np.mean([r["decoder_js_res"] for r in results[m]])),
            decoder_js_res_rw=float(np.mean([r["decoder_js_res_rw"] for r in results[m]])),
            decoder_valid_frac=float(np.mean([float(r.get("decoder_valid", False)) for r in results[m]])),
            decoder_min_domain_classes=int(np.min([r.get("decoder_min_domain_classes", 0) for r in results[m]])),
            decoder_single_class_frac=float(np.mean([r.get("decoder_single_class_frac", 1.0) for r in results[m]])),
            leakage_advantage=float(np.mean([r["leakage_advantage"] for r in results[m]])),
            leakage_advantage_rw=float(np.mean([r["leakage_advantage_rw"] for r in results[m]])),
            label_sep=float(np.mean([r["label_sep"] for r in results[m]])),
            inloop_reg=float(np.mean([r["inloop_reg"] for r in results[m]])),
            inloop_dec=float(np.mean([r.get("inloop_dec", 0.0) for r in results[m]])),
            inloop_dec_loss=float(np.mean([r.get("inloop_dec_loss", 0.0) for r in results[m]])),
            train_dec_margin=float(np.mean([r.get("train_dec_margin", 0.0) for r in results[m]])),
            stepA_dom_acc=float(np.mean([r["stepA_dom_acc"] for r in results[m]])),
            # Graph-DualCMI gate aggregates (source retention + graph-usage ablation); target side is
            # per_target_balanced_acc_mean / worst_target_balanced_acc above.
            source_bacc_mean=float(np.mean([r.get("source_bacc", float("nan")) for r in results[m]])),
            worst_source_bacc=float(np.min([r.get("source_bacc", float("nan")) for r in results[m]])),
            n_folds=len(results[m]), per_target=results[m])
        if results[m]:                                   # aggregate every recorded ablation mode (incl zero_temporal)
            for _abl in [k for k in results[m][0] if k.startswith("ablate_") and k.endswith("_target_bacc")]:
                summary[m][_abl + "_mean"] = float(np.mean([r[_abl] for r in results[m]]))
        if results[m] and "ts_balanced_acc" in results[m][0]:          # CIPC transductive-corrected metrics
            summary[m]["transduct_balanced_acc_mean"] = float(np.mean([r["ts_balanced_acc"] for r in results[m]]))
            summary[m]["transduct_balanced_acc_std"] = float(np.std([r["ts_balanced_acc"] for r in results[m]]))
            summary[m]["probe_balanced_acc_mean"] = float(np.mean([r["probe_balanced_acc"] for r in results[m]]))
            summary[m]["transduct_worst"] = float(np.min([r["ts_balanced_acc"] for r in results[m]]))
            for k in [kk for kk in results[m][0] if kk.startswith("ts_") and kk.endswith("_balanced_acc")]:
                summary[m][k + "_mean"] = float(np.mean([r[k] for r in results[m]]))   # ablation-ladder modes
        add_decoder_valid_means(summary[m], results[m])
        for k in ("decoder_cmi_res_null_q", "decoder_cmi_res_excess",
                  "decoder_cmi_res_rw_null_q", "decoder_cmi_res_rw_excess",
                  "decoder_js_res_null_q", "decoder_js_res_excess",
                  "decoder_js_res_rw_null_q", "decoder_js_res_rw_excess"):
            vals = [r[k] for r in results[m] if k in r]
            if vals:
                summary[m][k] = float(np.mean(vals))
    return summary


def _save_out(out_path, args, classes, summary, preds, quiet=False):
    """Write the results JSON + per-fold-probability sidecar .npz. Safe to call repeatedly (overwrites)."""
    out = dict(config=vars(args), classes=classes, summary=summary)
    json.dump(out, open(out_path, "w"), indent=2)
    if not quiet:
        print(f"\nsaved -> {out_path}")
    try:                                              # sidecar: per-fold probs (ECE/NLL/new metrics, no retrain)
        blob = {}
        for lbl, folds in preds.items():
            if not folds:
                continue
            blob[f"{lbl}::prob"] = np.concatenate([p for p, _, _ in folds])
            blob[f"{lbl}::y"] = np.concatenate([yy for _, yy, _ in folds])
            blob[f"{lbl}::tgt"] = np.concatenate([np.array([t] * len(yy)) for _, yy, t in folds])
        if blob:
            np.savez_compressed(out_path.rsplit(".", 1)[0] + ".preds.npz", **blob)
            if not quiet:
                print(f"saved -> {out_path.rsplit('.', 1)[0]}.preds.npz (per-fold probabilities)")
    except Exception as e:
        if not quiet:
            print(f"[warn] prob sidecar not saved: {e}")
    return out


def run(args):
    run_git_sha = _git_sha()          # capture ONCE at launch so all folds share one provenance SHA
    from cmi.train.trainer import ALL_METHODS
    bad = [c for c in args.configs if c.split(":")[0] not in ALL_METHODS]
    if bad:                                          # fail fast, before the slow data load
        raise ValueError(f"unknown method(s) in configs: {bad}; allowed: {sorted(ALL_METHODS)}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    subjects = None
    X, y, meta, classes = load(args.dataset, subjects=subjects,
                               tmin=args.tmin, tmax=args.tmax, resample=args.resample)
    if args.max_subjects:
        keep = sorted(meta["subject"].unique())[:args.max_subjects]
        m = meta["subject"].isin(keep).to_numpy()
        X, y, meta = X[m], y[m], meta[m].reset_index(drop=True)
    # Protocol A (LOSO): domain=subject, leave one subject out.
    # Protocol B (cross-session): domain=subject*session, leave one session out.
    if args.protocol == "cross_session":
        dom_mode, splits = "subject_session", list(leave_one_session_splits(meta))
    else:
        dom_mode, splits = "subject", list(loso_splits(meta))
    # Explicit fold selection (pilot): keep only the requested LOSO indices, in split order. This is NOT
    # --max_subjects (which truncates the SOURCE pool); it picks exactly which held-out targets to run,
    # so e.g. BNCI2015_001 fold 9 can be named precisely. Recorded for provenance.
    requested_target_indices = None
    if args.target_indices is not None:
        requested_target_indices = list(args.target_indices)
        bad = [i for i in requested_target_indices if i < 0 or i >= len(splits)]
        if bad:
            raise ValueError(f"--target_indices {bad} out of range for {len(splits)} folds (0..{len(splits)-1})")
        splits = [splits[i] for i in requested_target_indices]
    args.requested_target_indices = requested_target_indices
    args.actual_target_subjects = [str(s[0]) for s in splits]
    print(f"  target selection: requested_indices={requested_target_indices} -> "
          f"targets={args.actual_target_subjects} ({len(splits)} folds)", flush=True)
    dom_all, _ = domain_labels(meta, dom_mode)
    if args.fmin or args.fmax:                       # dual-band: encoder broadband vs covariance 8-30
        from cmi.data.alignment import bandpass
        X = bandpass(X, args.fmin, args.fmax, args.resample)
    if args.align in ("ea", "ra"):                   # signal-level per-subject recentering (label-free)
        from cmi.data.alignment import euclidean_align, riemannian_align
        X = (euclidean_align if args.align == "ea" else riemannian_align)(X, dom_all)
        print(f"  applied {args.align.upper()} alignment per domain", flush=True)
    n_cls = len(classes); n_ch, n_t = X.shape[1], X.shape[2]
    print(f"[{args.dataset}] X={X.shape} classes={classes} subjects={sorted(meta['subject'].unique())} "
          f"device={device}", flush=True)

    configs = [parse_config(c, default_beta=args.beta) for c in args.configs]  # +node_w slot (lambda_node for graphdualpc)
    results = {lbl: [] for lbl, *_ in configs}
    pooled = {lbl: [] for lbl, *_ in configs}        # raw (y, pred, target) for global/subject metrics
    preds = {lbl: [] for lbl, *_ in configs}         # per-fold probabilities -> sidecar .npz
    t0 = time.time()
    for tgt, tr_mask, te_mask in splits:
        Xtr_all, ytr_all = X[tr_mask], y[tr_mask]
        dtr_all, n_dom = _remap(dom_all[tr_mask])
        if args.imbalance > 0:                       # induce label-domain imbalance in source
            Xtr_all, ytr_all, dtr_all = _imbalance_subsample(
                Xtr_all, ytr_all, dtr_all, args.imbalance, n_cls, args.seed)
        Xte, yte = X[te_mask], y[te_mask]
        if args.align == "ea_strict":                # strict DG: target whitened by source-pool only
            from cmi.data.alignment import euclidean_align_strict
            Xtr_all, Xte = euclidean_align_strict(Xtr_all, dtr_all, Xte)
        nch, nt = n_ch, n_t
        if args.backbone == "LogCov":                # frozen geometric features (fit on source)
            from cmi.data.geometric import tangent_features
            Xtr_all, Xte = tangent_features(Xtr_all, Xte)
            if args.align == "ha":                   # EXPLORATORY: hyperbolic per-domain recentering of tangent feats
                from cmi.data.alignment import hyperbolic_align
                te_dom = np.full(len(Xte), n_dom)    # target is its own (held-out) domain
                allF = hyperbolic_align(np.concatenate([Xtr_all, Xte]),
                                        np.concatenate([dtr_all, te_dom]))
                Xtr_all, Xte = allF[:len(Xtr_all)], allF[len(Xtr_all):]
            Xtr_all, Xte = Xtr_all[:, None, :], Xte[:, None, :]   # [N,1,D] for the MLP backbone
            nch, nt = 1, Xtr_all.shape[2]
        # source probe/eval split for leakage + label-sep diagnostics
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(Xtr_all)); cut = int(0.7 * len(idx))
        pi, ei = idx[:cut], idx[cut:]
        # P3-E: deterministic source-only early-stopping val subject (one held-out SOURCE domain, seeded).
        # First-scaffold heuristic (upgradeable to inner LOSO); independent of the target and its labels.
        sval_doms = None
        if args.source_val_early_stop:
            _srcdoms = np.unique(dtr_all)
            if len(_srcdoms) > 1:
                sval_doms = [int(np.random.default_rng(args.seed).permutation(_srcdoms)[0])]
        # FBLGGGraph electrode grouping: prefer the central_strip_v1 montage preset (P3-H); else the
        # 10-20 region/index builder (P3-F.3). ch_names come from a dataset preset (P3-F.3).
        ch_names, ch_names_source = (None, None)
        cs_groups = cs_named = grouping_scheme = grouping_warning = None
        if args.backbone in ("FBLGGGraph", "FBCSPLGGGraph"):
            ch_names, ch_names_source = _infer_ch_names(args.dataset, nch)
            from cmi.models.fb_lgg_dualcmi import central_strip_groups, _CENTRAL_STRIP_V1
            cs_groups, cs_named, cs_warn = central_strip_groups(args.dataset, ch_names)
            if cs_groups is not None:
                grouping_scheme = "central_strip_v1"
            elif args.dataset in _CENTRAL_STRIP_V1:
                # dataset HAS a preset but it did not resolve -> fail closed (never run F0 on a mis-grouped
                # MI montage, and never silently fall back to index partition).
                raise SystemExit(f"ABORT: central_strip_v1 preset for {args.dataset} did not resolve: {cs_warn}")
            else:
                grouping_scheme, grouping_warning = "region_or_index", cs_warn
        for lbl, method, lam, gamma, lam_edge, z_margin, dec_scale, node_w, lam_spatial in configs:
            bb = build_backbone(args.backbone, nch, nt, n_cls, device=device, ch_names=ch_names,
                                groups=cs_groups, group_names=cs_named, grouping_scheme=grouping_scheme,
                                fusion_floor=args.fusion_floor, spatial_mode=args.spatial_mode,
                                cov_shrinkage=args.cov_shrinkage, cov_eps=args.cov_eps)
            if args.beta > 0 and method not in ("graphdualpc", "fbdualpc"):   # VIB (graphdualpc/fbdualpc use beta=lambda_node, not VIB)
                from cmi.methods.vib import VIBBackbone
                bb = VIBBackbone(bb, n_cls).to(device)
            bb, _, diag = train_model(bb, Xtr_all, ytr_all, dtr_all, n_cls, method=method,
                                lam=lam, gamma=gamma, lam_edge=lam_edge, beta=node_w, balance=args.balance,
                                label_correct=args.label_correct, reweight_dual=args.reweight_dual,
                                dec_margin=resolve_dec_margin(method, args.dec_margin),
                                z_margin=z_margin, dec_scale=dec_scale, lam_spatial=lam_spatial,
                                epochs=args.epochs, bs=args.bs,
                                warmup=args.warmup, n_inner=args.n_inner, sampler=args.sampler,
                                prior_mode=args.prior, prior_alpha=args.prior_alpha,
                                early_stop=args.source_val_early_stop, source_val_domains=sval_doms,
                                weight_decay=args.weight_decay, device=device, seed=args.seed)
            prob = predict(bb, Xte, device)
            cm = classification_metrics(prob, yte)   # per-target (valid only for MCPS targets)
            ts = {}
            if args.transduct != "off":              # CIPC: transductive correction on penultimate features
                from cmi.train.trainer import embed
                from cmi.eval.label_shift import transduct_predict, transduct_all
                z_se = embed(bb, Xtr_all[ei], device); z_te = embed(bb, Xte, device)
                pi_S = np.bincount(ytr_all, minlength=n_cls).astype(float); pi_S /= pi_S.sum()
                if args.transduct == "all":          # ablation ladder in one pass
                    probs = transduct_all(z_se, ytr_all[ei], z_te, pi_S, n_cls, shrink=args.transduct_shrink)
                    ts = {f"ts_{md}_balanced_acc": classification_metrics(p, yte)["balanced_acc"]
                          for md, p in probs.items()}
                    ts["ts_balanced_acc"] = ts["ts_coral_balanced_acc"]          # headline = coral
                    ts["probe_balanced_acc"] = ts["ts_probe_balanced_acc"]
                else:
                    tp = transduct_predict(z_se, ytr_all[ei], z_te, pi_S, n_cls, mode=args.transduct,
                                           shrink=args.transduct_shrink, gate_l1=args.transduct_gate)
                    ts = dict(ts_balanced_acc=classification_metrics(tp["prob"], yte)["balanced_acc"],
                              probe_balanced_acc=classification_metrics(tp["prob_probe_raw"], yte)["balanced_acc"],
                              ts_pi_T=tp["pi_T"])
            lk = leakage_probe(bb, Xtr_all[pi], ytr_all[pi], dtr_all[pi],
                               Xtr_all[ei], ytr_all[ei], dtr_all[ei], n_cls, device=device)
            lk_rw = leakage_probe(bb, Xtr_all[pi], ytr_all[pi], dtr_all[pi],
                                  Xtr_all[ei], ytr_all[ei], dtr_all[ei], n_cls,
                                  device=device, reweight=True)
            mlk = marginal_leakage_probe(bb, Xtr_all[pi], ytr_all[pi], dtr_all[pi],
                                         Xtr_all[ei], ytr_all[ei], dtr_all[ei], n_cls, device=device)
            mlk_rw = marginal_leakage_probe(bb, Xtr_all[pi], ytr_all[pi], dtr_all[pi],
                                            Xtr_all[ei], ytr_all[ei], dtr_all[ei], n_cls,
                                            device=device, reweight=True)
            dlk = decoder_leakage_probe(bb, Xtr_all[pi], ytr_all[pi], dtr_all[pi],   # I(Y;D|Z)
                                        Xtr_all[ei], ytr_all[ei], dtr_all[ei], n_cls, device=device,
                                        null_perms=args.decoder_null_perms,
                                        null_quantile=args.decoder_null_quantile)
            dlk_rw = decoder_leakage_probe(bb, Xtr_all[pi], ytr_all[pi], dtr_all[pi],  # GLS-reweighted Itilde(Y;D|Z)
                                           Xtr_all[ei], ytr_all[ei], dtr_all[ei], n_cls, device=device, reweight=True,
                                           null_perms=args.decoder_null_perms,
                                           null_quantile=args.decoder_null_quantile)
            ls = label_separability(bb, Xtr_all[pi], ytr_all[pi], Xtr_all[ei], ytr_all[ei], device)
            rec = dict(target=str(tgt), accuracy=float((prob.argmax(1) == yte).mean()), **cm, **lk,
                       leakage_kl_rw=lk_rw["leakage_kl"],
                       leakage_advantage_rw=lk_rw["leakage_advantage"],
                       marginal_leakage_kl=mlk["marginal_leakage_kl"],
                       marginal_leakage_kl_rw=mlk_rw["marginal_leakage_kl"],
                       marginal_leakage_advantage=mlk["marginal_leakage_advantage"],
                       marginal_leakage_advantage_rw=mlk_rw["marginal_leakage_advantage"],
                       decoder_cmi=dlk["decoder_cmi"], decoder_cmi_rw=dlk_rw["decoder_cmi"],
                       decoder_cmi_res=dlk["decoder_cmi_res"], decoder_cmi_res_rw=dlk_rw["decoder_cmi_res"],
                       decoder_js_res=dlk["decoder_js_res"], decoder_js_res_rw=dlk_rw["decoder_js_res"],
                       decoder_valid=bool(dlk["decoder_valid"]),
                       decoder_n_domains=dlk["decoder_n_domains"],
                       decoder_min_domain_classes=dlk["decoder_min_domain_classes"],
                       decoder_mean_domain_classes=dlk["decoder_mean_domain_classes"],
                       decoder_single_class_frac=dlk["decoder_single_class_frac"],
                       decoder_domain_class_spans=dlk["decoder_domain_class_spans"],
                       decoder_domain_counts=dlk["decoder_domain_counts"],
                       label_sep=ls, inloop_reg=diag["inloop_reg"], inloop_dec=diag.get("inloop_dec", 0.0),
                       inloop_dec_loss=diag.get("inloop_dec_loss", 0.0),
                       train_dec_margin=diag.get("dec_margin", resolve_dec_margin(method, args.dec_margin)),
                       train_sampler=diag.get("sampler", args.sampler), stepA_dom_acc=diag["stepA_dom_acc"],
                       **ts)
            for src, prefix, key in ((dlk, "decoder_cmi_res", "decoder_cmi_res"),
                                     (dlk_rw, "decoder_cmi_res_rw", "decoder_cmi_res"),
                                     (dlk, "decoder_js_res", "decoder_js_res"),
                                     (dlk_rw, "decoder_js_res_rw", "decoder_js_res")):
                if f"{key}_null_q" in src:
                    rec[f"{prefix}_null_q"] = src[f"{key}_null_q"]
                    rec[f"{prefix}_excess"] = src[f"{key}_excess"]
            # Graph-DualCMI gate metrics: source retention, graph-usage ablation, git provenance, GLS diagnostics.
            rec["method_config"] = lbl
            rec["backbone"] = args.backbone
            rec["git_sha"] = run_git_sha
            if args.backbone in ("FBLGGGraph", "FBCSPLGGGraph"):   # grouping provenance (P3-H / P5)
                rec["ch_names_source"] = ch_names_source
                rec["grouping_scheme"] = getattr(bb, "grouping_scheme", grouping_scheme)
                # named montage groups when available (central_strip_v1); else raw index groups
                rec["channel_groups"] = (bb.group_names if getattr(bb, "group_names", None)
                                         else [list(g) for g in getattr(bb, "groups", [])])
                if grouping_warning:
                    rec["grouping_warning"] = grouping_warning
            if callable(getattr(bb, "gate_summary", None)):        # P5-D: fusion-gate instrumentation (aggregate)
                try:
                    rec.update(bb.gate_summary(torch.as_tensor(Xte[:256], dtype=torch.float32, device=device)))
                except Exception as _e:
                    rec["gate_summary_error"] = str(_e)
            if callable(getattr(bb, "cov_summary", None)):         # P7a: covariance-tangent conditioning diag
                try:
                    rec.update(bb.cov_summary(torch.as_tensor(Xte[:256], dtype=torch.float32, device=device)))
                except Exception as _e:
                    rec["cov_summary_error"] = str(_e)
            rec["source_bacc"] = float(classification_metrics(predict(bb, Xtr_all[ei], device),
                                                              ytr_all[ei])["balanced_acc"])
            if callable(getattr(bb, "ablate", None)):
                # Record every ablation the backbone declares. FB-LGG adds zero_temporal so we can see BOTH
                # branch contributions (graph vs temporal); static DGCNN declares only zero_graph/permute_nodes.
                for _mode in getattr(bb, "meta", {}).get("ablation_modes", ("zero_graph", "permute_nodes")):
                    rec[f"ablate_{_mode}_target_bacc"] = _ablate_bacc(bb, Xte, yte, _mode, device)
            if method in ("graphdualpc", "fbdualpc"):
                for k in ("lambda_g", "lambda_node", "lambda_edge", "gamma_dec", "reg_graph_gls",
                          "reg_node_gls", "reg_edge_gls", "dec_js_res", "dec_ce_res",
                          "stepA_graph_dom_acc_gls", "stepA_node_dom_acc_gls", "stepA_edge_dom_acc_gls",
                          # P3-D decoder-activation diagnostics
                          "loss_ce", "dec_js_res_raw", "dec_js_res_scaled", "loss_dec", "loss_dec_over_ce",
                          "dec_gate_active_frac",
                          # P6-A spatial encoder CMI diagnostics (fbdualpc)
                          "lambda_spatial", "reg_spatial_gls", "loss_spatial", "stepA_spatial_loss_gls"):
                    if k in diag:
                        rec[k] = diag[k]
            for k in ("source_val_subjects", "best_epoch", "best_source_val_bacc",   # P3-E early-stopping metadata
                      "final_val_source_bacc", "final_train_source_bacc"):
                if k in diag:
                    rec[k] = diag[k]
            results[lbl].append(rec)
            pooled[lbl].append((yte, prob.argmax(1), str(tgt)))
            preds[lbl].append((prob.astype("float32"), yte.astype("int16"), str(tgt)))
            print(f"  tgt={tgt} {lbl:15s} bAcc={cm['balanced_acc']*100:5.1f} "
                  f"F1={cm['macro_f1']*100:5.1f} leakKL={lk['leakage_kl']:.3f} "
                  f"labelSep={ls*100:4.1f} ({time.time()-t0:.0f}s)", flush=True)
        if args.out:                                  # INCREMENTAL: persist after EVERY fold so a wall-clock
            _save_out(args.out, args, classes, _summarize(results, pooled), preds, quiet=True)
            print(f"  [ckpt] {len(results[configs[0][0]])} folds saved -> {args.out}", flush=True)

    summary = _summarize(results, pooled)
    print(f"\n=== {args.dataset} / {args.backbone} (LOSO, {args.epochs} ep) ===")
    print(f"{'config':16s} {'PerTgtBAcc':>12s} {'PoolBAcc':>9s} {'SubjBAcc':>9s} {'Worst':>6s} "
          f"{'condKL':>8s} {'PzKLw':>8s} {'inloopKL':>8s} {'qDomAcc':>7s} {'LabelSep':>9s}")
    for m, s in summary.items():
        print(f"{m:16s} {s['per_target_balanced_acc_mean']*100:6.1f}±{s['per_target_balanced_acc_std']*100:4.1f} "
              f"{s['pooled_balanced_acc']*100:8.1f} {s['subject_balanced_acc']*100:8.1f} "
              f"{s['worst_target_acc']*100:6.1f} {s['leakage_kl']:8.3f} {s['marginal_leakage_kl_rw']:8.3f} {s['inloop_reg']:8.3f} "
              f"{s['stepA_dom_acc']*100:6.1f} {s['label_sep']*100:8.1f}")
    out = _save_out(args.out, args, classes, summary, preds) if args.out \
        else dict(config=vars(args), classes=classes, summary=summary)
    return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="EEGNet",
                    choices=["EEGNet", "ShallowConvNet", "Deep4Net", "EEGConformer", "LogCov", "TSMNet",
                             "GraphCMI", "DGCNN", "RGNN", "DGCNNGraph", "FBLGGGraph", "FBCSPLGGGraph"])   # DGCNNGraph = static adjacency; FBLGGGraph = CIGL_47 filterbank+local-global graph; FBCSPLGGGraph = CIGL_49 + FBCSP spatial branch
    ap.add_argument("--target_indices", type=int, nargs="+", default=None,
                    help="run only these LOSO fold indices (0-based, in split order), e.g. 0 1 for the pilot "
                         "or 0 9 for BNCI2015_001. Default: all folds. Recorded in the output as "
                         "requested_target_indices / actual_target_subjects.")
    ap.add_argument("--source_val_early_stop", action="store_true",
                    help="P3-E: hold out one SOURCE subject (deterministic, seeded) as an inner validation "
                         "set and restore the best-source-val-bAcc epoch. Target labels are never used. "
                         "Records source_val_subjects/best_epoch/best_source_val_bacc/final_{train,val}_"
                         "source_bacc per fold. Default OFF (fixed-epoch training, unchanged).")
    ap.add_argument("--fusion_floor", type=float, default=0.0,
                    help="P6-B: FBCSPLGGGraph 3-way gate floor eps -> (1-3eps)*softmax + eps, so no branch is "
                         "fully starved. 0.0 = plain softmax (off). Try 0.05 / 0.10. Only affects FBCSPLGGGraph.")
    ap.add_argument("--spatial_mode", choices=["logvar", "cov_tangent"], default="logvar",
                    help="P7a: FBCSPLGGGraph spatial branch feature. 'logvar' = P6 per-filter log-variance "
                         "(default, byte-identical). 'cov_tangent' = per-band C x C covariance -> SPD tangent "
                         "vech(logm(S)). Only affects FBCSPLGGGraph.")
    ap.add_argument("--cov_shrinkage", type=float, default=0.05,
                    help="P7a cov_tangent shrinkage alpha: S <- (1-a)*S + a*I/C (SPD floor, min eig >= a/C).")
    ap.add_argument("--cov_eps", type=float, default=1e-4,
                    help="P7a cov_tangent eigenvalue floor for logm and trace-normalization stability.")
    # configs = list of "method:lam" (one job can sweep lambda; splits shared across all)
    ap.add_argument("--configs", nargs="+",
                    default=["erm:0", "marginal:1", "chain:1", "lpc_uniform:1", "lpc_prior:1"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--n_inner", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--sampler", default="classbal", choices=["classbal", "raw", "domainbal"],
                    help="classbal/raw PRESERVE p(D|Y) (consistent with empirical pi_y); "
                         "domainbal uniformizes it (use --prior effective for consistency)")
    ap.add_argument("--prior", default="empirical", choices=["empirical", "effective", "subject"])
    ap.add_argument("--prior_alpha", type=float, default=1.0, help="Laplace smoothing of pi_y (zero-support)")
    ap.add_argument("--beta", type=float, default=0.0, help="VIB weight on I(X;Z) (variational KL); 0=off")
    ap.add_argument("--balance", action="store_true", help="class-balanced (BER) CE = label-shift correction")
    ap.add_argument("--label_correct", action="store_true",
                    help="GLS (A4) per-sample label-shift weight w_i=pi*(y)/pi_d(y) on task CE. "
                         "For full CMI-estimator reweighting use method dualpc or --reweight_dual with dual.")
    ap.add_argument("--dec_margin", type=float, default=None,
                    help="decoder gate tau. Default is method-specific: dualpc/dualpc_marginal=0, others=0.02")
    ap.add_argument("--decoder_null_perms", type=int, default=0,
                    help="source-only permutation-null repetitions for residual decoder CMI evaluation; 0=off")
    ap.add_argument("--decoder_null_quantile", type=float, default=0.95,
                    help="quantile of the residual decoder permutation null used for *_excess fields")
    ap.add_argument("--reweight_dual", action="store_true",
                    help="Route B (reweighted-dual): for method 'dual' only, apply the GLS weight "
                         "w_i=pi*(y)/pi_d(y) to BOTH CMI estimators (encoder I(Z;D|Y) weighted KL vs "
                         "the post-GLS domain marginal + decoder I(Y;D|Z) weighted H(Y|Z)-H(Y|Z,D)) so the two "
                         "terms decouple. No effect on non-dual methods or naive 'dual' (flag off).")
    ap.add_argument("--align", default="none", choices=["none", "ea", "ra", "ha", "ea_strict"],
                    help="per-subject recentering (accuracy pipeline only; keep OFF for the raw leakage probe): "
                         "ea=Euclidean, ra=Riemannian (AIRM), ha=hyperbolic (LogCov features, exploratory)")
    ap.add_argument("--fmin", type=float, default=None)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--protocol", default="loso", choices=["loso", "cross_session"])
    ap.add_argument("--imbalance", type=float, default=0.0,
                    help="0=balanced; >0 skews source class distribution per domain (pi_y test)")
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--transduct", default="off",
                    choices=["off", "probe", "coral", "prior", "coral_prior", "pmct", "all"],
                    help="CIPC transductive correction on penultimate features (coral=balanced-acc lever; all=ablation ladder)")
    ap.add_argument("--transduct_shrink", type=float, default=0.1, help="BBSE prior shrink toward source")
    ap.add_argument("--transduct_gate", type=float, default=0.0, help="L1 gate on |pi_T-pi_S| for prior correction")
    ap.add_argument("--max_subjects", type=int, default=0)
    ap.add_argument("--out", default="")
    return ap


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
