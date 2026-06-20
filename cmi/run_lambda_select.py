"""Constrained penalty selection via SOURCE-ONLY model selection (no target labels, strict DG).
For each LOSO target fold: hold out a fraction of SOURCE subjects as validation, train each candidate λ
(including λ=0=ERM) on the rest of source, pick the source-only candidate, then RETRAIN on ALL source and
test on the target. Because λ=0 is always a candidate, the selected method is ≈ ERM-or-better by
construction under the selected source-only criterion.

The default criterion is source-val balanced accuracy. For DualPC/Route-C experiments, `--select_rule
guarded_probe` keeps only candidates within a source-val bAcc tolerance and breaks ties by source-only
representation probes: GLS conditional leakage, GLS marginal P(z) leakage, and JS P(y|Z) consistency. The
raw conditional leakage and CE residual P(y|Z) probe are still recorded as diagnostics. This also supports
a grid over `--dec_margins`,
avoiding target-tuned decoder gates.
Also trains a fixed-ERM model per fold for a direct constrained-λ-vs-ERM comparison.

  python -m cmi.run_lambda_select --dataset BNCI2014_001 --lams 0 0.05 0.1 0.3 --out results/lamsel_2a.json
  python -m cmi.run_lambda_select --dataset MUMTAZ --method dualpc --lams 0 0.05 0.1 --gammas 0.05 0.1
  python -m cmi.run_lambda_select --dataset MUMTAZ --method dualpc_marginal --lams 0 0.05 --gammas 0.05
  python -m cmi.run_lambda_select --dataset BNCI2014_001 --backbone LogCov --method dualpc \
      --lams 0 0.05 --gammas 0.05 --dec_margins 0 0.02 --select_rule guarded_probe --device cpu
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from cmi.run_loso import load, _remap, _global_metrics
from cmi.data.moabb_data import domain_labels, loso_splits
from cmi.models.backbones import build_backbone
from cmi.train.trainer import train_model, predict, resolve_dec_margin
from cmi.eval.metrics import leakage_probe, marginal_leakage_probe, decoder_leakage_probe


def _candidates(args):
    lams = [float(x) for x in args.lams]
    if 0.0 not in lams:
        lams = [0.0] + lams
    default_margin = resolve_dec_margin(args.method, args.dec_margin)
    margins = [float(x) for x in (args.dec_margins or [default_margin])]
    if args.method in {"dual", "dualc", "dualpc", "dualpc_hinge", "dualpc_marginal"}:
        gammas = [float(x) for x in (args.gammas or args.lams)]
        z_margins = [float(x) for x in (args.z_margins or [0.0])]
        dec_scales = [float(x) for x in (args.dec_scales or [1.0])]
        out = [(0.0, 0.0, default_margin, 0.0, 1.0)]
        out += [
            (la, ga, tau, z, ds)
            for la in lams if la != 0.0
            for ga in gammas
            for tau in margins
            for z in (z_margins if args.method == "dualpc_hinge" else [0.0])
            for ds in (dec_scales if args.method == "dualpc_hinge" else [1.0])
        ]
        return out
    return [(la, 0.0, default_margin, 0.0, 1.0) for la in lams]


def _label(method, lam, gamma, dec_margin=None, default_margin=None, z_margin=0.0, dec_scale=1.0):
    if lam == 0 and gamma == 0:
        return "erm:0"
    if method == "dualpc_hinge":
        s = f"{method}:{lam:g}:{gamma:g}:{z_margin:g}:{dec_scale:g}"
        if dec_margin is not None and default_margin is not None and abs(dec_margin - default_margin) > 1e-12:
            s += f":tau={dec_margin:g}"
        return s
    if method in {"dual", "dualc", "dualpc", "dualpc_marginal"}:
        s = f"{method}:{lam:g}:{gamma:g}"
        if dec_margin is not None and default_margin is not None and abs(dec_margin - default_margin) > 1e-12:
            s += f":tau={dec_margin:g}"
        return s
    return f"{method}:{lam:g}"


def _fit(Xtr, ytr, dtr, n_cls, lam, gamma, dec_margin, z_margin, dec_scale, args, device, epochs):
    d, n_dom = _remap(dtr)
    bb = build_backbone(args.backbone, Xtr.shape[1], Xtr.shape[2], n_cls, device=device)
    method = "erm" if lam == 0 and gamma == 0 else args.method
    bb, _, _ = train_model(bb, Xtr, ytr, d, n_cls, method=method, lam=lam, gamma=gamma,
                           dec_margin=dec_margin, epochs=epochs,
                           z_margin=z_margin, dec_scale=dec_scale,
                           bs=args.bs, warmup=args.warmup, n_inner=args.n_inner, sampler=args.sampler,
                           device=device, seed=args.seed)
    return bb


def _source_probe_indices(d, frac, seed):
    """Split trials inside already-selected source domains for source-only representation probes."""
    uniq = np.unique(d)
    if len(uniq) < 2:
        return None, None
    rng = np.random.default_rng(30000 + seed)
    fit, ev = [], []
    for dom in uniq:
        idx = np.where(d == dom)[0]
        if len(idx) < 4:
            continue
        idx = rng.permutation(idx)
        n_ev = int(round(frac * len(idx)))
        n_ev = min(max(1, n_ev), len(idx) - 1)
        ev.append(idx[:n_ev])
        fit.append(idx[n_ev:])
    if not fit or not ev:
        return None, None
    fit = np.concatenate(fit)
    ev = np.concatenate(ev)
    if len(np.unique(d[fit])) < 2 or len(np.unique(d[ev])) < 2:
        return None, None
    return fit, ev


def _selection_probes(bb, Xsrc, ysrc, dsrc_raw, n_cls, args, device, seed, probe_epochs=None):
    """Held-out source probes used only for guarded source selection."""
    dsrc, _ = _remap(dsrc_raw)
    pi, ei = _source_probe_indices(dsrc, args.select_probe_frac, seed)
    out = dict(selection_probe_valid=False, selector_penalty=0.0,
               select_cond_kl=0.0, select_cond_kl_rw=0.0,
               select_pz_kl_rw=0.0, select_py_res_rw=0.0, select_py_js_rw=0.0,
               select_decoder_valid=False, select_decoder_min_domain_classes=0,
               select_decoder_single_class_frac=1.0)
    if pi is None:
        return out
    probe_epochs = int(probe_epochs or args.select_probe_epochs or 50)
    lk = leakage_probe(bb, Xsrc[pi], ysrc[pi], dsrc[pi], Xsrc[ei], ysrc[ei], dsrc[ei],
                       n_cls, device=device, epochs=probe_epochs, seed=seed)
    lk_rw = leakage_probe(bb, Xsrc[pi], ysrc[pi], dsrc[pi], Xsrc[ei], ysrc[ei], dsrc[ei],
                          n_cls, device=device, epochs=probe_epochs, seed=seed, reweight=True)
    mlk = marginal_leakage_probe(bb, Xsrc[pi], ysrc[pi], dsrc[pi], Xsrc[ei], ysrc[ei], dsrc[ei],
                                 n_cls, device=device, epochs=probe_epochs, seed=seed, reweight=True)
    dlk = decoder_leakage_probe(bb, Xsrc[pi], ysrc[pi], dsrc[pi], Xsrc[ei], ysrc[ei], dsrc[ei],
                                n_cls, device=device, epochs=probe_epochs, seed=seed, reweight=True)
    decoder_valid = bool(dlk["decoder_valid"])
    py_term = dlk["decoder_js_res"] if decoder_valid else 0.0
    penalty = (args.select_cond_weight * lk_rw["leakage_kl"]
               + args.select_pz_weight * mlk["marginal_leakage_kl"]
               + args.select_py_weight * py_term)
    out.update(selection_probe_valid=decoder_valid,
               selector_penalty=float(penalty),
               select_cond_kl=float(lk["leakage_kl"]),
               select_cond_kl_rw=float(lk_rw["leakage_kl"]),
               select_pz_kl_rw=float(mlk["marginal_leakage_kl"]),
               select_py_res_rw=float(dlk["decoder_cmi_res"]),
               select_py_js_rw=float(dlk["decoder_js_res"]),
               select_decoder_valid=decoder_valid,
               select_decoder_min_domain_classes=int(dlk["decoder_min_domain_classes"]),
               select_decoder_single_class_frac=float(dlk["decoder_single_class_frac"]))
    return out


def _source_domain_val_mask(dsraw, val_frac, rng):
    """Hold out whole source domains to mimic LOSO target selection without using target labels."""
    subj = np.unique(dsraw)
    if len(subj) < 2:
        raise RuntimeError("source-only selection needs at least two source domains; "
                           "increase --max_subjects or use a non-LOSO setting with more source domains")
    n_val = min(max(1, int(round(val_frac * len(subj)))), len(subj) - 1)
    val = rng.choice(subj, n_val, replace=False)
    return np.isin(dsraw, val), [str(x) for x in sorted(val)]


def _source_domain_val_splits(dsraw, val_frac, rng, mode):
    if mode == "random":
        return [_source_domain_val_mask(dsraw, val_frac, rng)]
    if mode == "leave_one_domain":
        return [(np.isin(dsraw, [dom]), [str(dom)]) for dom in np.unique(dsraw)]
    raise ValueError(mode)


def _aggregate_source_scores(vals, agg):
    if agg == "mean":
        return float(np.mean(vals))
    if agg == "min":
        return float(np.min(vals))
    raise ValueError(agg)


def _final_probe_fields(probes, prefix):
    """Rename source-probe fields for the final retrained selected/ERM model."""
    mapping = {
        "selection_probe_valid": f"{prefix}_probe_valid",
        "selector_penalty": f"{prefix}_probe_penalty",
        "select_cond_kl": f"{prefix}_cond_kl",
        "select_cond_kl_rw": f"{prefix}_cond_kl_rw",
        "select_pz_kl_rw": f"{prefix}_pz_kl_rw",
        "select_py_res_rw": f"{prefix}_py_res_rw",
        "select_py_js_rw": f"{prefix}_py_js_rw",
        "select_decoder_valid": f"{prefix}_decoder_valid",
        "select_decoder_min_domain_classes": f"{prefix}_decoder_min_domain_classes",
        "select_decoder_single_class_frac": f"{prefix}_decoder_single_class_frac",
    }
    return {dst: probes.get(src) for src, dst in mapping.items()}


def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    X, y, meta, classes = load(args.dataset, tmin=args.tmin, tmax=args.tmax, resample=args.resample)
    if args.max_subjects:
        keep = sorted(meta["subject"].unique())[:args.max_subjects]
        m = meta["subject"].isin(keep).to_numpy()
        X, y, meta = X[m], y[m], meta[m].reset_index(drop=True)
    dom_all, _ = domain_labels(meta, "subject")
    n_cls = len(classes)
    scps = args.dataset in {"ADFTD", "ADFTD_bin", "MUMTAZ", "TUAB"}
    candidates = _candidates(args)
    default_margin = resolve_dec_margin(args.method, args.dec_margin)
    sel_epochs = args.select_epochs or max(1, args.epochs // 2)
    print(f"[{args.dataset}] X={X.shape} classes={classes} method={args.method} "
          f"candidates={[_label(args.method, la, ga, tau, default_margin, z, ds) for la, ga, tau, z, ds in candidates]} "
          f"select_epochs={sel_epochs} final={args.epochs}", flush=True)

    pooled_sel, pooled_erm, selected = [], [], []
    selected_lam, selected_gamma, selected_margin, selection_records = [], [], [], []
    t0 = time.time()
    for fold_i, (tgt, tr, te) in enumerate(loso_splits(meta)):
        Xs, ys, dsraw = X[tr], y[tr], dom_all[tr]
        Xte, yte = X[te], y[te]
        # --- source-only λ selection ---
        candidate_scores = []
        for ci, (lam, gamma, dec_margin, z_margin, dec_scale) in enumerate(candidates):
            val_scores, val_domains, probe_fields = [], [], None
            for rep in range(max(1, args.select_repeats)):
                split_seed = args.seed + 100000 * fold_i + 1000 * ci + rep
                split_rng = np.random.default_rng(split_seed)
                splits = _source_domain_val_splits(dsraw, args.val_frac, split_rng, args.select_cv_mode)
                for split_i, (vmask, held_domains) in enumerate(splits):
                    bb = _fit(Xs[~vmask], ys[~vmask], dsraw[~vmask], n_cls, lam, gamma, dec_margin,
                              z_margin, dec_scale, args, device, sel_epochs)
                    va = balanced_accuracy_score(ys[vmask], predict(bb, Xs[vmask], device).argmax(1))
                    val_scores.append(float(va))
                    val_domains.append(held_domains)
                    if args.select_rule == "guarded_probe" and rep == 0 and split_i == 0:
                        probe_fields = _selection_probes(bb, Xs[~vmask], ys[~vmask], dsraw[~vmask], n_cls,
                                                         args, device, args.seed + 1000 * fold_i + ci)
            rec = dict(config=_label(args.method, lam, gamma, dec_margin, default_margin, z_margin, dec_scale),
                       lam=float(lam), gamma=float(gamma), dec_margin=float(dec_margin),
                       z_margin=float(z_margin), dec_scale=float(dec_scale),
                       source_val_bacc=_aggregate_source_scores(val_scores, args.select_agg),
                       source_val_bacc_mean=float(np.mean(val_scores)),
                       source_val_bacc_min=float(np.min(val_scores)),
                       source_val_bacc_repeats=val_scores,
                       source_val_domains=val_domains)
            if args.select_rule == "guarded_probe":
                rec.update(probe_fields or {})
            else:
                rec.update(selection_probe_valid=False, selector_penalty=0.0)
            candidate_scores.append(rec)
        if args.select_rule == "bacc":
            best_i = max(range(len(candidate_scores)), key=lambda i: (candidate_scores[i]["source_val_bacc"], -i))
        elif args.select_rule == "guarded_probe":
            best_bacc = max(x["source_val_bacc"] for x in candidate_scores)
            best_i, best_key = 0, None
            for i, rec in enumerate(candidate_scores):
                if rec["source_val_bacc"] + args.select_tolerance < best_bacc:
                    continue
                key = (0 if rec.get("selection_probe_valid") else 1,
                       rec.get("selector_penalty", 0.0), -rec["source_val_bacc"], i)
                if best_key is None or key < best_key:
                    best_i, best_key = i, key
        else:
            raise ValueError(args.select_rule)
        guard_reason = ""
        if args.erm_guard and candidate_scores[best_i]["config"] != "erm:0":
            erm = next((rec for rec in candidate_scores if rec["config"] == "erm:0"), None)
            if erm is not None:
                selected_candidate = candidate_scores[best_i]
                source_gain = selected_candidate["source_val_bacc"] - erm["source_val_bacc"]
                probe_drop = erm.get("selector_penalty", 0.0) - selected_candidate.get("selector_penalty", 0.0)
                selected_penalty = selected_candidate.get("selector_penalty", 0.0)
                max_pen = args.erm_guard_max_selected_penalty
                blocked = []
                if source_gain < args.erm_guard_min_source_gain:
                    blocked.append(f"source_gain={source_gain:.4g}<min")
                if probe_drop < args.erm_guard_min_probe_drop:
                    blocked.append(f"probe_drop={probe_drop:.4g}<min")
                if max_pen >= 0 and selected_penalty > max_pen:
                    blocked.append(f"selected_penalty={selected_penalty:.4g}>max")
                if blocked:
                    best_i = candidate_scores.index(erm)
                    guard_reason = ";".join(blocked)
        best = candidate_scores[best_i]
        best_lam, best_gamma, best_margin, best_acc = (
            best["lam"], best["gamma"], best["dec_margin"], best["source_val_bacc"])
        best_z_margin, best_dec_scale = best.get("z_margin", 0.0), best.get("dec_scale", 1.0)
        selected.append(best["config"])
        selected_lam.append(float(best_lam)); selected_gamma.append(float(best_gamma)); selected_margin.append(float(best_margin))
        sel_rec = dict(target=str(tgt), selected=best["config"], candidates=candidate_scores,
                       erm_guard_reason=guard_reason)
        # --- retrain on ALL source with selected λ, and a fixed-ERM, test target ---
        bb_sel = _fit(Xs, ys, dsraw, n_cls, best_lam, best_gamma, best_margin,
                      best_z_margin, best_dec_scale, args, device, args.epochs)
        bb_erm = bb_sel if best_lam == 0 and best_gamma == 0 else _fit(
            Xs, ys, dsraw, n_cls, 0.0, 0.0, default_margin, 0.0, 1.0, args, device, args.epochs)
        psel, perm = predict(bb_sel, Xte, device).argmax(1), predict(bb_erm, Xte, device).argmax(1)
        pooled_sel.append((yte, psel, str(tgt))); pooled_erm.append((yte, perm, str(tgt)))
        ba = balanced_accuracy_score(yte, psel)
        ba_erm = balanced_accuracy_score(yte, perm)
        sel_rec.update(target_bacc=float(ba), target_erm_bacc=float(ba_erm))
        if args.final_probe_epochs > 0:
            final_seed = args.seed + 50000 + 1000 * fold_i
            final_sel = _selection_probes(bb_sel, Xs, ys, dsraw, n_cls, args, device, final_seed,
                                          probe_epochs=args.final_probe_epochs)
            sel_rec.update(_final_probe_fields(final_sel, "final_selected"))
            if bb_erm is bb_sel:
                sel_rec.update(_final_probe_fields(final_sel, "final_erm"))
            else:
                final_erm = _selection_probes(bb_erm, Xs, ys, dsraw, n_cls, args, device, final_seed + 1,
                                              probe_epochs=args.final_probe_epochs)
                sel_rec.update(_final_probe_fields(final_erm, "final_erm"))
        selection_records.append(sel_rec)
        print(f"  tgt={tgt} cfg*={selected[-1]} (src-val {best_acc*100:.1f}, "
              f"pen {best.get('selector_penalty', 0.0):.3f}) -> target bAcc={ba*100:.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    def agg(pooled):
        m = _global_metrics(pooled)
        return (m["subject_balanced_acc"] if scps else m["pooled_balanced_acc"]) * 100
    out = dict(config=vars(args), classes=classes,
               selected_config=selected,
               selected_lambda=selected_lam,
               selected_gamma=selected_gamma,
               selected_dec_margin=selected_margin,
               config_hist={k: selected.count(k) for k in sorted(set(selected))},
               selection_records=selection_records,
               acc_constrained=agg(pooled_sel), acc_erm=agg(pooled_erm))
    print(f"\n=== {args.dataset} constrained-λ vs ERM ===")
    print(f"  config* histogram: {out['config_hist']}")
    print(f"  ERM            = {out['acc_erm']:.1f}")
    print(f"  constrained-λ  = {out['acc_constrained']:.1f}   (Δ={out['acc_constrained']-out['acc_erm']:+.1f})")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2); print(f"saved -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--method", default="lpc_prior", choices=["lpc_prior", "dualc", "dualpc", "dualpc_hinge", "dualpc_marginal"],
                    help="non-ERM method to source-select; λ=0 always means ERM")
    ap.add_argument("--lams", nargs="+", default=["0", "0.05", "0.1", "0.3"])
    ap.add_argument("--gammas", nargs="+", default=None,
                    help="decoder weights for dualc/dualpc variants; default mirrors --lams")
    ap.add_argument("--dec_margin", type=float, default=None,
                    help="decoder gate tau. Default is method-specific: dualpc/dualpc_marginal=0, others=0.02")
    ap.add_argument("--dec_margins", nargs="+", type=float, default=None,
                    help="candidate decoder gates for dualc/dualpc source selection; default uses method-specific tau")
    ap.add_argument("--z_margins", nargs="+", type=float, default=None,
                    help="candidate encoder hinge thresholds for dualpc_hinge")
    ap.add_argument("--dec_scales", nargs="+", type=float, default=None,
                    help="candidate decoder JS scales for dualpc_hinge")
    ap.add_argument("--select_rule", default="bacc", choices=["bacc", "guarded_probe"],
                    help="bacc: pick best source-val bAcc; guarded_probe: within tolerance, minimize source-only P(z)/P(y|Z) probes")
    ap.add_argument("--select_tolerance", type=float, default=0.02,
                    help="guarded_probe source-val bAcc slack below the best candidate")
    ap.add_argument("--erm_guard", action="store_true",
                    help="fallback to ERM unless the selected non-ERM candidate clears source/probe guards")
    ap.add_argument("--erm_guard_min_source_gain", type=float, default=0.0,
                    help="minimum selected source-val bAcc gain over ERM required by --erm_guard")
    ap.add_argument("--erm_guard_min_probe_drop", type=float, default=0.0,
                    help="minimum selected source-probe penalty drop versus ERM required by --erm_guard")
    ap.add_argument("--erm_guard_max_selected_penalty", type=float, default=-1.0,
                    help="optional maximum selected source-probe penalty under --erm_guard; negative disables")
    ap.add_argument("--select_probe_frac", type=float, default=0.3,
                    help="fraction of source-train trials per domain held out for guarded probes")
    ap.add_argument("--select_probe_epochs", type=int, default=50)
    ap.add_argument("--final_probe_epochs", type=int, default=0,
                    help="if >0, record source-only probes after retraining selected/ERM models on all source")
    ap.add_argument("--select_cond_weight", type=float, default=1.0,
                    help="guarded_probe weight on GLS conditional leakage I_w(Z;D|Y)")
    ap.add_argument("--select_pz_weight", type=float, default=1.0)
    ap.add_argument("--select_py_weight", type=float, default=1.0,
                    help="guarded_probe weight on held-out JS P(y|Z) consistency")
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--select_repeats", type=int, default=1,
                    help="number of repeated whole-source-domain validation splits per candidate")
    ap.add_argument("--select_agg", default="mean", choices=["mean", "min"],
                    help="aggregate repeated source-domain validation bAcc before guarded selection")
    ap.add_argument("--select_cv_mode", default="random", choices=["random", "leave_one_domain"],
                    help="source-domain validation split mode; leave_one_domain evaluates every source domain")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--select_epochs", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--n_inner", type=int, default=2); ap.add_argument("--sampler", default="classbal")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--tmin", type=float, default=0.5); ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=250); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_subjects", type=int, default=0)
    ap.add_argument("--out", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
