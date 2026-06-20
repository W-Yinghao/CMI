"""Cross-dataset SCPS — Protocol C, leave-one-COHORT-out.

Train on K-1 cohorts of a disease (e.g. PD), test on the held-out cohort = an UNSEEN site + device +
subjects. This is strict cross-site clinical DG. The domain D for the CMI penalty can be taken at the
fine (subject) or coarse (cohort) granularity; the hierarchical pi_y over (cohort, subject) is the
planned extension. Reads the npz cache built by scripts/build_scps_cache.py (falls back to live BIDS).

  python -m cmi.run_scps_crossdataset --condition PD --domain subject \
         --configs erm:0 lpc_prior:0.1 cdann:1 --out results/scps_PD.json
"""
import argparse, json, os
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from cmi.models.backbones import build_backbone
from cmi.train.trainer import train_model, predict, resolve_dec_margin
from cmi.eval.metrics import leakage_probe, marginal_leakage_probe, decoder_leakage_probe, add_decoder_valid_means

CACHE = "/projects/EEG-foundation-model/datalake/raw/scps/cache"


def _remap(arr):
    uniq = {v: i for i, v in enumerate(sorted(set(arr)))}
    return np.array([uniq[v] for v in arr], "int64"), len(uniq)


def load(condition, cohorts):
    p = f"{CACHE}/{condition}.npz"
    X = None
    if os.path.exists(p):
        try:                                              # robust to a half-written cache
            d = np.load(p, allow_pickle=True)
            X, y, subj, coh, classes = d["X"], d["y"], d["subject"], d["cohort"], list(d["classes"])
        except Exception as e:
            print(f"  [cache unreadable: {e}; falling back to live BIDS load]"); X = None
    if X is None:
        from cmi.data.bids_data import load_crossdataset
        X, y, meta, classes = load_crossdataset(condition, cohorts=cohorts)
        subj, coh = meta["subject"].values.astype(str), meta["cohort"].values.astype(str)
    if cohorts:
        m = np.isin(coh, cohorts)
        X, y, subj, coh = X[m], y[m], subj[m], coh[m]
    return X, y, subj.astype(str), coh.astype(str), classes


def parse_cfg(s):
    p = s.split(":")
    nums = [float(x) for x in p[1:]]
    lam = nums[0] if nums else 0.0
    gamma = nums[1] if len(nums) > 1 else lam    # dual:<lam_enc>:<gamma_dec>
    z_margin = nums[2] if p[0] == "dualpc_hinge" and len(nums) > 2 else 0.0
    dec_scale = nums[3] if p[0] == "dualpc_hinge" and len(nums) > 3 else 1.0
    return (s, p[0], lam, gamma, z_margin, dec_scale)


def _xs_save(out_path, args, cohorts, results):
    """Build summary over WHATEVER cohorts are done so far + write JSON. Safe to call per-fold (overwrites)."""
    summary = {}
    for lbl in results:
        r = results[lbl]
        if not r:
            continue
        summary[lbl] = dict(per_target_balanced_acc_mean=float(np.mean([x["balanced_acc"] for x in r])),
                            worst_target_balanced_acc=float(np.min([x["balanced_acc"] for x in r])),
                            leakage_kl=float(np.mean([x["leakage_kl"] for x in r])),
                            leakage_kl_rw=float(np.mean([x["leakage_kl_rw"] for x in r])),
                            marginal_leakage_kl=float(np.mean([x["marginal_leakage_kl"] for x in r])),
                            marginal_leakage_kl_rw=float(np.mean([x["marginal_leakage_kl_rw"] for x in r])),
                            marginal_leakage_advantage=float(np.mean([x["marginal_leakage_advantage"] for x in r])),
                            marginal_leakage_advantage_rw=float(np.mean([x["marginal_leakage_advantage_rw"] for x in r])),
                            leakage_advantage_rw=float(np.mean([x["leakage_advantage_rw"] for x in r])),
                            decoder_cmi=float(np.mean([x["decoder_cmi"] for x in r])),
                            decoder_cmi_rw=float(np.mean([x["decoder_cmi_rw"] for x in r])),
                            decoder_cmi_res=float(np.mean([x["decoder_cmi_res"] for x in r])),
                            decoder_cmi_res_rw=float(np.mean([x["decoder_cmi_res_rw"] for x in r])),
                            decoder_js_res=float(np.mean([x["decoder_js_res"] for x in r])),
                            decoder_js_res_rw=float(np.mean([x["decoder_js_res_rw"] for x in r])),
                            decoder_valid_frac=float(np.mean([float(x.get("decoder_valid", False)) for x in r])),
                            decoder_min_domain_classes=int(np.min([x.get("decoder_min_domain_classes", 0) for x in r])),
                            decoder_single_class_frac=float(np.mean([x.get("decoder_single_class_frac", 1.0) for x in r])),
                            inloop_reg=float(np.mean([x.get("inloop_reg", 0.0) for x in r])),
                            inloop_dec=float(np.mean([x.get("inloop_dec", 0.0) for x in r])),
                            inloop_dec_loss=float(np.mean([x.get("inloop_dec_loss", 0.0) for x in r])),
                            train_dec_margin=float(np.mean([x.get("train_dec_margin", 0.0) for x in r])),
                            per_cohort={x["held_out"]: round(x["balanced_acc"] * 100, 1) for x in r})
        if r and "ts_balanced_acc" in r[0]:                            # CIPC transductive-corrected
            summary[lbl]["transduct_balanced_acc_mean"] = float(np.mean([x["ts_balanced_acc"] for x in r]))
            summary[lbl]["probe_balanced_acc_mean"] = float(np.mean([x["probe_balanced_acc"] for x in r]))
            for k in [kk for kk in r[0] if kk.startswith("ts_") and kk.endswith("_balanced_acc")]:
                summary[lbl][k + "_mean"] = float(np.mean([x[k] for x in r]))
            for k in ("pmct_js_vs_probe", "pmct_flip_frac", "pmct_conf_change", "probe_vs_head_agree",
                      "pmct_w2c", "pmct_c2w", "pmct_net_correction"):                  # diagnostics (#4)
                if k in r[0]:
                    summary[lbl][k] = float(np.mean([x[k] for x in r]))
        add_decoder_valid_means(summary[lbl], r)
        for k in ("decoder_cmi_res_null_q", "decoder_cmi_res_excess",
                  "decoder_cmi_res_rw_null_q", "decoder_cmi_res_rw_excess",
                  "decoder_js_res_null_q", "decoder_js_res_excess",
                  "decoder_js_res_rw_null_q", "decoder_js_res_rw_excess"):
            vals = [x[k] for x in r if k in x]
            if vals:
                summary[lbl][k] = float(np.mean(vals))
    out = dict(condition=args.condition, domain=args.domain, dec_domain=(args.dec_domain or args.domain),
               cohorts=cohorts, config=vars(args), summary=summary, folds=results)
    if out_path:
        json.dump(out, open(out_path, "w"), indent=2)
    return out, summary


def _train_on(args, X, y, dom_all, mask, cfg, n_cls, device):
    """Train a backbone on the subset X[mask] with config cfg (used by the nested source-domain selector)."""
    _, method, lam, gamma, z_margin, dec_scale = cfg
    d, _ = _remap(dom_all[mask])
    bb = build_backbone(args.backbone, X.shape[1], X.shape[2], n_cls, device=device)
    if args.beta > 0:
        from cmi.methods.vib import VIBBackbone
        bb = VIBBackbone(bb, n_cls).to(device)
    bb, _, _ = train_model(bb, X[mask], y[mask], d, n_cls, method=method, lam=lam, gamma=gamma, beta=args.beta,
                           balance=args.balance, dec_margin=resolve_dec_margin(method, args.dec_margin),
                           z_margin=z_margin, dec_scale=dec_scale, label_correct=args.label_correct,
                           reweight_dual=args.reweight_dual, epochs=args.epochs, bs=args.bs, warmup=args.warmup,
                           n_inner=args.n_inner, sampler=args.sampler, prior_mode=args.prior,
                           device=device, seed=args.seed)
    return bb


def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    X, y, subj, coh, classes = load(args.condition, args.cohorts)
    cohorts = sorted(set(coh))
    n_cls = len(classes)
    print(f"[{args.condition}] X={X.shape} classes={classes} cohorts={cohorts} "
          f"y={np.bincount(y)} domain={args.domain}", flush=True)
    configs = [parse_cfg(c) for c in args.configs]
    results = {lbl: [] for lbl, *_ in configs}
    if args.select == "nested":
        results["CITA_nested"] = []                       # fixed selector picks lambda* per fold (no oracle)

    for hold in cohorts:                                  # leave-one-cohort-out
        te = coh == hold; tr = ~te
        Xtr, ytr = X[tr], y[tr]
        dom_raw = (subj if args.domain == "subject" else coh)[tr]
        dtr, n_dom = _remap(dom_raw)
        # HIERARCHICAL D: encoder penalty + encoder-leakage use --domain (e.g. site); the decoder concept
        # probe uses --dec_domain (e.g. subject) when set — decouples I(Z;D_site|Y) from I(Y;D_subj|Z).
        dec_name = args.dec_domain or args.domain
        ddec = dtr if dec_name == args.domain else _remap((subj if dec_name == "subject" else coh)[tr])[0]
        Xte, yte = X[te], y[te]
        if args.target_prior > 0 and n_cls == 2:          # STRESS TEST: subsample target to a skewed class prior
            r2 = np.random.default_rng(args.seed + 999)
            i0, i1 = np.where(yte == 0)[0], np.where(yte == 1)[0]
            n = 2 * min(len(i0), len(i1)); n0 = int(round(args.target_prior * n))
            keep = np.concatenate([r2.choice(i0, min(n0, len(i0)), replace=False),
                                   r2.choice(i1, min(n - n0, len(i1)), replace=False)])
            Xte, yte = Xte[keep], yte[keep]
        # source-internal split for the leakage probe
        rng = np.random.default_rng(args.seed); idx = rng.permutation(len(Xtr)); cut = int(0.7 * len(idx))
        pi, ei = idx[:cut], idx[cut:]
        # NESTED SOURCE-DOMAIN SELECTOR (reviewer §1): inner leave-one-source-cohort-out cross-validation.
        # The model is NEVER trained on its validation domain (fixes the in-sample sv_bacc). val bAcc measures
        # cross-DOMAIN generalization (the real selection target); leakage (lk_rw) is the tie-break within eps.
        inner_val = {}
        if args.select == "nested":
            src_cohorts = [c for c in cohorts if c != hold]
            dom_all = subj if args.domain == "subject" else coh
            if len(src_cohorts) >= 2:
                acc = {lbl: [] for lbl, *_ in configs}
                for dv in src_cohorts:                    # hold out one SOURCE cohort, train on the rest
                    fit = (coh != hold) & (coh != dv); ev = (coh == dv)
                    for cfg in configs:
                        bbv = _train_on(args, X, y, dom_all, fit, cfg, n_cls, device)
                        acc[cfg[0]].append(float(balanced_accuracy_score(y[ev],
                                                                         predict(bbv, X[ev], device).argmax(1))))
                inner_val = {lbl: float(np.mean(v)) for lbl, v in acc.items()}
                print(f"  [nested hold={hold}] inner LOSDO val bAcc: " +
                      " ".join(f"{l}={inner_val[l]*100:.1f}" for l in inner_val), flush=True)
            else:
                print(f"  [nested hold={hold}] only {len(src_cohorts)} source cohort(s) -> skip nested selection",
                      flush=True)
        for lbl, method, lam, gamma, z_margin, dec_scale in configs:
            bb = build_backbone(args.backbone, X.shape[1], X.shape[2], n_cls, device=device)
            if args.beta > 0:
                from cmi.methods.vib import VIBBackbone
                bb = VIBBackbone(bb, n_cls).to(device)
            bb, _, diag = train_model(bb, Xtr, ytr, dtr, n_cls, method=method, lam=lam, gamma=gamma,
                                      beta=args.beta, balance=args.balance,
                                      dec_margin=resolve_dec_margin(method, args.dec_margin),
                                      z_margin=z_margin, dec_scale=dec_scale,
                                      label_correct=args.label_correct, reweight_dual=args.reweight_dual,
                                      epochs=args.epochs, bs=args.bs, warmup=args.warmup,
                                      n_inner=args.n_inner, sampler=args.sampler, prior_mode=args.prior,
                                      device=device, seed=args.seed)
            prob = predict(bb, Xte, device)
            ba = balanced_accuracy_score(yte, prob.argmax(1))
            sv_bacc = float(balanced_accuracy_score(ytr[ei], predict(bb, Xtr[ei], device).argmax(1)))  # source-only val (for selector)
            ts = {}
            if args.transduct != "off":               # CIPC cohort-level transductive correction (target=held-out cohort)
                from cmi.train.trainer import embed
                from cmi.eval.label_shift import transduct_predict, transduct_all
                z_se = embed(bb, Xtr[ei], device); z_te = embed(bb, Xte, device)
                pi_S = np.bincount(ytr, minlength=n_cls).astype(float); pi_S /= pi_S.sum()
                if args.transduct == "all":
                    probs = transduct_all(z_se, ytr[ei], z_te, pi_S, n_cls, shrink=args.transduct_shrink)
                    ts = {f"ts_{md}_balanced_acc": float(balanced_accuracy_score(yte, p.argmax(1)))
                          for md, p in probs.items()}
                    ts["ts_balanced_acc"] = ts["ts_coral_balanced_acc"]
                    ts["probe_balanced_acc"] = ts["ts_probe_balanced_acc"]
                    # PREDICTOR DIAGNOSTICS (#4) — SAME-CLASSIFIER (reviewer §1.2): compare the frozen source
                    # readout on z (probe) vs on T(z) (pmct), so flips isolate the TRANSPORT, not a classifier
                    # swap. (Also report probe-vs-EEGNet-head agreement so 'predictor-preserving' is honest.)
                    pr0 = np.clip(probs["probe"], 1e-9, 1); pm = np.clip(probs["pmct"], 1e-9, 1)
                    mid = 0.5 * (pr0 + pm)
                    ts["pmct_js_vs_probe"] = float((0.5 * (pr0 * np.log(pr0 / mid)).sum(1)
                                                    + 0.5 * (pm * np.log(pm / mid)).sum(1)).mean())
                    bp_, ap_ = probs["probe"].argmax(1), probs["pmct"].argmax(1)
                    ts["pmct_flip_frac"] = float((bp_ != ap_).mean())
                    ts["pmct_conf_change"] = float(probs["pmct"].max(1).mean() - probs["probe"].max(1).mean())
                    ts["probe_vs_head_agree"] = float((bp_ == prob.argmax(1)).mean())
                    # NET CORRECTION (reviewer): is the alignment FIXING errors or just reshuffling predictions?
                    ts["pmct_w2c"] = float(((bp_ != yte) & (ap_ == yte)).mean())     # wrong -> correct
                    ts["pmct_c2w"] = float(((bp_ == yte) & (ap_ != yte)).mean())     # correct -> wrong
                    ts["pmct_net_correction"] = ts["pmct_w2c"] - ts["pmct_c2w"]
                else:
                    tp = transduct_predict(z_se, ytr[ei], z_te, pi_S, n_cls, mode=args.transduct,
                                           shrink=args.transduct_shrink, gate_l1=args.transduct_gate)
                    ts = dict(ts_balanced_acc=float(balanced_accuracy_score(yte, tp["prob"].argmax(1))),
                              probe_balanced_acc=float(balanced_accuracy_score(yte, tp["prob_probe_raw"].argmax(1))),
                              ts_pi_T=tp["pi_T"])
            lk = leakage_probe(bb, Xtr[pi], ytr[pi], dtr[pi], Xtr[ei], ytr[ei], dtr[ei], n_cls, device=device)
            lk_rw = leakage_probe(bb, Xtr[pi], ytr[pi], dtr[pi], Xtr[ei], ytr[ei], dtr[ei],
                                  n_cls, device=device, reweight=True)
            mlk = marginal_leakage_probe(bb, Xtr[pi], ytr[pi], dtr[pi], Xtr[ei], ytr[ei], dtr[ei],
                                         n_cls, device=device)
            mlk_rw = marginal_leakage_probe(bb, Xtr[pi], ytr[pi], dtr[pi], Xtr[ei], ytr[ei], dtr[ei],
                                            n_cls, device=device, reweight=True)
            dlk = decoder_leakage_probe(bb, Xtr[pi], ytr[pi], ddec[pi], Xtr[ei], ytr[ei], ddec[ei],
                                        n_cls, device=device,
                                        null_perms=args.decoder_null_perms,
                                        null_quantile=args.decoder_null_quantile)
            dlk_rw = decoder_leakage_probe(bb, Xtr[pi], ytr[pi], ddec[pi], Xtr[ei], ytr[ei], ddec[ei],
                                           n_cls, device=device, reweight=True,
                                           null_perms=args.decoder_null_perms,
                                           null_quantile=args.decoder_null_quantile)
            rec = dict(held_out=hold, balanced_acc=float(ba),
                       macro_f1=float(f1_score(yte, prob.argmax(1), average="macro")),
                       leakage_kl=lk["leakage_kl"],
                       leakage_kl_rw=lk_rw["leakage_kl"],
                       leakage_advantage_rw=lk_rw["leakage_advantage"],
                       marginal_leakage_kl=mlk["marginal_leakage_kl"],
                       marginal_leakage_kl_rw=mlk_rw["marginal_leakage_kl"],
                       marginal_leakage_advantage=mlk["marginal_leakage_advantage"],
                       marginal_leakage_advantage_rw=mlk_rw["marginal_leakage_advantage"],
                       decoder_cmi=dlk["decoder_cmi"],
                       decoder_cmi_rw=dlk_rw["decoder_cmi"], decoder_cmi_res=dlk["decoder_cmi_res"],
                       decoder_cmi_res_rw=dlk_rw["decoder_cmi_res"],
                       decoder_js_res=dlk["decoder_js_res"], decoder_js_res_rw=dlk_rw["decoder_js_res"],
                       decoder_valid=bool(dlk["decoder_valid"]),
                       decoder_n_domains=dlk["decoder_n_domains"],
                       decoder_min_domain_classes=dlk["decoder_min_domain_classes"],
                       decoder_mean_domain_classes=dlk["decoder_mean_domain_classes"],
                       decoder_single_class_frac=dlk["decoder_single_class_frac"],
                       decoder_domain_class_spans=dlk["decoder_domain_class_spans"],
                       decoder_domain_counts=dlk["decoder_domain_counts"],
                       inloop_reg=diag["inloop_reg"],
                       inloop_dec=diag.get("inloop_dec", 0.0),
                       inloop_dec_loss=diag.get("inloop_dec_loss", 0.0),
                       train_dec_margin=diag.get("dec_margin", resolve_dec_margin(method, args.dec_margin)),
                       train_sampler=diag.get("sampler", args.sampler), n_test=int(te.sum()),
                       source_val_bacc=sv_bacc, nested_val_bacc=inner_val.get(lbl), **ts)
            for src, prefix, key in ((dlk, "decoder_cmi_res", "decoder_cmi_res"),
                                     (dlk_rw, "decoder_cmi_res_rw", "decoder_cmi_res"),
                                     (dlk, "decoder_js_res", "decoder_js_res"),
                                     (dlk_rw, "decoder_js_res_rw", "decoder_js_res")):
                if f"{key}_null_q" in src:
                    rec[f"{prefix}_null_q"] = src[f"{key}_null_q"]
                    rec[f"{prefix}_excess"] = src[f"{key}_excess"]
            results[lbl].append(rec)
            print(f"  hold={hold:9s} {lbl:14s} bAcc={ba*100:5.1f} leakKL={lk['leakage_kl']:.3f}", flush=True)
        if args.select == "nested" and inner_val:     # FIXED selection: lambda* = argmin leakage s.t. valBAcc >= max-eps
            recs = {lbl: results[lbl][-1] for lbl, *_ in configs}
            best = max(inner_val.values())
            cand = [lbl for lbl in inner_val if inner_val[lbl] >= best - args.select_eps]
            lam_star = min(cand, key=lambda l: recs[l].get("leakage_advantage_rw", 0.0))
            sel = dict(recs[lam_star]); sel["selected_lbl"] = lam_star
            results["CITA_nested"].append(sel)
            print(f"  [nested hold={hold}] -> selected '{lam_star}' (cand={cand}, "
                  f"valBAcc={inner_val[lam_star]*100:.1f})", flush=True)
        if args.out:                                  # INCREMENTAL: persist after each held-out cohort
            _xs_save(args.out, args, cohorts, results)
            print(f"  [ckpt] cohort {hold} done -> {args.out}", flush=True)

    out, summary = _xs_save(args.out, args, cohorts, results)
    print(f"\n=== {args.condition} cross-dataset (leave-one-cohort-out, encD={args.domain} "
          f"decD={args.dec_domain or args.domain}) ===")
    for lbl, s in summary.items():
        print(f"  {lbl:14s} mean={s['per_target_balanced_acc_mean']*100:.1f} "
              f"worst={s['worst_target_balanced_acc']*100:.1f} leak={s['leakage_kl']:.3f}  {s['per_cohort']}")
    if args.out:
        print(f"saved -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="PD")
    ap.add_argument("--cohorts", nargs="*", default=None)
    ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--domain", default="subject", choices=["subject", "cohort"],
                    help="encoder D: the penalty I(Z;D|Y) and encoder-leakage probe use this granularity")
    ap.add_argument("--dec_domain", default="", choices=["", "subject", "cohort"],
                    help="decoder D for the concept probe I(Y;D|Z); empty=same as --domain. "
                         "Hierarchical: --domain cohort --dec_domain subject")
    ap.add_argument("--configs", nargs="+", default=["erm:0", "lpc_prior:0.1"])
    ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=30); ap.add_argument("--n_inner", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--sampler", default="classbal"); ap.add_argument("--prior", default="empirical")
    ap.add_argument("--beta", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance", action="store_true", help="class-balanced (BER) CE = label-shift correction")
    ap.add_argument("--label_correct", action="store_true",
                    help="GLS (A4) per-sample label-shift weight w_i=pi*(y)/pi_d(y) on task CE. "
                         "For full CMI-estimator reweighting use method dualpc or --reweight_dual with dual.")
    ap.add_argument("--reweight_dual", action="store_true",
                    help="Route B (reweighted-dual): for method 'dual' only, apply GLS weight "
                         "w_i=pi*(y)/pi_d(y) to BOTH CMI estimators so encoder/decoder terms decouple")
    ap.add_argument("--transduct", default="off",
                    choices=["off", "probe", "coral", "prior", "coral_prior", "pmct", "all"],
                    help="CIPC transductive correction (coral=balanced-acc lever; all=ablation ladder)")
    ap.add_argument("--transduct_shrink", type=float, default=0.1)
    ap.add_argument("--transduct_gate", type=float, default=0.0)
    ap.add_argument("--select", default="insample", choices=["insample", "nested"],
                    help="lambda selector: insample (legacy sv_bacc) | nested (leave-one-source-cohort-out CV, no oracle)")
    ap.add_argument("--select_eps", type=float, default=0.02,
                    help="nested selector: keep configs within eps of best val bAcc, then pick lowest leakage")
    ap.add_argument("--target_prior", type=float, default=-1.0,
                    help="stress test: subsample held-out cohort (binary) to this majority-class fraction")
    ap.add_argument("--dec_margin", type=float, default=None,
                    help="decoder gate tau. Default is method-specific: dualpc/dualpc_marginal=0, others=0.02")
    ap.add_argument("--decoder_null_perms", type=int, default=0,
                    help="source-only permutation-null repetitions for residual decoder CMI evaluation; 0=off")
    ap.add_argument("--decoder_null_quantile", type=float, default=0.95,
                    help="quantile of the residual decoder permutation null used for *_excess fields")
    ap.add_argument("--out", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
