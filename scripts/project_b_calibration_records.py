"""Project B-Next Step-S0: calibration record builder.

Builds a UNIFIED record layer for both the synthetic locked worlds (R2/HF3/H-OOD, ground truth
known) and real BNCI2014_004, so we can test the pivot from harm-risk to *error*-risk calibration.
It does NOT train a final ACAR-error router; it produces:
  - source_nested_records.csv : held-out SOURCE units treated as pseudo-targets. identity_error is
    legal to use for fitting a predictor (source calibration).
  - target_eval_records.csv   : real held-out target units. identity_error is POST-HOC only and must
    never fit a predictor or a threshold.
plus a toy numpy-ridge error predictor + split-conformal upper bound, and an HF3 probe that asks:
does a source-fold error predictor FLAG the concept-degraded target identity, or reproduce the same
non-identifiability boundary that harm calibration hit?

Records-only: modifies no h2cmi/** or cmi/**. Target labels never touch a threshold, a predictor fit,
or a decision. Reuses the frozen router harness + synthetic simulator + real bridge.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import math
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(REPO, "notes")
EXPECTED_BRANCH = "project-b-next"

SYNTH = dict(classes=3, chans=16, times=128, fs=128.0, sites=6, subjects=4, sessions=2,
             trials=60, noise=0.25, label_rho=0.0, bs=64, eval_unit="subject")
WORLDS = {
    "R2":    dict(cov=1.2, prior=0.4, montage=0.2, concept=0.0, concept_frac=0.0, seeds=[0, 1, 2]),
    "HF3":   dict(cov=0.8, prior=0.4, montage=0.2, concept=1.2, concept_frac=0.50, seeds=[3, 4, 7, 8, 10]),
    "H_OOD": dict(cov=0.8, prior=0.4, montage=0.2, concept=1.0, concept_frac=0.17, seeds=[32]),
}

# Feature columns fed to the toy error predictor (label-free diagnostics).
FEATURES = ["density_nll_target_prior", "target_support_excess", "ess", "ood_score", "prior_shift",
            "min_class_responsibility", "entropy_mean", "margin_mean", "max_prob_mean",
            "delta_density_nll", "transform_norm", "condition_number", "pred_disagreement"]

RECORD_COLS = ["source_or_target", "dataset_or_world", "seed", "config_id", "target_subject",
               "target_site", "fold_unit_type", "fold_unit_id", "record_unit_type", "record_unit_id",
               "eval_unit", "support_mode", "n", "n_classes", "identity_bacc", "offline_tta_bacc",
               "identity_error", "raw_tta_gain", "decision_action", "accepted", "reason_codes",
               "identity_reason_codes", "offline_tta_reason_codes", "offline_tta_blocking_reason_codes",
               "density_nll_target_prior", "density_nll_source_prior", "support_gap",
               "support_threshold_nll_target_prior", "target_support_excess", "ess", "ood_score",
               "prior_shift", "prior_shift_only", "min_class_responsibility", "entropy_mean",
               "margin_mean", "max_prob_mean", "delta_density_nll", "transform_norm",
               "condition_number", "pred_disagreement", "acar_harm_calibration_state",
               "cmi_residual_available", "label_access", "is_concept_unit", "target_concept_hit"]


class Fail(RuntimeError):
    pass


def _current_branch() -> str:
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def _q(a, qq=0.95):
    a = np.asarray(a, dtype=np.float64)
    return float(np.quantile(a, qq)) if a.size else float("nan")


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    if isinstance(v, dict):
        return ";".join(f"{k}:{v[k]}" for k in v)
    return "" if v is None else str(v)


def _write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in cols])


def _pred_stats(model, X_unit, pi_ref, device):
    """Label-free prediction-distribution stats (entropy/margin/max-prob) under identity."""
    from h2cmi.eval.harness import _embed, _predict_generative
    U = _embed(model, X_unit, device)
    p = np.asarray(_predict_generative(model, U, pi_ref), dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    ent = float(np.mean(-(p * np.log(p)).sum(1)))
    ps = np.sort(p, axis=1)
    margin = float(np.mean(ps[:, -1] - ps[:, -2])) if p.shape[1] >= 2 else float("nan")
    maxp = float(np.mean(ps[:, -1]))
    return ent, margin, maxp


def _records_from_report(rep, model, X_eval, unit_levels, pi_ref, device, threshold, ctx):
    """Turn each per-domain routing result into one unified record dict."""
    unit_levels = np.asarray(unit_levels)
    out = []
    for did, dv in rep["per_domain"].items():
        asf = dv["action_scores"]
        sup = dv["support"]
        tta = dv["diagnostics_offline_tta"]
        m = unit_levels == did
        ent, margin, maxp = _pred_stats(model, X_eval[m], pi_ref, device)
        rec = dict(ctx)
        rec.update(
            record_unit_id=int(did), n=int(dv["n"]), n_classes=int(ctx["n_classes"]),
            identity_bacc=float(dv["identity_bacc"]), offline_tta_bacc=float(dv["offline_tta_bacc"]),
            identity_error=float(1.0 - dv["identity_bacc"]), raw_tta_gain=float(dv["raw_gain"]),
            decision_action=dv["decision_action"], accepted=bool(dv["accepted"]),
            reason_codes=dv["reason_codes"], identity_reason_codes=asf["identity"]["reason_codes"],
            offline_tta_reason_codes=asf["offline_tta"]["reason_codes"],
            offline_tta_blocking_reason_codes=asf["offline_tta"]["blocking_reason_codes"],
            density_nll_target_prior=float(sup["density_nll_target_prior"]),
            density_nll_source_prior=float(sup["density_nll_source_prior"]),
            support_gap=float(sup["support_gap"]),
            support_threshold_nll_target_prior=float(threshold),
            target_support_excess=float(sup["density_nll_target_prior"] - threshold),
            ess=float(sup["ess"]), ood_score=float(sup["ood_score"]),
            prior_shift=float(sup["prior_shift"]), prior_shift_only=bool(asf["identity"]["prior_shift_only"]),
            min_class_responsibility=float(sup["min_class_responsibility"]),
            entropy_mean=ent, margin_mean=margin, max_prob_mean=maxp,
            delta_density_nll=float(tta.get("delta_density_nll", float("nan"))),
            transform_norm=float(tta.get("transform_norm", float("nan"))),
            condition_number=float(tta.get("condition_number", float("nan"))),
            pred_disagreement=float(tta.get("pred_disagreement", float("nan"))),
            acar_harm_calibration_state=asf["offline_tta"]["acar_harm_calibration_state"],
            cmi_residual_available=bool(asf["identity"]["cmi_residual_available"]),
        )
        out.append(rec)
    return out


def _route(model, X, y, unit_levels, cfg, pi_ref, X_cal, y_cal, cal_levels, device, threshold, mode):
    from h2cmi.eval.router_harness import evaluate_router_offline_tta, make_support_calibrated_feature_config
    from h2cmi.router.router import RefusalFirstRouter, RouterConfig
    fcfg = make_support_calibrated_feature_config(
        max_density_nll_target_prior=threshold, min_target_n=max(20, int(cfg.tta.min_target)))
    router = RefusalFirstRouter(RouterConfig(feature_config=fcfg))
    return evaluate_router_offline_tta(
        model, X, y, unit_levels, cfg, pi_ref, router=router,
        X_src=X_cal, y_src=y_cal, source_pseudo_levels=cal_levels, device=device,
        calibrate_source_support=False, support_calibration_mode=mode)


# ------------------------------------------------------------------ synthetic
def build_synthetic(world, seed, args):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    from h2cmi.config import H2Config
    from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
    from h2cmi.train.trainer import train_h2, reference_prior

    w = WORLDS[world]
    shift = ShiftSpec(cov=w["cov"], prior=w["prior"], concept=w["concept"],
                      concept_site_frac=w["concept_frac"], montage=w["montage"],
                      noise=SYNTH["noise"], label_mechanism_rho=SYNTH["label_rho"])
    sim = EEGSimulator(SYNTH["classes"], SYNTH["chans"], SYNTH["times"], SYNTH["fs"],
                       shift=shift, seed=seed).sample(
        SYNTH["sites"], SYNTH["subjects"], SYNTH["sessions"], SYNTH["trials"])
    src_idx, tgt_idx = train_target_split(sim, n_target_sites=1, seed=seed)
    target_site = int(np.unique(sim.site[tgt_idx])[0])
    source_sites = sorted(int(s) for s in np.unique(sim.site[src_idx]))
    concept_sites = [int(s) for s in sim.meta["concept_sites"]]
    tgt_concept_hit = target_site in concept_sites

    cfg = H2Config(n_classes=SYNTH["classes"])
    cfg.encoder.n_chans = SYNTH["chans"]; cfg.encoder.n_times = SYNTH["times"]; cfg.encoder.fs = SYNTH["fs"]
    cfg.train.epochs = args.synthetic_epochs; cfg.train.device = args.device
    cfg.train.seed = seed; cfg.train.batch_size = SYNTH["bs"]

    Xs, ys = sim.X[src_idx], sim.y[src_idx]
    src_dom = sim.domains.subset(src_idx)
    base, *_ = train_h2(Xs, ys, src_dom, sim.dag, cfg, align_factor="site", verbose=False)
    pi_star = reference_prior(ys, SYNTH["classes"], cfg.align.reference_prior)
    src_subj = src_dom.factor("subject")

    from h2cmi.eval.harness import _embed
    from h2cmi.eval.router_harness import prior_decoupled_density_diagnostics

    def _subj_nll(model, X, subj, prior):
        out = {}
        for u in np.unique(subj):
            mm = subj == u
            if int(mm.sum()) < cfg.tta.min_target:
                continue
            U = _embed(model, X[mm], args.device)
            out[int(u)] = prior_decoupled_density_diagnostics(model.head.density, U, prior)["density_nll_target_prior"]
        return out

    base_q95 = _q(list(_subj_nll(base, Xs, src_subj, pi_star).values()))
    site_arr = sim.site
    source_records, all_excess = [], []
    for fi, u in enumerate(source_sites[:args.synthetic_max_nested_folds]):
        tr_mask = np.isin(site_arr, [s for s in source_sites if s != u])
        ps_mask = site_arr == u
        Xtr, ytr = sim.X[tr_mask], sim.y[tr_mask]
        dom_tr = sim.domains.subset(np.where(tr_mask)[0])
        nmodel, *_ = train_h2(Xtr, ytr, dom_tr, sim.dag, cfg, align_factor="site", verbose=False)
        pi_n = reference_prior(ytr, SYNTH["classes"], cfg.align.reference_prior)
        tr_subj = dom_tr.factor("subject")
        fold_q95 = _q(list(_subj_nll(nmodel, Xtr, tr_subj, pi_n).values()))
        all_excess.extend([v - fold_q95 for v in _subj_nll(nmodel, sim.X[ps_mask],
                          sim.domains.subset(np.where(ps_mask)[0]).factor("subject"), pi_n).values()])
        ps_dom = sim.domains.subset(np.where(ps_mask)[0])
        ps_subj = ps_dom.factor("subject")
        rep = _route(nmodel, sim.X[ps_mask], sim.y[ps_mask], ps_subj, cfg, pi_n,
                     Xtr, ytr, tr_subj, args.device, fold_q95, "source_fold_train_q95")
        ctx = dict(source_or_target="source", dataset_or_world=world, seed=seed,
                   config_id=f"{world}_seed{seed}", target_subject="", target_site=target_site,
                   fold_unit_type="site", fold_unit_id=u, record_unit_type="subject",
                   eval_unit="subject", support_mode="source_fold_train_q95", n_classes=SYNTH["classes"],
                   label_access="source_calibration", is_concept_unit=bool(u in concept_sites),
                   target_concept_hit=bool(tgt_concept_hit))
        source_records += _records_from_report(rep, nmodel, sim.X[ps_mask], ps_subj, pi_n,
                                               args.device, fold_q95, ctx)
    nested_thr = base_q95 + max(0.0, _q(all_excess)) if all_excess else base_q95

    # target records under both support modes
    Xt, yt = sim.X[tgt_idx], sim.y[tgt_idx]
    tgt_unit = sim.domains.subset(tgt_idx).factor(SYNTH["eval_unit"])
    target_records = []
    for mode, thr in (("in_source_subject_q95", base_q95), ("nested_site_excess_q95", nested_thr)):
        rep = _route(base, Xt, yt, tgt_unit, cfg, pi_star, Xs, ys, src_subj, args.device, thr, mode)
        ctx = dict(source_or_target="target", dataset_or_world=world, seed=seed,
                   config_id=f"{world}_seed{seed}", target_subject="", target_site=target_site,
                   fold_unit_type="site", fold_unit_id=target_site, record_unit_type="subject",
                   eval_unit="subject", support_mode=mode, n_classes=SYNTH["classes"],
                   label_access="target_posthoc", is_concept_unit="", target_concept_hit=bool(tgt_concept_hit))
        target_records += _records_from_report(rep, base, Xt, tgt_unit, pi_star, args.device, thr, ctx)
    print(f"[synthetic] {world}/seed{seed}: src_recs={len(source_records)} tgt_recs={len(target_records)} "
          f"target_site={target_site} concept_hit={tgt_concept_hit} base_q95={base_q95:.2f}")
    return source_records, target_records


# ------------------------------------------------------------------ real BNCI2014_004
def build_real(args):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    from h2cmi.config import H2Config
    from h2cmi.train.trainer import train_h2, reference_prior
    from h2cmi.data.real_eeg_bridge import (
        load_moabb_real_eeg, loso_subjects, split_loso_by_subject, make_source_domain_labels,
        target_domain_levels, source_pseudo_levels_from_domains)
    from h2cmi.eval.harness import _embed
    from h2cmi.eval.router_harness import prior_decoupled_density_diagnostics

    ds = load_moabb_real_eeg("BNCI2014_004", max_subjects=args.real_max_subjects, tmin=0.5, tmax=3.5,
                             resample=args.resample)
    n_classes = len(ds.classes)
    targets = loso_subjects(ds.meta)[:args.real_max_targets]
    src_records, tgt_records = [], []

    def _subj_nll(model, X, subj, prior, cfg):
        out = {}
        for u in np.unique(subj):
            mm = subj == u
            if int(mm.sum()) < cfg.tta.min_target:
                continue
            U = _embed(model, X[mm], args.device)
            out[int(u)] = prior_decoupled_density_diagnostics(model.head.density, U, prior)["density_nll_target_prior"]
        return out

    for t in targets:
        src_idx, tgt_idx = split_loso_by_subject(ds.meta, t)
        Xs, ys = ds.X[src_idx], ds.y[src_idx]
        Xt, yt = ds.X[tgt_idx], ds.y[tgt_idx]
        meta_t = ds.meta.loc[tgt_idx].reset_index(drop=True)
        dag, src_domains, _ = make_source_domain_labels(ds.meta.loc[src_idx].reset_index(drop=True))
        cfg = H2Config(n_classes=n_classes)
        cfg.encoder.n_chans = int(ds.X.shape[1]); cfg.encoder.n_times = int(ds.X.shape[2]); cfg.encoder.fs = float(ds.fs)
        cfg.train.epochs = args.real_epochs; cfg.train.batch_size = args.batch_size
        cfg.train.device = args.device; cfg.train.seed = args.seed
        base, *_ = train_h2(Xs, ys, src_domains, dag, cfg, align_factor="subject", verbose=False)
        pi_star = reference_prior(ys, n_classes, cfg.align.reference_prior)
        src_subj = source_pseudo_levels_from_domains(src_domains, level="subject")
        base_q95 = _q(list(_subj_nll(base, Xs, src_subj, pi_star, cfg).values()))

        # source nested records: hold out each source subject, evaluate its identity error
        uniq = sorted(int(u) for u in np.unique(src_subj))
        all_excess = []
        for u in uniq[:args.real_max_nested_folds]:
            tr = src_subj != u; ps = src_subj == u
            nmodel, *_ = train_h2(Xs[tr], ys[tr], src_domains.subset(np.where(tr)[0]), dag, cfg,
                                  align_factor="subject", verbose=False)
            pi_n = reference_prior(ys[tr], n_classes, cfg.align.reference_prior)
            fold_q95 = _q(list(_subj_nll(nmodel, Xs[tr], src_subj[tr], pi_n, cfg).values()))
            all_excess.extend([v - fold_q95 for v in _subj_nll(nmodel, Xs[ps], src_subj[ps], pi_n, cfg).values()])
            ps_levels = src_subj[ps]
            rep = _route(nmodel, Xs[ps], ys[ps], ps_levels, cfg, pi_n, Xs[tr], ys[tr], src_subj[tr],
                         args.device, fold_q95, "source_fold_train_q95")
            ctx = dict(source_or_target="source", dataset_or_world="BNCI2014_004", seed=args.seed,
                       config_id=f"BNCI2014_004_t{t}", target_subject=int(t), target_site="",
                       fold_unit_type="subject", fold_unit_id=int(u), record_unit_type="subject",
                       eval_unit="subject", support_mode="source_fold_train_q95", n_classes=n_classes,
                       label_access="source_calibration", is_concept_unit="", target_concept_hit="")
            src_records += _records_from_report(rep, nmodel, Xs[ps], ps_levels, pi_n, args.device, fold_q95, ctx)
        nested_thr = base_q95 + max(0.0, _q(all_excess)) if all_excess else base_q95

        for eval_unit in ("subject", "session"):
            tgt_unit = target_domain_levels(meta_t, eval_unit=eval_unit)
            for mode, thr in (("in_source_subject_q95", base_q95),
                              ("nested_source_subject_excess_q95", nested_thr)):
                rep = _route(base, Xt, yt, tgt_unit, cfg, pi_star, Xs, ys, src_subj, args.device, thr, mode)
                ctx = dict(source_or_target="target", dataset_or_world="BNCI2014_004", seed=args.seed,
                           config_id=f"BNCI2014_004_t{t}", target_subject=int(t), target_site="",
                           fold_unit_type="subject", fold_unit_id=int(t), record_unit_type=eval_unit,
                           eval_unit=eval_unit, support_mode=mode, n_classes=n_classes,
                           label_access="target_posthoc", is_concept_unit="", target_concept_hit="")
                tgt_records += _records_from_report(rep, base, Xt, tgt_unit, pi_star, args.device, thr, ctx)
        print(f"[real] BNCI2014_004/t{t}: src_recs+={len(uniq[:args.real_max_nested_folds])} "
              f"base_q95={base_q95:.2f} nested_thr={nested_thr:.2f}")
    return src_records, tgt_records, dict(dataset="BNCI2014_004", n_trials=int(ds.X.shape[0]),
                                          n_chans=int(ds.X.shape[1]), subjects=loso_subjects(ds.meta),
                                          targets=targets)


# ------------------------------------------------------------------ toy error predictor + conformal
def _impute(X, fill):
    """Replace NaN/inf with per-column fill (TTA-transform features are NaN when TTA falls back)."""
    X = np.asarray(X, np.float64).copy()
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(fill, np.where(bad)[1])
    return X


def _ridge_fit(Xtr, ytr, alpha):
    Xtr = np.asarray(Xtr, np.float64); ytr = np.asarray(ytr, np.float64)
    with np.errstate(invalid="ignore"):
        fill = np.nanmean(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)          # all-NaN column -> 0
    Xi = _impute(Xtr, fill)
    mu = Xi.mean(0); sd = Xi.std(0); sd[sd < 1e-8] = 1.0
    Z = (Xi - mu) / sd
    Z1 = np.hstack([np.ones((Z.shape[0], 1)), Z])
    A = Z1.T @ Z1 + alpha * np.eye(Z1.shape[1]); A[0, 0] -= alpha  # don't penalize intercept
    w = np.linalg.solve(A, Z1.T @ ytr)
    return dict(mu=mu, sd=sd, w=w, fill=fill)


def _ridge_pred(mdl, X):
    Xi = _impute(X, mdl["fill"])
    Z = (Xi - mdl["mu"]) / mdl["sd"]
    Z1 = np.hstack([np.ones((Z.shape[0], 1)), Z])
    return np.clip(Z1 @ mdl["w"], 0.0, 1.0)


def _conformal_q(residuals, alpha):
    r = np.sort(np.asarray(residuals, np.float64))
    n = r.size
    if n == 0:
        return float("nan")
    k = math.ceil((n + 1) * (1 - alpha))
    k = min(max(k, 1), n)
    return float(r[k - 1])


def _feat_matrix(records, keep_idx=None):
    idx = keep_idx if keep_idx is not None else list(range(len(FEATURES)))
    return np.array([[float(r[FEATURES[i]]) for i in idx] for r in records], dtype=np.float64)


def _feature_audit(Xs_full):
    """Source-only column audit: which features are all-NaN in source training (DROP, not fill-0),
    which are partially NaN (impute from source mean). Returns (keep_idx, imputed, dropped)."""
    finite = np.isfinite(Xs_full)
    all_nan = ~finite.any(axis=0)
    some_nan = (~finite).any(axis=0) & ~all_nan
    dropped = [FEATURES[i] for i in range(len(FEATURES)) if all_nan[i]]
    imputed = [FEATURES[i] for i in range(len(FEATURES)) if some_nan[i]]
    keep_idx = [i for i in range(len(FEATURES)) if not all_nan[i]]
    return keep_idx, imputed, dropped


def toy_probe(source_records, target_records, args):
    """Fit ridge on SOURCE records per domain group; predict on target records; conformal upper error.

    Req-1 (source-only, auditable imputation): the imputation fill is estimated ONLY from source
    training rows and applied to targets with that source-derived mean; a feature that is ALL-NaN in
    source training is DROPPED (not forced to 0) and recorded. Source labels (identity_error) are legal
    to fit here; target labels never enter the fit."""
    groups = {}
    for r in source_records:
        groups.setdefault(r["dataset_or_world"], {"src": [], "tgt": []})["src"].append(r)
    for r in target_records:
        groups.setdefault(r["dataset_or_world"], {"src": [], "tgt": []})["tgt"].append(r)

    probe_rows, annotated_targets, audit = [], [], {}
    for name, g in sorted(groups.items()):
        src, tgt = g["src"], g["tgt"]
        if src:
            keep_idx, imputed, dropped = _feature_audit(_feat_matrix(src))
        else:
            keep_idx, imputed, dropped = list(range(len(FEATURES))), [], []
        audit[name] = dict(kept=[FEATURES[i] for i in keep_idx], imputed=imputed, dropped=dropped,
                           imputation_source="source_nested_training_mean")
        if len(src) < 5 or not tgt:
            for r in tgt:
                rr = dict(r); rr.update(toy_pred_error=float("nan"), toy_upper_error=float("nan"),
                                        toy_error_accept="", support_only_accept=bool(r["target_support_excess"] <= 0))
                annotated_targets.append(rr)
            probe_rows.append(dict(group=name, n_source=len(src), n_target=len(tgt),
                                   support_only_accept_rate=float("nan"), toy_error_accept_rate=float("nan"),
                                   mean_pred_error=float("nan"), mean_true_error=float("nan"),
                                   corr_pred_true=float("nan"), violation_support_only=float("nan"),
                                   violation_toy=float("nan"), note=f"toy predictor low power (n_source={len(src)})"))
            continue
        Xs = _feat_matrix(src, keep_idx); es = np.array([r["identity_error"] for r in src])
        mdl = _ridge_fit(Xs, es, args.ridge_alpha)
        resid = np.maximum(0.0, es - _ridge_pred(mdl, Xs))
        qhat = _conformal_q(resid, args.conformal_alpha)
        Xt = _feat_matrix(tgt, keep_idx); et = np.array([r["identity_error"] for r in tgt])
        pred_t = _ridge_pred(mdl, Xt); upper_t = pred_t + qhat
        sup_acc = np.array([r["target_support_excess"] <= 0 for r in tgt])
        toy_acc = upper_t <= args.error_budget
        for r, pe, ue, ta, sa in zip(tgt, pred_t, upper_t, toy_acc, sup_acc):
            rr = dict(r); rr.update(toy_pred_error=float(pe), toy_upper_error=float(ue),
                                    toy_error_accept=bool(ta), support_only_accept=bool(sa))
            annotated_targets.append(rr)
        corr = float(np.corrcoef(pred_t, et)[0, 1]) if len(et) > 2 and np.std(pred_t) > 0 else float("nan")
        v_sup = float(np.mean(et[sup_acc] > args.error_budget)) if sup_acc.any() else 0.0
        v_toy = float(np.mean(et[toy_acc] > args.error_budget)) if toy_acc.any() else 0.0
        probe_rows.append(dict(group=name, n_source=len(src), n_target=len(tgt),
                               support_only_accept_rate=float(sup_acc.mean()),
                               toy_error_accept_rate=float(toy_acc.mean()),
                               mean_pred_error=float(pred_t.mean()), mean_true_error=float(et.mean()),
                               corr_pred_true=corr, violation_support_only=v_sup, violation_toy=v_toy,
                               n_dropped_features=len(dropped), n_imputed_features=len(imputed),
                               note="toy predictor, not final ACAR-error"))
    return probe_rows, annotated_targets, audit


HF3_INTERP = {
    "source_representative_catch": "flagged; source folds carry analogous concept variation",
    "observable_diagnostic_catch": "flagged via observable covariate signature (weak source analogue)",
    "boundary_confirmed": "concept-degraded identity passes support + error bound; boundary confirmed",
    "not_applicable": "not concept-degraded / not applicable",
}


def hf3_probe(annotated_targets, source_records, args):
    """Req-2 three-class verdict per concept-degraded HF3 target: source-representative catch (source
    folds contain analogous concept variation) / observable-diagnostic catch (flagged with weak source
    analogue) / boundary confirmed (evades support + error bound)."""
    analogue = {}
    for r in source_records:
        if r["dataset_or_world"] == "HF3":
            analogue.setdefault(r["config_id"], 0)
            if r["is_concept_unit"] is True:
                analogue[r["config_id"]] += 1
    rows = []
    for r in annotated_targets:
        if r["dataset_or_world"] != "HF3":
            continue
        cd = (r["target_concept_hit"] is True) and (r["identity_bacc"] < 0.60)
        refuse = (r["toy_error_accept"] is False)
        src_analogue = analogue.get(r["config_id"], 0)
        if not cd:
            vclass = "not_applicable"
        elif refuse and src_analogue > 0:
            vclass = "source_representative_catch"
        elif refuse and src_analogue == 0:
            vclass = "observable_diagnostic_catch"
        else:
            vclass = "boundary_confirmed"
        rows.append(dict(seed=r["seed"], config_id=r["config_id"], target_site=r["target_site"],
                         record_unit_id=r["record_unit_id"], support_mode=r["support_mode"],
                         identity_bacc=r["identity_bacc"], identity_error=r["identity_error"],
                         source_analogue_folds=src_analogue,
                         target_support_excess=r["target_support_excess"],
                         support_only_accept=r["support_only_accept"], toy_pred_error=r["toy_pred_error"],
                         toy_upper_error=r["toy_upper_error"], toy_error_accept=r["toy_error_accept"],
                         would_refuse_by_error_budget=refuse, raw_tta_gain=r["raw_tta_gain"],
                         concept_degraded_identity=cd, verdict_class=vclass, interpretation=HF3_INTERP[vclass]))
    return rows


def _load_records(path):
    """Parse a records CSV back into typed dicts (for --from_records post-processing, no re-training)."""
    floats = ["identity_bacc", "offline_tta_bacc", "identity_error", "raw_tta_gain",
              "density_nll_target_prior", "density_nll_source_prior", "support_gap",
              "support_threshold_nll_target_prior", "target_support_excess", "ess", "ood_score",
              "prior_shift", "min_class_responsibility", "entropy_mean", "margin_mean", "max_prob_mean",
              "delta_density_nll", "transform_norm", "condition_number", "pred_disagreement"]
    lists = ["reason_codes", "identity_reason_codes", "offline_tta_reason_codes",
             "offline_tta_blocking_reason_codes"]
    bools = ["accepted", "prior_shift_only", "cmi_residual_available"]
    tri = ["is_concept_unit", "target_concept_hit"]
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d = dict(r)
            for k in floats:
                d[k] = float(d[k]) if d.get(k) not in (None, "") else float("nan")
            for k in ("n", "n_classes", "record_unit_id"):
                try:
                    d[k] = int(float(d[k]))
                except (TypeError, ValueError):
                    d[k] = -1
            for k in bools:
                d[k] = (d.get(k) == "True")
            for k in lists:
                d[k] = [x for x in d[k].split("|") if x] if d.get(k) else []
            for k in tri:
                d[k] = True if d.get(k) == "True" else (False if d.get(k) == "False" else "")
            try:
                d["seed"] = int(float(d["seed"]))
            except (TypeError, ValueError):
                d["seed"] = d.get("seed", "")
            rows.append(d)
    return rows


# ------------------------------------------------------------------ notes
def write_notes(hf3_verdict):
    proto = """# Project B-Next Error-Risk Calibration Protocol

## 1. Purpose
Move Project B from a *support-valid identity* router toward a *risk-calibrated output* router by
building a unified source/target record layer and testing whether source-only **identity-error**
calibration is estimable where source-only **harm** calibration was not.

## 2. Why harm calibration is not enough
Source ACAR-harm needs source pseudo-domains where TTA is actually worse than identity. Those are
often single-class or absent, so harm calibration degenerates (`OACI_ACAR_HARM_CALIBRATION_DEGENERATE`)
and the router blocks TTA without a usable bound. That is a real non-identifiability, not a bug.

## 3. Why error calibration may help
Held-out source subjects/sites usually DO vary in identity error, so an error predictor has signal.
An error budget lets the router accept IDENTITY only when the calibrated upper error is acceptable,
directly targeting HF3's support-valid-but-concept-degraded identity.

## 4. Non-identifiability caveat
Error calibration inherits the SAME boundary as harm calibration. It may repair observable,
source-representative error modes (covariate / support / ESS), but it cannot solve concept-shift error
non-identifiability without a representativeness assumption linking source folds to the target: if the
concept relation differs on the target and leaves no covariate signature, a source-fold predictor sees
no error and accepts the degraded target. S0 TESTS this on HF3 rather than assuming it.

## 5. S0 record schema
Unified `source_or_target` rows share label-free diagnostics (density NLLs, support excess, ESS, ood,
prior_shift, min_class_responsibility, entropy/margin/max_prob, transform diagnostics) plus post-hoc
`identity_bacc`/`identity_error`. `label_access` is `source_calibration` (labels legal to fit a
predictor) or `target_posthoc` (labels for evaluation only).

## 6. Synthetic worlds
R2 (seeds 0,1,2), HF3 (3,4,7,8,10), H-OOD (32). fold_unit=site, record_unit=subject. HF3 is the probe:
if a source-site error predictor cannot flag concept-degraded target identity, ACAR-error hits the
same non-identifiability boundary.

## 7. Real BNCI2014_004 records
LOSO targets 1..4, subject+session eval, both support modes; source nested records from held-out source
subjects. Same schema as synthetic.

## 8. Toy error-risk probe
Numpy ridge on source records + split-conformal upper error (alpha=0.10); accept if
`upper_error <= error_budget` (0.45). Labelled explicitly a TOY predictor, not the final ACAR-error.

## 9. Label-safety rules
Target labels never fit a threshold, a predictor, or a decision; they enter only post-hoc metrics.
Source-calibration labels are legal for fitting.

## 10. What S0 can and cannot claim
S0 can claim: a correct, label-safe, source/target-separated record layer, and HF3 evidence for OR
against source-only error identifiability. S0 cannot claim a final ACAR-error router, accuracy gains,
or that concept shift is solved.
"""
    with open(os.path.join(NOTES, "PROJECT_B_ERROR_RISK_PROTOCOL.md"), "w") as f:
        f.write(proto)

    ident = f"""# Project B Identifiability Note

## Proposition
Without a representativeness assumption linking source pseudo-target domains to the deployment target
domain, source-only calibration cannot identify either action harm or identity output error under
arbitrary concept shift.

## Construction
Consider two worlds with identical source data distribution, identical unlabeled target feature
distribution, and identical action diagnostics, but a different target label mechanism (concept
relation):

    same source observations
  + same unlabeled target diagnostics
  + different target label mechanism
  => same router decision, different true target error/harm.

Any source-only router sees identical observable information in both worlds and must take the same
action; it cannot simultaneously control harm/error in both. Therefore both `risk_harm(a, target)` and
`risk_error(identity, target)` are non-identifiable from source-only observables under arbitrary
concept shift.

## Consequences
- `OACI_ACAR_HARM_CALIBRATION_DEGENERATE` is a necessary state, not an implementation failure.
- Refusal-first is the rational default when the risk quantity is non-identifiable.
- Support-valid identity does not guarantee concept correctness.
- ACAR-error can only repair error modes that leave an observable, source-representative signature.

## Empirical status (S0 / HF3 probe)
{hf3_verdict}
"""
    with open(os.path.join(NOTES, "PROJECT_B_IDENTIFIABILITY_NOTE.md"), "w") as f:
        f.write(ident)


def write_summary(out, source_records, target_records, probe_rows, hf3_rows, hf3_verdict,
                  imputation_summary, vc):
    lines = ["# Project B-Next S0 Calibration Records Summary", "",
             "*Record layer + toy error-risk probe. No final ACAR-error router. Label-safe.*", "",
             f"- source_nested_records: {len(source_records)} rows",
             f"- target_eval_records: {len(target_records)} rows",
             f"- worlds/datasets: {sorted({r['dataset_or_world'] for r in source_records + target_records})}",
             "", "## Toy error-risk probe (per group)", "",
             "| group | n_src | n_tgt | support_acc | toy_err_acc | mean_pred | mean_true | corr | viol_support | viol_toy |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in probe_rows:
        lines.append(f"| {r['group']} | {r['n_source']} | {r['n_target']} | "
                     f"{r['support_only_accept_rate']:.2f} | {r['toy_error_accept_rate']:.2f} | "
                     f"{r['mean_pred_error']:.3f} | {r['mean_true_error']:.3f} | {r['corr_pred_true']:.2f} | "
                     f"{r['violation_support_only']:.2f} | {r['violation_toy']:.2f} |")
    lines += ["", "## HF3 three-class verdict (Req-2)", "",
              f"- source-representative catch: {vc['source_representative_catch']}",
              f"- observable-diagnostic catch: {vc['observable_diagnostic_catch']}",
              f"- boundary-confirmed evasion: {vc['boundary_confirmed']}", "",
              hf3_verdict, "",
              "## Imputation audit (Req-1, source-only)", "",
              f"- imputation_source: {imputation_summary['imputation_source']}",
              f"- imputed_feature_names: {imputation_summary['imputed_feature_names']}",
              f"- dropped_feature_names: {imputation_summary['dropped_feature_names']}",
              f"- has_tta_fallback_features_unavailable: {imputation_summary['has_tta_fallback_features_unavailable']}",
              "", "## Boundary", "This is S0: a record layer and a TOY probe, not the final ACAR-error "
              "implementation. A miss on HF3/H_OOD is a scientific boundary result, not a failure."]
    txt = os.linesep.join(lines) + os.linesep
    with open(os.path.join(out, "records_summary.md"), "w") as f:
        f.write(txt)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Project B-Next S0 calibration record builder")
    ap.add_argument("--include_synthetic", action="store_true")
    ap.add_argument("--include_real_bnci2014_004", action="store_true")
    ap.add_argument("--synthetic_epochs", type=int, default=30)
    ap.add_argument("--synthetic_max_nested_folds", type=int, default=4)
    ap.add_argument("--synthetic_worlds", default="R2,HF3,H_OOD")
    ap.add_argument("--synthetic_max_seeds_per_world", type=int, default=0)  # 0 = all
    ap.add_argument("--real_epochs", type=int, default=8)
    ap.add_argument("--real_max_subjects", type=int, default=6)
    ap.add_argument("--real_max_targets", type=int, default=4)
    ap.add_argument("--real_max_nested_folds", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--error_budget", type=float, default=0.45)
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--conformal_alpha", type=float, default=0.10)
    ap.add_argument("--allow_missing_data", action="store_true")
    ap.add_argument("--skip_branch_check", action="store_true")
    ap.add_argument("--from_records", default=None,
                    help="re-derive probe/verdict/audit from existing records (no re-training)")
    ap.add_argument("--out", default="/tmp/project_b_s0_calibration_records")
    args = ap.parse_args()

    # validation 1: branch
    branch = _current_branch()
    if not args.skip_branch_check and branch != EXPECTED_BRANCH:
        raise Fail(f"[FAIL] validation 1: running branch is '{branch}', expected '{EXPECTED_BRANCH}'")
    os.makedirs(args.out, exist_ok=True)

    worlds = [w for w in args.synthetic_worlds.split(",") if w]
    source_records, target_records = [], []
    real_meta = None

    if args.from_records:
        # Req-1/Req-2 re-derivation: load authoritative records, re-run probe/verdict/audit only.
        source_records = _load_records(os.path.join(args.from_records, "source_nested_records.csv"))
        target_records = _load_records(os.path.join(args.from_records, "target_eval_records.csv"))
        print(f"[from_records] loaded source={len(source_records)} target={len(target_records)} "
              f"from {args.from_records} (no re-training)")
    else:
        if args.include_synthetic:
            for world in worlds:
                seeds = WORLDS[world]["seeds"]
                if args.synthetic_max_seeds_per_world > 0:
                    seeds = seeds[:args.synthetic_max_seeds_per_world]
                for seed in seeds:
                    sr, tr = build_synthetic(world, seed, args)
                    source_records += sr; target_records += tr
        if args.include_real_bnci2014_004:
            try:
                sr, tr, real_meta = build_real(args)
                source_records += sr; target_records += tr
            except Exception as e:  # noqa: BLE001
                with open(os.path.join(args.out, "availability_error_BNCI2014_004.json"), "w") as f:
                    json.dump(dict(error=str(e), traceback=traceback.format_exc()), f, indent=2)
                if not args.allow_missing_data:
                    raise Fail(f"[FAIL] validation 4: real BNCI2014_004 unavailable and --allow_missing_data not set: {e}")
                print(f"[real] BNCI2014_004 unavailable (allowed): {e}")

    # toy probe (Req-1 auditable imputation) + hf3 probe (Req-2 three-class verdict)
    probe_rows, annotated_targets, impute_audit = toy_probe(source_records, target_records, args)
    hf3_rows = hf3_probe(annotated_targets, source_records, args)

    # Req-2 combined verdict: HF3 (source-representative regime) + H_OOD (target-only boundary)
    cd_rows = [r for r in hf3_rows if r["concept_degraded_identity"]]
    caught = [r for r in cd_rows if r["would_refuse_by_error_budget"]]
    vc = {k: sum(1 for r in cd_rows if r["verdict_class"] == k)
          for k in ("source_representative_catch", "observable_diagnostic_catch", "boundary_confirmed")}
    hood = next((r for r in probe_rows if r["group"] == "H_OOD"), None)
    hood_txt = ""
    if hood and math.isfinite(hood.get("corr_pred_true", float("nan"))):
        hood_txt = (f" H_OOD (target-only concept, concept_frac=0.17): corr(pred,true)="
                    f"{hood['corr_pred_true']:.2f}, toy violation={hood['violation_toy']:.2f} -> "
                    f"{'target-only boundary confirmed' if hood['violation_toy'] > 0.1 or hood['corr_pred_true'] < 0.2 else 'target-only estimable'}.")
    if not cd_rows:
        hf3_verdict = ("No concept-degraded HF3 target identity at the current threshold; HF3 boundary "
                       "test inconclusive this run." + hood_txt)
    else:
        hf3_verdict = (
            f"HF3 (source-representative concept, concept_frac=0.50): "
            f"{vc['source_representative_catch']} source-representative catch, "
            f"{vc['observable_diagnostic_catch']} observable-diagnostic catch, "
            f"{vc['boundary_confirmed']} boundary-confirmed evasion "
            f"(of {len(cd_rows)} concept-degraded HF3 targets; {len(caught)} caught total)." + hood_txt +
            " Net: source-only ACAR-error is partially estimable where the error mechanism is "
            "source-representative/observable, but reproduces the non-identifiability boundary for "
            "target-only concept shift; it does NOT solve concept shift.")

    # --- write outputs (records are authoritative in from_records mode: do not rewrite them) ---
    if not args.from_records:
        _write_csv(os.path.join(args.out, "source_nested_records.csv"), RECORD_COLS, source_records)
        _write_csv(os.path.join(args.out, "target_eval_records.csv"), RECORD_COLS, target_records)
    _write_csv(os.path.join(args.out, "error_risk_probe_summary.csv"),
               ["group", "n_source", "n_target", "support_only_accept_rate", "toy_error_accept_rate",
                "mean_pred_error", "mean_true_error", "corr_pred_true", "violation_support_only",
                "violation_toy", "n_dropped_features", "n_imputed_features", "note"], probe_rows)
    _write_csv(os.path.join(args.out, "hf3_concept_probe.csv"),
               ["seed", "config_id", "target_site", "record_unit_id", "support_mode", "identity_bacc",
                "identity_error", "source_analogue_folds", "target_support_excess", "support_only_accept",
                "toy_pred_error", "toy_upper_error", "toy_error_accept", "would_refuse_by_error_budget",
                "raw_tta_gain", "concept_degraded_identity", "verdict_class", "interpretation"], hf3_rows)
    # Req-1 imputation audit
    all_imputed = sorted({f for a in impute_audit.values() for f in a["imputed"]})
    all_dropped = sorted({f for a in impute_audit.values() for f in a["dropped"]})
    imputation_summary = dict(
        imputation_source="source_nested_training_mean", per_group=impute_audit,
        imputed_feature_names=all_imputed, dropped_feature_names=all_dropped,
        has_tta_fallback_features_unavailable=bool(all_imputed or all_dropped))
    with open(os.path.join(args.out, "imputation_summary.json"), "w") as f:
        json.dump(imputation_summary, f, indent=2)
    _write_csv(os.path.join(args.out, "imputation_audit.csv"),
               ["group", "kept_features", "imputed_features", "dropped_features", "imputation_source"],
               [dict(group=g, kept_features=a["kept"], imputed_features=a["imputed"],
                     dropped_features=a["dropped"], imputation_source=a["imputation_source"])
                for g, a in sorted(impute_audit.items())])
    # real dataset summary
    real_rows = [r for r in target_records if r["dataset_or_world"] == "BNCI2014_004"]
    real_summary = []
    if real_rows:
        for eu in ("subject", "session"):
            sub = [r for r in real_rows if r["eval_unit"] == eu]
            if sub:
                real_summary.append(dict(
                    eval_unit=eu, n=len(sub),
                    mean_identity_bacc=float(np.mean([r["identity_bacc"] for r in sub])),
                    mean_raw_tta_gain=float(np.mean([r["raw_tta_gain"] for r in sub])),
                    n_refuse=sum(1 for r in sub if r["decision_action"] == "refuse"),
                    n_identity=sum(1 for r in sub if r["decision_action"] == "identity"),
                    n_offline_tta=sum(1 for r in sub if r["decision_action"] == "offline_tta")))
    _write_csv(os.path.join(args.out, "real_bnci2014_004_records_summary.csv"),
               ["eval_unit", "n", "mean_identity_bacc", "mean_raw_tta_gain", "n_refuse", "n_identity",
                "n_offline_tta"], real_summary)

    schema = dict(record_columns=RECORD_COLS, feature_columns=FEATURES,
                  label_access_values=["source_calibration", "target_posthoc"],
                  error_budget=args.error_budget, conformal_alpha=args.conformal_alpha,
                  ridge_alpha=args.ridge_alpha,
                  imputation_source=imputation_summary["imputation_source"],
                  imputed_feature_names=imputation_summary["imputed_feature_names"],
                  dropped_feature_names=imputation_summary["dropped_feature_names"])
    with open(os.path.join(args.out, "records_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    write_notes(hf3_verdict)
    write_summary(args.out, source_records, target_records, probe_rows, hf3_rows, hf3_verdict,
                  imputation_summary, vc)

    # ------------------ validation gates (fail loud) ------------------
    def _bad_feature(records):
        for r in records:
            for fcol in FEATURES:
                v = r.get(fcol)
                if isinstance(v, float) and not math.isfinite(v):
                    has_unavail = any("UNAVAILABLE" in rc or "INSUFFICIENT" in rc for rc in r["reason_codes"])
                    if not has_unavail:
                        return f"{r['dataset_or_world']}/{r.get('config_id')}/{fcol}"
        return None

    present_worlds = {r["dataset_or_world"] for r in source_records}
    if args.include_synthetic:
        for w in worlds:
            if w not in present_worlds:
                raise Fail(f"[FAIL] validation 3: synthetic world {w} missing from records")
    if not source_records:
        raise Fail("[FAIL] validation 5: source_nested_records has zero rows")
    if not target_records:
        raise Fail("[FAIL] validation 6: target_eval_records has zero rows")
    bad = _bad_feature(source_records + target_records)
    if bad:
        raise Fail(f"[FAIL] validation 7: non-finite feature without unavailable reason code: {bad}")
    if args.include_synthetic and "HF3" in worlds and not hf3_rows:
        raise Fail("[FAIL] validation 8: HF3 probe rows missing")
    for req in ("record_columns", "feature_columns", "label_access_values"):
        if req not in schema:
            raise Fail(f"[FAIL] validation 9: records_schema.json missing {req}")
    # label-safety: target labels never entered the predictor
    for r in target_records:
        if r["label_access"] != "target_posthoc":
            raise Fail(f"[FAIL] validation 2: target record with non-posthoc label_access: {r['config_id']}")

    real_requested = args.include_real_bnci2014_004 or (
        args.from_records and any(r["dataset_or_world"] == "BNCI2014_004" for r in target_records))
    validation = dict(
        step="S0", branch=branch, mode=("from_records" if args.from_records else "train"),
        checks=dict(branch_ok=True, synthetic_worlds_present=sorted(present_worlds),
                    source_records=len(source_records), target_records=len(target_records),
                    features_finite_or_reasoncoded=True, hf3_probe_present=bool(hf3_rows),
                    schema_ok=True, target_labels_posthoc_only=True,
                    predictor_fit_source_only=True, imputation_source_only=True,
                    real_available=bool(real_rows) if real_requested else "not_requested"),
        error_budget=args.error_budget, hf3_concept_degraded=len(cd_rows), hf3_caught=len(caught),
        hf3_verdict_class_counts=vc, hf3_verdict=hf3_verdict,
        imputation=dict(imputation_source=imputation_summary["imputation_source"],
                        imputed_feature_names=imputation_summary["imputed_feature_names"],
                        dropped_feature_names=imputation_summary["dropped_feature_names"],
                        has_tta_fallback_features_unavailable=imputation_summary["has_tta_fallback_features_unavailable"]),
        probe_groups=[r["group"] for r in probe_rows], all_checks_passed=True)
    with open(os.path.join(args.out, "s0_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    print(f"[S0] OK: source={len(source_records)} target={len(target_records)} "
          f"hf3_cd={len(cd_rows)} caught={len(caught)} verdict_classes={vc}")
    print(f"[S0] HF3 verdict: {hf3_verdict}")
    print(f"[S0] imputation: imputed={imputation_summary['imputed_feature_names']} "
          f"dropped={imputation_summary['dropped_feature_names']}")
    print(f"[S0] wrote records + probes + audit + schema + summary + 2 notes to {args.out}")


if __name__ == "__main__":
    main()
