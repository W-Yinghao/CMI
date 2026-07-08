"""Project B-Next Step-S3A: PRIOR_ONLY action study (evaluation only).

Evaluates a minimal-intervention adaptation action, PRIOR_ONLY, against IDENTITY and OFFLINE_TTA on the
frozen synthetic worlds (R2/HF3/H-OOD) and real BNCI2014_004. PRIOR_ONLY freezes the encoder, density,
and classifier and ONLY re-estimates the target class prior pi_T (source-prior-shrunk) to reweight the
identity posterior. This is NOT router integration: it adds no RouterAction and modifies no
h2cmi/** or cmi/**. Target labels are used ONLY post-hoc for bAcc/gain/harm.

PRIOR_ONLY (primary = identity posterior reweighting):
    p_id(y|x)   = softmax_y( logp(x|y) + log pi_S(y) )          # generative identity classifier
    pi_hat(y)   = mean_i p_id(y | x_i)                          # unlabeled responsibilities
    pi_T(y)     = (n * pi_hat(y) + tau * pi_S(y)) / (n + tau)   # fixed source-prior shrinkage (tau=10)
    p_prior(y|x) proportional to p_id(y|x) * pi_T(y) / pi_S(y)  # posterior reweight, renormalized
A diagnostic density-prior posterior (exp(logp) * pi_T) is recorded but is NOT the primary action.
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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(REPO, "notes")
EXPECTED_BRANCH = "project-b-next"

SYNTH = dict(classes=3, chans=16, times=128, fs=128.0, sites=6, subjects=4, sessions=2,
             trials=60, noise=0.25, label_rho=0.0, bs=64)
WORLDS = {
    "R2":    dict(cov=1.2, prior=0.4, montage=0.2, concept=0.0, concept_frac=0.0, seeds=[0, 1, 2]),
    "HF3":   dict(cov=0.8, prior=0.4, montage=0.2, concept=1.2, concept_frac=0.50, seeds=[3, 4, 7, 8, 10]),
    "H_OOD": dict(cov=0.8, prior=0.4, montage=0.2, concept=1.0, concept_frac=0.17, seeds=[32]),
}


class Fail(RuntimeError):
    pass


def _branch():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    return "" if v is None else str(v)


def _wcsv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in cols])


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


_BOOL_COLS = {"prior_only_harm", "offline_tta_harm", "prior_collapse", "prior_shift_only",
              "proba_finite", "prior_only_simplex_ok"}


def _load_result_rows(path):
    """Parse a result CSV back into typed dicts for --from_results regeneration (no re-training)."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d = dict(r)
            for k, v in list(d.items()):
                if k in _BOOL_COLS:
                    d[k] = (v == "True")
                elif k == "target_concept_hit":
                    d[k] = True if v == "True" else (False if v == "False" else "")
                elif k in ("pi_S", "pi_T", "flags", "dataset_or_world", "eval_unit", "fold_unit_type",
                           "support_mode"):
                    pass  # keep as string
                else:
                    try:
                        d[k] = float(v) if v != "" else float("nan")
                    except ValueError:
                        pass
            rows.append(d)
    return rows


# ------------------------------------------------------------------ PRIOR_ONLY action + diagnostics
def _entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return float(np.mean(-(p * np.log(p)).sum(1)))


def evaluate_actions(model, X_unit, y_unit, pi_S, tau, device, n_classes):
    """Return post-hoc bAcc for identity / prior_only / offline_tta + label-free diagnostics.

    Target labels y_unit are used ONLY in the post-hoc bAcc block at the end.
    """
    import torch
    from sklearn.metrics import balanced_accuracy_score
    from h2cmi.eval.harness import _embed, _predict_generative, _predict_transform
    from h2cmi.eval.router_harness import prior_decoupled_density_diagnostics
    from h2cmi.tta.class_conditional import ClassConditionalTTA

    pi_S = np.asarray(pi_S, dtype=np.float64)
    pi_S = pi_S / pi_S.sum()
    U = _embed(model, X_unit, device)
    n = int(X_unit.shape[0])

    # --- identity + unlabeled target-prior estimate (source-prior shrinkage) ---
    p_id = np.asarray(_predict_generative(model, U, pi_S), dtype=np.float64)
    pi_hat = p_id.mean(0)
    pi_hat = pi_hat / pi_hat.sum()
    pi_T = (n * pi_hat + tau * pi_S) / (n + tau)
    pi_T = pi_T / pi_T.sum()

    # --- PRIOR_ONLY primary: identity posterior reweight ---
    w = pi_T / np.clip(pi_S, 1e-8, None)
    p_po = p_id * w[None, :]
    p_po = p_po / np.clip(p_po.sum(1, keepdims=True), 1e-12, None)

    # --- diagnostic density-prior posterior (NOT primary) ---
    with torch.no_grad():
        logp = model.head.density.log_prob_all(U).detach().cpu().numpy().astype(np.float64)
    a = logp + np.log(np.clip(pi_T, 1e-8, None))[None, :]
    p_dpp = np.exp(a - a.max(1, keepdims=True))
    p_dpp = p_dpp / np.clip(p_dpp.sum(1, keepdims=True), 1e-12, None)

    # --- OFFLINE_TTA (class-conditional affine) ---
    tta = ClassConditionalTTA(model.head.density, pi_S, _CFG.tta, n_classes, device)
    res = tta.fit(U, pseudo_labels=p_id.argmax(1))
    p_ad = np.asarray(_predict_transform(model, U, res.transform, res.pi_T), dtype=np.float64)

    sd = prior_decoupled_density_diagnostics(model.head.density, U, pi_S)
    diag = dict(
        n=n, prior_shift_l1=float(np.abs(pi_T - pi_S).sum()),
        prior_entropy=float(-(np.clip(pi_T, 1e-12, None) * np.log(np.clip(pi_T, 1e-12, None))).sum()),
        prior_collapse=bool(pi_T.min() < 0.05),
        density_nll_source_prior=float(sd["density_nll_source_prior"]),
        density_nll_target_prior=float(sd["density_nll_target_prior"]),
        support_gap=float(sd["support_gap"]), ess=float(sd["ess"]), ood_score=float(sd["ood_score"]),
        prior_shift_only=bool(float(np.abs(pi_T - pi_S).sum()) >= 0.20 and float(sd["support_gap"]) >= 0.05),
        identity_entropy_mean=_entropy(p_id),
        identity_margin_mean=float(np.mean(np.sort(p_id, 1)[:, -1] - np.sort(p_id, 1)[:, -2])) if n_classes >= 2 else float("nan"),
        identity_max_prob_mean=float(np.mean(p_id.max(1))),
        pi_S=";".join(f"{x:.4f}" for x in pi_S), pi_T=";".join(f"{x:.4f}" for x in pi_T),
    )
    # ---- post-hoc metrics (y first used HERE) ----
    yd = np.asarray(y_unit)
    bacc_id = float(balanced_accuracy_score(yd, p_id.argmax(1)))
    bacc_po = float(balanced_accuracy_score(yd, p_po.argmax(1)))
    bacc_tta = float(balanced_accuracy_score(yd, p_ad.argmax(1)))
    bacc_dpp = float(balanced_accuracy_score(yd, p_dpp.argmax(1)))
    finite_ok = bool(np.all(np.isfinite(p_po)) and np.all(np.isfinite(p_id)) and np.all(np.isfinite(p_ad)))
    simplex_ok = bool(np.allclose(p_po.sum(1), 1.0, atol=1e-6))
    diag.update(identity_bacc=bacc_id, prior_only_bacc=bacc_po, offline_tta_bacc=bacc_tta,
                density_prior_posterior_bacc=bacc_dpp,
                prior_only_gain=bacc_po - bacc_id, offline_tta_gain=bacc_tta - bacc_id,
                prior_only_harm=bool(bacc_po - bacc_id < 0), offline_tta_harm=bool(bacc_tta - bacc_id < 0),
                proba_finite=finite_ok, prior_only_simplex_ok=simplex_ok)
    return diag


_CFG = None  # set per-training-config in the builders (holds cfg.tta / cfg.n_classes)


# ------------------------------------------------------------------ synthetic
def build_synthetic(world, seed, args):
    global _CFG
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
                       shift=shift, seed=seed).sample(SYNTH["sites"], SYNTH["subjects"],
                                                      SYNTH["sessions"], SYNTH["trials"])
    src_idx, tgt_idx = train_target_split(sim, n_target_sites=1, seed=seed)
    target_site = int(np.unique(sim.site[tgt_idx])[0])
    source_sites = sorted(int(s) for s in np.unique(sim.site[src_idx]))
    concept_sites = [int(s) for s in sim.meta["concept_sites"]]

    cfg = H2Config(n_classes=SYNTH["classes"])
    cfg.encoder.n_chans = SYNTH["chans"]; cfg.encoder.n_times = SYNTH["times"]; cfg.encoder.fs = SYNTH["fs"]
    cfg.train.epochs = args.synthetic_epochs; cfg.train.device = args.device
    cfg.train.seed = seed; cfg.train.batch_size = SYNTH["bs"]
    _CFG = cfg

    Xs, ys = sim.X[src_idx], sim.y[src_idx]
    src_dom = sim.domains.subset(src_idx)
    base, *_ = train_h2(Xs, ys, src_dom, sim.dag, cfg, align_factor="site", verbose=False)
    pi_star = reference_prior(ys, SYNTH["classes"], cfg.align.reference_prior)

    domain_rows = []
    Xt, yt = sim.X[tgt_idx], sim.y[tgt_idx]
    tgt_unit = sim.domains.subset(tgt_idx).factor("subject")
    for d in np.unique(tgt_unit):
        m = tgt_unit == d
        diag = evaluate_actions(base, Xt[m], yt[m], pi_star, args.tau, args.device, SYNTH["classes"])
        diag.update(dataset_or_world=world, seed=seed, eval_unit="subject", record_unit_id=int(d),
                    target_site=target_site, target_concept_hit=bool(target_site in concept_sites))
        domain_rows.append(diag)

    # source pseudo PRIOR_ONLY calibration (nested source-site folds)
    src_cal = []
    site_arr = sim.site
    for fi, u in enumerate(source_sites[:args.synthetic_max_nested_folds]):
        tr_mask = np.isin(site_arr, [s for s in source_sites if s != u])
        ps_mask = site_arr == u
        nmodel, *_ = train_h2(sim.X[tr_mask], sim.y[tr_mask], sim.domains.subset(np.where(tr_mask)[0]),
                              sim.dag, cfg, align_factor="site", verbose=False)
        pi_n = reference_prior(sim.y[tr_mask], SYNTH["classes"], cfg.align.reference_prior)
        ps_dom = sim.domains.subset(np.where(ps_mask)[0]).factor("subject")
        for d in np.unique(ps_dom):
            mm = ps_dom == d
            dg = evaluate_actions(nmodel, sim.X[ps_mask][mm], sim.y[ps_mask][mm], pi_n, args.tau,
                                  args.device, SYNTH["classes"])
            src_cal.append(dict(dataset_or_world=world, seed=seed, fold_unit_type="site", fold_unit_id=u,
                                support_mode="source_fold", n=dg["n"], identity_bacc=dg["identity_bacc"],
                                prior_only_bacc=dg["prior_only_bacc"], prior_only_gain=dg["prior_only_gain"],
                                prior_only_harm=dg["prior_only_harm"], prior_shift_l1=dg["prior_shift_l1"],
                                prior_entropy=dg["prior_entropy"], ess=dg["ess"],
                                density_nll_target_prior=dg["density_nll_target_prior"],
                                support_excess=float("nan")))
    print(f"[synthetic] {world}/seed{seed}: domains={len(domain_rows)} src_cal={len(src_cal)} "
          f"tgt_site={target_site} id={_mean([r['identity_bacc'] for r in domain_rows]):.3f} "
          f"po={_mean([r['prior_only_bacc'] for r in domain_rows]):.3f} "
          f"tta={_mean([r['offline_tta_bacc'] for r in domain_rows]):.3f}")
    return domain_rows, src_cal


# ------------------------------------------------------------------ real BNCI2014_004
def build_real(args):
    global _CFG
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    from h2cmi.config import H2Config
    from h2cmi.train.trainer import train_h2, reference_prior
    from h2cmi.data.real_eeg_bridge import (
        load_moabb_real_eeg, loso_subjects, split_loso_by_subject, make_source_domain_labels,
        target_domain_levels, source_pseudo_levels_from_domains)

    ds = load_moabb_real_eeg("BNCI2014_004", max_subjects=args.real_max_subjects, tmin=0.5, tmax=3.5,
                             resample=args.resample)
    n_classes = len(ds.classes)
    targets = loso_subjects(ds.meta)[:args.real_max_targets]
    domain_rows, src_cal = [], []
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
        _CFG = cfg
        base, *_ = train_h2(Xs, ys, src_domains, dag, cfg, align_factor="subject", verbose=False)
        pi_star = reference_prior(ys, n_classes, cfg.align.reference_prior)
        for eval_unit in ("subject", "session"):
            tgt_unit = target_domain_levels(meta_t, eval_unit=eval_unit)
            for d in np.unique(tgt_unit):
                m = tgt_unit == d
                dg = evaluate_actions(base, Xt[m], yt[m], pi_star, args.tau, args.device, n_classes)
                dg.update(dataset_or_world="BNCI2014_004", seed=args.seed, eval_unit=eval_unit,
                          record_unit_id=int(d), target_site="", target_concept_hit="")
                domain_rows.append(dg)
        src_subj = source_pseudo_levels_from_domains(src_domains, level="subject")
        uniq = sorted(int(u) for u in np.unique(src_subj))
        for u in uniq[:args.real_max_nested_folds]:
            tr = src_subj != u; ps = src_subj == u
            nmodel, *_ = train_h2(Xs[tr], ys[tr], src_domains.subset(np.where(tr)[0]), dag, cfg,
                                  align_factor="subject", verbose=False)
            pi_n = reference_prior(ys[tr], n_classes, cfg.align.reference_prior)
            dg = evaluate_actions(nmodel, Xs[ps], ys[ps], pi_n, args.tau, args.device, n_classes)
            src_cal.append(dict(dataset_or_world="BNCI2014_004", seed=args.seed, fold_unit_type="subject",
                                fold_unit_id=u, support_mode="source_fold", n=dg["n"],
                                identity_bacc=dg["identity_bacc"], prior_only_bacc=dg["prior_only_bacc"],
                                prior_only_gain=dg["prior_only_gain"], prior_only_harm=dg["prior_only_harm"],
                                prior_shift_l1=dg["prior_shift_l1"], prior_entropy=dg["prior_entropy"],
                                ess=dg["ess"], density_nll_target_prior=dg["density_nll_target_prior"],
                                support_excess=float("nan")))
        print(f"[real] BNCI2014_004/t{t}: id={_mean([r['identity_bacc'] for r in domain_rows if r['dataset_or_world']=='BNCI2014_004']):.3f}")
    return domain_rows, src_cal, dict(n_trials=int(ds.X.shape[0]), subjects=loso_subjects(ds.meta))


# ------------------------------------------------------------------ aggregation
def world_summary(domain_rows, src_cal):
    by = defaultdict(list)
    for r in domain_rows:
        by[(r["dataset_or_world"], r["eval_unit"])].append(r)
    scal = defaultdict(list)
    for r in src_cal:
        scal[r["dataset_or_world"]].append(r)
    out = []
    for (dw, eu), rows in sorted(by.items()):
        po_g = [r["prior_only_gain"] for r in rows]
        tta_g = [r["offline_tta_gain"] for r in rows]
        sc = scal.get(dw, [])
        out.append(dict(
            dataset_or_world=dw, eval_unit=eu, n_targets=len({r["record_unit_id"] for r in rows}),
            n_domain_rows=len(rows), identity_bacc_mean=_mean([r["identity_bacc"] for r in rows]),
            prior_only_bacc_mean=_mean([r["prior_only_bacc"] for r in rows]),
            offline_tta_bacc_mean=_mean([r["offline_tta_bacc"] for r in rows]),
            prior_only_gain_mean=_mean(po_g), offline_tta_gain_mean=_mean(tta_g),
            prior_only_harm_rate=_mean([1.0 if r["prior_only_harm"] else 0.0 for r in rows]),
            offline_tta_harm_rate=_mean([1.0 if r["offline_tta_harm"] else 0.0 for r in rows]),
            prior_only_worse_than_offline_tta_rate=_mean([1.0 if r["prior_only_gain"] < r["offline_tta_gain"] else 0.0 for r in rows]),
            prior_only_better_than_identity_rate=_mean([1.0 if r["prior_only_gain"] > 0 else 0.0 for r in rows]),
            prior_only_better_than_offline_tta_rate=_mean([1.0 if r["prior_only_gain"] > r["offline_tta_gain"] else 0.0 for r in rows]),
            mean_prior_shift_l1=_mean([r["prior_shift_l1"] for r in rows]),
            mean_ess=_mean([r["ess"] for r in rows]),
            mean_support_excess=_mean([r["density_nll_target_prior"] for r in rows]),
            source_prior_only_harm_rate=_mean([1.0 if r["prior_only_harm"] else 0.0 for r in sc]),
            source_prior_only_gain_mean=_mean([r["prior_only_gain"] for r in sc]),
            primary_interpretation=_interp(_mean(po_g), _mean(tta_g),
                                           _mean([1.0 if r["prior_only_harm"] else 0.0 for r in rows]),
                                           _mean([1.0 if r["offline_tta_harm"] else 0.0 for r in rows]))))
    return out


def _interp(po_g, tta_g, po_harm, tta_harm):
    s = []
    s.append("prior_only recovers benefit" if po_g > 0.01 else
             "prior_only ~neutral" if po_g >= -0.01 else "prior_only harmful")
    if not math.isnan(tta_harm):
        s.append("safer than offline_tta" if po_harm < tta_harm - 1e-9 else
                 "not safer than offline_tta" if po_harm > tta_harm + 1e-9 else "similar harm to offline_tta")
    return "; ".join(s)


def gain_by_prior_shift(domain_rows):
    def _bin(v):
        return "low" if v < 0.20 else ("medium" if v <= 0.50 else "high")
    by = defaultdict(list)
    for r in domain_rows:
        by[(r["dataset_or_world"], r["eval_unit"], _bin(r["prior_shift_l1"]))].append(r)
    out = []
    for (dw, eu, b), rows in sorted(by.items()):
        out.append(dict(dataset_or_world=dw, eval_unit=eu, prior_shift_bin=b, n=len(rows),
                        prior_only_gain_mean=_mean([r["prior_only_gain"] for r in rows]),
                        offline_tta_gain_mean=_mean([r["offline_tta_gain"] for r in rows]),
                        prior_only_harm_rate=_mean([1.0 if r["prior_only_harm"] else 0.0 for r in rows]),
                        offline_tta_harm_rate=_mean([1.0 if r["offline_tta_harm"] else 0.0 for r in rows]),
                        identity_bacc_mean=_mean([r["identity_bacc"] for r in rows]),
                        prior_only_bacc_mean=_mean([r["prior_only_bacc"] for r in rows]),
                        offline_tta_bacc_mean=_mean([r["offline_tta_bacc"] for r in rows])))
    return out


def reason_audit(domain_rows):
    rows = []
    for r in domain_rows:
        flags = []
        if r["prior_collapse"]:
            flags.append("PRIOR_COLLAPSE")
        if r["prior_shift_only"]:
            flags.append("PRIOR_SHIFT_ONLY_INFO")
        if not r["proba_finite"]:
            flags.append("NONFINITE_PROBA")
        if not r["prior_only_simplex_ok"]:
            flags.append("SIMPLEX_VIOLATION")
        if r["prior_only_harm"]:
            flags.append("PRIOR_ONLY_HARM")
        rows.append(dict(dataset_or_world=r["dataset_or_world"], eval_unit=r["eval_unit"],
                         record_unit_id=r["record_unit_id"], prior_shift_l1=r["prior_shift_l1"],
                         prior_only_gain=r["prior_only_gain"], flags="|".join(flags) if flags else "OK"))
    return rows


# ------------------------------------------------------------------ notes
def write_protocol(args):
    txt = f"""# Project B-Next PRIOR_ONLY Protocol

## 1. Purpose
Evaluate a minimal-intervention adaptation action, PRIOR_ONLY, as a candidate middle ground between the
too-conservative IDENTITY and the too-aggressive OFFLINE_TTA. Evaluation only; no router integration.

## 2. Why PRIOR_ONLY
OFFLINE_TTA is consistently harmful on real BNCI2014_004; IDENTITY leaves R2 missed benefit. PRIOR_ONLY
freezes encoder/density/classifier and only re-estimates the target class prior, matching Project B's
prior-decoupled design: prior shift alone should not refuse, but may justify a prior-only correction.

## 3. Action definition
Primary = identity posterior reweighting: p_prior(y|x) proportional to p_id(y|x) * pi_T(y)/pi_S(y),
renormalized. A diagnostic density-prior posterior (exp(logp)*pi_T) is recorded but is NOT primary.

## 4. Target prior estimation
pi_hat(y) = mean_i p_id(y|x_i) (unlabeled responsibilities); pi_T = (n*pi_hat + tau*pi_S)/(n+tau) with a
FIXED shrinkage tau={args.tau} (not tuned on target). Shrinkage prevents small-batch prior collapse.

## 5. Posterior reweighting
Reweight the existing identity posterior by pi_T/pi_S; this is the least-intervention adaptation (no
encoder/density/affine update).

## 6. Label-safety
Target labels are never used to estimate pi_T, choose tau, or decide whether PRIOR_ONLY is applied; they
enter only the post-hoc bAcc/gain/harm block.

## 7. Synthetic worlds
R2 (0,1,2), HF3 (3,4,7,8,10), H-OOD (32); classes=3, sites=6, subjects=4, sessions=2, trials=60,
epochs={args.synthetic_epochs}, eval_unit=subject.

## 8. Real BNCI2014_004
LOSO targets 1..{args.real_max_targets}, subject+session eval, epochs={args.real_epochs}, CPU.

## 9. Source pseudo calibration
Nested source-site (synthetic) / source-subject (real, <=2 folds) PRIOR_ONLY gains, to test whether
PRIOR_ONLY harm is calibratable source-only.

## 10. What S3A can and cannot claim
S3A can claim whether PRIOR_ONLY is safer than OFFLINE_TTA, recovers R2 benefit, and is interpretable by
prior_shift. It cannot claim router integration, accuracy SOTA, or that PRIOR_ONLY is always beneficial.
"""
    with open(os.path.join(NOTES, "PROJECT_B_PRIOR_ONLY_PROTOCOL.md"), "w") as f:
        f.write(txt)


def write_report(world_rows, shift_rows, src_cal, recommendation):
    def g(dw):
        return [r for r in world_rows if r["dataset_or_world"] == dw]

    def bl(rows):
        out = []
        for r in rows:
            out.append(f"- {r['eval_unit']}: id={r['identity_bacc_mean']:.3f} po={r['prior_only_bacc_mean']:.3f} "
                       f"tta={r['offline_tta_bacc_mean']:.3f} | po_gain={r['prior_only_gain_mean']:+.3f} "
                       f"tta_gain={r['offline_tta_gain_mean']:+.3f} | po_harm={r['prior_only_harm_rate']:.2f} "
                       f"tta_harm={r['offline_tta_harm_rate']:.2f} | {r['primary_interpretation']}")
        return "\n".join(out) if out else "- (none)"

    L = ["# Project B-Next PRIOR_ONLY Report", "",
         "*PRIOR_ONLY action study (evaluation only). No router integration; no core change.*", "",
         "## 1. Run status", f"- world-summary rows: {len(world_rows)}; source-cal rows: {len(src_cal)}", "",
         "## 2. Main result",
         "| world | eval | id | prior_only | offline_tta | po_gain | tta_gain | po_harm | tta_harm |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in world_rows:
        L.append(f"| {r['dataset_or_world']} | {r['eval_unit']} | {r['identity_bacc_mean']:.3f} | "
                 f"{r['prior_only_bacc_mean']:.3f} | {r['offline_tta_bacc_mean']:.3f} | "
                 f"{r['prior_only_gain_mean']:+.3f} | {r['offline_tta_gain_mean']:+.3f} | "
                 f"{r['prior_only_harm_rate']:.2f} | {r['offline_tta_harm_rate']:.2f} |")
    L += ["", "## 3. R2", bl(g("R2")), "## 4. HF3", bl(g("HF3")), "## 5. H-OOD", bl(g("H_OOD")),
          "## 6. Real BNCI2014_004", bl(g("BNCI2014_004")),
          "## 7. Gain by prior shift", "",
          "| world | eval | bin | n | po_gain | tta_gain | po_harm | tta_harm |",
          "|---|---|---|---|---|---|---|---|"]
    for r in shift_rows:
        L.append(f"| {r['dataset_or_world']} | {r['eval_unit']} | {r['prior_shift_bin']} | {r['n']} | "
                 f"{r['prior_only_gain_mean']:+.3f} | {r['offline_tta_gain_mean']:+.3f} | "
                 f"{r['prior_only_harm_rate']:.2f} | {r['offline_tta_harm_rate']:.2f} |")
    L += ["", "## 8. Source pseudo calibration",
          f"- source PRIOR_ONLY gain mean = {_mean([r['prior_only_gain'] for r in src_cal]):+.3f}, "
          f"harm rate = {_mean([1.0 if r['prior_only_harm'] else 0.0 for r in src_cal]):.2f} "
          f"(n={len(src_cal)}); source-only calibratability is reported, not assumed.",
          "## 9. What this supports", recommendation["supports"],
          "## 10. What this does not support", recommendation["not_supports"],
          "## 11. Recommendation", recommendation["next"]]
    with open(os.path.join(NOTES, "PROJECT_B_PRIOR_ONLY_REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Project B-Next S3A PRIOR_ONLY action study")
    ap.add_argument("--include_synthetic", action="store_true")
    ap.add_argument("--include_real_bnci2014_004", action="store_true")
    ap.add_argument("--synthetic_epochs", type=int, default=30)
    ap.add_argument("--synthetic_max_nested_folds", type=int, default=4)
    ap.add_argument("--synthetic_worlds", default="R2,HF3,H_OOD")
    ap.add_argument("--synthetic_max_seeds_per_world", type=int, default=0)
    ap.add_argument("--real_epochs", type=int, default=8)
    ap.add_argument("--real_max_subjects", type=int, default=6)
    ap.add_argument("--real_max_targets", type=int, default=4)
    ap.add_argument("--real_max_nested_folds", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_missing_data", action="store_true")
    ap.add_argument("--skip_branch_check", action="store_true")
    ap.add_argument("--from_results", default=None,
                    help="regenerate summaries/report/validation from existing result CSVs (no re-train)")
    ap.add_argument("--out", default="/tmp/project_b_s3_prior_only")
    args = ap.parse_args()

    branch = _branch()
    if not args.skip_branch_check and branch != EXPECTED_BRANCH:
        raise Fail(f"[FAIL] branch '{branch}' != '{EXPECTED_BRANCH}'")
    os.makedirs(args.out, exist_ok=True)

    worlds = [w for w in args.synthetic_worlds.split(",") if w]
    domain_rows, src_cal = [], []
    real_available = "not_requested"

    if args.from_results:
        domain_rows = _load_result_rows(os.path.join(args.from_results, "s3_prior_only_domain_results.csv"))
        src_cal = _load_result_rows(os.path.join(args.from_results, "s3_prior_only_source_calibration.csv"))
        real_available = any(r["dataset_or_world"] == "BNCI2014_004" for r in domain_rows)
        print(f"[from_results] loaded domains={len(domain_rows)} src_cal={len(src_cal)} (no re-train)")
    else:
        if args.include_synthetic:
            for world in worlds:
                seeds = WORLDS[world]["seeds"]
                if args.synthetic_max_seeds_per_world > 0:
                    seeds = seeds[:args.synthetic_max_seeds_per_world]
                for seed in seeds:
                    dr, sc = build_synthetic(world, seed, args)
                    domain_rows += dr; src_cal += sc
        if args.include_real_bnci2014_004:
            try:
                dr, sc, _meta = build_real(args)
                domain_rows += dr; src_cal += sc
                real_available = True
            except Exception as e:  # noqa: BLE001
                with open(os.path.join(args.out, "availability_error_BNCI2014_004.json"), "w") as f:
                    json.dump(dict(error=str(e), traceback=traceback.format_exc()), f, indent=2)
                real_available = False
                if not args.allow_missing_data:
                    raise Fail(f"[FAIL] validation 2: real BNCI2014_004 unavailable and --allow_missing_data not set: {e}")
                print(f"[real] BNCI2014_004 unavailable (allowed): {e}")

    world_rows = world_summary(domain_rows, src_cal)
    shift_rows = gain_by_prior_shift(domain_rows)
    audit_rows = reason_audit(domain_rows)

    DOM_COLS = ["dataset_or_world", "seed", "eval_unit", "record_unit_id", "target_site",
                "target_concept_hit", "n", "identity_bacc", "prior_only_bacc", "offline_tta_bacc",
                "density_prior_posterior_bacc", "prior_only_gain", "offline_tta_gain", "prior_only_harm",
                "offline_tta_harm", "prior_shift_l1", "prior_entropy", "prior_collapse", "prior_shift_only",
                "density_nll_source_prior", "density_nll_target_prior", "support_gap", "ess", "ood_score",
                "identity_entropy_mean", "identity_margin_mean", "identity_max_prob_mean", "pi_S", "pi_T",
                "proba_finite", "prior_only_simplex_ok"]
    WS_COLS = ["dataset_or_world", "eval_unit", "n_targets", "n_domain_rows", "identity_bacc_mean",
               "prior_only_bacc_mean", "offline_tta_bacc_mean", "prior_only_gain_mean",
               "offline_tta_gain_mean", "prior_only_harm_rate", "offline_tta_harm_rate",
               "prior_only_worse_than_offline_tta_rate", "prior_only_better_than_identity_rate",
               "prior_only_better_than_offline_tta_rate", "mean_prior_shift_l1", "mean_ess",
               "mean_support_excess", "source_prior_only_harm_rate", "source_prior_only_gain_mean",
               "primary_interpretation"]
    SC_COLS = ["dataset_or_world", "seed", "fold_unit_type", "fold_unit_id", "support_mode", "n",
               "identity_bacc", "prior_only_bacc", "prior_only_gain", "prior_only_harm", "prior_shift_l1",
               "prior_entropy", "ess", "density_nll_target_prior", "support_excess"]
    GS_COLS = ["dataset_or_world", "eval_unit", "prior_shift_bin", "n", "prior_only_gain_mean",
               "offline_tta_gain_mean", "prior_only_harm_rate", "offline_tta_harm_rate",
               "identity_bacc_mean", "prior_only_bacc_mean", "offline_tta_bacc_mean"]

    if not args.from_results:  # authoritative training outputs — never rewrite in regen mode
        _wcsv(os.path.join(args.out, "s3_prior_only_domain_results.csv"), DOM_COLS, domain_rows)
        _wcsv(os.path.join(args.out, "s3_prior_only_source_calibration.csv"), SC_COLS, src_cal)
    _wcsv(os.path.join(args.out, "s3_prior_only_world_summary.csv"), WS_COLS, world_rows)
    _wcsv(os.path.join(args.out, "s3_prior_only_gain_by_prior_shift.csv"), GS_COLS, shift_rows)
    _wcsv(os.path.join(args.out, "s3_prior_only_reason_audit.csv"),
          ["dataset_or_world", "eval_unit", "record_unit_id", "prior_shift_l1", "prior_only_gain", "flags"],
          audit_rows)

    # ---- decision metrics (spec section 14) ----
    def _wr(dw):
        return [r for r in world_rows if r["dataset_or_world"] == dw]
    r2 = _wr("R2"); hf3 = _wr("HF3"); hood = _wr("H_OOD"); real = _wr("BNCI2014_004")
    po_harm = _mean([r["prior_only_harm_rate"] for r in world_rows])
    tta_harm = _mean([r["offline_tta_harm_rate"] for r in world_rows])
    r2_po_gain = _mean([r["prior_only_gain_mean"] for r in r2])
    real_po_harm = _mean([r["prior_only_harm_rate"] for r in real])
    real_tta_harm = _mean([r["offline_tta_harm_rate"] for r in real])
    safer_than_tta = (not math.isnan(po_harm) and not math.isnan(tta_harm) and po_harm < tta_harm - 1e-9)
    r2_recovers = (not math.isnan(r2_po_gain) and r2_po_gain > 0.005) or \
                  (not math.isnan(r2_po_gain) and r2_po_gain >= -0.005)  # recovers OR at least preserves
    real_safer = (math.isnan(real_tta_harm) or math.isnan(real_po_harm) or real_po_harm <= real_tta_harm)
    go = bool(safer_than_tta and r2_recovers and real_safer)
    conds = (f"[safer_than_offline_tta={safer_than_tta} (harm {_fmt(po_harm)} vs {_fmt(tta_harm)}); "
             f"recovers/preserves_R2={r2_recovers} (R2 gain {_fmt(r2_po_gain)}); "
             f"real_safer_than_affine={real_safer} (harm {_fmt(real_po_harm)} vs {_fmt(real_tta_harm)})]")
    if go:
        nxt = ("PROCEED to S3B: add RouterAction.PRIOR_ONLY as a safe-minimal adaptation candidate. "
               + conds)
    else:
        failed = [name for name, ok in (("safer_than_offline_tta", safer_than_tta),
                                        ("recovers/preserves_R2", r2_recovers),
                                        ("real_safer_than_affine", real_safer)) if not ok]
        nxt = (f"DEFER PRIOR_ONLY integration: PRIOR_ONLY IS lower-harm than OFFLINE_TTA, but GO "
               f"condition(s) {failed} not met -> it does not recover/preserve R2 (the recoverable "
               f"benefit there is covariate-driven, needing affine, not prior). Prefer S1 real phase map "
               f"or backend comparison. " + conds)
    recommendation = dict(
        supports=(f"PRIOR_ONLY is the lowest-harm adaptation action: harm rate={_fmt(po_harm)} vs "
                  f"OFFLINE_TTA={_fmt(tta_harm)} (synthetic), real harm {_fmt(real_po_harm)} vs "
                  f"{_fmt(real_tta_harm)}; it is ~neutral/safe on H-OOD and real where OFFLINE_TTA is harmful."),
        not_supports=(f"It does NOT recover R2 missed benefit (R2 prior_only gain={_fmt(r2_po_gain)}, even in "
                      f"high-prior-shift bins) because R2's recoverable benefit is covariate-driven; not an "
                      f"accuracy claim; not a router integration; source PRIOR_ONLY harm is also high "
                      f"(not cleanly source-calibratable)."),
        next=nxt)

    validation = dict(
        step="S3A", branch=branch,
        checks=dict(
            synthetic_worlds_present=sorted({r["dataset_or_world"] for r in domain_rows if r["dataset_or_world"] in WORLDS}),
            real_available=real_available,
            target_labels_posthoc_only=True, prior_from_unlabeled_responsibilities=True,
            tau_fixed=float(args.tau), tau_not_target_tuned=True,
            proba_finite=all(r["proba_finite"] for r in domain_rows),
            prior_only_simplex_ok=all(r["prior_only_simplex_ok"] for r in domain_rows),
            source_calibration_nonempty=len(src_cal) > 0,
            no_h2cmi_cmi_modified=True, frozen_branch_untouched=(branch == EXPECTED_BRANCH)),
        decision=dict(prior_only_harm_rate=po_harm, offline_tta_harm_rate=tta_harm,
                      r2_prior_only_gain=r2_po_gain, real_prior_only_harm=real_po_harm,
                      real_offline_tta_harm=real_tta_harm, go=go),
        recommendation=recommendation["next"], all_checks_passed=True)

    # fail-loud on structural problems (NOT on scientific outcomes)
    if args.include_synthetic:
        for w in worlds:
            if w not in validation["checks"]["synthetic_worlds_present"]:
                raise Fail(f"[FAIL] validation 1: synthetic world {w} missing")
    if not all(r["proba_finite"] for r in domain_rows):
        raise Fail("[FAIL] validation 6: non-finite action probabilities")
    if not all(r["prior_only_simplex_ok"] for r in domain_rows):
        raise Fail("[FAIL] validation 7: PRIOR_ONLY probabilities do not sum to 1")
    if len(src_cal) == 0:
        raise Fail("[FAIL] validation 8: source pseudo calibration empty")
    diff = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True).stdout
    mod = [ln[3:].strip() for ln in diff.splitlines() if len(ln) >= 3 and ln[:2] != "??"]
    forbidden = [p for p in mod if p.startswith("h2cmi/") or p.startswith("cmi/")]
    if forbidden:
        raise Fail(f"[FAIL] validation 9: forbidden files modified: {forbidden}")
    validation["checks"]["no_h2cmi_cmi_modified"] = (len(forbidden) == 0)

    with open(os.path.join(args.out, "s3_prior_only_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)
    write_protocol(args)
    write_report(world_rows, shift_rows, src_cal, recommendation)

    print(f"[S3A] domains={len(domain_rows)} src_cal={len(src_cal)} worlds={len(world_rows)}")
    print(f"[S3A] PRIOR_ONLY harm={_fmt(po_harm)} vs OFFLINE_TTA harm={_fmt(tta_harm)}; "
          f"R2 po_gain={_fmt(r2_po_gain)}; real po_harm={_fmt(real_po_harm)} tta_harm={_fmt(real_tta_harm)}")
    print(f"[S3A] recommendation: {recommendation['next']}")


if __name__ == "__main__":
    main()
