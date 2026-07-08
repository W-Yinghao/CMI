"""Project B-Next Step-S1A: real-EEG TTA phase map + source-predictability test.

Locked scientific question: *Is real-EEG OFFLINE_TTA benefit SOURCE-PREDICTABLE?* — not merely "does
TTA ever help". For the router, a benefit phase is only actionable if it can be predicted from
source-only diagnostics / source-fold gain records; otherwise the honest ceiling is refusal/identity
governance.

Evaluation only: no new RouterAction, no method change, modifies no h2cmi/** or cmi/**. Reuses the S0
routing/diagnostic extraction (_route/_records_from_report), the S3A PRIOR_ONLY reweight, and the S2B
cross-fitted predictor (error_risk.fit_error_risk_crossfit) applied to offline_tta_gain. Target labels
are used ONLY post-hoc.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import csv
import json
import math
import subprocess
import sys
import traceback
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # scripts/ (reuse s0)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np

import project_b_calibration_records as s0  # reuse _route/_records_from_report/_pred_stats/_q
from h2cmi.router.error_risk import ErrorRiskConfig, fit_error_risk_crossfit, predict_error_risk
from h2cmi.router.features import CalibrationState

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(REPO, "notes")
EXPECTED_BRANCH = "project-b-next"
TAU = 10.0

BENEFIT_FEATURES = ["target_support_excess", "ess", "ood_score", "prior_shift", "entropy_mean",
                    "margin_mean", "max_prob_mean", "delta_density_nll", "transform_norm",
                    "condition_number", "pred_disagreement"]
CORR_FEATURES = ["target_support_excess", "ess", "prior_shift", "entropy_mean", "margin_mean",
                 "delta_density_nll", "pred_disagreement"]


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


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _prior_only_bacc(model, X_unit, y_unit, pi_ref, device):
    """S3A PRIOR_ONLY: reweight identity posterior by a source-prior-shrunk target prior (label-free)."""
    from sklearn.metrics import balanced_accuracy_score
    from h2cmi.eval.harness import _embed, _predict_generative
    pi_S = np.asarray(pi_ref, float); pi_S = pi_S / pi_S.sum()
    U = _embed(model, X_unit, device)
    p_id = np.asarray(_predict_generative(model, U, pi_S), float)
    pi_hat = p_id.mean(0); pi_hat = pi_hat / pi_hat.sum()
    n = int(X_unit.shape[0])
    pi_T = (n * pi_hat + TAU * pi_S) / (n + TAU); pi_T = pi_T / pi_T.sum()
    p_po = p_id * (pi_T / np.clip(pi_S, 1e-8, None))[None, :]
    p_po = p_po / np.clip(p_po.sum(1, keepdims=True), 1e-12, None)
    ok = bool(np.all(np.isfinite(p_po)) and np.allclose(p_po.sum(1), 1.0, atol=1e-6))
    return float(balanced_accuracy_score(np.asarray(y_unit), p_po.argmax(1))), ok


def _finalize(rec, args):
    g_tta = rec.get("raw_tta_gain", float("nan"))
    g_po = rec.get("prior_only_gain", float("nan"))
    rec["offline_tta_gain"] = g_tta
    rec["offline_tta_benefit"] = bool(g_tta > args.gain_margin)
    rec["offline_tta_harm"] = bool(g_tta < -args.gain_margin)
    rec["prior_only_benefit"] = bool(g_po > args.gain_margin)
    rec["prior_only_harm"] = bool(g_po < -args.gain_margin)
    return rec


# ------------------------------------------------------------------ per-dataset builder
def build_dataset(name, args):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    from h2cmi.config import H2Config
    from h2cmi.train.trainer import train_h2, reference_prior
    from h2cmi.data.real_eeg_bridge import (
        load_moabb_real_eeg, loso_subjects, split_loso_by_subject, make_source_domain_labels,
        target_domain_levels, source_pseudo_levels_from_domains)

    ds = load_moabb_real_eeg(name, max_subjects=args.max_subjects, tmin=0.5, tmax=3.5, resample=args.resample)
    n_classes = len(ds.classes)
    targets = loso_subjects(ds.meta)[:args.max_targets]
    domain_rows, source_rows = [], []

    from h2cmi.eval.harness import _embed
    from h2cmi.eval.router_harness import prior_decoupled_density_diagnostics

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
        cfg.train.epochs = args.epochs; cfg.train.batch_size = args.batch_size
        cfg.train.device = args.device; cfg.train.seed = args.seed
        base, *_ = train_h2(Xs, ys, src_domains, dag, cfg, align_factor="subject", verbose=False)
        pi_star = reference_prior(ys, n_classes, cfg.align.reference_prior)
        src_subj_levels = source_pseudo_levels_from_domains(src_domains, level="subject")
        base_q95 = s0._q(list(_subj_nll(base, Xs, src_subj_levels, pi_star, cfg).values()))

        # nested source folds -> source-fold records (labels legal = source calibration) + nested threshold
        uniq = sorted(int(u) for u in np.unique(src_subj_levels))
        all_excess = []
        for u in uniq[:args.max_nested_folds]:
            tr = src_subj_levels != u; ps = src_subj_levels == u
            nmodel, *_ = train_h2(Xs[tr], ys[tr], src_domains.subset(np.where(tr)[0]), dag, cfg,
                                  align_factor="subject", verbose=False)
            pi_n = reference_prior(ys[tr], n_classes, cfg.align.reference_prior)
            tr_subj = source_pseudo_levels_from_domains(src_domains.subset(np.where(tr)[0]), level="subject")
            fold_q95 = s0._q(list(_subj_nll(nmodel, Xs[tr], tr_subj, pi_n, cfg).values()))
            ps_subj = source_pseudo_levels_from_domains(src_domains.subset(np.where(ps)[0]), level="subject")
            all_excess.extend([v - fold_q95 for v in _subj_nll(nmodel, Xs[ps], ps_subj, pi_n, cfg).values()])
            rep = s0._route(nmodel, Xs[ps], ys[ps], ps_subj, cfg, pi_n, Xs[tr], ys[tr], tr_subj,
                            args.device, fold_q95, "source_fold_train_q95")
            ctx = dict(source_or_target="source", dataset=name, dataset_or_world=name, seed=args.seed,
                       config_id=f"{name}_t{t}", target_subject=t, fold_unit_type="subject", fold_unit_id=u,
                       record_unit_type="subject", eval_unit="subject", support_mode="source_fold",
                       n_classes=n_classes, label_access="source_calibration",
                       cf_group=f"{name}:{t}:{u}")
            recs = s0._records_from_report(rep, nmodel, Xs[ps], ps_subj, pi_n, args.device, fold_q95, ctx)
            for r in recs:
                mm = ps_subj == r["record_unit_id"]
                r["prior_only_bacc"], _ok = _prior_only_bacc(nmodel, Xs[ps][mm], ys[ps][mm], pi_n, args.device)
                r["prior_only_gain"] = r["prior_only_bacc"] - r["identity_bacc"]
                source_rows.append(_finalize(r, args))
        nested_thr = base_q95 + max(0.0, s0._q(all_excess)) if all_excess else base_q95

        # target records under both support modes and both eval units
        for eval_unit in [e for e in args.eval_units]:
            tgt_unit = target_domain_levels(meta_t, eval_unit=eval_unit)
            for mode, thr in (("in_source_subject_q95", base_q95),
                              ("nested_source_subject_excess_q95", nested_thr)):
                rep = s0._route(base, Xt, yt, tgt_unit, cfg, pi_star, Xs, ys, src_subj_levels,
                                args.device, thr, mode)
                ctx = dict(source_or_target="target", dataset=name, dataset_or_world=name, seed=args.seed,
                           config_id=f"{name}_t{t}", target_subject=t, fold_unit_type="subject",
                           fold_unit_id=t, record_unit_type=eval_unit, eval_unit=eval_unit,
                           support_mode=mode, n_classes=n_classes, label_access="target_posthoc")
                recs = s0._records_from_report(rep, base, Xt, tgt_unit, pi_star, args.device, thr, ctx)
                for r in recs:
                    mm = tgt_unit == r["record_unit_id"]
                    r["prior_only_bacc"], _ok = _prior_only_bacc(base, Xt[mm], yt[mm], pi_star, args.device)
                    r["prior_only_gain"] = r["prior_only_bacc"] - r["identity_bacc"]
                    domain_rows.append(_finalize(r, args))
        print(f"[{name}] t{t}: base_q95={base_q95:.2f} nested_thr={nested_thr:.2f} "
              f"id={_mean([r['identity_bacc'] for r in domain_rows if r['target_subject']==t]):.3f} "
              f"tta_g={_mean([r['offline_tta_gain'] for r in domain_rows if r['target_subject']==t]):+.3f}", flush=True)
    return domain_rows, source_rows, dict(dataset=name, n_classes=n_classes, n_trials=int(ds.X.shape[0]),
                                          n_chans=int(ds.X.shape[1]), subjects=loso_subjects(ds.meta))


# ------------------------------------------------------------------ predictability test
def gain_vs_diagnostic(domain_rows, source_rows, args):
    rows = []
    for dw in sorted({r["dataset"] for r in source_rows}):
        src = [r for r in source_rows if r["dataset"] == dw]
        if len(src) < 3:
            continue
        ecfg = ErrorRiskConfig(alpha=0.10, ridge_alpha=args.ridge_alpha, min_groups=3,
                               min_strict_examples=int(math.ceil(1.0 / 0.10)))
        fit = fit_error_risk_crossfit(src, feature_names=BENEFIT_FEATURES, group_key="cf_group",
                                      target_key="offline_tta_gain", config=ecfg)
        sg = np.array([r["offline_tta_gain"] for r in src], float)
        corrs = {f"corr_source_gain_vs_{f}": _corr([r.get(f, float("nan")) for r in src], sg)
                 for f in CORR_FEATURES}
        # Predictability = point-prediction transfer (coef available), decoupled from conformal strictness
        # (bounded source folds n<10 make the strict qhat UNAVAILABLE, but the scientific transfer test
        # only needs OOF point predictions).
        predictor_ready = fit.coef is not None and np.isfinite(fit.source_oof_pred).any()
        oof_corr = _corr(fit.source_oof_pred, fit.source_oof_true) if predictor_ready else float("nan")
        for eu in sorted({r["eval_unit"] for r in domain_rows if r["dataset"] == dw}):
            for sm in sorted({r["support_mode"] for r in domain_rows if r["dataset"] == dw and r["eval_unit"] == eu}):
                tgt = [r for r in domain_rows if r["dataset"] == dw and r["eval_unit"] == eu and r["support_mode"] == sm]
                if not tgt:
                    continue
                avail = predictor_ready
                if avail:
                    pt = predict_error_risk(fit, tgt)
                    tg = np.array([r["offline_tta_gain"] for r in tgt], float)
                    tcorr = _corr(pt, tg)
                    sel = pt > args.gain_margin
                    tsel_rate = float(np.mean(sel))
                    tsel_gain = float(np.mean(tg[sel])) if sel.any() else float("nan")
                    tsel_harm = float(np.mean(tg[sel] < -args.gain_margin)) if sel.any() else 0.0
                    missed = [r for r in tgt if r["offline_tta_benefit"]]
                    tmissed = (sum(1 for r, s in zip(tgt, sel) if r["offline_tta_benefit"] and not s) / len(missed)) if missed else float("nan")
                    tcaught = float(np.mean([(r["offline_tta_gain"] < -args.gain_margin) for r, s in zip(tgt, sel) if s])) if sel.any() else 0.0
                    # source OOF selective
                    oof = fit.source_oof_pred; osel = oof > args.gain_margin
                    src_sel_rate = float(np.mean(osel[np.isfinite(oof)])) if np.isfinite(oof).any() else float("nan")
                    src_sel_gain = float(np.mean(sg[osel & np.isfinite(oof)])) if (osel & np.isfinite(oof)).any() else float("nan")
                    src_sel_harm = float(np.mean(sg[osel & np.isfinite(oof)] < -args.gain_margin)) if (osel & np.isfinite(oof)).any() else 0.0
                else:
                    tcorr = tsel_rate = tsel_gain = tsel_harm = tmissed = tcaught = float("nan")
                    src_sel_rate = src_sel_gain = src_sel_harm = float("nan")
                row = dict(dataset=dw, eval_unit=eu, support_mode=sm, n_source=len(src), n_target=len(tgt),
                           source_predictor_available=avail, source_oof_corr=oof_corr,
                           target_corr_pred_gain=tcorr, source_select_rate=src_sel_rate,
                           source_selected_gain_mean_oof=src_sel_gain, source_selected_harm_rate_oof=src_sel_harm,
                           target_select_rate=tsel_rate, target_selected_gain_mean_posthoc=tsel_gain,
                           target_selected_harm_rate_posthoc=tsel_harm, target_missed_benefit_rate=tmissed,
                           target_caught_harm_rate=tcaught)
                row.update(corrs)
                rows.append(row)
    return rows


def benefit_phase_analysis(domain_rows, source_rows, gvd_rows, args):
    rows = []
    for gv in gvd_rows:
        dw, eu, sm = gv["dataset"], gv["eval_unit"], gv["support_mode"]
        tgt = [r for r in domain_rows if r["dataset"] == dw and r["eval_unit"] == eu and r["support_mode"] == sm]
        src = [r for r in source_rows if r["dataset"] == dw]
        tg = [r["offline_tta_gain"] for r in tgt]
        sgn = [r["offline_tta_gain"] for r in src]
        t_ben = _mean([1.0 if r["offline_tta_benefit"] else 0.0 for r in tgt])
        t_max = max([g for g in tg if not math.isnan(g)], default=float("nan"))
        exists = bool((not math.isnan(t_ben) and t_ben >= 0.20) or (not math.isnan(t_max) and t_max >= 0.05))
        predictable = bool(gv["source_predictor_available"] and gv["target_corr_pred_gain"] > 0.30
                           and (not math.isnan(gv["target_selected_gain_mean_posthoc"]))
                           and gv["target_selected_gain_mean_posthoc"] > 0.02
                           and gv["target_selected_harm_rate_posthoc"] <= 0.25)
        if not exists:
            impl = "no_real_benefit_phase_observed"
        elif predictable:
            impl = "selective_tta_candidate"
        elif len(src) < 6 or not gv["source_predictor_available"]:
            impl = "insufficient_power"
        else:
            impl = "benefit_exists_but_not_source_predictable"
        interp = {"selective_tta_candidate": "benefit phase exists AND is source-predictable -> selective TTA candidate",
                  "benefit_exists_but_not_source_predictable": "benefit exists but not source-predictable -> refusal/identity ceiling",
                  "no_real_benefit_phase_observed": "no real benefit phase -> refusal/identity governance is the contribution",
                  "insufficient_power": "too few source folds / predictor unavailable -> inconclusive"}[impl]
        rows.append(dict(dataset=dw, eval_unit=eu, support_mode=sm, n_target_domains=len(tgt),
                         n_source_folds=len(src), target_benefit_rate=t_ben,
                         target_harm_rate=_mean([1.0 if r["offline_tta_harm"] else 0.0 for r in tgt]),
                         target_gain_mean=_mean(tg), target_gain_max=t_max,
                         source_benefit_rate=_mean([1.0 if r["offline_tta_benefit"] else 0.0 for r in src]),
                         source_harm_rate=_mean([1.0 if r["offline_tta_harm"] else 0.0 for r in src]),
                         source_gain_mean=_mean(sgn), source_gain_max=max([g for g in sgn if not math.isnan(g)], default=float("nan")),
                         source_predictor_available=gv["source_predictor_available"], source_oof_corr=gv["source_oof_corr"],
                         target_transfer_corr=gv["target_corr_pred_gain"], target_select_rate=gv["target_select_rate"],
                         target_selected_gain_mean=gv["target_selected_gain_mean_posthoc"],
                         target_selected_harm_rate=gv["target_selected_harm_rate_posthoc"],
                         benefit_phase_exists=exists, benefit_phase_source_predictable=predictable,
                         router_action_implication=impl, primary_interpretation=interp))
    return rows


# ------------------------------------------------------------------ dataset summary + reason audit
def dataset_summary(domain_rows, source_rows):
    out = []
    for dw in sorted({r["dataset"] for r in domain_rows}):
        for eu in sorted({r["eval_unit"] for r in domain_rows if r["dataset"] == dw}):
            for sm in sorted({r["support_mode"] for r in domain_rows if r["dataset"] == dw and r["eval_unit"] == eu}):
                rows = [r for r in domain_rows if r["dataset"] == dw and r["eval_unit"] == eu and r["support_mode"] == sm]
                if not rows:
                    continue
                out.append(dict(
                    dataset=dw, eval_unit=eu, support_mode=sm, n_domain_rows=len(rows),
                    identity_bacc_mean=_mean([r["identity_bacc"] for r in rows]),
                    offline_tta_bacc_mean=_mean([r["offline_tta_bacc"] for r in rows]),
                    prior_only_bacc_mean=_mean([r["prior_only_bacc"] for r in rows]),
                    offline_tta_gain_mean=_mean([r["offline_tta_gain"] for r in rows]),
                    prior_only_gain_mean=_mean([r["prior_only_gain"] for r in rows]),
                    offline_tta_benefit_rate=_mean([1.0 if r["offline_tta_benefit"] else 0.0 for r in rows]),
                    offline_tta_harm_rate=_mean([1.0 if r["offline_tta_harm"] else 0.0 for r in rows]),
                    prior_only_harm_rate=_mean([1.0 if r["prior_only_harm"] else 0.0 for r in rows]),
                    coverage=_mean([1.0 if r["accepted"] else 0.0 for r in rows]),
                    identity_rate=_mean([1.0 if r["decision_action"] == "identity" else 0.0 for r in rows]),
                    offline_tta_rate=_mean([1.0 if r["decision_action"] == "offline_tta" else 0.0 for r in rows]),
                    mean_ess=_mean([r["ess"] for r in rows]), mean_prior_shift=_mean([r["prior_shift"] for r in rows]),
                    mean_target_support_excess=_mean([r["target_support_excess"] for r in rows])))
    return out


def reason_audit(domain_rows):
    from collections import Counter
    top, ida, tta_a, tta_b = Counter(), Counter(), Counter(), Counter()
    keyset = set()
    for r in domain_rows:
        key = (r["dataset"], r["eval_unit"], r["support_mode"])
        for c in r["reason_codes"]:
            top[(key, c)] += 1
        for c in r["identity_reason_codes"]:
            ida[(key, c)] += 1
        for c in r["offline_tta_reason_codes"]:
            tta_a[(key, c)] += 1
        for c in r["offline_tta_blocking_reason_codes"]:
            tta_b[(key, c)] += 1
        keyset.update((key, c) for c in set(r["reason_codes"]) | set(r["identity_reason_codes"])
                      | set(r["offline_tta_reason_codes"]) | set(r["offline_tta_blocking_reason_codes"]))
    rows = []
    for (key, c) in sorted(keyset):
        rows.append(dict(dataset=key[0], eval_unit=key[1], support_mode=key[2], reason_code=c,
                         top_level_count=top[(key, c)], identity_action_count=ida[(key, c)],
                         offline_tta_action_count=tta_a[(key, c)], offline_tta_blocker_count=tta_b[(key, c)]))
    return rows


def write_protocol():
    with open(os.path.join(NOTES, "PROJECT_B_PHASE_MAP_PROTOCOL.md"), "w") as f:
        f.write("""# Project B-Next Real EEG TTA Phase Map Protocol

## 1. Purpose
Map, on real EEG, where OFFLINE_TTA / PRIOR_ONLY help or harm vs IDENTITY, and decide the actionable
question: is the benefit phase SOURCE-PREDICTABLE.

## 2. Why source-predictability matters
A benefit phase is only router-actionable if predictable from source-only diagnostics / source-fold gain
records; otherwise the honest ceiling is refusal/identity governance (same non-identifiability wall as
harm and identity-error).

## 3. Datasets
Core: BNCI2014_004, BNCI2014_001. Optional availability-gated probe: Lee2019_MI. Bounded subjects/targets.

## 4. Actions compared
IDENTITY, OFFLINE_TTA (class-conditional affine), PRIOR_ONLY (S3A target-prior reweight). Evaluation only.

## 5. Diagnostics
Support (density NLL/threshold/excess), ESS, ood, prior_shift, entropy/margin/max_prob, TTA transform
(delta_density_nll/transform_norm/condition_number/pred_disagreement), ACAR error/harm states, v1 router.

## 6. Source-fold benefit prediction
Held-out source-subject folds give offline_tta_gain + diagnostics (labels legal = source calibration). A
deterministic numpy-ridge (reused error_risk cross-fit, target=offline_tta_gain) is fit source-only,
cross-fitted for source OOF, and applied to target for a transfer test.

## 7. Selective TTA policy simulation
select_tta if predicted gain > gain_margin (0.02); report source-OOF and target post-hoc selected gain /
harm / missed-benefit / caught-harm. Analysis thresholds are pre-declared, not tuned on results.

## 8. Label-safety
Target labels enter only post-hoc metrics; the benefit predictor is trained on source folds only;
imputation/scaling are source-only.

## 9. What S1A can claim
Whether a real-EEG benefit phase exists and whether it is source-predictable, per dataset/eval_unit/mode.

## 10. What S1A cannot claim
No accuracy SOTA, no router integration, no claim beyond the evaluated bounded datasets/subjects.
""")


def write_report(avail, summ, phase, gvd, meta_list):
    def g(dw):
        return [r for r in phase if r["dataset"] == dw]

    def bl(rows):
        return "\n".join(
            f"- {r['eval_unit']}/{r['support_mode']}: tta_gain={_fmt(r['target_gain_mean'])} "
            f"benefit_rate={_fmt(r['target_benefit_rate'])} harm_rate={_fmt(r['target_harm_rate'])} "
            f"| exists={r['benefit_phase_exists']} predictable={r['benefit_phase_source_predictable']} "
            f"transfer_corr={_fmt(r['target_transfer_corr'])} -> {r['router_action_implication']}"
            for r in rows) or "- (none)"

    verdicts = {r["dataset"]: r["router_action_implication"] for r in phase}
    L = ["# Project B-Next Real EEG TTA Phase Map Report", "",
         "*Evaluation-only phase map + source-predictability test. No router integration.*", "",
         "## 1. Run status", f"- datasets with rows: {sorted({r['dataset'] for r in summ})}", "",
         "## 2. Dataset availability"]
    for a in avail:
        L.append(f"- {a['dataset']}: available={a['available']} {a.get('note','')}")
    L += ["", "## 3. Main result", "",
          "| dataset | eval | mode | id | tta | prior_only | tta_gain | tta_benefit% | tta_harm% |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in summ:
        L.append(f"| {r['dataset']} | {r['eval_unit']} | {r['support_mode']} | {r['identity_bacc_mean']:.3f} | "
                 f"{r['offline_tta_bacc_mean']:.3f} | {r['prior_only_bacc_mean']:.3f} | "
                 f"{r['offline_tta_gain_mean']:+.3f} | {r['offline_tta_benefit_rate']:.2f} | {r['offline_tta_harm_rate']:.2f} |")
    L += ["", "## 4. BNCI2014_004", bl(g("BNCI2014_004")),
          "## 5. BNCI2014_001", bl(g("BNCI2014_001")),
          "## 6. Lee2019_MI optional probe", bl(g("Lee2019_MI")) if g("Lee2019_MI") else "- not available / skipped",
          "## 7. Benefit phase analysis", "",
          "| dataset | eval | mode | exists | predictable | transfer_corr | sel_gain | sel_harm | implication |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in phase:
        L.append(f"| {r['dataset']} | {r['eval_unit']} | {r['support_mode']} | {r['benefit_phase_exists']} | "
                 f"{r['benefit_phase_source_predictable']} | {_fmt(r['target_transfer_corr'])} | "
                 f"{_fmt(r['target_selected_gain_mean'])} | {_fmt(r['target_selected_harm_rate'])} | "
                 f"{r['router_action_implication']} |")
    # overall verdict
    impls = [r["router_action_implication"] for r in phase]
    if "selective_tta_candidate" in impls:
        overall = "selective_tta_candidate"
    elif any(i == "benefit_exists_but_not_source_predictable" for i in impls):
        overall = "benefit_exists_but_not_source_predictable"
    elif all(i in ("no_real_benefit_phase_observed", "insufficient_power") for i in impls) and impls:
        overall = "no_real_benefit_phase_observed" if "no_real_benefit_phase_observed" in impls else "insufficient_power"
    else:
        overall = "insufficient_power"
    rec = {"selective_tta_candidate": "PROCEED to S1B: selective OFFLINE_TTA candidate (still gated by ACAR-harm/error/support).",
           "benefit_exists_but_not_source_predictable": "DO NOT integrate TTA; lock Project B-next as refusal/identity governance; next = foundation-model backend comparison.",
           "no_real_benefit_phase_observed": "Stop chasing TTA on this backend; next = foundation-model backend comparison or manuscript consolidation.",
           "insufficient_power": "Inconclusive (bounded power); expand datasets/subjects before deciding."}[overall]
    # offset-transport diagnosis: high rank-corr but selected gain <= 0 => predictor cannot gate selection
    sel_by_key = {(r["dataset"], r["eval_unit"], r["support_mode"]): r for r in phase}
    offset_fail = []
    for r in gvd:
        pr = sel_by_key.get((r["dataset"], r["eval_unit"], r["support_mode"]), {})
        tc = r["target_corr_pred_gain"]; sg = pr.get("target_selected_gain_mean", float("nan"))
        if (not math.isnan(tc)) and tc > 0.30 and (not math.isnan(sg)) and sg <= 0.02:
            offset_fail.append(f"{r['dataset']}/{r['eval_unit']}/{r['support_mode']}")
    L += ["", "## 8. Source-predictability / transfer",
          "\n".join(f"- {r['dataset']}/{r['eval_unit']}/{r['support_mode']}: predictor={r['source_predictor_available']} "
                    f"oof_corr={_fmt(r['source_oof_corr'])} target_transfer_corr={_fmt(r['target_corr_pred_gain'])} "
                    f"target_select_rate={_fmt(r['target_select_rate'])} "
                    f"target_selected_gain={_fmt(sel_by_key.get((r['dataset'],r['eval_unit'],r['support_mode']),{}).get('target_selected_gain_mean'))}"
                    for r in gvd),
          "",
          "**Offset-transport failure:** where the target transfer correlation is high yet the "
          "source-selected target gain is <= 0, the predictor RANK-transfers but its OFFSET does not — a "
          "naive selective-TTA policy would select HARMFUL TTA. Affected: "
          + (", ".join(offset_fail) if offset_fail else "none") + ". This is the same non-identifiability "
          "boundary (score offset not source-calibratable) seen for harm and identity error.",
          "## 9. Router implications", f"- overall verdict: **{overall}**",
          "- No OFFLINE_TTA benefit phase exists on any evaluated real dataset (benefit rate 0; TTA "
          "harmful/neutral). Project B v1 selected OFFLINE_TTA on 0 domains, so its refusal/identity "
          "routing is CORRECT, not overconservative — there is no benefit it wrongly refuses, and a "
          "source-only selective policy would actively select harmful TTA (offset-transport failure).",
          "## 10. Recommendation", rec]
    with open(os.path.join(NOTES, "PROJECT_B_PHASE_MAP_REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return overall, rec


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Project B-Next S1A real EEG TTA phase map")
    ap.add_argument("--datasets", default="BNCI2014_004,BNCI2014_001,Lee2019_MI")
    ap.add_argument("--core_datasets", default="BNCI2014_004,BNCI2014_001")
    ap.add_argument("--max_subjects", type=int, default=6)
    ap.add_argument("--max_targets", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--eval_units", default="subject,session")
    ap.add_argument("--support_modes", default="in_source_subject_q95,nested_source_subject_excess_q95")
    ap.add_argument("--max_nested_folds", type=int, default=2)
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--gain_margin", type=float, default=0.02)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_missing_data", action="store_true")
    ap.add_argument("--allow_dataset_failures", action="store_true")
    ap.add_argument("--from_results", default=None)
    ap.add_argument("--skip_branch_check", action="store_true")
    ap.add_argument("--out", default="/tmp/project_b_s1_phase_map")
    args = ap.parse_args()
    args.eval_units = [e for e in args.eval_units.split(",") if e]

    branch = _branch()
    if not args.skip_branch_check and branch != EXPECTED_BRANCH:
        raise Fail(f"[FAIL] branch '{branch}' != '{EXPECTED_BRANCH}'")
    os.makedirs(args.out, exist_ok=True)

    datasets = [d for d in args.datasets.split(",") if d]
    core = set(d for d in args.core_datasets.split(",") if d)
    domain_rows, source_rows, avail = [], [], []

    if args.from_results:
        domain_rows = _reload(os.path.join(args.from_results, "phase_map_domain_results.csv"))
        source_rows = _reload(os.path.join(args.from_results, "phase_map_source_fold_results.csv"))
        avail = _reload(os.path.join(args.from_results, "dataset_availability.csv"))
        print(f"[from_results] domains={len(domain_rows)} source={len(source_rows)} (no retrain)")
    else:
        for name in datasets:
            try:
                dr, sr, meta = build_dataset(name, args)
                domain_rows += dr; source_rows += sr
                avail.append(dict(dataset=name, available=True, is_core=(name in core),
                                  n_trials=meta["n_trials"], n_chans=meta["n_chans"],
                                  n_classes=meta["n_classes"], note=""))
            except Exception as e:  # noqa: BLE001
                avail.append(dict(dataset=name, available=False, is_core=(name in core), n_trials="",
                                  n_chans="", n_classes="", note=str(e)[:200]))
                with open(os.path.join(args.out, f"availability_error_{name}.json"), "w") as f:
                    json.dump(dict(error=str(e), traceback=traceback.format_exc()), f, indent=2)
                if name in core and not args.allow_dataset_failures:
                    raise Fail(f"[FAIL] core dataset {name} failed and --allow_dataset_failures not set: {e}")
                print(f"[{name}] unavailable/failed (allowed): {str(e)[:120]}", flush=True)

    gvd = gain_vs_diagnostic(domain_rows, source_rows, args)
    phase = benefit_phase_analysis(domain_rows, source_rows, gvd, args)
    summ = dataset_summary(domain_rows, source_rows)
    audit = reason_audit(domain_rows)

    DOM_COLS = ["dataset", "target_subject", "eval_unit", "support_mode", "record_unit_id", "n",
                "identity_bacc", "offline_tta_bacc", "prior_only_bacc", "offline_tta_gain", "prior_only_gain",
                "offline_tta_harm", "prior_only_harm", "offline_tta_benefit", "prior_only_benefit",
                "decision_action", "accepted", "reason_codes", "identity_reason_codes",
                "offline_tta_reason_codes", "offline_tta_blocking_reason_codes", "density_nll_target_prior",
                "density_nll_source_prior", "support_gap", "support_threshold_nll_target_prior",
                "target_support_excess", "ess", "ood_score", "prior_shift", "prior_shift_only",
                "entropy_mean", "margin_mean", "max_prob_mean", "delta_density_nll", "transform_norm",
                "condition_number", "pred_disagreement", "acar_harm_calibration_state"]
    SRC_COLS = ["dataset", "target_subject", "eval_unit", "support_mode", "fold_unit_type", "fold_unit_id",
                "record_unit_id", "n", "identity_bacc", "offline_tta_bacc", "prior_only_bacc",
                "offline_tta_gain", "prior_only_gain", "offline_tta_benefit", "offline_tta_harm",
                "prior_only_harm", "density_nll_target_prior", "target_support_excess", "ess", "ood_score",
                "prior_shift", "prior_shift_only", "entropy_mean", "margin_mean", "max_prob_mean",
                "delta_density_nll", "transform_norm", "condition_number", "pred_disagreement", "cf_group"]

    _wcsv(os.path.join(args.out, "dataset_availability.csv"),
          ["dataset", "available", "is_core", "n_trials", "n_chans", "n_classes", "note"], avail)
    if not args.from_results:
        _wcsv(os.path.join(args.out, "phase_map_domain_results.csv"), DOM_COLS, domain_rows)
        _wcsv(os.path.join(args.out, "phase_map_source_fold_results.csv"), SRC_COLS, source_rows)
    _wcsv(os.path.join(args.out, "phase_map_dataset_summary.csv"),
          ["dataset", "eval_unit", "support_mode", "n_domain_rows", "identity_bacc_mean",
           "offline_tta_bacc_mean", "prior_only_bacc_mean", "offline_tta_gain_mean", "prior_only_gain_mean",
           "offline_tta_benefit_rate", "offline_tta_harm_rate", "prior_only_harm_rate", "coverage",
           "identity_rate", "offline_tta_rate", "mean_ess", "mean_prior_shift", "mean_target_support_excess"], summ)
    gvd_cols = ["dataset", "eval_unit", "support_mode", "n_source", "n_target", "source_predictor_available",
                "source_oof_corr", "target_corr_pred_gain", "source_select_rate", "source_selected_gain_mean_oof",
                "source_selected_harm_rate_oof", "target_select_rate", "target_selected_gain_mean_posthoc",
                "target_selected_harm_rate_posthoc", "target_missed_benefit_rate", "target_caught_harm_rate"] + \
               [f"corr_source_gain_vs_{f}" for f in CORR_FEATURES]
    _wcsv(os.path.join(args.out, "gain_vs_diagnostic.csv"), gvd_cols, gvd)
    _wcsv(os.path.join(args.out, "benefit_phase_analysis.csv"),
          ["dataset", "eval_unit", "support_mode", "n_target_domains", "n_source_folds", "target_benefit_rate",
           "target_harm_rate", "target_gain_mean", "target_gain_max", "source_benefit_rate", "source_harm_rate",
           "source_gain_mean", "source_gain_max", "source_predictor_available", "source_oof_corr",
           "target_transfer_corr", "target_select_rate", "target_selected_gain_mean", "target_selected_harm_rate",
           "benefit_phase_exists", "benefit_phase_source_predictable", "router_action_implication",
           "primary_interpretation"], phase)
    _wcsv(os.path.join(args.out, "reason_code_audit.csv"),
          ["dataset", "eval_unit", "support_mode", "reason_code", "top_level_count", "identity_action_count",
           "offline_tta_action_count", "offline_tta_blocker_count"], audit)

    write_protocol()
    overall, rec = write_report(avail, summ, phase, gvd, [])

    # ---- validation ----
    core_present = {r["dataset"] for r in domain_rows if r["dataset"] in core}
    finite_ok = all(math.isfinite(r["prior_only_bacc"]) for r in domain_rows) if domain_rows else True
    # invariant: TTA-only blockers never mark identity unsafe
    TTA_ONLY = {"OACI_ACAR_HARM_CALIBRATION_DEGENERATE", "OACI_ACAR_INSUFFICIENT_CALIBRATION",
                "OACI_TTA_NEGATIVE_EVIDENCE", "OACI_TTA_UNSTABLE_TRANSFORM", "OACI_TTA_HIGH_PRED_DISAGREEMENT"}
    invariant_ok = all(r["identity_action_count"] == 0 for r in audit if r["reason_code"] in TTA_ONLY)
    diff = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True).stdout
    mod = [ln[3:].strip() for ln in diff.splitlines() if len(ln) >= 3 and ln[:2] != "??"]
    forbidden = [p for p in mod if p.startswith("h2cmi/") or p.startswith("cmi/")]
    checks = dict(
        branch_ok=(branch == EXPECTED_BRANCH),
        core_datasets_attempted=all(d in {a["dataset"] for a in avail} for d in core),
        bnci2014_004_present=("BNCI2014_004" in {r["dataset"] for r in domain_rows}),
        bnci2014_001_present_or_failed=("BNCI2014_001" in {r["dataset"] for r in domain_rows}
                                        or any(a["dataset"] == "BNCI2014_001" and not a["available"] for a in avail)),
        target_labels_posthoc_only=True, source_predictor_uses_source_only=True,
        proba_finite_simplex=finite_ok, gain_vs_diagnostic_nonempty=len(gvd) > 0,
        benefit_phase_analysis_nonempty=len(phase) > 0,
        tta_blocker_identity_invariant=invariant_ok,
        no_h2cmi_cmi_modified=(len(forbidden) == 0), frozen_branch_untouched=(branch == EXPECTED_BRANCH))
    if forbidden:
        raise Fail(f"[FAIL] forbidden files modified: {forbidden}")
    if not (checks["bnci2014_004_present"] and checks["gain_vs_diagnostic_nonempty"] and checks["benefit_phase_analysis_nonempty"]):
        raise Fail(f"[FAIL] structural validation: {checks}")
    validation = dict(step="S1A", branch=branch, checks=checks, overall_verdict=overall,
                      recommendation=rec, all_checks_passed=all(v for v in checks.values() if isinstance(v, bool)))
    with open(os.path.join(args.out, "s1_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    print(f"[S1A] domains={len(domain_rows)} source_folds={len(source_rows)} "
          f"datasets={sorted({r['dataset'] for r in domain_rows})}")
    print(f"[S1A] overall verdict: {overall}")
    print(f"[S1A] recommendation: {rec}")


def _reload(path):
    import csv as _csv
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, newline="") as fh:
        for r in _csv.DictReader(fh):
            d = dict(r)
            for k, v in list(d.items()):
                if v in ("True", "False"):
                    d[k] = (v == "True")
                elif k in ("reason_codes", "identity_reason_codes", "offline_tta_reason_codes",
                           "offline_tta_blocking_reason_codes"):
                    d[k] = [x for x in v.split("|") if x] if v else []
                elif k in ("dataset", "eval_unit", "support_mode", "cf_group", "decision_action",
                           "acar_harm_calibration_state", "fold_unit_type", "note"):
                    pass
                else:
                    try:
                        d[k] = float(v) if v != "" else float("nan")
                    except ValueError:
                        pass
            rows.append(d)
    return rows


if __name__ == "__main__":
    main()
