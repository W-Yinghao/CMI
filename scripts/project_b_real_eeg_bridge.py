"""Project B Step-3A: real-EEG LOSO router bridge smoke.

Loads a MOABB motor-imagery dataset (via the unmodified cmi/ loader), builds a subject->session
H2-CMI DomainDAG, trains H2-CMI on source subjects, source-only-calibrates a support threshold
(baseline + bounded nested source-subject excess), and runs the Step-2 router harness on each
held-out target subject. Target labels are used only post-hoc (inside the harness). Modifies no
h2cmi core / router / cmi loader.

If the dataset cannot be loaded (no cache / no network) and --allow_missing_data is set, writes
availability_error.json and exits 0 (this is a bridge smoke, not a benchmark).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import math
import sys
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

FOLD_COLS = ["dataset", "target_subject", "support_mode", "n_source", "n_target", "n_classes",
             "n_chans", "n_times", "n_target_domains", "strict_bacc", "raw_offline_delta_bacc",
             "router_coverage", "router_refusal_rate", "router_identity_rate", "router_offline_tta_rate",
             "router_accepted_bacc", "router_missed_benefit", "router_avoided_harm",
             "source_acar_harm_state", "source_pseudo_gain_min", "source_pseudo_gain_mean",
             "source_pseudo_gain_max", "support_threshold_nll_target_prior",
             "base_source_q95_nll_target_prior", "nested_excess_q95", "reason_hist",
             "action_counts", "status", "error"]

DOMAIN_COLS = ["dataset", "target_subject", "support_mode", "domain_id", "n", "decision_action",
               "accepted", "reason_codes", "identity_bacc", "offline_tta_bacc", "raw_gain",
               "selected_bacc", "selected_gain_vs_identity", "offline_tta_admissible",
               "offline_tta_reason_codes", "offline_tta_blocking_reason_codes", "identity_admissible",
               "identity_reason_codes", "prior_shift_only", "cmi_residual_available",
               "acar_harm_calibration_state", "density_nll_target_prior",
               "support_threshold_nll_target_prior", "target_support_excess", "ess", "ood_score"]

DETAIL_COLS = ["dataset", "target_subject", "support_mode", "fold", "pseudo_subject_level",
               "n_train_subject_units", "n_pseudo_subject_units", "fold_train_q95_nll_target_prior",
               "pseudo_excess_min", "pseudo_excess_mean", "pseudo_excess_max",
               "base_source_q95_nll_target_prior", "nested_excess_q95", "support_threshold_nll_target_prior"]


def _q95(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.quantile(a, 0.95)) if a.size else float("nan")


def _fmt(v):
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


def _per_subject_nll_target(model, X, subj_levels, source_prior, cfg, device):
    from h2cmi.eval.router_harness import prior_decoupled_density_diagnostics
    from h2cmi.eval.harness import _embed
    out = {}
    for u in np.unique(subj_levels):
        m = subj_levels == u
        if int(m.sum()) < cfg.tta.min_target:
            continue
        U = _embed(model, X[m], device)
        out[int(u)] = prior_decoupled_density_diagnostics(model.head.density, U, source_prior)["density_nll_target_prior"]
    return out


# ------------------------------------------------------------------ one held-out target subject
def run_one_target(ds, target_subject, args, out_dir):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    from h2cmi.config import H2Config
    from h2cmi.train.trainer import train_h2, reference_prior
    from h2cmi.eval.router_harness import evaluate_router_offline_tta, make_support_calibrated_feature_config
    from h2cmi.router.router import RefusalFirstRouter, RouterConfig
    from h2cmi.data.real_eeg_bridge import (
        split_loso_by_subject, make_source_domain_labels, target_domain_levels,
        source_pseudo_levels_from_domains)

    src_idx, tgt_idx = split_loso_by_subject(ds.meta, target_subject)
    Xs, ys = ds.X[src_idx], ds.y[src_idx]
    Xt, yt = ds.X[tgt_idx], ds.y[tgt_idx]
    meta_s = ds.meta.loc[src_idx].reset_index(drop=True)
    meta_t = ds.meta.loc[tgt_idx].reset_index(drop=True)
    n_classes = len(ds.classes)

    dag, src_domains, sinfo = make_source_domain_labels(meta_s)
    cfg = H2Config(n_classes=n_classes)
    cfg.encoder.n_chans = int(ds.X.shape[1])
    cfg.encoder.n_times = int(ds.X.shape[2])
    cfg.encoder.fs = float(ds.fs)
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.device = args.device
    cfg.train.seed = args.seed

    model, *_ = train_h2(Xs, ys, src_domains, dag, cfg, align_factor="subject", verbose=False)
    pi_star = reference_prior(ys, n_classes, cfg.align.reference_prior)

    src_subj = source_pseudo_levels_from_domains(src_domains, level="subject")
    tgt_unit = target_domain_levels(meta_t, eval_unit=args.eval_unit)

    # --- support thresholds (source-only) ---
    base_q95 = _q95(list(_per_subject_nll_target(model, Xs, src_subj, pi_star, cfg, args.device).values()))
    fold_rows, all_excess = [], []
    nested_excess_q95, nested_threshold = 0.0, base_q95
    if args.support_mode in ("nested_source_subject_excess_q95", "both"):
        uniq = sorted(int(u) for u in np.unique(src_subj))
        for fi, u in enumerate(uniq[:args.max_nested_folds]):
            tr = src_subj != u
            ps = src_subj == u
            nmodel, *_ = train_h2(Xs[tr], ys[tr], src_domains.subset(np.where(tr)[0]), dag, cfg,
                                  align_factor="subject", verbose=False)
            pi_n = reference_prior(ys[tr], n_classes, cfg.align.reference_prior)
            tr_nll = list(_per_subject_nll_target(nmodel, Xs[tr], src_subj[tr], pi_n, cfg, args.device).values())
            ps_nll = list(_per_subject_nll_target(nmodel, Xs[ps], src_subj[ps], pi_n, cfg, args.device).values())
            fq = _q95(tr_nll)
            exc = [float(x - fq) for x in ps_nll]
            all_excess.extend(exc)
            fold_rows.append(dict(fold=fi, pseudo_subject_level=u, n_train_subject_units=len(tr_nll),
                                  n_pseudo_subject_units=len(ps_nll), fold_train_q95_nll_target_prior=fq,
                                  pseudo_excess_min=(min(exc) if exc else float("nan")),
                                  pseudo_excess_mean=(float(np.mean(exc)) if exc else float("nan")),
                                  pseudo_excess_max=(max(exc) if exc else float("nan"))))
        nested_excess_q95 = max(0.0, _q95(all_excess)) if all_excess else 0.0
        nested_threshold = base_q95 + nested_excess_q95

    thresholds = {"in_source_subject_q95": base_q95, "nested_source_subject_excess_q95": nested_threshold}
    modes = (["in_source_subject_q95", "nested_source_subject_excess_q95"]
             if args.support_mode == "both" else [args.support_mode])

    fold_summ, domain_rows, detail_rows = [], [], []
    for mode in modes:
        thr = thresholds[mode]
        fcfg = make_support_calibrated_feature_config(
            max_density_nll_target_prior=thr, min_target_n=max(20, int(cfg.tta.min_target)))
        router = RefusalFirstRouter(RouterConfig(feature_config=fcfg))
        rep = evaluate_router_offline_tta(
            model, Xt, yt, tgt_unit, cfg, pi_star, router=router, X_src=Xs, y_src=ys,
            source_pseudo_levels=src_subj, device=args.device,
            calibrate_source_support=False, support_calibration_mode=mode)
        s = rep["router_summary"]
        gains = s["source_pseudo_gains"] or {}
        out = dict(dataset=ds.dataset, target_subject=target_subject, support_mode=mode,
                   threshold=thr, base_source_q95_nll_target_prior=base_q95,
                   nested_excess_q95=nested_excess_q95, report=rep)
        with open(os.path.join(out_dir, f"{ds.dataset}_target{target_subject}_{mode}_router_report.json"), "w") as f:
            json.dump(out, f, indent=2, default=float)

        fold_summ.append(dict(
            dataset=ds.dataset, target_subject=target_subject, support_mode=mode,
            n_source=int(len(src_idx)), n_target=int(len(tgt_idx)), n_classes=n_classes,
            n_chans=int(ds.X.shape[1]), n_times=int(ds.X.shape[2]), n_target_domains=len(rep["per_domain"]),
            strict_bacc=float(rep["identity"]["balanced_acc"]),
            raw_offline_delta_bacc=float(rep["delta_raw_offline_tta"]["d_balanced_acc"]),
            router_coverage=s["coverage"], router_refusal_rate=s["refusal_rate"],
            router_identity_rate=s["identity_rate"], router_offline_tta_rate=s["offline_tta_rate"],
            router_accepted_bacc=s["accepted_bacc"], router_missed_benefit=s["missed_benefit"],
            router_avoided_harm=s["avoided_harm"], source_acar_harm_state=s["source_acar_harm_calibration_state"],
            source_pseudo_gain_min=gains.get("gain_min"), source_pseudo_gain_mean=gains.get("gain_mean"),
            source_pseudo_gain_max=gains.get("gain_max"), support_threshold_nll_target_prior=thr,
            base_source_q95_nll_target_prior=base_q95, nested_excess_q95=nested_excess_q95,
            reason_hist=s["reason_hist"], action_counts=s["action_counts"], status="ok", error=""))
        for did, dv in rep["per_domain"].items():
            asf = dv["action_scores"]; sup = dv["support"]
            domain_rows.append(dict(
                dataset=ds.dataset, target_subject=target_subject, support_mode=mode, domain_id=did,
                n=dv["n"], decision_action=dv["decision_action"], accepted=dv["accepted"],
                reason_codes=dv["reason_codes"], identity_bacc=dv["identity_bacc"],
                offline_tta_bacc=dv["offline_tta_bacc"], raw_gain=dv["raw_gain"],
                selected_bacc=dv["selected_bacc"], selected_gain_vs_identity=dv["selected_gain_vs_identity"],
                offline_tta_admissible=asf["offline_tta"]["admissible"],
                offline_tta_reason_codes=asf["offline_tta"]["reason_codes"],
                offline_tta_blocking_reason_codes=asf["offline_tta"]["blocking_reason_codes"],
                identity_admissible=asf["identity"]["admissible"],
                identity_reason_codes=asf["identity"]["reason_codes"],
                prior_shift_only=asf["identity"]["prior_shift_only"],
                cmi_residual_available=asf["identity"]["cmi_residual_available"],
                acar_harm_calibration_state=asf["offline_tta"]["acar_harm_calibration_state"],
                density_nll_target_prior=sup["density_nll_target_prior"],
                support_threshold_nll_target_prior=thr,
                target_support_excess=sup["density_nll_target_prior"] - thr,
                ess=sup["ess"], ood_score=sup["ood_score"]))
        base_detail = dict(dataset=ds.dataset, target_subject=target_subject, support_mode=mode,
                           base_source_q95_nll_target_prior=base_q95, nested_excess_q95=nested_excess_q95,
                           support_threshold_nll_target_prior=thr)
        if mode == "nested_source_subject_excess_q95":
            for fr in fold_rows:
                detail_rows.append({**base_detail, **fr})
        else:
            detail_rows.append({**base_detail, "fold": "", "pseudo_subject_level": ""})
    return fold_summ, domain_rows, detail_rows


# ------------------------------------------------------------------ CLI / main
def main():
    ap = argparse.ArgumentParser(description="Project B Step-3A real-EEG LOSO router bridge")
    ap.add_argument("--dataset", default="BNCI2014_004")
    ap.add_argument("--max_subjects", type=int, default=4)
    ap.add_argument("--max_targets", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--eval_unit", default="subject", choices=["subject", "session", "run"])
    ap.add_argument("--support_mode", default="both",
                    choices=["in_source_subject_q95", "nested_source_subject_excess_q95", "both"])
    ap.add_argument("--max_nested_folds", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/project_b_step3a_real_bridge")
    ap.add_argument("--allow_missing_data", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- load real data (the only step that may be unavailable) ---
    try:
        from h2cmi.data.real_eeg_bridge import load_moabb_real_eeg, loso_subjects
        ds = load_moabb_real_eeg(args.dataset, max_subjects=args.max_subjects, tmin=args.tmin,
                                 tmax=args.tmax, resample=args.resample)
    except Exception as e:  # noqa: BLE001 - bridge must degrade cleanly on missing data
        err = dict(dataset=args.dataset, status="data_unavailable", error=str(e),
                   traceback=traceback.format_exc(),
                   note="MOABB data not cached / not downloadable; bridge smoke degraded cleanly.")
        with open(os.path.join(args.out, "availability_error.json"), "w") as f:
            json.dump(err, f, indent=2)
        print(f"[bridge] data unavailable for {args.dataset}: {e}")
        if args.allow_missing_data:
            print("[bridge] --allow_missing_data set; wrote availability_error.json; exit 0")
            return
        raise

    targets = loso_subjects(ds.meta)[:args.max_targets]
    print(f"[bridge] loaded {ds.dataset}: X={ds.X.shape} classes={ds.classes} "
          f"subjects={loso_subjects(ds.meta)} -> targets={targets}")

    fold_summ, domain_rows, detail_rows = [], [], []
    for t in targets:
        try:
            fs, dr, det = run_one_target(ds, t, args, args.out)
            fold_summ += fs; domain_rows += dr; detail_rows += det
            for r in fs:
                print(f"[bridge] target{t}/{r['support_mode']}: strict={r['strict_bacc']:.3f} "
                      f"cov={r['router_coverage']:.2f} actions={r['action_counts']} "
                      f"acar={r['source_acar_harm_state']} thr={r['support_threshold_nll_target_prior']:.2f}")
        except Exception as e:  # noqa: BLE001 - one target failing must not kill the run
            fold_summ.append(dict(dataset=ds.dataset, target_subject=t, support_mode=args.support_mode,
                                  status="error", error=str(e)))
            print(f"[bridge] target{t} FAILED: {e}")

    _write_csv(os.path.join(args.out, "fold_summary.csv"), FOLD_COLS, fold_summ)
    _write_csv(os.path.join(args.out, "per_domain_decisions.csv"), DOMAIN_COLS, domain_rows)
    _write_csv(os.path.join(args.out, "support_calibration_details.csv"), DETAIL_COLS, detail_rows)
    summary = dict(dataset=ds.dataset, X_shape=list(ds.X.shape), classes=ds.classes, fs=ds.fs,
                   subjects=loso_subjects(ds.meta), targets=targets, eval_unit=args.eval_unit,
                   support_mode=args.support_mode, max_nested_folds=args.max_nested_folds,
                   n_fold_rows=len(fold_summ), n_domain_rows=len(domain_rows),
                   status="ok" if fold_summ and all(r.get("status") == "ok" for r in fold_summ) else "partial",
                   folds=fold_summ)
    with open(os.path.join(args.out, "real_bridge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[bridge] wrote fold_summary/per_domain/support_details + real_bridge_summary.json ({summary['status']})")


if __name__ == "__main__":
    main()
