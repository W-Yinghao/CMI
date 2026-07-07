"""Project B Step-3C: bounded real-EEG benchmark expansion.

Scales the Step-3A LOSO bridge to more targets, subject+session eval units, and both source-only
support modes, with an OACI reason-code audit. NOT a full MOABB benchmark. Target labels are used only
post-hoc (inside the harness). Modifies no h2cmi/** or cmi/**; reuses the Step-3A bridge helpers and the
Step-2 router harness. Primary pass/fail depends only on BNCI2014_004.
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
import time
import traceback
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

DEGENERATE = ("degenerate", "unavailable")

FOLD_COLS = ["dataset", "target_subject", "eval_unit", "support_mode", "n_source", "n_target",
             "n_classes", "n_chans", "n_times", "n_target_domains", "strict_bacc",
             "raw_offline_delta_bacc", "router_coverage", "router_refusal_rate", "router_identity_rate",
             "router_offline_tta_rate", "router_accepted_bacc", "router_missed_benefit",
             "router_avoided_harm", "source_acar_harm_state", "source_pseudo_gain_min",
             "source_pseudo_gain_mean", "source_pseudo_gain_max", "support_threshold_nll_target_prior",
             "base_source_q95_nll_target_prior", "nested_excess_q95", "reason_hist", "action_counts",
             "status", "error"]

AGG_COLS = ["dataset", "eval_unit", "support_mode", "n_targets", "n_domain_rows", "mean_strict_bacc",
            "mean_raw_offline_delta_bacc", "mean_router_coverage", "mean_router_identity_rate",
            "mean_router_offline_tta_rate", "mean_router_accepted_bacc", "mean_router_missed_benefit",
            "mean_router_avoided_harm", "n_refused_domains", "n_identity_domains",
            "n_offline_tta_domains", "n_support_mismatch_domains", "n_low_ess_domains",
            "n_acar_harm_degenerate_or_unavailable", "primary_interpretation"]

DOMAIN_COLS = ["dataset", "target_subject", "eval_unit", "support_mode", "domain_id", "n",
               "decision_action", "accepted", "reason_codes", "identity_bacc", "offline_tta_bacc",
               "raw_gain", "selected_bacc", "selected_gain_vs_identity", "offline_tta_admissible",
               "offline_tta_reason_codes", "offline_tta_blocking_reason_codes", "identity_admissible",
               "identity_reason_codes", "prior_shift_only", "cmi_residual_available",
               "acar_harm_calibration_state", "density_nll_target_prior",
               "support_threshold_nll_target_prior", "target_support_excess", "ess", "ood_score"]

DETAIL_COLS = ["dataset", "target_subject", "support_mode", "fold", "pseudo_subject_level",
               "n_train_subject_units", "n_pseudo_subject_units", "fold_train_q95_nll_target_prior",
               "pseudo_excess_min", "pseudo_excess_mean", "pseudo_excess_max",
               "base_source_q95_nll_target_prior", "nested_excess_q95", "support_threshold_nll_target_prior"]

AUDIT_COLS = ["dataset", "eval_unit", "support_mode", "reason_code", "top_level_count",
              "identity_action_count", "offline_tta_action_count", "offline_tta_blocker_count"]


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
        w = csv.writer(f); w.writerow(cols)
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


# ------------------------------------------------------------------ one target subject (all units/modes)
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
    meta_t = ds.meta.loc[tgt_idx].reset_index(drop=True)
    n_classes = len(ds.classes)

    dag, src_domains, _ = make_source_domain_labels(ds.meta.loc[src_idx].reset_index(drop=True))
    cfg = H2Config(n_classes=n_classes)
    cfg.encoder.n_chans = int(ds.X.shape[1]); cfg.encoder.n_times = int(ds.X.shape[2]); cfg.encoder.fs = float(ds.fs)
    cfg.train.epochs = args.epochs; cfg.train.batch_size = args.batch_size
    cfg.train.device = args.device; cfg.train.seed = args.seed

    base, *_ = train_h2(Xs, ys, src_domains, dag, cfg, align_factor="subject", verbose=False)
    pi_star = reference_prior(ys, n_classes, cfg.align.reference_prior)
    src_subj = source_pseudo_levels_from_domains(src_domains, level="subject")

    base_q95 = _q95(list(_per_subject_nll_target(base, Xs, src_subj, pi_star, cfg, args.device).values()))
    fold_rows, all_excess, nested_excess_q95, nested_threshold = [], [], 0.0, base_q95
    if "nested_source_subject_excess_q95" in args.support_modes:
        uniq = sorted(int(u) for u in np.unique(src_subj))
        for fi, u in enumerate(uniq[:args.max_nested_folds]):
            tr = src_subj != u; ps = src_subj == u
            nmodel, *_ = train_h2(Xs[tr], ys[tr], src_domains.subset(np.where(tr)[0]), dag, cfg,
                                  align_factor="subject", verbose=False)
            pi_n = reference_prior(ys[tr], n_classes, cfg.align.reference_prior)
            tr_nll = list(_per_subject_nll_target(nmodel, Xs[tr], src_subj[tr], pi_n, cfg, args.device).values())
            ps_nll = list(_per_subject_nll_target(nmodel, Xs[ps], src_subj[ps], pi_n, cfg, args.device).values())
            fq = _q95(tr_nll); exc = [float(x - fq) for x in ps_nll]; all_excess.extend(exc)
            fold_rows.append(dict(fold=fi, pseudo_subject_level=u, n_train_subject_units=len(tr_nll),
                                  n_pseudo_subject_units=len(ps_nll), fold_train_q95_nll_target_prior=fq,
                                  pseudo_excess_min=(min(exc) if exc else float("nan")),
                                  pseudo_excess_mean=(float(np.mean(exc)) if exc else float("nan")),
                                  pseudo_excess_max=(max(exc) if exc else float("nan"))))
        nested_excess_q95 = max(0.0, _q95(all_excess)) if all_excess else 0.0
        nested_threshold = base_q95 + nested_excess_q95

    thresholds = {"in_source_subject_q95": base_q95, "nested_source_subject_excess_q95": nested_threshold}
    fold_summ, domain_rows, detail_rows = [], [], []
    for eval_unit in args.eval_units:
        tgt_unit = target_domain_levels(meta_t, eval_unit=eval_unit)
        for mode in args.support_modes:
            thr = thresholds[mode]
            fcfg = make_support_calibrated_feature_config(
                max_density_nll_target_prior=thr, min_target_n=max(20, int(cfg.tta.min_target)))
            router = RefusalFirstRouter(RouterConfig(feature_config=fcfg))
            rep = evaluate_router_offline_tta(
                base, Xt, yt, tgt_unit, cfg, pi_star, router=router, X_src=Xs, y_src=ys,
                source_pseudo_levels=src_subj, device=args.device, calibrate_source_support=False,
                support_calibration_mode=mode)
            s = rep["router_summary"]; g = s["source_pseudo_gains"] or {}
            with open(os.path.join(out_dir, f"{ds.dataset}_target{target_subject}_{eval_unit}_{mode}_router_report.json"), "w") as f:
                json.dump(dict(dataset=ds.dataset, target_subject=target_subject, eval_unit=eval_unit,
                               support_mode=mode, threshold=thr, base_source_q95_nll_target_prior=base_q95,
                               nested_excess_q95=nested_excess_q95, report=rep), f, indent=2, default=float)
            fold_summ.append(dict(
                dataset=ds.dataset, target_subject=target_subject, eval_unit=eval_unit, support_mode=mode,
                n_source=int(len(src_idx)), n_target=int(len(tgt_idx)), n_classes=n_classes,
                n_chans=int(ds.X.shape[1]), n_times=int(ds.X.shape[2]), n_target_domains=len(rep["per_domain"]),
                strict_bacc=float(rep["identity"]["balanced_acc"]),
                raw_offline_delta_bacc=float(rep["delta_raw_offline_tta"]["d_balanced_acc"]),
                router_coverage=s["coverage"], router_refusal_rate=s["refusal_rate"],
                router_identity_rate=s["identity_rate"], router_offline_tta_rate=s["offline_tta_rate"],
                router_accepted_bacc=s["accepted_bacc"], router_missed_benefit=s["missed_benefit"],
                router_avoided_harm=s["avoided_harm"], source_acar_harm_state=s["source_acar_harm_calibration_state"],
                source_pseudo_gain_min=g.get("gain_min"), source_pseudo_gain_mean=g.get("gain_mean"),
                source_pseudo_gain_max=g.get("gain_max"), support_threshold_nll_target_prior=thr,
                base_source_q95_nll_target_prior=base_q95, nested_excess_q95=nested_excess_q95,
                reason_hist=s["reason_hist"], action_counts=s["action_counts"], status="ok", error=""))
            for did, dv in rep["per_domain"].items():
                asf = dv["action_scores"]; sup = dv["support"]
                domain_rows.append(dict(
                    dataset=ds.dataset, target_subject=target_subject, eval_unit=eval_unit, support_mode=mode,
                    domain_id=did, n=dv["n"], decision_action=dv["decision_action"], accepted=dv["accepted"],
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
            if mode == "nested_source_subject_excess_q95" and eval_unit == args.eval_units[0]:
                for fr in fold_rows:
                    detail_rows.append({**base_detail, **fr})
            elif mode == "in_source_subject_q95" and eval_unit == args.eval_units[0]:
                detail_rows.append({**base_detail, "fold": "", "pseudo_subject_level": ""})
    return fold_summ, domain_rows, detail_rows


# ------------------------------------------------------------------ aggregation
def aggregate(fold_rows, domain_rows):
    def _mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(xs)) if xs else float("nan")
    agg = []
    by = defaultdict(list)
    for r in fold_rows:
        if r.get("status") == "ok":
            by[(r["dataset"], r["eval_unit"], r["support_mode"])].append(r)
    dby = defaultdict(list)
    for d in domain_rows:
        dby[(d["dataset"], d["eval_unit"], d["support_mode"])].append(d)
    for key, rs in sorted(by.items()):
        ds_, eu, mode = key
        doms = dby.get(key, [])
        n_ref = sum(1 for d in doms if d["decision_action"] == "refuse")
        n_id = sum(1 for d in doms if d["decision_action"] == "identity")
        n_off = sum(1 for d in doms if d["decision_action"] == "offline_tta")
        n_sm = sum(1 for d in doms if "OACI_TOS_SUPPORT_MISMATCH" in d["identity_reason_codes"])
        n_le = sum(1 for d in doms if "OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE" in d["identity_reason_codes"])
        n_acar = sum(1 for r in rs if r["source_acar_harm_state"] in DEGENERATE)
        md = _mean([r["raw_offline_delta_bacc"] for r in rs])
        interp = ("raw TTA harmful; " if md < -0.01 else "raw TTA beneficial; " if md > 0.01 else "raw TTA ~neutral; ")
        interp += ("router mostly identity" if n_id >= max(1, len(doms)) * 0.5 else
                   "router mostly refuse" if n_ref >= max(1, len(doms)) * 0.5 else "router mixed")
        interp += ("; TTA blocked (ACAR degenerate/unavailable)" if n_off == 0 else "; some ACAR-available TTA")
        agg.append(dict(dataset=ds_, eval_unit=eu, support_mode=mode, n_targets=len({r["target_subject"] for r in rs}),
                        n_domain_rows=len(doms), mean_strict_bacc=_mean([r["strict_bacc"] for r in rs]),
                        mean_raw_offline_delta_bacc=md,
                        mean_router_coverage=_mean([r["router_coverage"] for r in rs]),
                        mean_router_identity_rate=_mean([r["router_identity_rate"] for r in rs]),
                        mean_router_offline_tta_rate=_mean([r["router_offline_tta_rate"] for r in rs]),
                        mean_router_accepted_bacc=_mean([r["router_accepted_bacc"] for r in rs]),
                        mean_router_missed_benefit=_mean([r["router_missed_benefit"] for r in rs]),
                        mean_router_avoided_harm=_mean([r["router_avoided_harm"] for r in rs]),
                        n_refused_domains=n_ref, n_identity_domains=n_id, n_offline_tta_domains=n_off,
                        n_support_mismatch_domains=n_sm, n_low_ess_domains=n_le,
                        n_acar_harm_degenerate_or_unavailable=n_acar, primary_interpretation=interp))
    return agg


def reason_audit(domain_rows):
    agg = defaultdict(lambda: dict(top=0, idc=0, offc=0, offb=0))
    for d in domain_rows:
        k0 = (d["dataset"], d["eval_unit"], d["support_mode"])
        for rc in d["reason_codes"]:
            agg[(*k0, rc)]["top"] += 1
        for rc in d["identity_reason_codes"]:
            agg[(*k0, rc)]["idc"] += 1
        for rc in d["offline_tta_reason_codes"]:
            agg[(*k0, rc)]["offc"] += 1
        for rc in d["offline_tta_blocking_reason_codes"]:
            agg[(*k0, rc)]["offb"] += 1
    rows = [dict(dataset=k[0], eval_unit=k[1], support_mode=k[2], reason_code=k[3],
                 top_level_count=v["top"], identity_action_count=v["idc"],
                 offline_tta_action_count=v["offc"], offline_tta_blocker_count=v["offb"])
            for k, v in agg.items()]
    rows.sort(key=lambda r: (r["dataset"], r["eval_unit"], r["support_mode"], -r["top_level_count"], r["reason_code"]))
    return rows


# ------------------------------------------------------------------ report + main
def write_report(out_dir, avail_rows, fold_rows, agg_rows, partial):
    lines = ["# Project B Step-3C Real EEG Benchmark Expansion Report", "",
             "*Bounded real benchmark expansion, NOT a full MOABB benchmark. Auto-generated.*", "",
             "## 1. Run status", f"- runtime_bounded_partial = {partial}",
             "- datasets: " + ", ".join(f"{r['dataset']}({'ok' if r.get('available') else 'unavailable'})"
                                        for r in avail_rows), "",
             "## 2. Aggregate summary", "",
             "| dataset | eval_unit | mode | n_tgt | mean_strict | mean_rawdTTA | cov | id_rate | off_tta | acc_bAcc | avoided | interpretation |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg_rows:
        lines.append(f"| {r['dataset']} | {r['eval_unit']} | {r['support_mode']} | {r['n_targets']} | "
                     f"{r['mean_strict_bacc']:.3f} | {r['mean_raw_offline_delta_bacc']:+.3f} | "
                     f"{r['mean_router_coverage']:.2f} | {r['mean_router_identity_rate']:.2f} | "
                     f"{r['mean_router_offline_tta_rate']:.2f} | {r['mean_router_accepted_bacc']:.3f} | "
                     f"{r['mean_router_avoided_harm']:.3f} | {r['primary_interpretation']} |")
    lines += ["", "## 3. Subject-level routing", "See rows with eval_unit=subject above.",
              "## 4. Session-level routing", "See rows with eval_unit=session above.",
              "## 5. TTA harm / benefit", "raw_offline_delta_bacc per fold in fold_summary.csv.",
              "## 6. Router action distribution", "action_counts per fold in fold_summary.csv; per-domain in per_domain_decisions.csv.",
              "## 7. Reason-code audit", "reason_code_audit.csv (top-level vs identity-action vs TTA-blocker).",
              "## 8. Comparison to Step-3A", "Step-3A: 2 targets, subject only. Step-3C: more targets + session-level; "
              "OFFLINE_TTA remains blocked under degenerate/unavailable ACAR-harm.",
              "## 9. Claim boundary update", "See step3c_claim_boundary_update.json. This remains a bounded "
              "real benchmark expansion, not a full benchmark; no target-label-tuned thresholds."]
    with open(os.path.join(out_dir, "step3c_real_benchmark_report.md"), "w") as f:
        f.write(os.linesep.join(lines) + os.linesep)
    repo_notes = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes")
    with open(os.path.join(repo_notes, "PROJECT_B_STEP3C_REAL_BENCHMARK_REPORT.md"), "w") as f:
        f.write(os.linesep.join(lines) + os.linesep)


def main():
    ap = argparse.ArgumentParser(description="Project B Step-3C bounded real-EEG benchmark")
    ap.add_argument("--datasets", default="BNCI2014_004")
    ap.add_argument("--max_subjects", type=int, default=6)
    ap.add_argument("--max_targets", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--eval_units", default="subject,session")
    ap.add_argument("--support_modes", default="in_source_subject_q95,nested_source_subject_excess_q95")
    ap.add_argument("--max_nested_folds", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/project_b_step3c_real_benchmark")
    ap.add_argument("--allow_missing_data", action="store_true")
    ap.add_argument("--allow_dataset_failures", action="store_true")
    ap.add_argument("--max_runtime_seconds", type=float, default=2400.0)
    args = ap.parse_args()
    args.eval_units = [x for x in args.eval_units.split(",") if x]
    args.support_modes = [x for x in args.support_modes.split(",") if x]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    deadline = t0 + args.max_runtime_seconds

    from h2cmi.data.real_eeg_bridge import load_moabb_real_eeg, loso_subjects
    datasets = [d for d in args.datasets.split(",") if d]
    primary = datasets[0]

    avail_rows, fold_rows, domain_rows, detail_rows = [], [], [], []
    partial = False
    for di, dataset in enumerate(datasets):
        is_primary = (dataset == primary)
        try:
            ds = load_moabb_real_eeg(dataset, max_subjects=args.max_subjects, tmin=0.5, tmax=3.5,
                                     resample=args.resample)
        except Exception as e:  # noqa: BLE001
            avail_rows.append(dict(dataset=dataset, available=False, error=str(e),
                                   is_primary=is_primary))
            with open(os.path.join(args.out, f"availability_error_{dataset}.json"), "w") as f:
                json.dump(dict(dataset=dataset, error=str(e), traceback=traceback.format_exc()), f, indent=2)
            if is_primary and not args.allow_missing_data:
                sys.exit(f"[FAIL] primary dataset {dataset} unavailable and --allow_missing_data not set")
            if not is_primary and not args.allow_dataset_failures:
                raise
            print(f"[bench] dataset {dataset} unavailable: {e}")
            continue
        avail_rows.append(dict(dataset=dataset, available=True, n_trials=int(ds.X.shape[0]),
                               n_chans=int(ds.X.shape[1]), n_times=int(ds.X.shape[2]),
                               subjects=loso_subjects(ds.meta), is_primary=is_primary, error=""))
        targets = loso_subjects(ds.meta)[:args.max_targets]
        print(f"[bench] {dataset}: X={ds.X.shape} classes={ds.classes} targets={targets}")
        for t in targets:
            if time.time() > deadline:
                partial = True
                print(f"[bench] runtime budget reached; stopping after completed targets")
                break
            try:
                fs, dr, det = run_one_target(ds, t, args, args.out)
                fold_rows += fs; domain_rows += dr; detail_rows += det
                for r in fs:
                    print(f"[bench] {dataset}/t{t}/{r['eval_unit']}/{r['support_mode']}: "
                          f"strict={r['strict_bacc']:.3f} cov={r['router_coverage']:.2f} "
                          f"acts={r['action_counts']} acar={r['source_acar_harm_state']}")
            except Exception as e:  # noqa: BLE001
                fold_rows.append(dict(dataset=dataset, target_subject=t, status="error", error=str(e)))
                print(f"[bench] {dataset}/t{t} FAILED: {e}")
        if partial:
            break

    # --- validation: no OFFLINE_TTA under degenerate/unavailable ACAR-harm; no NaN/inf vectors ---
    for r in fold_rows:
        if r.get("status") == "ok" and r["source_acar_harm_state"] in DEGENERATE \
                and r["action_counts"].get("offline_tta", 0) > 0:
            sys.exit(f"[FAIL] OFFLINE_TTA selected under degenerate ACAR-harm: {r['dataset']}/{r['target_subject']}")
    for d in domain_rows:
        for k in ("density_nll_target_prior", "ess", "ood_score", "target_support_excess"):
            v = d[k]
            if isinstance(v, float) and not math.isfinite(v):
                sys.exit(f"[FAIL] non-finite diagnostic {k} in {d['dataset']}/{d['target_subject']}")
    prim_ok = [r for r in fold_rows if r.get("status") == "ok" and r["dataset"] == primary]
    prim_targets = {r["target_subject"] for r in prim_ok}
    prim_units = {r["eval_unit"] for r in prim_ok}
    prim_modes = {r["support_mode"] for r in prim_ok}

    agg_rows = aggregate(fold_rows, domain_rows)
    audit_rows = reason_audit(domain_rows)
    _write_csv(os.path.join(args.out, "dataset_availability.csv"),
               ["dataset", "available", "is_primary", "n_trials", "n_chans", "n_times", "subjects", "error"], avail_rows)
    _write_csv(os.path.join(args.out, "fold_summary.csv"), FOLD_COLS, fold_rows)
    _write_csv(os.path.join(args.out, "aggregate_summary.csv"), AGG_COLS, agg_rows)
    _write_csv(os.path.join(args.out, "per_domain_decisions.csv"), DOMAIN_COLS, domain_rows)
    _write_csv(os.path.join(args.out, "support_calibration_details.csv"), DETAIL_COLS, detail_rows)
    _write_csv(os.path.join(args.out, "reason_code_audit.csv"), AUDIT_COLS, audit_rows)
    write_report(args.out, avail_rows, fold_rows, agg_rows, partial)

    cbu = dict(step="3C", primary_dataset=primary, runtime_bounded_partial=partial,
               n_primary_targets=len(prim_targets), eval_units=sorted(prim_units), support_modes=sorted(prim_modes),
               claimable_addition="Bounded real benchmark expansion on BNCI2014_004: subject- and "
                                  "session-level LOSO routing, source-only support calibration, OFFLINE_TTA "
                                  "never selected under degenerate/unavailable ACAR-harm.",
               not_claimable_addition="Not a full MOABB benchmark; bounded subjects/targets/epochs.")
    with open(os.path.join(args.out, "step3c_claim_boundary_update.json"), "w") as f:
        json.dump(cbu, f, indent=2)

    print(f"[bench] wrote 6 CSVs + report + claim update; primary targets={len(prim_targets)} "
          f"units={sorted(prim_units)} modes={sorted(prim_modes)} partial={partial}")
    if len(prim_targets) < 3:
        sys.exit(f"[FAIL] primary completed only {len(prim_targets)} targets (<3)")


if __name__ == "__main__":
    main()
