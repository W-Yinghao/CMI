"""Project B Step-2F: nested source-site support calibration on the locked worlds.

Compares two source-only, label-safe support-threshold modes per (world, seed):
  in_source_subject_q95 : Step-2E baseline = q95(base-model in-source subject density_nll_target_prior)
  nested_site_excess_q95: base_source_q95 + q95 over source-site folds of (held-out-site subject NLL
                          - that fold's q95(in-training subject NLL)); i.e. a SCALE-NORMALISED excess
                          measured on each nested model's own scale, then added to the base scale.

The nested folds only estimate "how much to widen the held-out support boundary"; the widening is added
to the base model's own NLL scale, so raw NLLs across differently-trained encoders are never compared.
Target labels are used only for post-hoc bAcc (via the router harness). Trains the base model ONCE per
(world, seed) and reuses it for both modes; nested models are used only for the threshold.

Does NOT modify any h2cmi router/core file.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import glob
import json
import math
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

COMMON = dict(classes=3, chans=16, times=128, fs=128.0, sites=6, subjects=4, sessions=2,
              trials=60, noise=0.25, label_rho=0.0, epochs=30, bs=64, device="cpu",
              eval_unit="subject")
WORLDS = {
    "R2":    dict(cov=1.2, prior=0.4, montage=0.2, concept=0.0, concept_frac=0.0, seeds=[0, 1, 2]),
    "HF3":   dict(cov=0.8, prior=0.4, montage=0.2, concept=1.2, concept_frac=0.50, seeds=[3, 4, 7, 8, 10]),
    "H_OOD": dict(cov=0.8, prior=0.4, montage=0.2, concept=1.0, concept_frac=0.17, seeds=[32]),
}
MODES = ["in_source_subject_q95", "nested_site_excess_q95"]
SUPPORT_QUANTILE = 0.95

WORLD_COLS = [
    "world", "seed", "support_mode", "target_site", "target_concept_hit", "source_concept_count",
    "strict_bacc", "raw_offline_delta_bacc",
    "base_source_q95_nll_target_prior", "nested_excess_q95", "support_threshold_nll_target_prior",
    "mean_target_density_nll_target_prior", "mean_target_support_excess",
    "n_support_mismatch_domains", "n_low_ess_domains",
    "router_coverage", "router_refusal_rate", "router_identity_rate", "router_offline_tta_rate",
    "router_accepted_bacc", "router_selected_mean_gain_vs_identity",
    "router_missed_benefit", "router_avoided_harm",
    "source_acar_harm_state", "reason_hist", "tta_block_reason_hist", "action_counts",
]
DOMAIN_COLS = [
    "world", "seed", "support_mode", "domain_id", "n", "decision_action", "accepted", "reason_codes",
    "identity_bacc", "offline_tta_bacc", "raw_gain", "selected_bacc", "selected_gain_vs_identity",
    "identity_admissible", "offline_tta_admissible", "offline_tta_blocking_reason_codes",
    "prior_shift_only", "cmi_residual_available", "acar_harm_calibration_state",
    "density_nll_source_prior", "density_nll_target_prior", "support_gap", "ess", "ood_score",
    "support_threshold_nll_target_prior", "target_support_excess", "support_mismatch", "low_ess",
]
DETAIL_COLS = [
    "world", "seed", "support_mode", "target_site", "pseudo_site", "fold",
    "n_train_subject_units", "n_pseudo_subject_units", "fold_train_q95_nll_target_prior",
    "pseudo_nll_min", "pseudo_nll_mean", "pseudo_nll_max",
    "pseudo_excess_min", "pseudo_excess_mean", "pseudo_excess_max",
    "base_source_q95_nll_target_prior", "nested_excess_q95", "support_threshold_nll_target_prior",
]


def _report_path(out_dir, world, seed, mode):
    return os.path.join(out_dir, f"{world}_seed{seed}_{mode}_router_report.json")


def _q95(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.quantile(a, SUPPORT_QUANTILE)) if a.size else float("nan")


def _per_subject_nll_target(model, X, subj_levels, source_prior, cfg, device):
    """density_nll_target_prior per subject-domain (>= min_target samples). No labels."""
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


# ------------------------------------------------------------------ one (world, seed): both modes
def run_one(world: str, seed: int, out_dir: str) -> None:
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    from h2cmi.config import H2Config
    from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
    from h2cmi.train.trainer import train_h2, reference_prior
    from h2cmi.eval.router_harness import (
        evaluate_router_offline_tta, make_support_calibrated_feature_config)
    from h2cmi.router.router import RefusalFirstRouter, RouterConfig

    def _cfg():
        c = H2Config(n_classes=COMMON["classes"])
        c.encoder.n_chans = COMMON["chans"]; c.encoder.n_times = COMMON["times"]; c.encoder.fs = COMMON["fs"]
        c.train.epochs = COMMON["epochs"]; c.train.device = COMMON["device"]
        c.train.seed = seed; c.train.batch_size = COMMON["bs"]
        return c

    w = WORLDS[world]
    shift = ShiftSpec(cov=w["cov"], prior=w["prior"], concept=w["concept"],
                      concept_site_frac=w["concept_frac"], montage=w["montage"],
                      noise=COMMON["noise"], label_mechanism_rho=COMMON["label_rho"])
    sim = EEGSimulator(COMMON["classes"], COMMON["chans"], COMMON["times"], COMMON["fs"],
                       shift=shift, seed=seed).sample(
        COMMON["sites"], COMMON["subjects"], COMMON["sessions"], COMMON["trials"])
    src_idx, tgt_idx = train_target_split(sim, n_target_sites=1, seed=seed)
    target_site = int(np.unique(sim.site[tgt_idx])[0])
    source_sites = sorted(int(s) for s in np.unique(sim.site[src_idx]))
    cfg = _cfg()

    # --- base model on all source sites ---
    Xs, ys = sim.X[src_idx], sim.y[src_idx]
    src_dom = sim.domains.subset(src_idx)
    base, *_ = train_h2(Xs, ys, src_dom, sim.dag, cfg, align_factor="site", verbose=False)
    pi_star = reference_prior(ys, COMMON["classes"], cfg.align.reference_prior)
    src_subj = src_dom.factor("subject")
    base_source_q95 = _q95(list(_per_subject_nll_target(base, Xs, src_subj, pi_star, cfg, COMMON["device"]).values()))

    # --- nested source-site folds -> scale-normalised excess ---
    fold_details, all_excesses = [], []
    site_arr = sim.site
    for fi, u in enumerate(source_sites):
        tr_mask = np.isin(site_arr, [s for s in source_sites if s != u])
        ps_mask = site_arr == u
        Xtr, ytr = sim.X[tr_mask], sim.y[tr_mask]
        dom_tr = sim.domains.subset(np.where(tr_mask)[0])
        nmodel, *_ = train_h2(Xtr, ytr, dom_tr, sim.dag, cfg, align_factor="site", verbose=False)
        pi_n = reference_prior(ytr, COMMON["classes"], cfg.align.reference_prior)
        tr_nll = list(_per_subject_nll_target(nmodel, Xtr, dom_tr.factor("subject"), pi_n, cfg, COMMON["device"]).values())
        ps_dom = sim.domains.subset(np.where(ps_mask)[0])
        ps_nll = list(_per_subject_nll_target(nmodel, sim.X[ps_mask], ps_dom.factor("subject"), pi_n, cfg, COMMON["device"]).values())
        fold_train_q95 = _q95(tr_nll)
        excesses = [float(x - fold_train_q95) for x in ps_nll]
        all_excesses.extend(excesses)
        fold_details.append(dict(
            fold=fi, pseudo_site=u, n_train_subject_units=len(tr_nll), n_pseudo_subject_units=len(ps_nll),
            fold_train_q95_nll_target_prior=fold_train_q95,
            pseudo_nll_min=(min(ps_nll) if ps_nll else float("nan")),
            pseudo_nll_mean=(float(np.mean(ps_nll)) if ps_nll else float("nan")),
            pseudo_nll_max=(max(ps_nll) if ps_nll else float("nan")),
            pseudo_excess_min=(min(excesses) if excesses else float("nan")),
            pseudo_excess_mean=(float(np.mean(excesses)) if excesses else float("nan")),
            pseudo_excess_max=(max(excesses) if excesses else float("nan"))))
    nested_excess_q95 = max(0.0, _q95(all_excesses))
    nested_threshold = base_source_q95 + nested_excess_q95

    thresholds = {"in_source_subject_q95": base_source_q95, "nested_site_excess_q95": nested_threshold}

    # --- route target under both modes (base model reused; no retraining) ---
    Xt, yt = sim.X[tgt_idx], sim.y[tgt_idx]
    tgt_unit = sim.domains.subset(tgt_idx).factor(COMMON["eval_unit"])
    concept_sites = [int(s) for s in sim.meta["concept_sites"]]
    for mode in MODES:
        thr = thresholds[mode]
        fcfg = make_support_calibrated_feature_config(
            max_density_nll_target_prior=thr, min_target_n=max(20, int(cfg.tta.min_target)))
        router = RefusalFirstRouter(RouterConfig(feature_config=fcfg))
        rep = evaluate_router_offline_tta(
            base, Xt, yt, tgt_unit, cfg, pi_star, router=router,
            X_src=Xs, y_src=ys, source_pseudo_levels=src_subj, device=COMMON["device"],
            calibrate_source_support=False, support_calibration_mode=mode)
        out = dict(
            world=world, seed=seed, support_mode=mode, threshold=thr,
            base_source_q95_nll_target_prior=base_source_q95, nested_excess_q95=nested_excess_q95,
            target_sites=[target_site], concept_sites=concept_sites,
            target_concept_hit=bool(target_site in concept_sites),
            source_concept_count=len(set(concept_sites) - {target_site}),
            strict_bacc=float(rep["identity"]["balanced_acc"]),
            raw_offline_delta_bacc=float(rep["delta_raw_offline_tta"]["d_balanced_acc"]),
            fold_details=(fold_details if mode == "nested_site_excess_q95" else []),
            report=rep)
        os.makedirs(out_dir, exist_ok=True)
        with open(_report_path(out_dir, world, seed, mode), "w") as f:
            json.dump(out, f, indent=2, default=float)
        s = rep["router_summary"]
        print(f"[support] {world}/seed{seed}/{mode}: thr={thr:.2f} (base_q95={base_source_q95:.2f} "
              f"+excess={nested_excess_q95:.2f}) actions={dict(s['action_counts'])} "
              f"coverage={s['coverage']:.2f} identity_rate={s['identity_rate']:.2f}")


# ------------------------------------------------------------------ aggregate
def _fmt(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    if isinstance(v, dict):
        return ";".join(f"{k}:{v[k]}" for k in v)
    return str(v)


def aggregate(run_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(run_dir, "*_router_report.json")))
    if not files:
        raise SystemExit(f"no reports in {run_dir}")
    world_rows, domain_rows, detail_rows = [], [], []
    for fp in files:
        out = json.load(open(fp))
        rep = out["report"]; s = rep["router_summary"]; mode = out["support_mode"]; thr = out["threshold"]
        per = rep["per_domain"]
        tgt_nlls = [dv["support"]["density_nll_target_prior"] for dv in per.values()]
        n_sm = sum(1 for dv in per.values() if "OACI_TOS_SUPPORT_MISMATCH" in dv["action_scores"]["identity"]["reason_codes"])
        n_le = sum(1 for dv in per.values() if "OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE" in dv["action_scores"]["identity"]["reason_codes"])
        world_rows.append({
            "world": out["world"], "seed": out["seed"], "support_mode": mode,
            "target_site": "|".join(map(str, out["target_sites"])),
            "target_concept_hit": out["target_concept_hit"], "source_concept_count": out["source_concept_count"],
            "strict_bacc": out["strict_bacc"], "raw_offline_delta_bacc": out["raw_offline_delta_bacc"],
            "base_source_q95_nll_target_prior": out["base_source_q95_nll_target_prior"],
            "nested_excess_q95": out["nested_excess_q95"], "support_threshold_nll_target_prior": thr,
            "mean_target_density_nll_target_prior": float(np.mean(tgt_nlls)),
            "mean_target_support_excess": float(np.mean([x - thr for x in tgt_nlls])),
            "n_support_mismatch_domains": n_sm, "n_low_ess_domains": n_le,
            "router_coverage": s["coverage"], "router_refusal_rate": s["refusal_rate"],
            "router_identity_rate": s["identity_rate"], "router_offline_tta_rate": s["offline_tta_rate"],
            "router_accepted_bacc": s["accepted_bacc"],
            "router_selected_mean_gain_vs_identity": s["selected_mean_gain_vs_identity"],
            "router_missed_benefit": s["missed_benefit"], "router_avoided_harm": s["avoided_harm"],
            "source_acar_harm_state": s["source_acar_harm_calibration_state"],
            "reason_hist": s["reason_hist"], "tta_block_reason_hist": s["tta_block_reason_hist"],
            "action_counts": s["action_counts"],
        })
        for did, dv in per.items():
            asf = dv["action_scores"]; sup = dv["support"]
            idr = asf["identity"]["reason_codes"]
            domain_rows.append({
                "world": out["world"], "seed": out["seed"], "support_mode": mode, "domain_id": did, "n": dv["n"],
                "decision_action": dv["decision_action"], "accepted": dv["accepted"], "reason_codes": dv["reason_codes"],
                "identity_bacc": dv["identity_bacc"], "offline_tta_bacc": dv["offline_tta_bacc"],
                "raw_gain": dv["raw_gain"], "selected_bacc": dv["selected_bacc"],
                "selected_gain_vs_identity": dv["selected_gain_vs_identity"],
                "identity_admissible": asf["identity"]["admissible"],
                "offline_tta_admissible": asf["offline_tta"]["admissible"],
                "offline_tta_blocking_reason_codes": asf["offline_tta"]["blocking_reason_codes"],
                "prior_shift_only": asf["identity"]["prior_shift_only"],
                "cmi_residual_available": asf["identity"]["cmi_residual_available"],
                "acar_harm_calibration_state": asf["offline_tta"]["acar_harm_calibration_state"],
                "density_nll_source_prior": sup["density_nll_source_prior"],
                "density_nll_target_prior": sup["density_nll_target_prior"],
                "support_gap": sup["support_gap"], "ess": sup["ess"], "ood_score": sup["ood_score"],
                "support_threshold_nll_target_prior": thr,
                "target_support_excess": sup["density_nll_target_prior"] - thr,
                "support_mismatch": ("OACI_TOS_SUPPORT_MISMATCH" in idr),
                "low_ess": ("OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE" in idr),
            })
        base = {"world": out["world"], "seed": out["seed"], "support_mode": mode,
                "target_site": "|".join(map(str, out["target_sites"])),
                "base_source_q95_nll_target_prior": out["base_source_q95_nll_target_prior"],
                "nested_excess_q95": out["nested_excess_q95"], "support_threshold_nll_target_prior": thr}
        for fd in out.get("fold_details", []):
            detail_rows.append({**base, **fd})
        if not out.get("fold_details"):
            detail_rows.append({**base, "pseudo_site": "", "fold": "",
                                "n_train_subject_units": "", "n_pseudo_subject_units": ""})

    def _write(path, cols, rows, key):
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for r in sorted(rows, key=key):
                w.writerow([_fmt(r.get(c)) for c in cols])

    _write(os.path.join(run_dir, "world_summary.csv"), WORLD_COLS, world_rows,
           lambda r: (r["world"], r["seed"], r["support_mode"]))
    _write(os.path.join(run_dir, "per_domain_decisions.csv"), DOMAIN_COLS, domain_rows,
           lambda r: (r["world"], r["seed"], r["support_mode"], r["domain_id"]))
    _write(os.path.join(run_dir, "support_calibration_details.csv"), DETAIL_COLS, detail_rows,
           lambda r: (r["world"], r["seed"], r["support_mode"], str(r.get("fold"))))
    with open(os.path.join(run_dir, "world_summary.json"), "w") as f:
        json.dump(world_rows, f, indent=2, default=float)
    print(f"[aggregate] {len(world_rows)} world rows, {len(domain_rows)} domain rows -> {run_dir}")

    print("\n===== baseline vs nested (coverage / identity_rate / refusal) =====")
    by = {}
    for r in world_rows:
        by.setdefault((r["world"], r["seed"]), {})[r["support_mode"]] = r
    for (world, seed), d in sorted(by.items()):
        b = d.get("in_source_subject_q95", {}); n = d.get("nested_site_excess_q95", {})
        print(f"  {world:6s} seed{seed:<2d} thr base={b.get('support_threshold_nll_target_prior',float('nan')):.2f}"
              f"->nested={n.get('support_threshold_nll_target_prior',float('nan')):.2f} | "
              f"coverage {b.get('router_coverage',float('nan')):.2f}->{n.get('router_coverage',float('nan')):.2f} | "
              f"identity {b.get('router_identity_rate',float('nan')):.2f}->{n.get('router_identity_rate',float('nan')):.2f} | "
              f"SM {b.get('n_support_mismatch_domains')}->{n.get('n_support_mismatch_domains')} "
              f"lowESS {b.get('n_low_ess_domains')}->{n.get('n_low_ess_domains')} | "
              f"off_tta {b.get('router_offline_tta_rate',float('nan')):.2f}->{n.get('router_offline_tta_rate',float('nan')):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Project B Step-2F nested support calibration eval")
    ap.add_argument("--mode", choices=["run", "aggregate"], default=None)
    ap.add_argument("--world"); ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="/tmp/project_b_step2f_support")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    if args.mode == "run":
        if not (args.world and args.seed is not None):
            raise SystemExit("--world and --seed required for run")
        run_one(args.world, args.seed, args.out)
    elif args.mode == "aggregate":
        aggregate(args.dir or args.out)
    else:
        for world, cfgw in WORLDS.items():
            for seed in cfgw["seeds"]:
                if all(os.path.exists(_report_path(args.out, world, seed, m)) for m in MODES):
                    print(f"[skip] {world}/seed{seed} (reports exist)")
                    continue
                run_one(world, seed, args.out)
        aggregate(args.out)


if __name__ == "__main__":
    main()
