"""Project B Step-2E: locked-world router evaluation on the synthetic substrate.

Trains a full (NO --fast) H2-CMI model per (world, seed), runs the router integration harness
(source-only calibration; target labels used only post-hoc), and writes per-domain decisions +
per-world summaries for the three locked worlds:
  R2    (recoverable)                     seeds 0,1,2
  HF3   (source-calibratable-attempt)     seeds 3,4,7,8,10
  H_OOD (target-only stress, seed 32)     seed 32

Does NOT modify any h2cmi core (tta/gate/harness/trainer/config) or the router package.
Modes: --mode run --world W --seed S ; --mode aggregate --dir D ; default (no --mode) runs all
locked (world,seed) pairs — reusing any existing per-seed report JSON — then aggregates.
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

WORLD_COLS = [
    "world", "seed", "target_site", "target_concept_hit", "source_concept_count",
    "strict_bacc", "raw_offline_delta_bacc",
    "router_coverage", "router_refusal_rate", "router_identity_rate", "router_offline_tta_rate",
    "router_accepted_bacc", "router_selected_mean_gain_vs_identity",
    "router_avoided_harm", "router_missed_benefit",
    "source_support_threshold_nll_target_prior", "source_acar_harm_state",
    "source_pseudo_gain_min", "source_pseudo_gain_mean", "source_pseudo_gain_max",
    "reason_hist", "tta_block_reason_hist", "action_counts",
]

DOMAIN_COLS = [
    "world", "seed", "domain_id", "n", "decision_action", "accepted", "reason_codes",
    "identity_bacc", "offline_tta_bacc", "raw_gain", "selected_bacc", "selected_gain_vs_identity",
    "offline_tta_admissible", "offline_tta_reason_codes", "offline_tta_blocking_reason_codes",
    "identity_admissible", "identity_reason_codes",
    "prior_shift_only", "cmi_residual_available", "acar_harm_calibration_state",
    "density_nll_source_prior", "density_nll_target_prior", "support_gap", "ess", "ood_score",
]


def _tasks():
    return [(w, s) for w, cfgw in WORLDS.items() for s in cfgw["seeds"]]


def _report_path(out_dir, world, seed):
    return os.path.join(out_dir, f"{world}_seed{seed}_router_report.json")


# ------------------------------------------------------------------ one (world, seed)
def run_one(world: str, seed: int, out_dir: str) -> dict:
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    from h2cmi.config import H2Config
    from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
    from h2cmi.train.trainer import train_h2, reference_prior
    from h2cmi.eval.router_harness import evaluate_router_offline_tta

    w = WORLDS[world]
    shift = ShiftSpec(cov=w["cov"], prior=w["prior"], concept=w["concept"],
                      concept_site_frac=w["concept_frac"], montage=w["montage"],
                      noise=COMMON["noise"], label_mechanism_rho=COMMON["label_rho"])
    sim = EEGSimulator(COMMON["classes"], COMMON["chans"], COMMON["times"], COMMON["fs"],
                       shift=shift, seed=seed).sample(
        COMMON["sites"], COMMON["subjects"], COMMON["sessions"], COMMON["trials"])
    src_idx, tgt_idx = train_target_split(sim, n_target_sites=1, seed=seed)

    cfg = H2Config(n_classes=COMMON["classes"])
    cfg.encoder.n_chans = COMMON["chans"]; cfg.encoder.n_times = COMMON["times"]; cfg.encoder.fs = COMMON["fs"]
    cfg.train.epochs = COMMON["epochs"]; cfg.train.device = COMMON["device"]
    cfg.train.seed = seed; cfg.train.batch_size = COMMON["bs"]

    Xs, ys = sim.X[src_idx], sim.y[src_idx]
    src_dom = sim.domains.subset(src_idx)
    model, *_ = train_h2(Xs, ys, src_dom, sim.dag, cfg, align_factor="site", verbose=False)
    pi_star = reference_prior(ys, COMMON["classes"], cfg.align.reference_prior)

    Xt, yt = sim.X[tgt_idx], sim.y[tgt_idx]
    tgt_unit = sim.domains.subset(tgt_idx).factor(COMMON["eval_unit"])
    src_unit = src_dom.factor("subject")

    rep = evaluate_router_offline_tta(
        model, Xt, yt, tgt_unit, cfg, pi_star,
        X_src=Xs, y_src=ys, source_pseudo_levels=src_unit, device=cfg.train.device)

    tgt_sites = sorted(np.unique(sim.site[tgt_idx]).tolist())
    concept_sites = [int(s) for s in sim.meta["concept_sites"]]
    out = dict(
        world=world, seed=seed,
        target_sites=tgt_sites, concept_sites=concept_sites,
        target_concept_hit=bool(set(tgt_sites) & set(concept_sites)),
        source_concept_count=len(set(concept_sites) - set(tgt_sites)),
        strict_bacc=float(rep["identity"]["balanced_acc"]),
        raw_offline_delta_bacc=float(rep["delta_raw_offline_tta"]["d_balanced_acc"]),
        report=rep,
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(_report_path(out_dir, world, seed), "w") as f:
        json.dump(out, f, indent=2, default=float)
    s = rep["router_summary"]
    print(f"[router] {world}/seed{seed}: strict={out['strict_bacc']:.3f} "
          f"raw_dTTA={out['raw_offline_delta_bacc']:+.3f} actions={dict(s['action_counts'])} "
          f"acar={s['source_acar_harm_calibration_state']} coverage={s['coverage']:.2f}")
    return out


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
        raise SystemExit(f"no router reports in {run_dir}")
    world_rows, domain_rows = [], []
    for fp in files:
        out = json.load(open(fp))
        rep = out["report"]; s = rep["router_summary"]
        world_rows.append({
            "world": out["world"], "seed": out["seed"],
            "target_site": "|".join(map(str, out["target_sites"])),
            "target_concept_hit": out["target_concept_hit"],
            "source_concept_count": out["source_concept_count"],
            "strict_bacc": out["strict_bacc"], "raw_offline_delta_bacc": out["raw_offline_delta_bacc"],
            "router_coverage": s["coverage"], "router_refusal_rate": s["refusal_rate"],
            "router_identity_rate": s["identity_rate"], "router_offline_tta_rate": s["offline_tta_rate"],
            "router_accepted_bacc": s["accepted_bacc"],
            "router_selected_mean_gain_vs_identity": s["selected_mean_gain_vs_identity"],
            "router_avoided_harm": s["avoided_harm"], "router_missed_benefit": s["missed_benefit"],
            "source_support_threshold_nll_target_prior": s.get("source_support_threshold_nll_target_prior"),
            "source_acar_harm_state": s["source_acar_harm_calibration_state"],
            "source_pseudo_gain_min": (s["source_pseudo_gains"] or {}).get("gain_min"),
            "source_pseudo_gain_mean": (s["source_pseudo_gains"] or {}).get("gain_mean"),
            "source_pseudo_gain_max": (s["source_pseudo_gains"] or {}).get("gain_max"),
            "reason_hist": s["reason_hist"], "tta_block_reason_hist": s["tta_block_reason_hist"],
            "action_counts": s["action_counts"],
        })
        for did, dv in rep["per_domain"].items():
            asf = dv["action_scores"]; sup = dv["support"]
            domain_rows.append({
                "world": out["world"], "seed": out["seed"], "domain_id": did, "n": dv["n"],
                "decision_action": dv["decision_action"], "accepted": dv["accepted"],
                "reason_codes": dv["reason_codes"],
                "identity_bacc": dv["identity_bacc"], "offline_tta_bacc": dv["offline_tta_bacc"],
                "raw_gain": dv["raw_gain"], "selected_bacc": dv["selected_bacc"],
                "selected_gain_vs_identity": dv["selected_gain_vs_identity"],
                "offline_tta_admissible": asf["offline_tta"]["admissible"],
                "offline_tta_reason_codes": asf["offline_tta"]["reason_codes"],
                "offline_tta_blocking_reason_codes": asf["offline_tta"]["blocking_reason_codes"],
                "identity_admissible": asf["identity"]["admissible"],
                "identity_reason_codes": asf["identity"]["reason_codes"],
                "prior_shift_only": asf["identity"]["prior_shift_only"],
                "cmi_residual_available": asf["identity"]["cmi_residual_available"],
                "acar_harm_calibration_state": asf["offline_tta"]["acar_harm_calibration_state"],
                "density_nll_source_prior": sup["density_nll_source_prior"],
                "density_nll_target_prior": sup["density_nll_target_prior"],
                "support_gap": sup["support_gap"], "ess": sup["ess"], "ood_score": sup["ood_score"],
            })

    with open(os.path.join(run_dir, "world_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(WORLD_COLS)
        for r in sorted(world_rows, key=lambda r: (r["world"], r["seed"])):
            w.writerow([_fmt(r.get(c)) for c in WORLD_COLS])
    with open(os.path.join(run_dir, "per_domain_decisions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(DOMAIN_COLS)
        for r in sorted(domain_rows, key=lambda r: (r["world"], r["seed"], r["domain_id"])):
            w.writerow([_fmt(r.get(c)) for c in DOMAIN_COLS])
    with open(os.path.join(run_dir, "world_summary.json"), "w") as f:
        json.dump(world_rows, f, indent=2, default=float)
    print(f"[aggregate] {len(world_rows)} world-seed rows, {len(domain_rows)} domain rows -> {run_dir}")

    # per-world readout
    print("\n===== per (world, seed) =====")
    for r in sorted(world_rows, key=lambda r: (r["world"], r["seed"])):
        print(f"  {r['world']:6s} seed{r['seed']:<2d} strict={r['strict_bacc']:.3f} "
              f"raw_dTTA={r['raw_offline_delta_bacc']:+.3f} hit={r['target_concept_hit']} "
              f"acar={r['source_acar_harm_state']} actions={r['action_counts']} "
              f"missed={r['router_missed_benefit']:.3f} avoided={r['router_avoided_harm']:.3f}")


# ------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Project B Step-2E locked-world router evaluation")
    ap.add_argument("--mode", choices=["run", "aggregate"], default=None)
    ap.add_argument("--world")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="/tmp/project_b_step2e_router")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    if args.mode == "run":
        if not (args.world and args.seed is not None):
            raise SystemExit("--world and --seed required for run")
        run_one(args.world, args.seed, args.out)
    elif args.mode == "aggregate":
        aggregate(args.dir or args.out)
    else:
        # default: run all locked (world, seed) pairs (reuse existing reports), then aggregate
        for world, seed in _tasks():
            if os.path.exists(_report_path(args.out, world, seed)):
                print(f"[skip] {world}/seed{seed} (report exists)")
                continue
            run_one(world, seed, args.out)
        aggregate(args.out)


if __name__ == "__main__":
    main()
