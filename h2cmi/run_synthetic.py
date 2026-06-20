"""End-to-end H2-CMI on the EEG mechanism simulator.

Trains the H2-CMI model on SOURCE sites, then evaluates the held-out TARGET site under all
three settings (strict DG / offline transductive TTA / online streaming TTA), trains the
source-only safety gate, and reports cross-fitted conditional leakage on a source split.

This is the integration entry point and the substrate for the orthogonal-shift study
(vary ShiftSpec to reproduce the review's R0/R3/R6 boundaries).

Usage:
    python -m h2cmi.run_synthetic --epochs 25 --concept 0.0
    python -m h2cmi.run_synthetic --epochs 25 --concept 0.6 --concept_frac 0.5   # harmful shift
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from h2cmi.config import H2Config
from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
from h2cmi.train.trainer import train_h2, reference_prior
from h2cmi.eval.harness import run_three_settings
from h2cmi.eval.leakage import crossfit_conditional_leakage


def build_config(args) -> H2Config:
    cfg = H2Config(n_classes=args.classes)
    cfg.encoder.n_chans = args.chans
    cfg.encoder.n_times = args.times
    cfg.encoder.fs = args.fs
    cfg.train.epochs = args.epochs
    cfg.train.device = args.device
    cfg.train.seed = args.seed
    cfg.train.batch_size = args.bs
    if args.fast:
        cfg.small()
    return cfg


def main():
    ap = argparse.ArgumentParser(description="H2-CMI synthetic end-to-end")
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--chans", type=int, default=16)
    ap.add_argument("--times", type=int, default=128)
    ap.add_argument("--fs", type=float, default=128.0)
    ap.add_argument("--sites", type=int, default=5)
    ap.add_argument("--subjects", type=int, default=3, help="subjects per site")
    ap.add_argument("--sessions", type=int, default=2, help="sessions per subject")
    ap.add_argument("--trials", type=int, default=40, help="trials per session")
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--prior", type=float, default=0.4)
    ap.add_argument("--concept", type=float, default=0.0)
    ap.add_argument("--concept_frac", type=float, default=0.0)
    ap.add_argument("--montage", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--label_rho", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_unit", default="subject", choices=["subject", "session", "site"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    shift = ShiftSpec(cov=args.cov, prior=args.prior, concept=args.concept,
                      concept_site_frac=args.concept_frac, montage=args.montage,
                      noise=args.noise, label_mechanism_rho=args.label_rho)
    sim = EEGSimulator(args.classes, args.chans, args.times, args.fs, shift=shift,
                       seed=args.seed).sample(args.sites, args.subjects, args.sessions, args.trials)

    src_idx, tgt_idx = train_target_split(sim, n_target_sites=1, seed=args.seed)
    cfg = build_config(args)

    # train on source sites
    Xs, ys = sim.X[src_idx], sim.y[src_idx]
    src_domains = sim.domains.subset(src_idx)
    model, hcmi, dual, hist = train_h2(Xs, ys, src_domains, sim.dag, cfg,
                                       align_factor="site", verbose=not args.quiet)

    pi_star = reference_prior(ys, args.classes, cfg.align.reference_prior)

    # held-out target site, evaluated by the chosen domain unit
    Xt, yt = sim.X[tgt_idx], sim.y[tgt_idx]
    tgt_unit = sim.domains.subset(tgt_idx).factor(args.eval_unit)
    src_unit = src_domains.factor("subject")          # gate pseudo-targets = source subjects

    results = run_three_settings(model, Xt, yt, tgt_unit, cfg, pi_star,
                                 X_src=Xs, y_src=ys, gate_pseudo_levels=src_unit,
                                 device=cfg.train.device)

    # cross-fitted leakage on a source split (frozen encoder)
    Zs = model.embed(Xs, device=cfg.train.device)
    leak = crossfit_conditional_leakage(Zs, ys, src_domains, sim.dag, args.classes,
                                        device=cfg.train.device, n_perm=20, seed=args.seed)

    report = dict(
        config=dict(classes=args.classes, sites=args.sites, shift=vars(shift),
                    epochs=cfg.train.epochs, eval_unit=args.eval_unit,
                    concept_sites=sim.meta["concept_sites"]),
        strict_dg={k: v for k, v in results["strict_dg"].items() if k != "per_domain_bacc"},
        offline_tta=dict(delta_adapt=results["offline_tta"]["delta_adapt"],
                         delta_selective=results["offline_tta"]["delta_selective"],
                         gain_bootstrap=results["offline_tta"]["gain_bootstrap"],
                         selective_risk=results["offline_tta"]["selective_risk"]),
        online_tta={k: v for k, v in results["online_tta"].items() if k != "per_domain_bacc"},
        gate_info=results["gate_info"],
        leakage=leak,
    )
    print("\n===== H2-CMI synthetic report =====")
    print(json.dumps(report, indent=2, default=float))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print("saved ->", args.out)
    return report


if __name__ == "__main__":
    main()
