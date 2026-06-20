"""Ablation harness (review section 10.3) on the EEG mechanism simulator.

Runs a panel of ablations on ONE shared simulated dataset (same source/target split) and
reports strict-DG balanced accuracy + offline-TTA Δ for each. Demonstrates that every
H²-CMI toggle is wired and that the core design choices matter.

    python -m h2cmi.run_ablation --epochs 15
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from h2cmi.config import H2Config
from h2cmi.domains import DomainDAG, DomainFactor, DomainLabels, compact_domain_labels
from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
from h2cmi.train.trainer import train_h2, reference_prior
from h2cmi.eval.harness import evaluate_strict_dg, evaluate_offline_tta


def base_config(args) -> H2Config:
    cfg = H2Config(n_classes=args.classes)
    cfg.encoder.n_chans = args.chans
    cfg.encoder.n_times = args.times
    cfg.train.epochs = args.epochs
    cfg.train.seed = args.seed
    return cfg


def flat_domain_labels(domains: DomainLabels) -> tuple[DomainDAG, DomainLabels]:
    """Collapse the DAG to a single flat JOINT 'domain' = (site,subject,session) tuple,
    relabelled contiguously (review: a true flat-joint baseline, not site-only)."""
    import numpy as np
    keys = domains.levels  # [N, n_factors]
    # encode the full tuple as one categorical, then relabel to 0..K-1
    enc = np.zeros(domains.n, dtype=np.int64)
    for j, f in enumerate(domains.dag.factors):
        enc = enc * f.n_levels + keys[:, j]
    _, inv = np.unique(enc, return_inverse=True)
    dag = DomainDAG([DomainFactor("domain", int(inv.max()) + 1, (), "invariant", 0.02)])
    return dag, DomainLabels(dag, inv.reshape(-1, 1))


# each ablation: name -> (config-mutator, use_flat_D)
def ablations() -> dict:
    def mut(**kw):
        def f(c: H2Config):
            for k, v in kw.items():
                obj, attr = k.rsplit(".", 1)
                setattr(getattr(c, obj), attr, v)
        return f
    def no_cmi(c):           # lambda stays 0 -> no leakage penalty
        c.cmi.lambda_init = 0.0; c.cmi.dual_lr = 0.0
    def hard_inv(c):         # tiny budgets + strong start -> near-hard invariance
        c.cmi.lambda_init = 5.0
        for f in ("encoder",): pass
    return {
        "full":            (lambda c: None, False),
        "flat_D":          (lambda c: None, True),
        "no_CMI":          (no_cmi, False),
        "no_align":        (mut(**{"align.enabled": False}), False),
        "no_disentangle":  (mut(**{"disentangle.enabled": False}), False),
        "no_SSL":          (mut(**{"ssl.enabled": False}), False),
        "hard_invariance": (hard_inv, False),
        "temporal_only":   (mut(**{"encoder.use_spd": False, "encoder.use_graph": False}), False),
        "SPD_only":        (mut(**{"encoder.use_temporal": False, "encoder.use_graph": False}), False),
        "graph_only":      (mut(**{"encoder.use_temporal": False, "encoder.use_spd": False}), False),
        "gaussian_density":(mut(**{"density.df": 1e6}), False),
    }


def run_one(name, mutate, use_flat, sim, src_idx, tgt_idx, base, args):
    cfg = copy.deepcopy(base)
    mutate(cfg)
    Xs, ys = sim.X[src_idx], sim.y[src_idx]
    src_raw = sim.domains.subset(src_idx)
    if use_flat:
        dag, dom_for_train = flat_domain_labels(src_raw)     # true flat-joint baseline
        align_factor = "domain"
    else:
        dag, dom_for_train, _ = compact_domain_labels(src_raw)   # source-only compact DAG (P0-1)
        align_factor = "site"
    model, hcmi, dual, hist = train_h2(Xs, ys, dom_for_train, dag, cfg,
                                       align_factor=align_factor, verbose=False)
    pi_star = reference_prior(ys, args.classes, cfg.align.reference_prior)
    Xt, yt = sim.X[tgt_idx], sim.y[tgt_idx]
    tgt_unit = sim.domains.subset(tgt_idx).factor("subject")
    sdg = evaluate_strict_dg(model, Xt, yt, tgt_unit, prior=None)
    off = evaluate_offline_tta(model, Xt, yt, tgt_unit, cfg, pi_star, gate=None)
    return dict(name=name,
                strict_bacc=round(sdg["balanced_acc"], 3),
                worst_dom=round(sdg["worst_domain_bacc"], 3),
                ece=round(sdg["ece"], 3),
                tta_d_bacc=round(off["delta_adapt"]["d_balanced_acc"], 3),
                tta_ci=[round(off["gain_bootstrap"]["lo"], 3), round(off["gain_bootstrap"]["hi"], 3)],
                final_Ihat={f: round(hist[-1]["ihat"].get(f, float("nan")), 3) for f in hcmi.factors})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--chans", type=int, default=16)
    ap.add_argument("--times", type=int, default=128)
    ap.add_argument("--sites", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list to restrict ablations")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sim = EEGSimulator(args.classes, args.chans, args.times,
                       shift=ShiftSpec(cov=1.0, prior=0.4, concept=0.0, montage=0.2, noise=0.3),
                       seed=args.seed).sample(args.sites, 3, 2, 40)
    src_idx, tgt_idx = train_target_split(sim, 1, seed=args.seed)
    base = base_config(args)

    todo = ablations()
    if args.only:
        keep = set(args.only.split(","))
        todo = {k: v for k, v in todo.items() if k in keep}

    rows = []
    print(f"{'ablation':<17} {'strictDG':>8} {'worst':>6} {'ece':>6} {'TTA Δ':>7} {'TTA CI':>16}")
    for name, (mut, flat) in todo.items():
        r = run_one(name, mut, flat, sim, src_idx, tgt_idx, base, args)
        rows.append(r)
        print(f"{r['name']:<17} {r['strict_bacc']:>8} {r['worst_dom']:>6} {r['ece']:>6} "
              f"{r['tta_d_bacc']:>+7} {str(r['tta_ci']):>16}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2, default=float)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
