"""
CSC Route B3-P2.4b POST-HOC AUDIT (does NOT change the method or any decision). Addresses the reviewer's
three pre-freeze audit points for the fixed-margin h0 bootstrap null:

  A1 MCMC MIXING: per-cell sampler diagnostics (acceptance rate, changed-label fraction, condition-wise
     changed fraction) + n_swaps SENSITIVITY (base x {1,5,10}) -> does the fixed-margin p-value / decision
     move materially with more swaps?
  A2 ANALYSIS UNIT: the fixed margins are over the ROWS handed to the test = trials/epochs. The label
     acquisition is "query m paired SUBJECTS, label ALL their condition epochs as trials"; the test
     aggregates per-epoch nll to a subject-condition vote. So margin_unit = label_unit = TRIAL (epoch);
     decision_aggregation = subject_condition_vote. Reported explicitly below (no ambiguity).
  A3 NULL GENERATOR: compare full-audit h0 (default) vs fold-local h0 generators for the fixed-margin p.

  python -m csc.mininfo.audit_p24b_sampler --out csc/results/b3_p24b_sampler_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

from csc.sim.shift_simulator import SimConfig, make_geom
from .paired_sim import make_paired_target
from .paired_calibrated import (eligible_complete_pairs, _h0_full_logp, condition_code,
                                sample_h0_fixed_condition_margins, paired_cv_test,
                                MIN_EPOCHS_PER_CONDITION)
from .run_b3_p23 import SCENARIOS

UNIT_INFO = dict(margin_unit="trial", label_unit="trial",
                 label_acquisition="query m paired subjects; label ALL their condition epochs (trials)",
                 decision_aggregation="subject_condition_vote (class-balanced)")


def _queried_eligible(kind, scen, seed, m, n_subjects=36):
    cfg = SimConfig(seed=seed, subject_tau=scen.get("cfg_subject_tau", SimConfig.subject_tau),
                    epochs_min=scen.get("cfg_epochs_min", SimConfig.epochs_min),
                    epochs_max=scen.get("cfg_epochs_max", SimConfig.epochs_max))
    geom = make_geom(cfg, np.random.default_rng(seed))
    Z, Y, D, G, _ = make_paired_target(kind, geom, cfg, n_subjects=n_subjects, seed=10_000 + seed,
                                       cov_scale=scen.get("cov_scale", 10.0),
                                       base_prior=scen.get("base_prior"),
                                       label_noise=scen.get("label_noise", 0.0))
    elig = eligible_complete_pairs(D, G, MIN_EPOCHS_PER_CONDITION)
    rng = np.random.default_rng(seed)
    pick = rng.choice(np.array(sorted(elig)), size=min(int(m), len(elig)), replace=False)
    mask = np.isin(G, pick)
    return Z[mask], Y[mask], D[mask], G[mask]


def audit_cell(kind, scen_name, seed, m, n_boot):
    scen = SCENARIOS[scen_name]
    Zq, Yq, Dq, Gq = _queried_eligible(kind, scen, seed, m)
    elig = eligible_complete_pairs(Dq, Gq, MIN_EPOCHS_PER_CONDITION)
    me = np.isin(Gq, elig); Ze, Ye, De, Ge = Zq[me], Yq[me], Dq[me], Gq[me]
    cl = np.array(sorted(np.unique(Ye)))
    logp0 = _h0_full_logp(Ze, Ye, De, Ge, cl, "centered", 0.5)
    y0 = np.searchsorted(cl, Ye)
    nsw = max(20 * len(Ye), 300)
    _, diag = sample_h0_fixed_condition_margins(logp0, De, y0, np.random.default_rng(seed + 5), nsw,
                                                return_diag=True)
    # A1 n_swaps sensitivity (fixed-margin p at base x {1,5,10})
    def pfm(ns, gen="full_audit"):
        return paired_cv_test(Zq, Yq, Dq, Gq, n_boot=n_boot, seed=seed, also_standard=False,
                              n_swaps=ns, null_generator=gen)["fixed_margin_null_p"]
    p1, p5, p10 = pfm(nsw), pfm(5 * nsw), pfm(10 * nsw)
    # A3 generator comparison
    p_full, p_fold = pfm(nsw, "full_audit"), pfm(nsw, "fold_local")
    return dict(kind=kind, scenario=scen_name, seed=int(seed), m=int(m), n_eligible=int(len(elig)),
                n_rows=int(len(Ye)), **diag,
                p_nsw1=float(p1), p_nsw5=float(p5), p_nsw10=float(p10),
                p_swaps_range=float(max(p1, p5, p10) - min(p1, p5, p10)),
                p_full_audit=float(p_full), p_fold_local=float(p_fold),
                p_generator_diff=float(abs(p_full - p_fold)))


def run(out=None, n_boot=150, n_jobs=1):
    warnings.filterwarnings("ignore")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    cells = [(k, sc, s) for k in ("random_label", "paired_label", "paired_concept")
             for sc in ("baseline", "label_noise") for s in (1000, 1001)]
    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=n_jobs)(delayed(audit_cell)(k, sc, s, 24, n_boot) for k, sc, s in cells)
    else:
        recs = [audit_cell(k, sc, s, 24, n_boot) for k, sc, s in cells]
    payload = dict(kind="B3-P2.4b post-hoc sampler/generator audit (no decision change)", unit_info=UNIT_INFO,
                   n_boot=n_boot, cells=recs,
                   max_p_swaps_range=float(max(r["p_swaps_range"] for r in recs)),
                   max_p_generator_diff=float(max(r["p_generator_diff"] for r in recs)),
                   acceptance_rates=[round(r["acceptance_rate"], 3) for r in recs],
                   changed_fractions=[round(r["changed_fraction"], 3) for r in recs])
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[audit] wrote {out}")
    print("UNIT:", UNIT_INFO)
    print("kind                 scen        seed  accept  chg   p(1x/5x/10x)       range   p(full/fold) gdiff")
    for r in recs:
        print("  %-18s %-10s %4d  %.2f   %.2f  %.3f/%.3f/%.3f  %.3f   %.3f/%.3f %.3f" % (
            r["kind"], r["scenario"], r["seed"], r["acceptance_rate"], r["changed_fraction"],
            r["p_nsw1"], r["p_nsw5"], r["p_nsw10"], r["p_swaps_range"], r["p_full_audit"],
            r["p_fold_local"], r["p_generator_diff"]))
    print(f"MAX p-range over n_swaps x{{1,5,10}} = {payload['max_p_swaps_range']:.3f}")
    print(f"MAX |p_full_audit - p_fold_local|   = {payload['max_p_generator_diff']:.3f}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="B3-P2.4b post-hoc sampler/generator audit.")
    ap.add_argument("--n_boot", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=str, default="csc/results/b3_p24b_sampler_audit.json")
    a = ap.parse_args()
    run(out=a.out, n_boot=a.n_boot, n_jobs=a.jobs)


if __name__ == "__main__":
    main()
