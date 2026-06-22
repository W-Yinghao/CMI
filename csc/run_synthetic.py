"""
csc.run_synthetic — multi-seed evaluation of the three-state certificate (CSC-P0).

Per seed: analyze the source (atlas + residual evidence + direction-linked concept
pruning), then certify a fresh target of EACH shift type. Reports the pre-registered
quantities (PREREGISTRATION.md) over the taxonomy:

  clean / pure_conditional / label_shift / label_covariate_mixed -> must ABSTAIN
  covariate                                                      -> COVARIATE_COMPATIBLE
  boundary_coupled (visible concept)                            -> CONCEPT_SUSPECT

HONESTY: the unit is an INDEPENDENT source seed (the 4 must-abstain targets share a source).
The reported bound is the EXACT one-sided 95% Clopper-Pearson upper bound on the cluster-level
rate -- NOT the cruder Rule-of-Three approximation (3/N). It does not claim "rate <= 0.05".

DEVELOPMENT vs CONFIRMATORY: seeds run here while iterating on the method are DEVELOPMENT
seeds (they have informed code/threshold choices). A frozen confirmatory run must use a
SEPARATE, previously-unseen set of source-target cluster seeds (PREREGISTRATION §3, §5).

  conda run -n icml python -m csc.run_synthetic --seeds 30 --n_boot 100 --out audit.json
"""
from __future__ import annotations

import argparse
import json
import warnings
from math import comb, log
import numpy as np

from csc.sim.shift_simulator import SimConfig, make_source, make_target, _TRUTH
from csc.certificate import (
    analyze_source, certify, CertifierConfig,
    FORBIDDEN, UNIDENTIFIABLE, COVARIATE_COMPATIBLE, CONCEPT_SUSPECT,
)

STATES = [COVARIATE_COMPATIBLE, CONCEPT_SUSPECT, UNIDENTIFIABLE]
KINDS = list(_TRUTH)
MUST_ABSTAIN = ["clean", "pure_conditional", "label_shift", "label_covariate_mixed"]


def _clopper_pearson_upper(failures, n, conf=0.95):
    """EXACT one-sided (upper) Clopper-Pearson bound on a Bernoulli rate given failures/n.
    For failures=0 this is 1-(1-conf)^(1/n) (e.g. 0.259 at n=10, conf=0.95); the
    Rule-of-Three 3/n=0.30 is only its crude approximation."""
    if n == 0:
        return 1.0
    if failures >= n:
        return 1.0
    lo, hi = failures / n, 1.0
    for _ in range(80):                                # bisection on the exact binomial CDF
        mid = (lo + hi) / 2
        cdf = sum(comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(failures + 1))
        if cdf < 1 - conf:
            hi = mid
        else:
            lo = mid
    return hi


def run(seeds=30, n_boot=100, n_dir_boot=200, alpha=0.05,
        n_domains=8, concept_domains=3, seed_offset=0, label="DEVELOPMENT",
        out=None, quiet=True):
    if quiet:
        warnings.filterwarnings("ignore")
    confusion = {k: {s: 0 for s in STATES} for k in KINDS}
    forbidden = {k: 0 for k in KINDS}
    sig_count = 0
    # cluster unit = one source seed: the 4 must-abstain targets SHARE this source, so they
    # are NOT independent Bernoulli trials. We bound the rate at the SEED level.
    seed_cluster_failures = 0
    seed_list = list(range(seed_offset, seed_offset + seeds))

    for s in seed_list:
        cfg = SimConfig(seed=s)
        src = make_source(cfg, n_domains=n_domains, concept_domains=concept_domains, seed=s)
        sa = analyze_source(src.Z, src.Y, src.D, n_boot=n_boot, n_dir_boot=n_dir_boot,
                            alpha=alpha, seed=s)
        sig_count += int(sa.concept_evidenced)
        any_abstain_forbidden = False
        for kind in KINDS:
            tb = make_target(kind, cfg, geom=src.geom, seed=1000 + s)
            c = certify(sa, tb.Z, CertifierConfig())
            confusion[kind][c.state] += 1
            if c.state in FORBIDDEN[tb.truth]:
                forbidden[kind] += 1
                if kind in MUST_ABSTAIN:
                    any_abstain_forbidden = True
        seed_cluster_failures += int(any_abstain_forbidden)

    print(f"\n=== csc certificate (CSC-P1.1) [{label}] — {seeds} seeds "
          f"{seed_list[0]}..{seed_list[-1]}, n_boot={n_boot}, alpha={alpha} ===")
    print(f"source concept evidence valid: {sig_count}/{seeds} = {sig_count/seeds:.2f}\n")
    hdr = f"{'truth / shift kind':30s}" + "".join(f"{s.split('_')[0][:6]:>8s}" for s in STATES) + "  forbid"
    print(hdr); print("-" * len(hdr))
    for kind in KINDS:
        row = f"{kind+' ('+_TRUTH[kind]+')':30s}"
        for st in STATES:
            row += f"{confusion[kind][st]:>8d}"
        row += f"{forbidden[kind]:>8d}"
        print(row)

    # forbidden = false certifications (the rate we must control)
    tot_forbidden = sum(forbidden.values())
    abstain_forbidden = sum(forbidden[k] for k in MUST_ABSTAIN)
    power = confusion["boundary_coupled"][CONCEPT_SUSPECT] / seeds
    cov_ok = confusion["covariate"][COVARIATE_COMPATIBLE] / seeds
    # CLUSTER-LEVEL bound: the unit is an independent source seed, NOT a per-target
    # certificate. (The 4 must-abstain targets share one source -> correlated.)
    ub_cluster = _clopper_pearson_upper(seed_cluster_failures, seeds)
    need_n = 1  # seeds needed for the 0-failure exact-CP bound to reach alpha
    if seed_cluster_failures == 0 and 0 < alpha < 1:
        need_n = int(np.ceil(log(0.05) / log(1 - alpha)))

    print("\n--- pre-registered quantities (synthetic, DEVELOPMENT) ---")
    print(f"false certifications, total                 : {tot_forbidden}"
          f"  (across {seeds*len(KINDS)} certificates)")
    print(f"per-SEED clusters with any must-abstain miss : {seed_cluster_failures}/{seeds}"
          f"  -> exact 95% Clopper-Pearson upper bound = {ub_cluster:.3f}")
    print(f"power on VISIBLE concept (boundary_coupled) : {power:.3f}")
    print(f"covariate -> COVARIATE_COMPATIBLE           : {cov_ok:.3f}")
    print(f"\nNOTE: bound is the EXACT one-sided Clopper-Pearson at the SEED-CLUSTER level "
          f"(Rule-of-Three 3/N is only its approximation). Reaching a 0.05 bound at 0 "
          f"failures needs >= {need_n} INDEPENDENT clusters (currently {seeds}). These are "
          f"{label} seeds.")

    result = dict(label=label, seeds=seeds, seed_list=seed_list, n_boot=n_boot,
                  n_dir_boot=n_dir_boot, alpha=alpha, n_domains=n_domains,
                  concept_domains=concept_domains,
                  confusion={k: dict(v) for k, v in confusion.items()},
                  forbidden=dict(forbidden), source_concept_evidenced=sig_count,
                  power_visible_concept=power, covariate_compatible_coverage=cov_ok,
                  false_cert_total=tot_forbidden,
                  seed_cluster_failures=seed_cluster_failures,
                  exact_cp_upper_bound_cluster=ub_cluster,
                  clusters_needed_for_0p05=need_n)
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[audit] wrote {out}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=80)
    ap.add_argument("--n_dir_boot", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_domains", type=int, default=8)
    ap.add_argument("--concept_domains", type=int, default=3)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--label", type=str, default="DEVELOPMENT")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run(seeds=args.seeds, n_boot=args.n_boot, n_dir_boot=args.n_dir_boot, alpha=args.alpha,
        n_domains=args.n_domains, concept_domains=args.concept_domains,
        seed_offset=args.seed_offset, label=args.label, out=args.out)
