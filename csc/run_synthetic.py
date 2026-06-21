"""
csc.run_synthetic — multi-seed evaluation of the three-state certificate (CSC-P0).

Per seed: analyze the source (atlas + residual evidence + direction-linked concept
pruning), then certify a fresh target of EACH shift type. Reports the pre-registered
quantities (PREREGISTRATION.md) over the taxonomy:

  clean / pure_conditional / label_shift / label_covariate_mixed -> must ABSTAIN
  covariate                                                      -> COVARIATE_COMPATIBLE
  boundary_coupled (visible concept)                            -> CONCEPT_SUSPECT

HONESTY: a run of N zero-failure seeds only bounds the false-certification rate by the
binomial upper limit 1-(alpha_CI)^(1/N) ~ 3/N (Rule of Three). This script PRINTS that
bound; it does not claim "rate <= 0.05" from a handful of seeds.

  conda run -n icml python -m csc.run_synthetic --seeds 30 --n_boot 100
"""
from __future__ import annotations

import argparse
import warnings
import numpy as np

from csc.sim.shift_simulator import SimConfig, make_source, make_target, _TRUTH
from csc.certificate import (
    analyze_source, certify, CertifierConfig,
    FORBIDDEN, UNIDENTIFIABLE, COVARIATE_COMPATIBLE, CONCEPT_SUSPECT,
)

STATES = [COVARIATE_COMPATIBLE, CONCEPT_SUSPECT, UNIDENTIFIABLE]
KINDS = list(_TRUTH)
MUST_ABSTAIN = ["clean", "pure_conditional", "label_shift", "label_covariate_mixed"]


def _rule_of_three(failures, n, conf=0.95):
    """One-sided upper confidence bound on a rate given `failures`/`n`."""
    if n == 0:
        return 1.0
    if failures == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)         # = 3/n at conf=0.95 (Rule of Three)
    # crude Clopper-Pearson-ish upper bound via the same exponent form
    from math import comb
    # bisection on p for P(X<=failures) = 1-conf
    lo, hi = failures / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf = sum(comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(failures + 1))
        if cdf < 1 - conf:
            hi = mid
        else:
            lo = mid
    return hi


def run(seeds=30, n_boot=100, n_dir_boot=200, alpha=0.05,
        n_domains=8, concept_domains=3, quiet=True):
    if quiet:
        warnings.filterwarnings("ignore")
    confusion = {k: {s: 0 for s in STATES} for k in KINDS}
    forbidden = {k: 0 for k in KINDS}
    sig_count = 0

    for s in range(seeds):
        cfg = SimConfig(seed=s)
        src = make_source(cfg, n_domains=n_domains, concept_domains=concept_domains, seed=s)
        sa = analyze_source(src.Z, src.Y, src.D, n_boot=n_boot, n_dir_boot=n_dir_boot,
                            alpha=alpha, seed=s)
        sig_count += int(sa.concept_evidenced)
        for kind in KINDS:
            tb = make_target(kind, cfg, geom=src.geom, seed=1000 + s)
            c = certify(sa, tb.Z, CertifierConfig())
            confusion[kind][c.state] += 1
            if c.state in FORBIDDEN[tb.truth]:
                forbidden[kind] += 1

    print(f"\n=== csc certificate (CSC-P0) — {seeds} seeds, n_boot={n_boot}, alpha={alpha} ===")
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
    ub = _rule_of_three(abstain_forbidden, seeds * len(MUST_ABSTAIN))

    print("\n--- pre-registered quantities (synthetic) ---")
    print(f"false certifications, total                 : {tot_forbidden}"
          f"  (across {seeds*len(KINDS)} certificates)")
    print(f"false 'safe/suspect' on must-abstain shifts : {abstain_forbidden}/"
          f"{seeds*len(MUST_ABSTAIN)}  -> 95% upper bound on rate = {ub:.3f}")
    print(f"power on VISIBLE concept (boundary_coupled) : {power:.3f}")
    print(f"covariate -> COVARIATE_COMPATIBLE           : {cov_ok:.3f}")
    print(f"\nNOTE: zero observed failures does NOT prove rate<=alpha; the upper bound above "
          f"is what {seeds} seeds can support (Rule of Three).")
    return dict(confusion=confusion, forbidden=forbidden, power=power, cov_ok=cov_ok,
                false_cert_total=tot_forbidden, abstain_ub=ub)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=80)
    ap.add_argument("--n_dir_boot", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_domains", type=int, default=8)
    ap.add_argument("--concept_domains", type=int, default=3)
    args = ap.parse_args()
    run(seeds=args.seeds, n_boot=args.n_boot, n_dir_boot=args.n_dir_boot, alpha=args.alpha,
        n_domains=args.n_domains, concept_domains=args.concept_domains)
