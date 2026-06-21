"""
csc.run_synthetic — end-to-end multi-seed evaluation of the certificate.

Builds a source atlas + residual test per seed, then certifies a fresh target of each
shift type, and reports the two PRE-REGISTERED quantities (PREREGISTRATION.md):

  (A) FALSE-CERTIFICATION RATE on pure conditional (invisible) shift
        = P(certificate != UNIDENTIFIABLE | truth = CONCEPT_INVISIBLE)
      plus the *strict* false-certification rate (the FORBIDDEN map): how often a
      forbidden outcome (e.g. "safe" on a real concept shift) is emitted.
  (B) POWER on the visible-concept positive control
        = P(certificate == CONCEPT_SUSPECT | truth = CONCEPT_VISIBLE)

Run:
  conda run -n icml python -m csc.run_synthetic --seeds 30 --n_perm 100
"""
from __future__ import annotations

import argparse
import warnings
import numpy as np

from csc.sim.shift_simulator import SimConfig, make_source, make_target, _TRUTH
from csc.certificate import (
    build_atlas, residual_decoder_test, certify, CertifierConfig,
    ACCEPTABLE, FORBIDDEN, UNIDENTIFIABLE, COVARIATE_ADAPTABLE, CONCEPT_SUSPECT,
)

STATES = [COVARIATE_ADAPTABLE, CONCEPT_SUSPECT, UNIDENTIFIABLE]
KINDS = list(_TRUTH)


def run(seeds=30, n_perm=100, alpha=0.05, n_domains=8, concept_domains=3, quiet=True):
    if quiet:
        warnings.filterwarnings("ignore")
    confusion = {k: {s: 0 for s in STATES} for k in KINDS}
    forbidden_hits = {k: 0 for k in KINDS}
    sig_count = 0

    for s in range(seeds):
        cfg = SimConfig(seed=s)
        src = make_source(cfg, n_domains=n_domains, concept_domains=concept_domains, seed=s)
        atlas = build_atlas(src.Z, src.Y, src.D)
        rt = residual_decoder_test(src.Z, src.Y, src.D, n_perm=n_perm, alpha=alpha, seed=s)
        sig_count += int(rt.significant)
        for kind in KINDS:
            tb = make_target(kind, cfg, geom=src.geom, seed=1000 + s)
            cert = certify(atlas, rt, tb.Z, CertifierConfig())
            confusion[kind][cert.state] += 1
            if cert.state in FORBIDDEN[tb.truth]:
                forbidden_hits[kind] += 1

    print(f"\n=== csc certificate — {seeds} seeds, n_perm={n_perm}, alpha={alpha} ===")
    print(f"source residual test significant (concept atlas valid): "
          f"{sig_count}/{seeds} = {sig_count/seeds:.2f}\n")

    hdr = f"{'truth / shift kind':28s}" + "".join(f"{s.split('_')[0][:6]:>8s}" for s in STATES)
    print(hdr); print("-" * len(hdr))
    for kind in KINDS:
        row = f"{kind+' ('+_TRUTH[kind]+')':28s}"
        for st in STATES:
            row += f"{confusion[kind][st]:>8d}"
        print(row)

    # --- the two falsification numbers --------------------------------------------------
    inv = "pure_conditional"
    false_cert = 1.0 - confusion[inv][UNIDENTIFIABLE] / seeds
    strict_false = forbidden_hits[inv] / seeds
    power = confusion["boundary_coupled"][CONCEPT_SUSPECT] / seeds
    cov_ok = confusion["covariate"][COVARIATE_ADAPTABLE] / seeds
    clean_alarm = confusion["clean"][CONCEPT_SUSPECT] / seeds

    print("\n--- PRE-REGISTERED criteria ---")
    print(f"(A) false-certification on INVISIBLE concept shift : {false_cert:.3f}"
          f"   (target <= {alpha:.2f};  strict/forbidden = {strict_false:.3f})")
    print(f"(B) power on VISIBLE concept positive control       : {power:.3f}"
          f"   (target >= 0.80)")
    print(f"    covariate correctly ADAPTABLE                   : {cov_ok:.3f}")
    print(f"    false concept alarm on CLEAN                    : {clean_alarm:.3f}"
          f"   (target <= {alpha:.2f})")

    verdict = "PASS" if (strict_false <= alpha and power >= 0.80 and clean_alarm <= alpha) \
        else "FAIL"
    print(f"\nSYNTHETIC VERDICT: {verdict}  "
          f"(synthetic only — real-data PD ON/OFF + cross-site is the actual test)")
    return dict(confusion=confusion, false_cert=false_cert, strict_false=strict_false,
                power=power, cov_ok=cov_ok, clean_alarm=clean_alarm, verdict=verdict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_domains", type=int, default=8)
    ap.add_argument("--concept_domains", type=int, default=3)
    args = ap.parse_args()
    run(seeds=args.seeds, n_perm=args.n_perm, alpha=args.alpha,
        n_domains=args.n_domains, concept_domains=args.concept_domains)
