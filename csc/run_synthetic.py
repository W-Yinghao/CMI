"""
csc.run_synthetic — multi-seed DEVELOPMENT evaluation of the FROZEN-PATH certificate (CSC-P1.3).

EVERY certificate here goes through csc.protocol.run_frozen_protocol -- the SAME calibrated
robust path (analyze_source -> source-only calibrate_thresholds -> certify_robust) that any
confirmatory/freeze run uses. (The v0 called certify(...) directly, so its numbers did not
characterise the deployable algorithm.)

Two cluster-level (per independent source seed) FORBIDDEN endpoints, with EXACT one-sided 95%
Clopper-Pearson upper bounds:
  any_false_positive_must_abstain : a forbidden outcome on clean/pure/label/label_cov
  any_forbidden_full_suite        : ANY forbidden outcome across ALL shift kinds
                                    (covariate->CONCEPT_SUSPECT and visible-concept->
                                     COVARIATE_COMPATIBLE are forbidden too)
The confirmatory PRIMARY endpoint is `any_forbidden_full_suite`. concept power and covariate
coverage are DESCRIPTIVE here; deployment power is validated by csc.protocol.ood_power_bank.

DEVELOPMENT seeds informed tuning; a frozen confirmatory run needs a SEPARATE unseen seed set.

  conda run -n icml python -m csc.run_synthetic --seeds 30 --out audit.json
"""
from __future__ import annotations

import argparse
import json
import warnings
from math import comb, log
import numpy as np

from csc.sim.shift_simulator import SimConfig, make_source, make_target, _TRUTH
from csc.protocol import ProtocolConfig, run_frozen_protocol
from csc.certificate import FORBIDDEN, UNIDENTIFIABLE, COVARIATE_COMPATIBLE, CONCEPT_SUSPECT

STATES = [COVARIATE_COMPATIBLE, CONCEPT_SUSPECT, UNIDENTIFIABLE]
KINDS = list(_TRUTH)
MUST_ABSTAIN = ["clean", "pure_conditional", "label_shift", "label_covariate_mixed"]


def _clopper_pearson_upper(failures, n, conf=0.95):
    """EXACT one-sided (upper) Clopper-Pearson bound. failures=0 -> 1-(1-conf)^(1/n)
    (0.259 at n=10); the Rule-of-Three 3/n=0.30 is only its approximation."""
    if n == 0:
        return 1.0
    if failures >= n:
        return 1.0
    lo, hi = failures / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = sum(comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(failures + 1))
        if cdf < 1 - conf:
            hi = mid
        else:
            lo = mid
    return hi


def run(seeds=30, cfg: ProtocolConfig = None, seed_offset=0, label="DEVELOPMENT",
        out=None, quiet=True):
    if quiet:
        warnings.filterwarnings("ignore")
    cfg = cfg or ProtocolConfig()
    confusion = {k: {s: 0 for s in STATES} for k in KINDS}
    forbidden = {k: 0 for k in KINDS}
    seed_list = list(range(seed_offset, seed_offset + seeds))
    fp_must_abstain = 0          # per-seed: any forbidden on a must-abstain kind
    forbidden_full = 0           # per-seed: any forbidden across the FULL suite

    for s in seed_list:
        scfg = SimConfig(seed=s)
        src = make_source(scfg, n_domains=8, concept_domains=3, seed=s)
        any_fp_ma, any_forb = False, False
        for kind in KINDS:
            tb = make_target(kind, scfg, geom=src.geom, seed=1000 + s)
            state = run_frozen_protocol(src.Z, src.Y, src.D, tb.Z, cfg,
                                        src_group_ids=src.group_ids,
                                        tgt_group_ids=tb.group_ids, seed=s)["certificate"].state
            confusion[kind][state] += 1
            if state in FORBIDDEN[tb.truth]:
                forbidden[kind] += 1
                any_forb = True
                if kind in MUST_ABSTAIN:
                    any_fp_ma = True
        fp_must_abstain += int(any_fp_ma)
        forbidden_full += int(any_forb)

    n = seeds
    ub_full = _clopper_pearson_upper(forbidden_full, n)
    ub_ma = _clopper_pearson_upper(fp_must_abstain, n)
    power = confusion["boundary_coupled"][CONCEPT_SUSPECT] / n
    cov_ok = confusion["covariate"][COVARIATE_COMPATIBLE] / n
    need = int(np.ceil(log(0.05) / log(1 - cfg.alpha))) if 0 < cfg.alpha < 1 else 1

    print(f"\n=== csc FROZEN-PATH certificate [{label}] — {seeds} seeds "
          f"{seed_list[0]}..{seed_list[-1]}, manifest={cfg.hash()} ===")
    hdr = f"{'truth / shift kind':30s}" + "".join(f"{s.split('_')[0][:6]:>8s}" for s in STATES) + "  forbid"
    print(hdr); print("-" * len(hdr))
    for kind in KINDS:
        row = f"{kind+' ('+_TRUTH[kind]+')':30s}"
        for st in STATES:
            row += f"{confusion[kind][st]:>8d}"
        row += f"{forbidden[kind]:>8d}"
        print(row)
    print("\n--- cluster-level (per independent seed) endpoints, exact 95% Clopper-Pearson ---")
    print(f"PRIMARY any_forbidden_full_suite          : {forbidden_full}/{n}  -> UB {ub_full:.3f}")
    print(f"        any_false_positive_must_abstain   : {fp_must_abstain}/{n}  -> UB {ub_ma:.3f}")
    print(f"descriptive: concept power {power:.2f} | covariate->COMPATIBLE {cov_ok:.2f}")
    print(f"NOTE: DEVELOPMENT seeds (informed tuning). >= {need} independent clusters needed "
          f"for a 0.05 bound at 0 failures. Power is validated by ood_power_bank, not here.")

    result = dict(label=label, manifest_hash=cfg.hash(), protocol=cfg.manifest(),
                  seeds=seeds, seed_list=seed_list,
                  confusion={k: dict(v) for k, v in confusion.items()},
                  forbidden_per_kind=dict(forbidden),
                  any_forbidden_full_suite=forbidden_full,
                  any_false_positive_must_abstain=fp_must_abstain,
                  exact_cp_ub_full_suite=ub_full,
                  exact_cp_ub_must_abstain=ub_ma,
                  descriptive_concept_power=power,
                  descriptive_covariate_coverage=cov_ok,
                  clusters_needed_for_0p05=need,
                  primary_endpoint="any_forbidden_full_suite")
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[audit] wrote {out}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=40)
    ap.add_argument("--n_dir_boot", type=int, default=120)
    ap.add_argument("--target_n_boot", type=int, default=120)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--label", type=str, default="DEVELOPMENT")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    cfg = ProtocolConfig(n_boot=args.n_boot, n_dir_boot=args.n_dir_boot,
                         target_n_boot=args.target_n_boot)
    run(seeds=args.seeds, cfg=cfg, seed_offset=args.seed_offset, label=args.label, out=args.out)
