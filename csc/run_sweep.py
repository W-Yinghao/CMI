"""
csc.run_sweep — DEVELOPMENT lexicographic config selection (CSC-P1.3 #4/#7).

Sweeps a small grid of ProtocolConfigs through the FROZEN path and selects ONE config by the
pre-registered LEXICOGRAPHIC rule, using GENERATOR TRUTH only (we built the synthetic targets,
so we know each one's class). The oracle is NOT used to choose certificate parameters -- doing
so would tune predictor and ground truth together. The oracle's job (csc.calibration.lodo) is
only an estimator sanity-check on the null bank, with its bands FROZEN INDEPENDENTLY.

Selection order (a candidate to CARRY INTO the freeze, not a freeze itself):
  1. minimise the full-suite forbidden count (exact CP UB <= alpha is the FREEZE gate, which
     needs >= ~59 confirmatory clusters -- unreachable with dev seeds, so we minimise here);
  2. then maximise the WORST-CASE visible-concept power (ood_power_bank);
  3. then maximise covariate-compatible coverage.

DEVELOPMENT seeds only. The winner is a CANDIDATE; freezing requires the confirmatory run on
a separate unseen seed set.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import warnings
from math import comb
import numpy as np

from csc.protocol import ProtocolConfig, ood_power_bank
from csc.run_synthetic import run as run_syn


def _cp_upper(failures, n, conf=0.95):
    if n == 0:
        return 1.0
    if failures >= n:
        return 1.0
    lo, hi = failures / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf = sum(comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(failures + 1))
        hi, lo = (mid, lo) if cdf < 1 - conf else (hi, mid)
    return hi


def _grid(base: ProtocolConfig):
    out = []
    for consensus in (0.80, 0.90):
        for kappa in (1.3, 1.7):
            out.append(dataclasses.replace(base, consensus=consensus,
                                           cov_loading_margin_kappa=kappa))
    return out


def sweep(seeds=6, base: ProtocolConfig = None, out=None, quiet=True):
    if quiet:
        warnings.filterwarnings("ignore")
    base = base or ProtocolConfig(n_boot=20, n_dir_boot=80, target_n_boot=80,
                                  tau_n_pseudotargets=100)
    seed_list = list(range(seeds))
    results = []
    for cfg in _grid(base):
        syn = run_syn(seeds=seeds, cfg=cfg, label="DEV-SWEEP", quiet=True)
        ood = ood_power_bank(cfg, seed_list, min_visible=max(2, seeds // 2))
        worst_power = ood["concept_power"] if ood["concept_power"] is not None else 0.0
        cov_cov = ood["covariate_compatible_coverage"] or 0.0
        results.append(dict(
            manifest_hash=cfg.hash(), consensus=cfg.consensus,
            kappa=cfg.cov_loading_margin_kappa,
            forbidden_full=syn["any_forbidden_full_suite"],
            cp_ub_full=syn["exact_cp_ub_full_suite"],
            concept_power=worst_power, cov_coverage=cov_cov,
            ood_valid=ood["evaluable"], config=cfg.manifest()))

    # lexicographic: (forbidden asc, -power, -coverage)
    ranked = sorted(results, key=lambda r: (r["forbidden_full"], -r["concept_power"],
                                            -r["cov_coverage"]))
    winner = ranked[0]
    print(f"\n=== DEV sweep ({seeds} seeds) — lexicographic winner ===")
    for r in ranked:
        mark = "  <== WINNER" if r is winner else ""
        print(f"  cons={r['consensus']:.2f} kappa={r['kappa']:.1f} | forbidden_full="
              f"{r['forbidden_full']}/{seeds} (UB {r['cp_ub_full']:.2f}) power={r['concept_power']:.2f}"
              f" cov={r['cov_coverage']:.2f}{mark}")
    print(f"\nCANDIDATE manifest={winner['manifest_hash']} "
          f"(consensus={winner['consensus']}, kappa={winner['kappa']}). "
          f"NOT frozen: CP UB<=alpha is the freeze gate, needs the confirmatory unseen run.")
    payload = dict(seeds=seeds, base_manifest=base.hash(), grid=results,
                   winner_manifest=winner["manifest_hash"], winner=winner["config"],
                   selection_rule="lexicographic: forbidden_full asc, power desc, coverage desc",
                   note="DEVELOPMENT candidate; freeze requires confirmatory unseen-seed run")
    if out:
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[sweep] wrote {out}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    sweep(seeds=args.seeds, out=args.out)
