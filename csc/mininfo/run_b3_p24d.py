"""
CSC Route B3-P2.4d — cross-budget alpha-spending EVALUATION. Method LOCKED (P2.4c: pc_centered + fixed-
margin null + studentized subject-consistency gate); the ONLY change is alpha_budget = alpha_family/2 =
0.025 on the p-gates AND the LCB at 1-alpha_budget=0.975. Same grid (controls 48/cell, power 24/cell).
Reports P2.4c (alpha=0.05) vs P2.4d (alpha=0.025) decision delta. DEVELOPMENT diagnostic; NO
freeze/confirmatory/real-EEG.

  python -m csc.mininfo.run_b3_p24d --jobs 24 --out csc/results/b3_p24d.json
  python -m csc.mininfo.run_b3_p24d --canary
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

from csc.protocol import _cp_bound
from .paired_calibrated import CALIBRATION_VERSION
from .paired_certifier import CONCEPT_CONFIRMED
from .run_b3_p23 import CONTROLS, POSITIVES, SCENARIOS
from .run_b3_p24 import _one_cell, METHOD_LOCK

HARD_FAIL_CELL_FC = 6


def run(control_clusters=48, power_clusters=24, ms=(0, 20, 30), n_subjects=36, n_boot=200,
        control_seed=3000, power_seed=4000, n_jobs=1, out=None, quiet=True):
    if quiet:
        warnings.filterwarnings("ignore")
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, "1")
    tasks = [(k, sn, SCENARIOS[sn], m, control_seed + c)
             for sn in SCENARIOS for k in CONTROLS for m in ms for c in range(control_clusters)]
    tasks += [(k, sn, SCENARIOS[sn], m, power_seed + c)
              for sn in SCENARIOS for k in POSITIVES for m in ms for c in range(power_clusters)]
    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=n_jobs)(
            delayed(_one_cell)(k, sn, sp, m, s, n_subjects, n_boot) for k, sn, sp, m, s in tasks)
    else:
        recs = [_one_cell(k, sn, sp, m, s, n_subjects, n_boot) for k, sn, sp, m, s in tasks]

    NEW = lambda r: r["state"] == CONCEPT_CONFIRMED                       # P2.4d
    OLD = lambda r: r.get("old_p24c_decision") == CONCEPT_CONFIRMED       # P2.4c

    def pkrate(truth, kind, mm, pred):
        rs = [r for r in recs if r["truth"] == truth and r["kind"] == kind and r["label_budget_m"] == mm]
        n = len(rs); fc = sum(pred(r) for r in rs)
        return n, fc, (fc / n if n else None), (_cp_bound(fc, n, side="upper") if n else 1.0)
    ck = sorted({r["kind"] for r in recs if r["truth"] == "NO_CONCEPT"})
    pk = sorted({r["kind"] for r in recs if r["truth"] == "CONCEPT"})

    # per-budget control table (stratified -- the view the m=30 edge needs)
    ctrl_tab = {}
    for mm in (20, 30):
        for k in ck:
            n, fc, rate, cpu = pkrate("NO_CONCEPT", k, mm, NEW)
            _, fco, rateo, _ = pkrate("NO_CONCEPT", k, mm, OLD)
            ctrl_tab[f"{k}|m{mm}"] = dict(n=n, new_fc=fc, new_rate=rate, new_cp_upper=cpu, old_rate=rateo)
    pooled_new = sum(r["state"] == CONCEPT_CONFIRMED for r in recs
                     if r["truth"] == "NO_CONCEPT" and r["label_budget_m"] >= 20)
    pooled_n = sum(1 for r in recs if r["truth"] == "NO_CONCEPT" and r["label_budget_m"] >= 20)
    pow_tab = {}
    for k in pk:
        for mm in (20, 30):
            n, fc, rate, _ = pkrate("CONCEPT", k, mm, NEW)
            _, fco, rateo, _ = pkrate("CONCEPT", k, mm, OLD)
            pow_tab[f"{k}|m{mm}"] = dict(n=n, new_rate=rate, old_rate=rateo)

    # hard-flags on P2.4d per-budget (any control kind point > alpha at m20 OR m30)
    hard = []
    for kk, v in ctrl_tab.items():
        k = kk.split("|")[0]
        if k in ("missing_pair", "unequal_epochs_extreme") and v["new_fc"] > 0:
            hard.append(f"{kk} CONFIRM leak")
        if v["new_rate"] and v["new_rate"] > 0.05:
            hard.append(f"{kk} point {v['new_rate']:.3f} > alpha")
    # gate removals P2.4c->P2.4d
    rem_ctrl = sum(1 for r in recs if r["truth"] == "NO_CONCEPT" and r["label_budget_m"] >= 20
                   and OLD(r) and not NEW(r))
    rem_prim = sum(1 for r in recs if r["truth"] == "CONCEPT" and r["label_budget_m"] >= 20
                   and r["kind"] in ("paired_concept", "paired_concept_plus_cov") and OLD(r) and not NEW(r))
    rem_sec = sum(1 for r in recs if r["truth"] == "CONCEPT" and r["label_budget_m"] >= 20
                  and r["kind"] == "paired_pure_conditional" and OLD(r) and not NEW(r))

    payload = dict(kind="CSC Route B3-P2.4d cross-budget alpha-spending eval (pc_centered_calibrated)",
                   status="DEVELOPMENT diagnostic only -- NOT error control; NO freeze/confirmatory/real-EEG.",
                   method_lock=METHOD_LOCK, calibration_version=CALIBRATION_VERSION,
                   alpha_family=0.05, alpha_budget=0.025, lcb_level=0.975, positive_decision_budgets=[20, 30],
                   control_clusters=control_clusters, power_clusters=power_clusters, n_boot=n_boot,
                   control_seed=control_seed, power_seed=power_seed, hard_flags=hard,
                   pooled_new_controls=dict(fc=pooled_new, n=pooled_n, rate=pooled_new / pooled_n,
                                            cp_upper=_cp_bound(pooled_new, pooled_n, side="upper")),
                   control_table=ctrl_tab, power_table=pow_tab,
                   removed_controls=rem_ctrl, removed_primary=rem_prim, removed_secondary=rem_sec,
                   per_cluster=recs)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[b3p24d] wrote {out}")
    print(f"=== B3-P2.4d CONTROLS (48/cell; alpha_budget=0.025; per-budget NEW vs OLD=P2.4c) ===")
    print(f"pooled NEW {pooled_new}/{pooled_n}={pooled_new/pooled_n:.4f} "
          f"(CP-up {_cp_bound(pooled_new,pooled_n,side='upper'):.4f})   alpha_family=0.05 alpha_budget=0.025")
    for mm in (20, 30):
        print(f"-- m={mm} --")
        for k in ck:
            v = ctrl_tab[f"{k}|m{mm}"]
            print(f"  {k:26s} NEW {v['new_rate']:.4f} (CP-up {v['new_cp_upper']:.4f}) | OLD {v['old_rate']:.4f}"
                  f"  {'>A' if v['new_rate'] and v['new_rate']>0.05 else ''}")
    print(f"HARD-FAIL flags: {hard if hard else 'NONE'}")
    print(f"gate removals P2.4c->P2.4d (m>=20): controls={rem_ctrl} primary={rem_prim} secondary={rem_sec}")
    print("=== POWER (NEW P2.4d | OLD P2.4c) ===")
    for k in pk:
        print(f"  {k:26s} m20 {pow_tab[f'{k}|m20']['new_rate']:.3f}|{pow_tab[f'{k}|m20']['old_rate']:.3f}  "
              f"m30 {pow_tab[f'{k}|m30']['new_rate']:.3f}|{pow_tab[f'{k}|m30']['old_rate']:.3f}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="CSC Route B3-P2.4d cross-budget alpha-spending evaluation.")
    ap.add_argument("--control_clusters", type=int, default=48)
    ap.add_argument("--power_clusters", type=int, default=24)
    ap.add_argument("--ms", type=int, nargs="+", default=[0, 20, 30])
    ap.add_argument("--n_subjects", type=int, default=36)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--control_seed", type=int, default=3000)
    ap.add_argument("--power_seed", type=int, default=4000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--out", type=str, default="csc/results/b3_p24d.json")
    a = ap.parse_args()
    cc, pc = (4, 4) if a.canary else (a.control_clusters, a.power_clusters)
    run(control_clusters=cc, power_clusters=pc, ms=tuple(a.ms), n_subjects=a.n_subjects, n_boot=a.n_boot,
        control_seed=a.control_seed, power_seed=a.power_seed, n_jobs=a.jobs,
        out=(a.out.replace(".json", "_canary.json") if a.canary else a.out))


if __name__ == "__main__":
    main()
