"""
CSC Route B3-P2.4c — studentized subject-consistency gate EVALUATION. Method LOCKED (P2.4b fixed-margin
null + all guards); the ONLY change is the confirmation rule: CONCEPT_CONFIRMED now requires fixed-margin
mean-T p<=alpha AND studentized subject-consistency p<=alpha AND 95% LCB(delta_s)>0. Same grid as P2.4b:
CONTROLS 7x6x m{0,20,30} x48 (seed 1000) + POWER 3x6x m{0,20,30} x24 (seed 2000). Reports the OLD (mean-T
only) vs NEW (studentized-gated) decision delta -- the gate should remove noise-label confirmations, NOT
real concept positives. DEVELOPMENT diagnostic only; NO freeze/confirmatory/real-EEG.

  python -m csc.mininfo.run_b3_p24c --jobs 24 --out csc/results/b3_p24c.json
  python -m csc.mininfo.run_b3_p24c --canary
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

from csc.protocol import _cp_bound
from .paired_sim import PAIRED_TRUTH
from .paired_calibrated import CALIBRATION_VERSION
from .paired_certifier import CONCEPT_CONFIRMED
from .run_b3_p23 import CONTROLS, POSITIVES, SCENARIOS
from .run_b3_p24 import _one_cell, METHOD_LOCK

HARD_FAIL_CELL_FC = 6


def run(control_clusters=48, power_clusters=24, ms=(0, 20, 30), n_subjects=36, n_boot=200,
        control_seed=1000, power_seed=2000, n_jobs=1, out=None, quiet=True):
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

    dms = [m for m in ms if m >= 20]
    NEW = lambda r: r["state"] == CONCEPT_CONFIRMED
    OLD = lambda r: r.get("old_decision_without_studentized_gate") == CONCEPT_CONFIRMED

    def pooled(group, pred, conf):
        rs = [r for r in recs if r["label_budget_m"] in dms and r["truth"] == group and pred(r)]
        n = len(rs); fc = sum(conf(r) for r in rs)
        return dict(n=n, fc=fc, rate=(fc / n if n else None),
                    cp_upper=(_cp_bound(fc, n, side="upper") if n else 1.0))
    ck = sorted({r["kind"] for r in recs if r["truth"] == "NO_CONCEPT"})
    pk = sorted({r["kind"] for r in recs if r["truth"] == "CONCEPT"})
    by_kind_new = {k: pooled("NO_CONCEPT", lambda r, k=k: r["kind"] == k, NEW) for k in ck}
    by_kind_old = {k: pooled("NO_CONCEPT", lambda r, k=k: r["kind"] == k, OLD) for k in ck}
    pooled_all_new = pooled("NO_CONCEPT", lambda r: True, NEW)
    pooled_all_old = pooled("NO_CONCEPT", lambda r: True, OLD)
    pow_new = {k: pooled("CONCEPT", lambda r, k=k: r["kind"] == k, NEW) for k in pk}
    pow_old = {k: pooled("CONCEPT", lambda r, k=k: r["kind"] == k, OLD) for k in pk}

    # hard-flags on NEW gate (decision budgets)
    dcells = {}
    for r in recs:
        if r["truth"] == "NO_CONCEPT" and r["label_budget_m"] in dms:
            key = f"{r['scenario']}|{r['kind']}|m{r['label_budget_m']}"
            dcells.setdefault(key, []).append(r)
    cell_fc = {c: sum(NEW(r) for r in rs) for c, rs in dcells.items()}
    hard = []
    for k in ("missing_pair", "unequal_epochs_extreme"):
        if any(cell_fc[c] > 0 for c in cell_fc if c.split("|")[1] == k):
            hard.append(f"{k} CONFIRM leak")
    big = [c for c, v in cell_fc.items() if v >= HARD_FAIL_CELL_FC]
    if big:
        hard.append(f"cell>={HARD_FAIL_CELL_FC}/48: {big}")
    for k in ("random_label", "paired_label", "paired_covariate_plus_label"):
        if by_kind_new[k]["rate"] and by_kind_new[k]["rate"] > 0.05 and by_kind_new[k]["cp_upper"] > 0.05:
            hard.append(f"{k} pooled {by_kind_new[k]['fc']}/{by_kind_new[k]['n']}="
                        f"{by_kind_new[k]['rate']:.3f} (CP-up {by_kind_new[k]['cp_upper']:.3f}) > alpha")

    payload = dict(kind="CSC Route B3-P2.4c studentized subject-consistency gate eval (pc_centered_calibrated)",
                   status="DEVELOPMENT diagnostic only -- NOT error control; NO freeze/confirmatory/real-EEG.",
                   method_lock=METHOD_LOCK, calibration_version=CALIBRATION_VERSION,
                   control_clusters=control_clusters, power_clusters=power_clusters, label_budgets=list(ms),
                   decision_budgets=dms, n_boot=n_boot, hard_fail_cell_fc=HARD_FAIL_CELL_FC, hard_flags=hard,
                   pooled_all_controls_new=pooled_all_new, pooled_all_controls_old=pooled_all_old,
                   by_kind_controls_new=by_kind_new, by_kind_controls_old=by_kind_old,
                   power_new=pow_new, power_old=pow_old, per_cluster=recs)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[b3p24c] wrote {out}")
    print(f"=== B3-P2.4c CONTROLS (48/cell; m={dms}; NEW studentized gate vs OLD mean-T) ===")
    print(f"pooled ALL: NEW {pooled_all_new['fc']}/{pooled_all_new['n']}={pooled_all_new['rate']:.4f} "
          f"(CP-up {pooled_all_new['cp_upper']:.4f}) | OLD {pooled_all_old['rate']:.4f}   alpha=0.05")
    for k in ck:
        bn, bo = by_kind_new[k], by_kind_old[k]
        print(f"  {k:26s} NEW {bn['fc']:>3d}/{bn['n']}={bn['rate']:.4f} (CP-up {bn['cp_upper']:.3f}) | "
              f"OLD {bo['rate']:.4f}  {'>ALPHA' if bn['rate'] and bn['rate'] > 0.05 else ''}")
    print(f"HARD-FAIL flags: {hard if hard else 'NONE'}")
    print("=== POWER (NEW gate | OLD) ===")
    for k in pk:
        print(f"  {k:26s} NEW {pow_new[k]['rate']:.3f} (CP-low "
              f"{_cp_bound(pow_new[k]['fc'], pow_new[k]['n'], side='lower'):.3f}) | OLD {pow_old[k]['rate']:.3f}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="CSC Route B3-P2.4c studentized-gate evaluation.")
    ap.add_argument("--control_clusters", type=int, default=48)
    ap.add_argument("--power_clusters", type=int, default=24)
    ap.add_argument("--ms", type=int, nargs="+", default=[0, 20, 30])
    ap.add_argument("--n_subjects", type=int, default=36)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--out", type=str, default="csc/results/b3_p24c.json")
    a = ap.parse_args()
    cc, pc = (4, 4) if a.canary else (a.control_clusters, a.power_clusters)
    run(control_clusters=cc, power_clusters=pc, ms=tuple(a.ms), n_subjects=a.n_subjects, n_boot=a.n_boot,
        n_jobs=a.jobs, out=(a.out.replace(".json", "_canary.json") if a.canary else a.out))


if __name__ == "__main__":
    main()
