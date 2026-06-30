"""
CSC Route B3-P2.4b — fixed-margin-null EVALUATION. Method LOCKED `pc_centered_calibrated`; ONLY the null
is the condition-matched FIXED-MARGIN h0 bootstrap. Two phases in one run:
  CONTROLS  : 7 control kinds x 6 scenarios x m{0,20,30} x 48 fresh clusters (base_seed 1000, == P2.4a's
              clusters so the standard-vs-fixed-margin comparison is on identical data).
  POWER     : 3 positive kinds x 6 scenarios x m{0,20,30} x 24 clusters (base_seed 2000).
DEVELOPMENT diagnostic only (48/cell is NOT error control). NO freeze/confirmatory/real-EEG.

  python -m csc.mininfo.run_b3_p24b --jobs 24 --out csc/results/b3_p24b.json
  python -m csc.mininfo.run_b3_p24b --canary
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
from .paired_certifier import CONCEPT_CONFIRMED, NEED_MORE_LABELS, INVALID_PAIR
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
    ctrl = [r for r in recs if r["truth"] == "NO_CONCEPT"]
    pos = [r for r in recs if r["truth"] == "CONCEPT"]

    def cells_for(group):
        out = {}
        for r in group:
            key = f"{r['scenario']}|{r['kind']}|m{r['label_budget_m']}"
            out.setdefault(key, []).append(r)
        agg = {}
        for key, rs in out.items():
            n = len(rs); fc = sum(x["state"] == CONCEPT_CONFIRMED for x in rs)
            tr = rs[0]["truth"]
            d = dict(scenario=rs[0]["scenario"], kind=rs[0]["kind"], m=rs[0]["label_budget_m"],
                     truth=tr, n=n, confirm_count=fc, confirm_rate=fc / n if n else None,
                     would_std=sum(bool(x["would_confirm_under_standard_null"]) for x in rs))
            if tr == "NO_CONCEPT":
                d["false_confirm_cp_upper"] = _cp_bound(fc, n, side="upper") if n else 1.0
            else:
                d["power_cp_lower"] = _cp_bound(fc, n, side="lower") if n else 0.0
            agg[key] = d
        return agg
    ctrl_cells = cells_for(ctrl); pos_cells = cells_for(pos)

    def pooled(group, pred):
        rs = [r for r in group if r["label_budget_m"] in dms and pred(r)]
        n = len(rs); fc = sum(r["state"] == CONCEPT_CONFIRMED for r in rs)
        ws = sum(bool(r["would_confirm_under_standard_null"]) for r in rs)
        return dict(n=n, fc=fc, rate=(fc / n if n else None),
                    cp_upper=(_cp_bound(fc, n, side="upper") if n else 1.0),
                    would_std=ws, would_std_rate=(ws / n if n else None))
    pooled_all = pooled(ctrl, lambda r: True)
    by_kind = {k: pooled(ctrl, lambda r, k=k: r["kind"] == k) for k in CONTROLS}

    # hard-flags on the 48-cluster controls (decision budgets)
    dc = {c: v for c, v in ctrl_cells.items() if v["m"] in dms}
    hard = []
    for k in ("missing_pair", "unequal_epochs_extreme"):
        leak = [c for c, v in dc.items() if v["kind"] == k and v["confirm_count"] > 0]
        if leak:
            hard.append(f"{k} CONFIRM leak {leak}")
    big = [c for c, v in dc.items() if v["confirm_count"] >= HARD_FAIL_CELL_FC]
    if big:
        hard.append(f"cell>={HARD_FAIL_CELL_FC}/48 {big}")
    for k in ("random_label", "paired_label", "paired_covariate_plus_label"):
        hot = [c for c, v in dc.items() if v["kind"] == k and v["confirm_count"] >= 3]
        if len(hot) >= 2:
            hard.append(f"{k} kind-concentrated {hot}")
        if by_kind[k]["rate"] is not None and by_kind[k]["rate"] > 0.05 and by_kind[k]["cp_upper"] > 0.05:
            hard.append(f"{k} pooled {by_kind[k]['fc']}/{by_kind[k]['n']}={by_kind[k]['rate']:.3f} > alpha")
    for k in ("clean", "paired_covariate"):
        rep = [c for c, v in dc.items() if v["kind"] == k and v["confirm_count"] >= 1]
        if len(rep) >= 3:
            hard.append(f"{k} repeated {rep}")

    payload = dict(kind="CSC Route B3-P2.4b fixed-margin-null evaluation (pc_centered_calibrated)",
                   status="DEVELOPMENT diagnostic only -- NOT error control; NO freeze/confirmatory/real-EEG.",
                   method_lock=METHOD_LOCK, calibration_version=CALIBRATION_VERSION,
                   null_version="condition_matched_fixed_margin_h0_bootstrap",
                   control_clusters=control_clusters, power_clusters=power_clusters, label_budgets=list(ms),
                   decision_budgets=dms, n_boot=n_boot, control_seed=control_seed, power_seed=power_seed,
                   pooled_all_controls=pooled_all, by_kind_controls=by_kind, hard_fail_cell_fc=HARD_FAIL_CELL_FC,
                   hard_flags=hard, control_cells=ctrl_cells, power_cells=pos_cells, per_cluster=recs)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[b3p24b] wrote {out}")
    print(f"=== B3-P2.4b CONTROLS (48/cell; decision m={dms}; FIXED-MARGIN null) ===")
    print(f"pooled ALL controls: {pooled_all['fc']}/{pooled_all['n']} = {pooled_all['rate']:.4f} "
          f"(CP-up {pooled_all['cp_upper']:.4f})  [vs standard-null would: {pooled_all['would_std']}/"
          f"{pooled_all['n']} = {pooled_all['would_std_rate']:.4f}]   alpha=0.05")
    print("by-kind pooled (fixed-margin / standard-would):")
    for k in CONTROLS:
        b = by_kind[k]
        print(f"  {k:26s} {b['fc']:>3d}/{b['n']:<4d}={b['rate']:.4f} (CP-up {b['cp_upper']:.3f})  "
              f"std-would {b['would_std']}/{b['n']}={b['would_std_rate']:.4f}")
    print(f"HARD-FAIL flags: {hard if hard else 'NONE'}")
    print(f"=== B3-P2.4b POWER ({power_clusters}/cell; FIXED-MARGIN null) ===")
    for k in POSITIVES:
        row = "  %-24s " % k
        row += "  ".join("%s m%d=%.2f" % (sn[:4], m, (pos_cells.get(f"{sn}|{k}|m{m}", {}).get("confirm_rate") or 0.0))
                         for sn in SCENARIOS for m in dms)
        print(row)
    return payload


def main():
    ap = argparse.ArgumentParser(description="CSC Route B3-P2.4b fixed-margin-null evaluation.")
    ap.add_argument("--control_clusters", type=int, default=48)
    ap.add_argument("--power_clusters", type=int, default=24)
    ap.add_argument("--ms", type=int, nargs="+", default=[0, 20, 30])
    ap.add_argument("--n_subjects", type=int, default=36)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--out", type=str, default="csc/results/b3_p24b.json")
    a = ap.parse_args()
    cc, pc = (4, 4) if a.canary else (a.control_clusters, a.power_clusters)
    run(control_clusters=cc, power_clusters=pc, ms=tuple(a.ms), n_subjects=a.n_subjects, n_boot=a.n_boot,
        n_jobs=a.jobs, out=(a.out.replace(".json", "_canary.json") if a.canary else a.out))


if __name__ == "__main__":
    main()
