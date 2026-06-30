"""
CSC Route B3-P2.4a — CONTROL-RESOLUTION round. CONTROLS ONLY, 48 clusters/cell, to answer ONE question:
is P2.4's residual control false-confirm 24-cluster multiple-comparison NOISE, or method-level LEAKAGE?
Method LOCKED `pc_centered_calibrated` (no changes). DEVELOPMENT diagnostic only (48/cell is NOT
error-control: a 0/48 control still has CP-upper ~= 0.0605 > 0.05). NO freeze/confirmatory/real-EEG.

  python -m csc.mininfo.run_b3_p24a --clusters 48 --jobs 24 --out csc/results/b3_p24a_controls.json
  python -m csc.mininfo.run_b3_p24a --canary
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

from csc.protocol import _cp_bound
from .paired_sim import PAIRED_TRUTH
from .paired_calibrated import CALIBRATION_VERSION, PAIR_INTEGRITY_MIN, MIN_EPOCHS_PER_CONDITION, N_FOLDS
from .paired_certifier import CONCEPT_CONFIRMED, NEED_MORE_LABELS, INVALID_PAIR
from .run_b3_p23 import CONTROLS, SCENARIOS
from .run_b3_p24 import _one_cell, METHOD_LOCK

HARD_FAIL_CELL_FC = 6        # >=6/48 false-confirms in a cell: P(>=6|0.05)~=0.032 -> not ordinary noise


def run(clusters=48, ms=(0, 20, 30), n_subjects=36, n_boot=200, base_seed=1000, n_jobs=1, out=None,
        quiet=True):
    if quiet:
        warnings.filterwarnings("ignore")
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, "1")
    # CONTROLS only; fresh seeds (base_seed offset from P2.3/P2.4's 0) so clusters are independent.
    tasks = [(k, sn, SCENARIOS[sn], m, base_seed + c)
             for sn in SCENARIOS for k in CONTROLS for m in ms for c in range(clusters)]
    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=n_jobs)(
            delayed(_one_cell)(k, sn, sp, m, s, n_subjects, n_boot) for k, sn, sp, m, s in tasks)
    else:
        recs = [_one_cell(k, sn, sp, m, s, n_subjects, n_boot) for k, sn, sp, m, s in tasks]

    decision_ms = [m for m in ms if m >= 20]
    cells = {}
    for sn in SCENARIOS:
        for k in CONTROLS:
            for m in ms:
                rs = [r for r in recs if r["scenario"] == sn and r["kind"] == k
                      and r["label_budget_m"] == m]
                n = len(rs); fc = sum(r["state"] == CONCEPT_CONFIRMED for r in rs)
                states = {}
                for r in rs:
                    states[r["state"]] = states.get(r["state"], 0) + 1
                cells[f"{sn}|{k}|m{m}"] = dict(
                    scenario=sn, kind=k, m=m, n=n, false_confirm_count=int(fc),
                    false_confirm_cp_upper=(_cp_bound(fc, n, side="upper") if n else 1.0),
                    need_more_rate=states.get(NEED_MORE_LABELS, 0) / n if n else None,
                    invalid_pair_rate=states.get(INVALID_PAIR, 0) / n if n else None,
                    state_counts=states)

    def pooled(pred):
        rs = [r for r in recs if r["label_budget_m"] in decision_ms and pred(r)]
        n = len(rs); fc = sum(r["state"] == CONCEPT_CONFIRMED for r in rs)
        return dict(n=n, fc=fc, rate=(fc / n if n else None),
                    cp_upper=(_cp_bound(fc, n, side="upper") if n else 1.0))
    pooled_all = pooled(lambda r: True)
    by_kind = {k: pooled(lambda r, k=k: r["kind"] == k) for k in CONTROLS}
    by_scen = {s: pooled(lambda r, s=s: r["scenario"] == s) for s in SCENARIOS}
    by_budget = {m: pooled(lambda r, m=m: r["label_budget_m"] == m) for m in decision_ms}

    # ---- automated red-flags (reviewer's hard-fail / warning rules; reviewer makes the final call) ----
    dcells = {c: v for c, v in cells.items() if v["m"] in decision_ms}
    hard = []
    for k in ("missing_pair", "unequal_epochs_extreme"):
        leak = [c for c, v in dcells.items() if v["kind"] == k and v["false_confirm_count"] > 0]
        if leak:
            hard.append(f"{k} CONFIRM leak in {leak}")
    big = [c for c, v in dcells.items() if v["false_confirm_count"] >= HARD_FAIL_CELL_FC]
    if big:
        hard.append(f"cell >= {HARD_FAIL_CELL_FC}/48 in {big}")
    for k in ("random_label", "paired_label", "paired_covariate_plus_label"):
        hot = [c for c, v in dcells.items() if v["kind"] == k and v["false_confirm_count"] >= 3]
        if len(hot) >= 2:
            hard.append(f"{k} kind-concentrated elevation in {hot}")
    for k in ("clean", "paired_covariate"):
        rep = [c for c, v in dcells.items() if v["kind"] == k and v["false_confirm_count"] >= 1]
        if len(rep) >= 3:
            hard.append(f"{k} repeated elevation in {rep}")
    warnings_ = [c for c, v in dcells.items() if 1 <= v["false_confirm_count"] <= HARD_FAIL_CELL_FC - 1]

    payload = dict(kind="CSC Route B3-P2.4a CONTROL-RESOLUTION (pc_centered_calibrated; 48 clusters)",
                   status="DEVELOPMENT diagnostic only -- 48/cell is NOT error control; NO freeze/confirmatory; NO real EEG.",
                   method_lock=METHOD_LOCK, calibration_version=CALIBRATION_VERSION, controls=CONTROLS,
                   scenarios=list(SCENARIOS), clusters_per_cell=clusters, label_budgets=list(ms),
                   decision_budgets=decision_ms, n_subjects=n_subjects, n_boot=n_boot, base_seed=base_seed,
                   hard_fail_cell_fc=HARD_FAIL_CELL_FC, pooled_all=pooled_all, by_kind=by_kind,
                   by_scenario=by_scen, by_budget=by_budget, hard_flags=hard, warning_cells=warnings_,
                   cells=cells, per_cluster=recs)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[b3p24a] wrote {out}")
    # console
    print(f"=== B3-P2.4a CONTROL RESOLUTION (48/cell; decision m={decision_ms}) ===")
    print(f"pooled ALL controls: {pooled_all['fc']}/{pooled_all['n']} = {pooled_all['rate']:.4f} "
          f"(CP-up {pooled_all['cp_upper']:.4f})   [alpha=0.05]")
    print("by-kind pooled false-confirm (m>=20):")
    for k in CONTROLS:
        b = by_kind[k]; print(f"  {k:26s} {b['fc']:>3d}/{b['n']:<4d} = {b['rate']:.4f} (CP-up {b['cp_upper']:.3f})")
    print("by-scenario pooled:", {s: f"{by_scen[s]['fc']}/{by_scen[s]['n']}" for s in SCENARIOS})
    print("worst control cells (m>=20):")
    for c, v in sorted(dcells.items(), key=lambda kv: -kv[1]["false_confirm_count"])[:8]:
        if v["false_confirm_count"] > 0:
            print(f"  {c:48s} {v['false_confirm_count']}/48 CP-up={v['false_confirm_cp_upper']:.3f}")
    print(f"HARD-FAIL flags: {hard if hard else 'NONE'}")
    print(f"warning cells (1..{HARD_FAIL_CELL_FC - 1}): {len(warnings_)}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="CSC Route B3-P2.4a control-resolution (pc_centered_calibrated).")
    ap.add_argument("--clusters", type=int, default=48)
    ap.add_argument("--ms", type=int, nargs="+", default=[0, 20, 30])
    ap.add_argument("--n_subjects", type=int, default=36)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--base_seed", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--out", type=str, default="csc/results/b3_p24a_controls.json")
    a = ap.parse_args()
    run(clusters=(4 if a.canary else a.clusters), ms=tuple(a.ms), n_subjects=a.n_subjects,
        n_boot=a.n_boot, base_seed=a.base_seed, n_jobs=a.jobs,
        out=(a.out.replace(".json", "_canary.json") if a.canary else a.out))


if __name__ == "__main__":
    main()
