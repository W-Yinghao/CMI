"""Phase 1.3.2 -- operating frontier (task tolerance delta_Y x sample size) + ORACLE EFFICIENCY.

For each (delta_Y, n_eff, shape) cell, calibrate matched explaining-away controls at exact Bayes
effects {0, delta_Y} (null false-positive + boundary) and measure the detection power of BOTH:
  - the deployed nested CRITIC (probe_task_gain_ucb > delta_Y), and
  - the ORACLE info-density detector (true per-sample log p(y|u,n)-log p(y|u); the information
    limit at this n).
power_ok (critic) / oracle_ok = one-sided Wilson LCB of detection at the boundary >= 1-beta=0.80
(an 80% certified-power LOWER BOUND; needs ~28/30 detections at R=30 -- NOT a 90% point target).

Reading the frontier:
  oracle_ok & not critic_ok  -> ESTIMATOR bottleneck (pursue a lower-MDE estimator)
  neither ok                 -> intrinsic small-effect sample-complexity at this (delta_Y, n)
  both ok (+ null FP low)    -> candidate operating point (delta_Y must still be task-justified)

  python -m tos_cmi.run_frontier --cell "0.05,6000,23,1"     # one cell (parallel fan-out)
  python -m tos_cmi.run_frontier --merge
"""
import argparse
import glob
import json
import os
import numpy as np
import torch

torch.set_num_threads(1)

from dataclasses import replace
from tos_cmi.score_fisher import ScoreFisherConfig
from tos_cmi.eval.power_certificate import estimate_power, wilson_lcb, assert_power_feasible

FRONT_DIR = "tos_cmi/results/frontier_cells"
CALIB_SEED, N_DOM, BASE_SEP, SIGMA, R, BETA = 9001, 6, 1.5, 1.0, 30, 0.2


def run_cell(delta_Y, n_eff, d_base, d_extra, n_cls=3, light=False):
    assert_power_feasible(R, BETA)
    cfg = replace(ScoreFisherConfig(), task_protect=True, n_perm_null=2, delta_Y=delta_Y)
    if light:                                                # first-look config (plug-in is heavy);
        cfg = replace(cfg, task_gate_hidden=128, task_gate_epochs=300,  # if it clears HERE the full
                      task_gate_folds=3, task_gate_restarts=1)          # 256/600/5/3 only does better
    base = CALIB_SEED + int(1000 * delta_Y) + 7 * n_eff + d_base + d_extra
    out = {"delta_Y": delta_Y, "n_eff": n_eff, "d_base": d_base, "d_extra": d_extra, "n_cls": n_cls}
    for mult, tag in [(0.0, "null"), (1.0, "boundary")]:
        r = estimate_power(mult * delta_Y, d_base, d_extra, n_eff, n_cls, N_DOM, BASE_SEP, SIGMA,
                           cfg, R=R, seed=base, geom_seed=base + 13)
        out[tag] = r
    b = out["boundary"]
    out["critic_ok"] = bool(b["lcb"] >= 1 - BETA)
    out["oracle_ok"] = bool(b["lcb_oracle"] >= 1 - BETA)
    out["verdict"] = ("BOTH_OK" if out["critic_ok"] and out["oracle_ok"] else
                      "ESTIMATOR_BOTTLENECK" if out["oracle_ok"] else "INTRINSIC_HARD")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="", help="'delta_Y,n_eff,d_base,d_extra'")
    ap.add_argument("--light", action="store_true", help="first-look config (128/300/3/1)")
    ap.add_argument("--outdir", default="", help="cell dir (default frontier_cells); isolate sweeps")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default="tos_cmi/results/frontier.json")
    args = ap.parse_args()
    outdir = args.outdir or FRONT_DIR
    if args.cell:
        dY, ne, db, de = args.cell.split(",")
        c = run_cell(float(dY), int(ne), int(db), int(de), light=args.light)
        os.makedirs(outdir, exist_ok=True)
        path = "%s/cell_%s_%s_%s_%s.json" % (outdir, dY, ne, db, de)
        with open(path, "w") as f:
            json.dump(c, f, indent=1)
        nb, bo = c["boundary"], c["boundary"]
        print("dY=%.3f n=%d (%dx%d) | boundary critic %d/%d(lcb %.2f) oracle %d/%d(lcb %.2f) "
              "null critic %d/%d | delta_real=%.4f -> %s"
              % (c["delta_Y"], c["n_eff"], c["d_base"], c["d_extra"], bo["det"], bo["used"],
                 bo["lcb"], bo["det_oracle"], bo["used"], bo["lcb_oracle"], c["null"]["det"],
                 c["null"]["used"], bo["delta_real"], c["verdict"]), flush=True)
        print("FRONTIER_CELL_DONE"); return
    if args.merge:
        cells = [json.load(open(p)) for p in sorted(glob.glob("%s/cell_*.json" % outdir))]
        with open(args.out, "w") as f:
            json.dump({"meta": {"R": R, "beta": BETA, "base_sep": BASE_SEP}, "cells": cells}, f, indent=1)
        print("merged %d cells -> %s" % (len(cells), args.out))
        for c in sorted(cells, key=lambda c: (c["d_extra"], c["delta_Y"])):
            b = c["boundary"]
            print("dY=%.3f n=%d k=%d | critic_lcb=%.2f oracle_lcb=%.2f nullFP=%d/%d %s"
                  % (c["delta_Y"], c["n_eff"], c["d_extra"], b["lcb"], b["lcb_oracle"],
                     c["null"]["det"], c["null"]["used"], c["verdict"]))
        print("FRONTIER_DONE"); return


if __name__ == "__main__":
    main()
