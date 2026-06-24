"""Phase 1.3.1b -- build the OFFLINE task-gate competence table (the power floor).

For a grid of prefix shapes (n_eff x d_base x d_extra x n_cls) matched to what the gate sees,
estimate MDE_k(1-beta) with matched positive controls and record power_ok = MDE <= delta_Y. The
table is built on seeds DISJOINT from the Phase 1.3 evaluation grid (pre-registration), saved to
JSON, and the gate does a CONSERVATIVE lookup (uncovered -> power NOT ok -> abstain).

  python -m tos_cmi.run_power_table --smoke
  python -m tos_cmi.run_power_table
"""
import argparse
import json
import os
import numpy as np
import torch

torch.set_num_threads(1)

from dataclasses import replace
from tos_cmi.score_fisher import ScoreFisherConfig
from tos_cmi.eval.power_certificate import prefix_mde

# disjoint-from-eval calibration seed base (eval grid uses seeds 0,1,2)
CALIB_SEED = 9001


def build(n_effs, shapes, n_clss, n_dom, base_sep, sigma, cfg, R, beta):
    """shapes = list of (d_base, d_extra) prefix shapes (matched to z_dim - k, k)."""
    table = []
    for n_cls in n_clss:
        for (d_base, d_extra) in shapes:
            for n_eff in n_effs:
                r = prefix_mde(d_base, d_extra, n_eff, n_cls, n_dom, base_sep, sigma, cfg,
                               R=R, beta=beta, seed=CALIB_SEED + 7 * n_eff + d_base + d_extra)
                key = {"n_eff": n_eff, "d_base": d_base, "d_extra": d_extra, "n_cls": n_cls}
                table.append({**key, "mde": r["mde"], "power_ok": r["power_ok"], "rows": r["rows"]})
                print("n_eff=%-6d d_base=%d d_extra=%d n_cls=%d | MDE=%s power_ok=%s  pi@grid=%s"
                      % (n_eff, d_base, d_extra, n_cls, r["mde"], r["power_ok"],
                         [round(x["lcb"], 2) for x in r["rows"]]), flush=True)
    return table


CELL_DIR = "tos_cmi/results/power_cells"
# R_FULL=30: the Wilson LCB ceiling R/(R+z^2)=30/32.69=0.918 >= 1-beta=0.8 (R=8 was a BUG: its
# ceiling 0.748 < 0.8 made power_ok unreachable at any n). Do NOT lower R below ~12.
BASE_SEP, SIGMA, N_DOM, R_FULL, BETA = 1.5, 1.0, 6, 30, 0.2


def _meta(cfg):
    from tos_cmi.score_fisher import estimator_fingerprint
    return {"delta_Y": cfg.delta_Y, "beta": BETA, "calib_seed": CALIB_SEED, "base_sep": BASE_SEP,
            "sigma": SIGMA, "family": "gaussian_explaining_away",
            "fingerprint": estimator_fingerprint(cfg)}      # table valid ONLY for this estimator


def run_one_cell(cell, cfg, R=R_FULL, targets=None):
    """cell = (n_eff, d_base, d_extra, n_cls). Build one cell -> dict (for parallel fan-out).
    Both the sample seed and the geometry seed are DISJOINT from the eval grid (CALIB_SEED-based).
    `targets` (multipliers of delta_Y) lets the audit measure only the delta_Y target (1.0), which
    alone decides power_ok (MDE<=delta_Y <=> LCB(pi(delta_Y))>=1-beta)."""
    n_eff, d_base, d_extra, n_cls = cell
    base = CALIB_SEED + 7 * n_eff + d_base + d_extra
    grid = None if targets is None else [m * cfg.delta_Y for m in targets]
    r = prefix_mde(d_base, d_extra, n_eff, n_cls, N_DOM, BASE_SEP, SIGMA, cfg,
                   R=R, beta=BETA, seed=base, geom_seed=base + 13, grid=grid)
    return {"n_eff": n_eff, "d_base": d_base, "d_extra": d_extra, "n_cls": n_cls,
            "mde": r["mde"], "power_ok": r["power_ok"], "rows": r["rows"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cell", default="", help="'n_eff,d_base,d_extra,n_cls' -> build one cell")
    ap.add_argument("--targets", default="", help="comma multipliers of delta_Y (e.g. '1.0' audit)")
    ap.add_argument("--merge", action="store_true", help="merge results/power_cells/*.json -> --out")
    ap.add_argument("--out", default="tos_cmi/results/power_table.json")
    args = ap.parse_args()
    cfg = replace(ScoreFisherConfig(), task_protect=True, n_perm_null=2)

    if args.cell:                                            # PARALLEL: one cell per SLURM job
        cell = tuple(int(x) for x in args.cell.split(","))
        targets = [float(x) for x in args.targets.split(",")] if args.targets else None
        c = run_one_cell(cell, cfg, targets=targets)
        os.makedirs(CELL_DIR, exist_ok=True)
        path = "%s/cell_%d_%d_%d_%d.json" % (CELL_DIR, *cell)
        with open(path, "w") as f:
            json.dump({"meta": _meta(cfg), **c}, f, indent=1)
        dets = ["%d/%d@%.3f" % (x["det"], x["used"], x["delta_real"]) for x in c["rows"]]
        print("n_eff=%d d_base=%d d_extra=%d n_cls=%d | MDE=%s power_ok=%s det=%s lcb=%s -> %s"
              % (*cell, c["mde"], c["power_ok"], dets, [round(x["lcb"], 2) for x in c["rows"]],
                 path), flush=True)
        print("POWER_CELL_DONE"); return

    if args.merge:                                           # combine parallel cells -> table
        import glob
        cells = []
        for p in sorted(glob.glob("%s/*.json" % CELL_DIR)):
            with open(p) as f:
                d = json.load(f)
            cells.append({k: d[k] for k in ("n_eff", "d_base", "d_extra", "n_cls",
                                            "mde", "power_ok", "rows")})
        with open(args.out, "w") as f:
            json.dump({"meta": _meta(cfg), "table": cells}, f, indent=1)
        print("merged %d cells -> %s (%d power_ok)" %
              (len(cells), args.out, sum(c["power_ok"] for c in cells)))
        print("POWER_TABLE_DONE"); return
    # shapes matched to the synthetic z_dim=24: k=1 -> (23,1), k=2 -> (22,2), k=3 -> (21,3)
    if args.smoke:
        table = build([2000, 6000], [(23, 1), (22, 2)], [3], n_dom=6, base_sep=1.5, sigma=1.0,
                      cfg=cfg, R=15, beta=0.2)
    else:
        table = build(n_effs=[1500, 3000, 6000, 12000, 24000], shapes=[(23, 1), (22, 2), (21, 3)],
                      n_clss=[3], n_dom=6, base_sep=1.5, sigma=1.0, cfg=cfg, R=R_FULL, beta=0.2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": _meta(cfg), "table": table}, f, indent=1)
    n_ok = sum(t["power_ok"] for t in table)
    print("\nwrote %s : %d cells, %d power_ok" % (args.out, len(table), n_ok))
    print("POWER_TABLE_DONE")


if __name__ == "__main__":
    main()
