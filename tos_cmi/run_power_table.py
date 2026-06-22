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
BASE_SEP, SIGMA, N_DOM, R_FULL, BETA = 1.5, 1.0, 6, 8, 0.2


def _meta(cfg):
    return {"delta_Y": cfg.delta_Y, "beta": BETA, "task_gate_hidden": cfg.task_gate_hidden,
            "task_gate_epochs": cfg.task_gate_epochs, "task_gate_restarts": cfg.task_gate_restarts,
            "n_folds": cfg.n_folds, "gate_boot": cfg.gate_boot, "calib_seed": CALIB_SEED,
            "base_sep": BASE_SEP, "sigma": SIGMA, "family": "gaussian_explaining_away"}


def run_one_cell(cell, cfg, R=R_FULL):
    """cell = (n_eff, d_base, d_extra, n_cls). Build one cell -> dict (for parallel fan-out)."""
    n_eff, d_base, d_extra, n_cls = cell
    r = prefix_mde(d_base, d_extra, n_eff, n_cls, N_DOM, BASE_SEP, SIGMA, cfg,
                   R=R, beta=BETA, seed=CALIB_SEED + 7 * n_eff + d_base + d_extra)
    return {"n_eff": n_eff, "d_base": d_base, "d_extra": d_extra, "n_cls": n_cls,
            "mde": r["mde"], "power_ok": r["power_ok"], "rows": r["rows"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cell", default="", help="'n_eff,d_base,d_extra,n_cls' -> build one cell")
    ap.add_argument("--merge", action="store_true", help="merge results/power_cells/*.json -> --out")
    ap.add_argument("--out", default="tos_cmi/results/power_table.json")
    args = ap.parse_args()
    cfg = replace(ScoreFisherConfig(), task_protect=True, n_perm_null=2)

    if args.cell:                                            # PARALLEL: one cell per SLURM job
        cell = tuple(int(x) for x in args.cell.split(","))
        c = run_one_cell(cell, cfg)
        os.makedirs(CELL_DIR, exist_ok=True)
        path = "%s/cell_%d_%d_%d_%d.json" % (CELL_DIR, *cell)
        with open(path, "w") as f:
            json.dump({"meta": _meta(cfg), **c}, f, indent=1)
        print("n_eff=%d d_base=%d d_extra=%d n_cls=%d | MDE=%s power_ok=%s pi=%s -> %s"
              % (*cell, c["mde"], c["power_ok"], [round(x["lcb"], 2) for x in c["rows"]], path),
              flush=True)
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
                      cfg=cfg, R=4, beta=0.2)
    else:
        table = build(n_effs=[1500, 3000, 6000, 12000, 24000], shapes=[(23, 1), (22, 2), (21, 3)],
                      n_clss=[3], n_dom=6, base_sep=1.5, sigma=1.0, cfg=cfg, R=8, beta=0.2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta = {"delta_Y": cfg.delta_Y, "beta": 0.2, "task_gate_hidden": cfg.task_gate_hidden,
            "task_gate_epochs": cfg.task_gate_epochs, "task_gate_restarts": cfg.task_gate_restarts,
            "n_folds": cfg.n_folds, "gate_boot": cfg.gate_boot, "calib_seed": CALIB_SEED,
            "family": "gaussian_explaining_away"}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "table": table}, f, indent=1)
    n_ok = sum(t["power_ok"] for t in table)
    print("\nwrote %s : %d cells, %d power_ok" % (args.out, len(table), n_ok))
    print("POWER_TABLE_DONE")


if __name__ == "__main__":
    main()
