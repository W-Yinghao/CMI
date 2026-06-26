"""Phase 1.3.4 -- Independent-Seed Decision-Level Certification of the plug-in task gate.

Four DISJOINT seed groups (none used in any prior frontier / first-look / deployment run):
  G1 = 20000  power-table calibration   (run_power_table --calib-base 20000 --outdir cert_table_cells)
  G2 = 30000  held-out BOUNDARY confirmation   (--confirm here)
  G3 = 40000  oracle-injected UNSAFE grid       (--inject here)
  G4 = 50000  factorized SAFE controls          (--safe here)
All at R=50 (one-sided Wilson: need >=45/50 for LCB>=0.80) and the FROZEN deployment config
(tag plugin-logratio-v1-cert-candidate: hidden=256, epochs=600, folds=5, restarts=3, alpha grid,
floor, bootstrap, delta thresholds, generators, criteria -- NOT to be changed from confirmatory
results).

Exit conditions (per cell; NO extrapolation):
  boundary  : Δ*=δ_Y -> plug-in det>=45/50 (LCB>=0.80) AND oracle passes.
  unsafe    : Δ*∈{1.25,2}δ_Y, single/multi/mixed dirs, oracle-T & learned-T -> 0 UNSAFE_ACCEPT
              (report one-sided binomial UCB; zero-obs is NOT a distribution-free guarantee).
  safe      : factorized Δ*∈{0,0.5δ_Y} -> non-degenerate SAFE_ACCEPT: LCB[P(SAFE_ACCEPT)]>=0.5
              AND the accepted projector REDUCES conditional-domain advantage.
  null      : Δ*=0 -> report plug-in est/UCB, accept rate, alpha dist, q0/q1 outer NLL, clip, max|lr|.

  python -m tos_cmi.run_cert --confirm "0.10,6000,23,1"
  python -m tos_cmi.run_cert --safe    "0.10,6000,22,2" --table tos_cmi/results/cert_table.json
"""
import argparse
import json
import os
import numpy as np
import torch

torch.set_num_threads(1)

from dataclasses import replace
from tos_cmi.score_fisher import ScoreFisherConfig, ucb_rank_gate, _metric, _m_orthonormal
from tos_cmi.eval.power_certificate import (estimate_power, wilson_lcb, assert_power_feasible,
                                            make_control, tune_confound, _control_geometry)
from tos_cmi.eval.bayes_oracle import bayes_conditional_task_delta, classify_safety

G1, G2, G3, G4 = 20000, 30000, 40000, 50000
R_CERT, BETA, N_DOM, BASE_SEP, SIGMA = 50, 0.2, 6, 1.5, 1.0
CERT_DIR = "tos_cmi/results/cert_cells"


def _cfg(delta_Y):
    """FROZEN deployment config (cert-candidate). task_protect on; defaults 256/600/5/3."""
    return replace(ScoreFisherConfig(), task_protect=True, n_perm_null=2, delta_Y=delta_Y,
                   certificate_lookup="exact", certified_mode=True)


def confirm_boundary(delta_Y, n_eff, d_base, d_extra, n_cls):
    """G2: independent-seed boundary confirmation -- plug-in (and oracle) detection at Δ*=δ_Y."""
    assert_power_feasible(R_CERT, BETA)
    cfg = _cfg(delta_Y)
    base = G2 + 7 * n_eff + d_base + d_extra
    r = estimate_power(delta_Y, d_base, d_extra, n_eff, n_cls, N_DOM, BASE_SEP, SIGMA, cfg,
                       R=R_CERT, seed=base, geom_seed=base + 13)
    return {"mode": "confirm", "delta_Y": delta_Y, "n_eff": n_eff, "d_base": d_base,
            "d_extra": d_extra, "n_cls": n_cls, "R": R_CERT,
            "plugin_det": r["det"], "plugin_lcb": r["lcb"], "oracle_det": r["det_oracle"],
            "oracle_lcb": r["lcb_oracle"], "delta_real": r["delta_real"],
            "plugin_pass": bool(r["lcb"] >= 1 - BETA), "oracle_pass": bool(r["lcb_oracle"] >= 1 - BETA)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", default="", help="'delta_Y,n_eff,d_base,d_extra,n_cls' (G2)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.makedirs(CERT_DIR, exist_ok=True)
    if args.confirm:
        dY, ne, db, de, nc = args.confirm.split(",")
        r = confirm_boundary(float(dY), int(ne), int(db), int(de), int(nc))
        path = args.out or "%s/confirm_%s_%s_%s_%s_%s.json" % (CERT_DIR, dY, ne, db, de, nc)
        with open(path, "w") as f:
            json.dump(r, f, indent=1)
        print("CONFIRM dY=%.3f n=%d (%dx%d) | plug-in %d/%d (LCB %.2f, pass=%s) | oracle %d/%d "
              "(LCB %.2f) | delta_real=%.4f -> %s"
              % (r["delta_Y"], r["n_eff"], r["d_base"], r["d_extra"], r["plugin_det"], R_CERT,
                 r["plugin_lcb"], r["plugin_pass"], r["oracle_det"], R_CERT, r["oracle_lcb"],
                 r["delta_real"], path), flush=True)
        print("CERT_CONFIRM_DONE"); return


if __name__ == "__main__":
    main()
