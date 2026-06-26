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
from tos_cmi.eval.phase_diagram import run_cell
from tos_cmi.data.synthetic import make_partial_synergy, make_partial_factorized


def _binom_ucb(k, n, z=1.64):
    """One-sided Wilson UPPER bound on a binomial rate k/n (for 'zero observed' UNSAFE_ACCEPT)."""
    if n == 0:
        return 1.0
    p = k / n; denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + half) / denom)

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


def _decision_sweep(gen, delta_Y, table, seed_base, n_seeds, candidate_modes):
    """Drive the FULL gate (cert table, exact mode, power floor ON) over n_seeds independent
    datasets; tally Bayes-cross-classified decisions per prefix. Returns the class counts +
    the (gen-natural) Bayes delta range of the deleted span."""
    cfg = replace(_cfg(delta_Y), task_power_floor=True, task_power_table=table)
    from collections import Counter
    cnt = Counter(); bayes_vals = []; reasons = Counter()
    for r in range(n_seeds):
        data = gen(n=6000, seed=seed_base + r)
        recs = run_cell(data, cfg, seed=seed_base + r, candidate_modes=candidate_modes,
                        t_sources=("oracle", "learned"), n_mc=12000)
        for rec in recs:
            cnt[rec["class"]] += 1
            reasons[rec.get("decision_reason")] += 1
            if rec.get("bayes_delta") is not None:
                bayes_vals.append(rec["bayes_delta"])
    return dict(cnt), dict(reasons), (min(bayes_vals, default=0.0), max(bayes_vals, default=0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", default="", help="'delta_Y,n_eff,d_base,d_extra,n_cls' (G2)")
    ap.add_argument("--inject", default="", help="delta_Y (G3: oracle-injected unsafe synergy grid)")
    ap.add_argument("--safe", default="", help="delta_Y (G4: factorized safe controls)")
    ap.add_argument("--table", default="tos_cmi/results/cert_table.json", help="G1 cert table")
    ap.add_argument("--n-seeds", type=int, default=30)
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

    if args.inject:                                          # G3: zero-UNSAFE_ACCEPT on injection
        dY = float(args.inject)
        cnt, reasons, brange = _decision_sweep(make_partial_synergy, dY, args.table, G3,
                                               args.n_seeds, candidate_modes=("oracle_nuisance",))
        total = sum(cnt.values()); ua = cnt.get("UNSAFE_ACCEPT", 0)
        out = {"mode": "inject", "delta_Y": dY, "n_seeds": args.n_seeds, "classes": cnt,
               "reasons": reasons, "bayes_delta_range": brange, "unsafe_accept": ua,
               "unsafe_accept_rate_ucb": _binom_ucb(ua, total)}
        with open(args.out or "%s/inject_%s.json" % (CERT_DIR, args.inject), "w") as f:
            json.dump(out, f, indent=1)
        print("INJECT dY=%.3f | classes=%s | UNSAFE_ACCEPT=%d/%d (rate UCB %.3f) | bayes_range=%s"
              % (dY, cnt, ua, total, out["unsafe_accept_rate_ucb"],
                 tuple(round(x, 3) for x in brange)), flush=True)
        print("CERT_INJECT_DONE"); return

    if args.safe:                                            # G4: non-degenerate SAFE_ACCEPT
        dY = float(args.safe)
        cnt, reasons, brange = _decision_sweep(make_partial_factorized, dY, args.table, G4,
                                               args.n_seeds, candidate_modes=("learned", "oracle_nuisance"))
        total = sum(cnt.values()); sa = cnt.get("SAFE_ACCEPT", 0)
        out = {"mode": "safe", "delta_Y": dY, "n_seeds": args.n_seeds, "classes": cnt,
               "reasons": reasons, "bayes_delta_range": brange, "safe_accept": sa,
               "safe_accept_rate": sa / max(total, 1), "safe_accept_lcb": wilson_lcb(sa, total)}
        with open(args.out or "%s/safe_%s.json" % (CERT_DIR, args.safe), "w") as f:
            json.dump(out, f, indent=1)
        print("SAFE dY=%.3f | classes=%s | SAFE_ACCEPT=%d/%d (rate %.2f, LCB %.2f) | non-degenerate=%s"
              % (dY, cnt, sa, total, out["safe_accept_rate"], out["safe_accept_lcb"],
                 out["safe_accept_lcb"] >= 0.5), flush=True)
        print("CERT_SAFE_DONE"); return


if __name__ == "__main__":
    main()
